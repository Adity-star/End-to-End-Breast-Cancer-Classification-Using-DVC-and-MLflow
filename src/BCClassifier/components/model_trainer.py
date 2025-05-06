import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
from BCClassifier.entity.config_entity import TrainingConfig
from tqdm import tqdm
import time
import multiprocessing
from torch.cuda.amp import autocast, GradScaler
import gc

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_loader = None
        self.valid_loader = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def get_base_model(self):
        # Load the pre-trained model (VGG16 in this case)
        self.model = models.vgg16(pretrained=True)

        # Modify the classifier layer to match the number of classes
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, self.config.params_classes)

        # If include_top=False, we remove the classifier part
        if not self.config.params_include_top:
            self.model.classifier = nn.Identity()

        # Load the model to the device (CPU or GPU)
        self.model = self.model.to(self.device)
        
        # Enable cudnn benchmarking and deterministic mode for faster training
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            # Set memory allocation strategy
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
            torch.cuda.empty_cache()

    def train_valid_generator(self):
        # Define data augmentation and normalizing transforms
        transform_train = transforms.Compose([
            transforms.RandomRotation(40) if self.config.params_is_augmentation else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if self.config.params_is_augmentation else transforms.Lambda(lambda x: x),
            transforms.RandomResizedCrop(self.config.params_image_size[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_valid = transforms.Compose([
            transforms.Resize(self.config.params_image_size[0]),
            transforms.CenterCrop(self.config.params_image_size[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load datasets using separate paths
        full_train_dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=transform_train
        )
        
        # Calculate 30% of the dataset size
        train_size = int(0.1 * len(full_train_dataset))
        print(f"Using {train_size} images for training (30% of total)")
        
        # Create a subset of the training data
        indices = torch.randperm(len(full_train_dataset))[:train_size]
        train_dataset = torch.utils.data.Subset(full_train_dataset, indices)
        
        valid_dataset = datasets.ImageFolder(
            root=self.config.validation_data,
            transform=transform_valid
        )
        test_dataset = datasets.ImageFolder(
            root=self.config.test_data,
            transform=transform_valid
        )

        # Calculate optimal batch size based on GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            # For 8GB GPU, use larger batch size
            optimal_batch_size = min(128, self.config.params_batch_size)
        else:
            optimal_batch_size = self.config.params_batch_size

        # Create CPU generator for DataLoader
        generator = torch.Generator()
        generator.manual_seed(42)  # For reproducibility

        # Optimize DataLoader settings for maximum speed
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=optimal_batch_size,
            shuffle=True,
            num_workers=4,  
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=3,  
            drop_last=True,
            generator=generator
        )
        
        # Use larger batch size for validation/testing
        val_batch_size = optimal_batch_size * 2
        self.valid_loader = DataLoader(
            valid_dataset, 
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=3
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=3
        )

    def save_model(self, path: Path, model: torch.nn.Module):
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    def train(self):
        # Use an optimizer with momentum and weight decay
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.config.params_learning_rate, 
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True
        )
        
        # Learning rate scheduler with warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.params_learning_rate,
            epochs=self.config.params_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,  # Warm up for 30% of training
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        criterion = nn.CrossEntropyLoss()

        # Put model in training mode
        self.model.train()
        
        # Training loop
        for epoch in range(self.config.params_epochs):
            start_time = time.time()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Create progress bar
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.params_epochs}')
            
            for i, (inputs, labels) in enumerate(pbar):
                # Move data to GPU in a single operation
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                # Mixed precision training
                with autocast():
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                # Backward pass and optimize with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                # Update learning rate
                scheduler.step()

                # Update statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss/(i+1):.3f}',
                    'acc': f'{100*correct/total:.2f}%',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })

                # Clear memory efficiently
                del inputs, labels, outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Calculate epoch statistics
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100 * correct / total
            epoch_time = time.time() - start_time

            print(f'\nEpoch {epoch+1} Summary:')
            print(f'Time: {epoch_time:.2f}s')
            print(f'Loss: {epoch_loss:.3f}')
            print(f'Accuracy: {epoch_acc:.2f}%')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

            # Clear memory after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        # Save the final trained model
        self.save_model(path=self.config.trained_model_path, model=self.model)

    def validate(self):
        # Set the model to evaluation mode
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():  # No need to track gradients for validation
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

