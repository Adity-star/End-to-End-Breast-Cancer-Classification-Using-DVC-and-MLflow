import torch 
import torch.nn as nn
import torch.optim as optim
from pathlib import Path 
from urllib.parse import urlparse 
import mlflow
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from BCClassifier.entity.config_entity import EvaluationConfig
from BCClassifier.utils.common import read_yaml, create_directories, save_json
from BCClassifier.components.prepare_base_model import PrepareBaseModel

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        # Define transformations (rescale by 1./255 is handled by ToTensor)
        transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),  # (H, W)
            transforms.ToTensor(),
        ])

        # Load dataset from directory
        full_dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=transform
        )

        # Split 70% train, 30% validation
        val_size = int(0.3 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        _, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Validation DataLoader
        self.valid_generator = DataLoader(
            val_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    @staticmethod
    def load_model(path: Path) -> torch.nn.Module:
        return torch.load(path)

    def evaluation(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(self.config.path_to_model)
        self._valid_generator()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.valid_generator:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        self.score = (avg_loss, accuracy)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.pytorch.log_model(self.model, "model")

