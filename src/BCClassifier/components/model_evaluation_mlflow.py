import torch 
import torch.nn as nn
import torch.optim as optim
from pathlib import Path 
from urllib.parse import urlparse 
import mlflow
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from BCClassifier.entity.config_entity import EvaluationConfig,PrepareBaseModelConfig
from BCClassifier.utils.common import read_yaml, create_directories, save_json
from BCClassifier.components.prepare_base_model import PrepareBaseModel
import dagshub


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _valid_generator(self):
        """Set up the validation data generator"""
        # Define transformations (rescale by 1./255 is handled by ToTensor)
        transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),  # (H, W)
            transforms.ToTensor(),
        ])

        # Load dataset from directory
        try:
            full_dataset = datasets.ImageFolder(
                root=self.config.training_data,
                transform=transform
            )
            
            # Split 70% train, 30% validation
            val_size = int(0.01 * len(full_dataset)/2)
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
            
            print(f"Validation dataset size: {val_size}")
            
        except Exception as e:
            print(f"Error setting up validation data: {e}")
            raise e

    def load_model(self, path) -> torch.nn.Module:
        """Load a PyTorch model from the given path
        
        Args:
            path: Can be either a string path or a Path object
        """
        # Convert to Path object if it's a string
        if isinstance(path, str):
            path = Path(path)
            
        if not path.exists():
            print(f"Warning: Model path {path} does not exist")
            raise FileNotFoundError(f"Model not found at: {path}")
            
        # First, create a base model instance
        prepare_base_model = PrepareBaseModel(config=self.config)
        base_model = prepare_base_model.get_base_model()

        # Load the state dict
        try:
            print(f"Attempting to load model from: {path}")
            # Try loading as full model first
            model = torch.load(path, map_location=self.device)
            if isinstance(model, nn.Module):
                print("Loaded complete model")
                return model.to(self.device)
            
            # If it's a state dict or OrderedDict, load it into the base model
            elif isinstance(model, dict) or hasattr(model, 'items'):
                print("Loaded model state dictionary")
                base_model.load_state_dict(model)
                return base_model.to(self.device)
                
        except Exception as e:
            print(f"Error loading model: {e}")
            try:
                # Try loading as state dict only
                state_dict = torch.load(path, map_location=self.device)
                base_model.load_state_dict(state_dict)
                print("Loaded model using fallback method")
                return base_model.to(self.device)
            except Exception as e2:
                print(f"Failed to load model as state dict: {e2}")
                raise e2
    
    def evaluation(self):
        """Evaluate the model on validation data"""
        try:
            self.model = self.load_model(self.config.path_to_model)
            self._valid_generator()

            self.model.eval()  # Set model to evaluation mode
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
            
            print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            self.score = (avg_loss, accuracy)
            self.save_score()
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise e

    def save_score(self):
        """Save evaluation scores to a JSON file"""
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
        print(f"Scores saved to scores.json")

    def log_into_mlflow(self):
        """Log parameters, metrics, and model to MLFlow"""
        try:
            # Initialize Dagshub repository for MLflow
            dagshub.init(
                repo_owner='Adity-star',
                repo_name='End-to-End-Breast-Cancer-Classification-Using-DVC-and-MLflow',
                mlflow=True
            )
            
            # MLflow will use the Dagshub tracking URI automatically
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                # Log parameters from the config
                mlflow.log_params(self.config.all_params)
                
                # Log the metrics (loss, accuracy, etc.)
                mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

                # Log the model (either to file or remote)
                if tracking_url_type_store != "file":
                    mlflow.pytorch.log_model(self.model, "model", registered_model_name="VGG16Model")
                else:
                    mlflow.pytorch.log_model(self.model, "model")

            print("Model logged to MLflow successfully")
        
        except Exception as e:
            print(f"Error logging to MLflow: {e}")
            raise e