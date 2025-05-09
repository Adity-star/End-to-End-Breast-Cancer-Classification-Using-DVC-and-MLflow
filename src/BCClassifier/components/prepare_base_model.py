import os
import urllib.request as request
from zipfile import ZipFile
from pathlib import Path 
import torch
import torch.nn as nn
from torchvision import models
from BCClassifier.entity.config_entity import PrepareBaseModelConfig



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None

    def get_base_model(self):
        # Load pretrained VGG16
        self.model = models.vgg16(weights='IMAGENET1K_V1' if self.config.params_weights == 'imagenet' else None)
        
        # Remove classifier layers if include_top is False
        if not self.config.params_include_top:
            self.model.classifier = nn.Identity()

        # Save base model
        self._save_model(self.model, self.config.base_model_path)
        return self.model
    
    def _prepare_full_model(self, model, num_classes, freeze_all=True, freeze_till=None, learning_rate=0.01):
        # Freeze layers
        layers = list(model.features.children())

        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif freeze_till:
            for idx, layer in enumerate(layers[:-freeze_till]):
                for param in layer.parameters():
                    param.requires_grad = False

        # Replace classifier
        in_features = model.classifier[0].in_features if isinstance(model.classifier, nn.Sequential) else 512
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, num_classes)
        )
        model.classifier = classifier

        return model

    def update_base_model(self):
        if self.model is None:
            raise ValueError("Base model not loaded. Run get_base_model() first.")

        full_model = self._prepare_full_model(
            model=self.model,
            num_classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        # Save updated model
        self._save_model(full_model, self.config.update_base_model_path)
        return full_model
    
    @staticmethod
    def _save_model(model, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, path)
