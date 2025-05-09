import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load the model
        model_path = os.path.join("model", "model.pth")
        model = torch.load(model_path, map_location=torch.device('cpu'))  # use 'cuda' if GPU available
        model.eval()

        # Define image transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # use appropriate mean/std for your model
                                 std=[0.229, 0.224, 0.225])
        ])

        # Load and preprocess image
        img = Image.open(self.filename).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            outputs = model(img)
            result = torch.argmax(outputs, dim=1).item()

        # Interpret prediction
        if result == 1:
            prediction = 'Benign'
        else:
            prediction = 'Malignant'

        return [{"image": prediction}]
