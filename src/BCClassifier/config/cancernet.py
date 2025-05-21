import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CancerNet(nn.Module):
    def __init__(self, width, height, depth, classes):
        super(CancerNet, self).__init__()
        self.features = nn.Sequential(
            # Block 1: SeparableConv2D(32)
            SeparableConv2d(depth, 32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            # Block 2: SeparableConv2D(64) x2
            SeparableConv2d(32, 64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            SeparableConv2d(64, 64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            # Block 3: SeparableConv2D(128) x3
            SeparableConv2d(64, 128),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            SeparableConv2d(128, 128),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            SeparableConv2d(128, 128),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25)
        )

        # Compute the output size after convolutions using a dummy tensor
        self._to_linear = self._get_flatten_size(width, height, depth)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, classes),
            nn.Softmax(dim=1)  # softmax for multi-class classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _get_flatten_size(self, width, height, depth):
        with torch.no_grad():
            dummy = torch.zeros(1, depth, height, width)
            out = self.features(dummy)
            return out.view(1, -1).shape[1]
