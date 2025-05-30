import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class ColorClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.base(x))
