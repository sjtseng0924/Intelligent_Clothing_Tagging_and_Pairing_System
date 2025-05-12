import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class AttributeResNet(nn.Module):
    def __init__(self, num_labels=1000):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_labels)

    def forward(self, x):
        return self.backbone(x)
