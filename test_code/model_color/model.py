import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from label_train.model import AttributeResNet

class AttributeMatrixCompatibility(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_labels, num_labels) * 0.01)

    def forward(self, top_label, bottom_label):
        # top_label, bottom_label: (B, 12)
        x = torch.matmul(top_label, self.W)      # (B, 12)
        score = (x * bottom_label).sum(dim=1)    # (B,)
        return score