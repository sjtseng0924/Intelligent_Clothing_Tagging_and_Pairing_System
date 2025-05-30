# model.py

import torch
import torch.nn as nn
import torchvision.models as models

class ColorClassifier(nn.Module):
    def __init__(self, num_classes=11):  # 7 是顏色類別數，請確認
        super(ColorClassifier, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, num_classes),
            nn.Sigmoid()  # 因為 loss 用 BCELoss，所以用 Sigmoid 輸出
        )

    def forward(self, x):
        return self.backbone(x)


class CompatibilityNet(nn.Module):
    """
    Outfit 配對模型 (CompatibilityNet)

    給定 top 和 bottom 的特徵向量，輸出是否匹配的 logits 分數。
    這是一個 MLP 架構：
        - 輸入: p_top (B, D), p_bot (B, D)
        - 輸出: logits (B, 1)

    Args:
        in_dim (int): 單一圖像特徵維度 D (e.g., ResNet 抽出來的維度)
        hidden_dim (int): 隱藏層維度，預設 512
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512):
        super(CompatibilityNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),    # 輸入為 top+bottom 串接
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim // 2, 1)         # 二元分類 (logits)
        )
        self.norm = nn.LayerNorm(in_dim * 2)

    def forward(self, p_top: torch.Tensor, p_bot: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            p_top: Tensor of shape [B, D]
            p_bot: Tensor of shape [B, D]

        Returns:
            logits: Tensor of shape [B, 1]
        """
        x = torch.cat([p_top, p_bot], dim=1)  # [B, 2D]
        x = self.norm(x)
        logits = self.mlp(x)                  # [B, 1]
        return logits


class ColorClassifier(nn.Module):
    def __init__(self, model_path='color_label/color_classifier.pt'):
        super(ColorClassifier, self).__init__()
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.eval()

    def predict(self, image_tensor):
        with torch.no_grad():
            output = self.model(image_tensor)
            predicted = torch.argmax(output, dim=1)
        return predicted.item()