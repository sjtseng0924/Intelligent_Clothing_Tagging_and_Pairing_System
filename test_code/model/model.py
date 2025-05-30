import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# 你的屬性分類模型（之前訓練好的 AttributeResNet，這裡簡化為範例）
class AttributeResNet(nn.Module):
    def __init__(self, num_labels=1000):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_labels)

    def forward(self, x):
        return self.backbone(x)


class CompatibilityModel(nn.Module):
    def __init__(self, image_emb_dim=512, label_emb_dim=1000, hidden_dim=256):
        super().__init__()
        backbone = resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.fc_img = nn.Linear(image_emb_dim, hidden_dim)
        self.fc_label = nn.Linear(label_emb_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, top_img, bottom_img, top_label_emb, bottom_label_emb):
        top_img_emb = self.image_encoder(top_img).squeeze(-1).squeeze(-1)  # (B, 2048)
        bottom_img_emb = self.image_encoder(bottom_img).squeeze(-1).squeeze(-1)

        top_img_emb = self.fc_img(top_img_emb)
        bottom_img_emb = self.fc_img(bottom_img_emb)

        top_label_emb = self.fc_label(top_label_emb)
        bottom_label_emb = self.fc_label(bottom_label_emb)

        features = torch.cat([
            top_img_emb,
            bottom_img_emb,
            torch.abs(top_img_emb - bottom_img_emb),
            torch.abs(top_label_emb - bottom_label_emb)
        ], dim=1)

        score = self.classifier(features)
        return score.squeeze(1)
