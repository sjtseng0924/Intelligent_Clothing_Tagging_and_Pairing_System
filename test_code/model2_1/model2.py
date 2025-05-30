# model2.py
import torch
import torch.nn as nn

class CompatibilityNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, p_top: torch.Tensor, p_bot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_top: Tensor of shape [B, D]
            p_bot: Tensor of shape [B, D]
        Returns:
            raw logits Tensor of shape [B, 1]
        """
        # 特徵合併 (串接)
        x = torch.cat([p_top, p_bot], dim=1)  # → [B, 2*D]
        logits = self.mlp(x)                 # → [B, 1]
        return logits
