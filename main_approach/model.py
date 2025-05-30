import torch
import torch.nn as nn

class AttributeMLPCompatibility(nn.Module):
    def __init__(self, num_labels, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_labels * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, top_label, bottom_label):
        # top_label, bottom_label: (B, num_labels)
        x = torch.cat([top_label, bottom_label], dim=1)  # (B, num_labels*2)
        score = self.mlp(x).squeeze(1)  # (B,)
        return score
