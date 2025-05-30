import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torchvision.transforms as T
import matplotlib.pyplot as plt

from dataset import OutfitPairDataset
from model import CompatibilityNet
from label_train.model import AttributeResNet
from color_label.model import ColorClassifier

# ===== 參數設定 =====
cleaned_root = './Cleaned-Maryland-Dataset'
random_root  = './Random-Maryland-Dataset'
model1_ckpt  = './label_train/saved_models/best_tagger_top200.pth'
color_ckpt   = './color_label/color_classifier.pt'
batch_size   = 32
epochs       = 10
lr           = 1e-4
save_path    = './checkpoints/model2_best.pth'
num_labels   = 200
num_colors   = 12
# ====================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device = ", device)

# Transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Dataset & Loader
dataset = OutfitPairDataset(pos_root=cleaned_root, neg_root=random_root, transform=transform)
n_val = int(len(dataset) * 0.2)
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# Models
attr_model = AttributeResNet(num_labels=num_labels).to(device).eval()
attr_model.load_state_dict(torch.load(model1_ckpt, map_location=device))

color_model = ColorClassifier(num_classes=num_colors).to(device).eval()
color_model.load_state_dict(torch.load(color_ckpt, map_location=device))

# Auto in_dim
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224).to(device)
    feat_attr = attr_model(dummy)
    feat_color = color_model(dummy)
    in_dim = feat_attr.shape[1] + feat_color.shape[1]

compat_model = CompatibilityNet(in_dim=in_dim).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(compat_model.parameters(), lr=lr)

# Training loop
best_loss = float('inf')
train_losses, val_losses = [], []

for epoch in range(1, epochs+1):
    compat_model.train()
    running_train = 0.0
    for img_t, img_b, label in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} [Train]'):
        img_t, img_b, label = img_t.to(device), img_b.to(device), label.float().unsqueeze(1).to(device)
        with torch.no_grad():
            p_t = torch.cat([attr_model(img_t), color_model(img_t)], dim=1)
            p_b = torch.cat([attr_model(img_b), color_model(img_b)], dim=1)
        logits = compat_model(p_t, p_b)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_train += loss.item() * img_t.size(0)
    epoch_train_loss = running_train / n_train
    train_losses.append(epoch_train_loss)

    # Validation
    compat_model.eval()
    running_val = 0.0
    with torch.no_grad():
        for img_t, img_b, label in val_loader:
            img_t, img_b, label = img_t.to(device), img_b.to(device), label.float().unsqueeze(1).to(device)
            p_t = torch.cat([attr_model(img_t), color_model(img_t)], dim=1)
            p_b = torch.cat([attr_model(img_b), color_model(img_b)], dim=1)
            logits = compat_model(p_t, p_b)
            loss = criterion(logits, label)
            running_val += loss.item() * img_t.size(0)
    epoch_val_loss = running_val / n_val
    val_losses.append(epoch_val_loss)

    print(f'Epoch {epoch} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}')

    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        torch.save({'model2_state_dict': compat_model.state_dict()}, save_path)
        print(f'✅ Saved best model to {save_path}')