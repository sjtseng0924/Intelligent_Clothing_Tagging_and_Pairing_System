import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_attr import ColorDataset
from model import ColorClassifier

# === 參數設定 ===
data_path = "color_dataset"
class_names = sorted(os.listdir(data_path))  # 資料夾名稱就是顏色
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
epochs = 10
lr = 0.001

# === 資料載入與模型 ===
dataset = ColorDataset(data_path, class_names)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = ColorClassifier(num_classes=len(class_names)).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# === 訓練迴圈 ===
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(dataset):.4f}")

# === 儲存模型 ===
torch.save(model.state_dict(), "color_classifier.pt")
print("✅ 模型已儲存")
