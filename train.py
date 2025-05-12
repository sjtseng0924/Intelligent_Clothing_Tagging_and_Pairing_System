import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset_attr import DeepFashionAttrDataset
from model import AttributeResNet
import torch.nn as nn
import torch.optim as optim
import os

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用裝置：", device)

# 資料設定
img_root = 'DeepFashion'
attr_file = 'DeepFashion/Anno_coarse/list_attr_img.txt'
eval_file = 'DeepFashion/Eval/list_eval_partition.txt'

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Dataset & Dataloader
train_dataset = DeepFashionAttrDataset(img_root, attr_file, eval_file, mode='train', transform=transform, max_items=5000)
val_dataset = DeepFashionAttrDataset(img_root, attr_file, eval_file, mode='val', transform=transform, max_items=1000)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 模型與訓練參數
model = AttributeResNet(num_labels=1000).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 訓練流程
best_val_loss = float('inf')
for epoch in range(5):  # 可以自行調整 epoch 數
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Training Loss: {total_loss:.4f}")

    # 驗證
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}")

    # 儲存模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs('saved_models', exist_ok=True)
        torch.save(model.state_dict(), 'saved_models/best_tagger.pth')
        print("✅ Saved best model!")

