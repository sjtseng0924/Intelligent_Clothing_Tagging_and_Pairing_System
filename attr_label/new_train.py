import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from new_dataset_attr import DeepFashionAttrDataset
from new_model import AttributeResNet
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("✅ 使用裝置：", device)
    if device.type == 'cuda':
        print("✅ GPU 名稱：", torch.cuda.get_device_name(0))

    img_root = 'DeepFashion'
    attr_file = 'DeepFashion/Anno_coarse/list_attr_img.txt'
    eval_file = 'DeepFashion/Eval/list_eval_partition.txt'

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dataset = DeepFashionAttrDataset(img_root, attr_file, eval_file, mode='train', transform=transform, top_k=200)
    val_dataset = DeepFashionAttrDataset(img_root, attr_file, eval_file, mode='val', transform=transform, top_k=200)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=0, pin_memory=True)

    model = AttributeResNet(num_labels=200).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    for epoch in range(10):
        print(f"\n📘 Epoch {epoch+1}")
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc="Training"):
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), 'saved_models/best_tagger_top200.pth')
            print("✅ Saved best model!")

if __name__ == '__main__':
    main()
