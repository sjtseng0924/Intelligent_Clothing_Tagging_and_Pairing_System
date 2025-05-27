# train_compat.py
"""
訓練相容度評分器 Model2 (所有參數已寫入程式中，不需額外傳參數)
整合：
  - Model1 (AttributeResNet)
  - Dataset (CombinedMarylandDataset)
  - Model2 (CompatibilityNet)

將以下參數直接硬編碼在程式中：
  cleaned_root = './Cleaned-Maryland-Dataset'
  random_root  = './Random-Maryland-Dataset'
  model1_ckpt   = './label_train/saved_models/best_tagger.pth'
  batch_size    = 32
  epochs        = 10
  lr            = 1e-4
  save_path     = './checkpoints/model2_best.pth'
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torchvision.transforms as T

from dataset import CombinedMarylandDataset
from model2 import CompatibilityNet
from label_train.model import AttributeResNet

# ===== 參數設定 (硬編碼) =====
cleaned_root = './Cleaned-Maryland-Dataset'   # 正例資料夾
random_root  = './Random-Maryland-Dataset'    # 負例資料夾
model1_ckpt  = './label_train/saved_models/best_tagger.pth'
batch_size   = 32
epochs       = 10
lr           = 1e-4
save_path    = './checkpoints/model2_best.pth'
# ============================

def main():
    # 裝置設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 圖像預處理
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # 載入 Combined Dataset
    dataset = CombinedMarylandDataset(
        pos_root=cleaned_root,
        neg_root=random_root,
        transform=transform
    )
    total = len(dataset)
    n_val = int(total * 0.2)
    n_train = total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # 載入 Model1
    model1 = AttributeResNet(num_labels=1000).to(device).eval()
    ck1 = torch.load(model1_ckpt, map_location=device)
    sd1 = ck1.get('model_state_dict', ck1)
    model1.load_state_dict(sd1)
    for p in model1.parameters():
        p.requires_grad = False

    # 初始化 Model2
    with torch.no_grad():
        dummy = torch.randn(1,3,224,224).to(device)
        D = model1(dummy).shape[1]
    compat_net = CompatibilityNet(in_dim=2*D).to(device)

    # 損失函數 + 優化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(compat_net.parameters(), lr=lr, weight_decay=1e-5)

    # 確保目錄存在
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    best_loss = float('inf')
    train_losses, val_losses = [], []

    # 訓練迴圈
    for epoch in range(1, epochs+1):
        # Train
        compat_net.train()
        running_train = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", ncols=80)
        for img_t, img_b, label in loop:
            img_t = img_t.to(device); img_b = img_b.to(device)
            label = label.float().to(device).unsqueeze(1)

            with torch.no_grad():
                p_t = model1(img_t); p_b = model1(img_b)

            logits = compat_net(p_t, p_b)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train += loss.item() * img_t.size(0)
            loop.set_postfix(loss=loss.item())
        epoch_train_loss = running_train / n_train
        train_losses.append(epoch_train_loss)

        # Validation
        compat_net.eval()
        running_val = 0.0
        with torch.no_grad():
            for img_t, img_b, label in val_loader:
                img_t = img_t.to(device); img_b = img_b.to(device)
                label = label.float().to(device).unsqueeze(1)
                p_t = model1(img_t); p_b = model1(img_b)
                logits = compat_net(p_t, p_b)
                vloss = criterion(logits, label)
                running_val += vloss.item() * img_t.size(0)
        epoch_val_loss = running_val / n_val
        val_losses.append(epoch_val_loss)

        # Epoch 結果
        print()  # 換行
        print(f"Epoch [{epoch}/{epochs}] train_loss: {epoch_train_loss:.4f} val_loss: {epoch_val_loss:.4f}")
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save({
                'epoch': epoch,
                'model2_state_dict': compat_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, save_path)
            print(f"  Saved best model to {save_path}")

    print("Training completed.")

    # 儲存損失曲線
    epochs_range = range(1, epochs+1)
    plt.figure()
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Train/Val Loss'); plt.legend(); plt.grid(True)
    plot_path = os.path.splitext(save_path)[0] + '_loss_curve.png'
    plt.savefig(plot_path)
    print(f"Saved loss curve to {plot_path}")

if __name__ == '__main__':
    main()
