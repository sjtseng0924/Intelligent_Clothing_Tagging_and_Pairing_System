import csv
import ast
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# === 1. 讀取 CSV 資料 ===
class OutfitDataset(Dataset):
    def __init__(self, csv_path):
        self.top_labels = []
        self.bottom_labels = []
        self.label_counts_per_item = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                top = ast.literal_eval(row['filtered_top_probs'])
                bottom = ast.literal_eval(row['filtered_bottom_probs'])
                if sum(bottom) == 0:  # 過濾掉無標記資料
                    continue
                
                top_tensor = torch.tensor(top, dtype=torch.float32)
                bottom_tensor = torch.tensor(bottom, dtype=torch.float32)

                self.top_labels.append(top_tensor)
                self.bottom_labels.append(bottom_tensor)
                
                count_ones = (bottom_tensor == 1).sum().item()
                # print('count_ones: ', count_ones)
                self.label_counts_per_item.append(count_ones)
            print('mean ones:', np.mean(self.label_counts_per_item))


    def __len__(self):
        return len(self.top_labels)

    def __getitem__(self, idx):
        return self.top_labels[idx], self.bottom_labels[idx]
    
    def get_label_frequencies(self):
        all_bottoms = torch.stack(self.bottom_labels)  # (num_samples, 404)
        pos_counts = all_bottoms.sum(dim=0)  # (404,)
        total = all_bottoms.size(0)
        freqs = pos_counts / total  # 每個 label 出現 1 的比例
        return freqs

# === 2. 定義模型 ===
class Top2BottomModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, output_dim),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
    
# FocalLoss 定義
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha])
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha = self.alpha.to(inputs.device)
        if alpha.ndim > 0:
            alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
        else:
            alpha_t = alpha
        loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# === 3. 訓練模型（含驗證）===
# 主要訓練函數
def train_model(csv_path, num_epochs=20, batch_size=32, lr=5e-4, val_split=0.2):
    dataset = OutfitDataset(csv_path)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    input_dim = len(dataset[0][0])
    output_dim = len(dataset[0][1])

    model = Top2BottomModel(input_dim, output_dim).to(device)

    freqs = dataset.get_label_frequencies()
    pos_weight = (1.0 - freqs) / freqs
    pos_weight = torch.clamp(pos_weight, min=1.0, max=1000.0).to(device)
    alpha = pos_weight / (pos_weight + 1.0)

    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 5
    wait = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for tops, bottoms in train_loader:
            tops, bottoms = tops.to(device), bottoms.to(device)
            preds = model(tops)
            loss = criterion(preds, bottoms)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_losses.append(total_train_loss / len(train_loader))

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        total_val_loss = 0  # ✅ 加上這行

        with torch.no_grad():
            for tops, bottoms in val_loader:
                tops, bottoms = tops.to(device), bottoms.to(device)
                logits = model(tops)
                preds = torch.sigmoid(logits)

                loss = criterion(logits, bottoms)  # ✅ 加上這行
                total_val_loss += loss.item()      # ✅ 累加 val loss

                all_preds.append(preds.cpu())
                all_labels.append(bottoms.cpu())

        val_loss = total_val_loss / len(val_loader)  # ✅ 計算平均
        val_losses.append(val_loss)                  # ✅ 加入記錄

        all_preds = torch.cat(all_preds).numpy() > 0.5
        all_labels = torch.cat(all_labels).numpy()
        # val_losses.append(0)  # 可選：未記錄 val loss，保留佔位

        print(f'Epoch {epoch + 1}/{num_epochs}: ')
        print("Precision:", precision_score(all_labels, all_preds, average='micro'))
        print("Recall:   ", recall_score(all_labels, all_preds, average='micro'))
        print("F1 Score: ", f1_score(all_labels, all_preds, average='micro'))

        # Validation loss 已經算好了
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), "best_model.pt")  # 儲存最好的模型
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # Plot training loss
    plt.semilogy(train_losses, label="Train Loss")
    plt.semilogy(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Curve")
    plt.show()

    return model

# === 主程式 ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = "filtered_probs.csv"
    model = train_model(csv_path)

    torch.save(model.state_dict(), "top2bottom.pth")