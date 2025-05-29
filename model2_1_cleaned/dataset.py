import os
from PIL import Image
from torch.utils.data import Dataset

class OutfitPairDataset(Dataset):
    """
    上下裝配對資料集（每個資料夾是一對正樣本，負樣本另指定根目錄）
    """
    def __init__(self, pos_root, neg_root=None, transform=None):
        self.transform = transform

        # 正樣本：資料夾內有 top.jpg 和 bottom.jpg
        self.pos_pairs = []
        for folder in os.listdir(pos_root):
            folder_path = os.path.join(pos_root, folder)
            top_path = os.path.join(folder_path, 'top.jpg')
            bottom_path = os.path.join(folder_path, 'bottom.jpg')
            if os.path.isfile(top_path) and os.path.isfile(bottom_path):
                self.pos_pairs.append((top_path, bottom_path, 1))  # 標籤 1

        # 負樣本：如果有提供 neg_root，隨機配對 top 與 bottom
        self.neg_pairs = []
        if neg_root:
            tops = []
            bottoms = []
            for folder in os.listdir(neg_root):
                folder_path = os.path.join(neg_root, folder)
                top_path = os.path.join(folder_path, 'top.jpg')
                bottom_path = os.path.join(folder_path, 'bottom.jpg')
                if os.path.isfile(top_path) and os.path.isfile(bottom_path):
                    tops.append(top_path)
                    bottoms.append(bottom_path)
            import random
            for _ in range(len(self.pos_pairs)):
                top = random.choice(tops)
                bottom = random.choice(bottoms)
                self.neg_pairs.append((top, bottom, 0))  # 標籤 0

        # 合併資料
        self.data = self.pos_pairs + self.neg_pairs
        print(f"Loaded {len(self.pos_pairs)} positive pairs, {len(self.neg_pairs)} negative pairs.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        top_path, bottom_path, label = self.data[idx]
        top_img = Image.open(top_path).convert('RGB')
        bottom_img = Image.open(bottom_path).convert('RGB')
        if self.transform:
            top_img = self.transform(top_img)
            bottom_img = self.transform(bottom_img)
        return top_img, bottom_img, label
