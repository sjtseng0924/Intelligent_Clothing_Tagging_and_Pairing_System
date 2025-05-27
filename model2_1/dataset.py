# dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class CombinedMarylandDataset(Dataset):
    """
    同時載入 Cleaned（正例）與 Random（負例）資料集
    資料夾結構:
      pos_root/<id>/top.jpg, bottom.jpg   → label=1
      neg_root/<id>/top.jpg, bottom.jpg   → label=0
    """
    def __init__(self,
                 pos_root: str,
                 neg_root: str,
                 transform: T.Compose = None):
        self.pos_root = pos_root
        self.neg_root = neg_root
        self.transform = transform or T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std =[0.229,0.224,0.225]),
        ])

        # 讀正例
        self.positive = []
        for _id in sorted(os.listdir(pos_root)):
            d = os.path.join(pos_root, _id)
            if os.path.isdir(d):
                t = os.path.join(d, "top.jpg")
                b = os.path.join(d, "bottom.jpg")
                if os.path.isfile(t) and os.path.isfile(b):
                    self.positive.append((t, b, 1))

        # 讀負例
        self.negative = []
        for _id in sorted(os.listdir(neg_root)):
            d = os.path.join(neg_root, _id)
            if os.path.isdir(d):
                t = os.path.join(d, "top.jpg")
                b = os.path.join(d, "bottom.jpg")
                if os.path.isfile(t) and os.path.isfile(b):
                    self.negative.append((t, b, 0))

        # 合併並打亂
        self.pairs = self.positive + self.negative
        import random; random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        t_path, b_path, label = self.pairs[idx]
        img_t = Image.open(t_path).convert("RGB")
        img_b = Image.open(b_path).convert("RGB")
        img_t = self.transform(img_t)
        img_b = self.transform(img_b)
        return img_t, img_b, float(label)
