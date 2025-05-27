import os
import torch
from torch.utils.data import Dataset

class OutfitPairDataset(Dataset):
    def __init__(self, cleaned_dir, random_dir, attr_extractor):
        self.samples = []
        self.attr_extractor = attr_extractor

        # 正例
        for folder in os.listdir(cleaned_dir):
            folder_path = os.path.join(cleaned_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            self.samples.append((folder_path, 1))
        # 負例
        for folder in os.listdir(random_dir):
            folder_path = os.path.join(random_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            self.samples.append((folder_path, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, pair_label = self.samples[idx]
        top_img_path = os.path.join(folder_path, 'top.jpg')
        bottom_img_path = os.path.join(folder_path, 'bottom.jpg')
        # 取得屬性機率
        top_label_emb = torch.tensor(self.attr_extractor.extract(top_img_path)).float()
        bottom_label_emb = torch.tensor(self.attr_extractor.extract(bottom_img_path)).float()
        # 二值化
        top_label_bin = (top_label_emb > 0.2).float()
        bottom_label_bin = (bottom_label_emb > 0.2).float()
        return {
            'top_label_emb': top_label_bin,
            'bottom_label_emb': bottom_label_bin,
            'pair_label': torch.tensor(pair_label).float()
        }