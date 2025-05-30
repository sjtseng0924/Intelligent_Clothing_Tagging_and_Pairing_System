import os
import torch
from torch.utils.data import Dataset
from model_test.color_util import ColorExtractor

class OutfitPairDataset(Dataset):
    def __init__(self, cleaned_dir, random_dir, attr_extractor, color_extractor):
        self.samples = []
        self.attr_extractor = attr_extractor
        self.color_extractor = color_extractor

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
        # 屬性
        top_attr = torch.tensor(self.attr_extractor.extract(top_img_path)).float()
        bottom_attr = torch.tensor(self.attr_extractor.extract(bottom_img_path)).float()
        # 顏色
        top_color = torch.tensor(self.color_extractor.extract(top_img_path)).float()
        bottom_color = torch.tensor(self.color_extractor.extract(bottom_img_path)).float()
        # 合併
        top_label = torch.cat([top_attr, top_color])
        bottom_label = torch.cat([bottom_attr, bottom_color])
        return {
            'top_label_emb': top_label,
            'bottom_label_emb': bottom_label,
            'pair_label': torch.tensor(pair_label).float()
        }