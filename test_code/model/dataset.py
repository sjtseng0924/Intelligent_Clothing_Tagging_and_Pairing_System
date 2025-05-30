import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class OutfitPairDataset(Dataset):
    def __init__(self, cleaned_dir, random_dir, attr_extractor, transform=None):
        self.transform = transform
        self.attr_extractor = attr_extractor
        self.device = attr_extractor.device

        self.samples = []
        for folder in os.listdir(cleaned_dir):
            folder_path = os.path.join(cleaned_dir, folder)
            if os.path.isdir(folder_path):
                self.samples.append((folder_path, 1))
        for folder in os.listdir(random_dir):
            folder_path = os.path.join(random_dir, folder)
            if os.path.isdir(folder_path):
                self.samples.append((folder_path, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, pair_label = self.samples[idx]

        top_img = Image.open(os.path.join(folder_path, 'top.jpg')).convert('RGB')
        bottom_img = Image.open(os.path.join(folder_path, 'bottom.jpg')).convert('RGB')

        if self.transform is None:
            raise ValueError("Please provide transform for image preprocessing.")

        # Apply transform
        top_img_t = self.transform(top_img)
        bottom_img_t = self.transform(bottom_img)
        top_img_path = os.path.join(folder_path, 'top.jpg')
        bottom_img_path = os.path.join(folder_path, 'bottom.jpg')
        top_label_emb = torch.tensor(self.attr_extractor.extract(top_img_path))
        bottom_label_emb = torch.tensor(self.attr_extractor.extract(bottom_img_path))

        if top_label_emb.ndim > 1:
            top_label_emb = top_label_emb.squeeze(0)
        if bottom_label_emb.ndim > 1:
            bottom_label_emb = bottom_label_emb.squeeze(0)
        return {
            'top_img': top_img_t,
            'bottom_img': bottom_img_t,
            'top_label_emb': top_label_emb,
            'bottom_label_emb': bottom_label_emb,
            'pair_label': torch.tensor(pair_label, dtype=torch.float32)
        }
