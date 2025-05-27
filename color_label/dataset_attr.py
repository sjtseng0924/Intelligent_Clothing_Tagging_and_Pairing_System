import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class ColorDataset(Dataset):
    def __init__(self, root_dir, class_names):
        self.samples = []
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

        for class_name in class_names:
            folder = os.path.join(root_dir, class_name)
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                label = torch.zeros(len(class_names))
                label[self.class_to_idx[class_name]] = 1.0
                self.samples.append((path, label))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label
