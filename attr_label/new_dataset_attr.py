import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from collections import Counter

class DeepFashionAttrDataset(Dataset):
    def __init__(self, img_root, attr_file, eval_file, mode='train', transform=None, max_items=None, top_k=200):
        self.img_root = img_root
        self.transform = transform
        self.mode = mode
        self.top_k = top_k

        # 讀取屬性標籤
        with open(attr_file, 'r') as f:
            lines = f.readlines()[2:]  # skip header
        attr_dict = {}
        attr_counter = Counter()

        for line in lines:
            parts = line.strip().split()
            name = parts[0]
            attrs = torch.tensor([int(p) for p in parts[1:]])
            attr_dict[name] = attrs
            attr_counter.update(torch.where(attrs == 1)[0].tolist())

        # 取出前 top_k 的 index
        self.topk_indices = sorted([idx for idx, _ in attr_counter.most_common(self.top_k)])

        # 讀取分割資料
        with open(eval_file, 'r') as f:
            lines = f.readlines()[2:]  # skip header
        self.samples = []
        for line in lines:
            name, split = line.strip().split()
            if split == self.mode:
                self.samples.append(name)

        if max_items:
            self.samples = self.samples[:max_items]

        self.attr_dict = attr_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.samples[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        attr = self.attr_dict[self.samples[idx]]
        attr = attr[self.topk_indices]
        attr = (attr == 1).float()

        return image, attr
