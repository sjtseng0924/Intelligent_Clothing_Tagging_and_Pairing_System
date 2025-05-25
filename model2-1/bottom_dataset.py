# bottom_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class BottomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.ids = sorted([d for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d))])
        self.transform = transform or T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        path = os.path.join(self.root_dir, _id, 'bottom.jpg')
        img = Image.open(path).convert('RGB')
        return self.transform(img), path
