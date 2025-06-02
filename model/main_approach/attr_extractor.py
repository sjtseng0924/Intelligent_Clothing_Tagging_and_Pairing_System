import torch
import torchvision.transforms as T
from PIL import Image
import os
import sys

# 為了能從 model 資料夾載入 AttributeResNet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from model.attr_label.model import AttributeResNet

class AttributeExtractor:
    def __init__(self, model_path, device, threshold=0.02):
        self.device = device
        self.model = AttributeResNet(num_labels=1000).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.threshold = threshold

        attr_file = os.path.join(os.path.dirname(__file__), 'list_attr_cloth.txt')
        self.attr_names = []
        self.attr_indices = []
        with open(attr_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[2:]
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                name = ' '.join(parts[:-1])
                attr_type = int(parts[-1])
                if attr_type in [1, 2, 3]:
                    self.attr_indices.append(idx)
                    self.attr_names.append(name)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def extract(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.sigmoid(logits).squeeze()
        filtered_probs = probs[self.attr_indices]
        filtered_probs = torch.where(
            filtered_probs >= self.threshold, filtered_probs, torch.zeros_like(filtered_probs)
        )
        return filtered_probs.cpu().tolist()
