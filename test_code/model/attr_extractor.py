import torch
import torchvision.transforms as T
from PIL import Image
from model import AttributeResNet
import os

class AttributeExtractor:
    def __init__(self, model_path, device, attr_file='list_attr_cloth.txt', threshold=0.2):
        self.device = device
        self.model = AttributeResNet(num_labels=1000).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.threshold = threshold

        # 讀取屬性名稱
        self.attr_names = []
        with open(attr_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[2:]
            for line in lines:
                parts = line.strip().split()
                name = ' '.join(parts[:-1])
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
        # 取出所有 >= threshold 的屬性
        # selected = [(self.attr_names[i], float(probs[i])) for i in range(len(probs)) if probs[i] >= self.threshold]
        # return selected
        return probs.cpu().tolist()