import torch
from PIL import Image
import torchvision.transforms as T
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from color_label.model import ColorClassifier

class ColorExtractor:
    def __init__(self, model_path, device, color_txt_path='color_list.txt', num_colors=12):
        self.device = device
        self.model = ColorClassifier(num_classes=num_colors).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.num_colors = num_colors
        self.color_names = []

        # 修正 color.txt 的路徑：使用與 color_util.py 相對的絕對路徑
        color_txt_path = os.path.join(os.path.dirname(__file__), 'color.txt')
        with open(color_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name:
                    self.color_names.append(name)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
    def extract(self, image_path):
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        return probs.tolist()