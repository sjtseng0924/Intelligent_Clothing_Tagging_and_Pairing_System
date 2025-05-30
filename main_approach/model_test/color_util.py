import torch
from PIL import Image
import torchvision.transforms as T
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from color_label.model import ColorClassifier

class ColorExtractor:
    def __init__(self, model_path, device, num_colors=12):
        self.device = device
        self.model = ColorClassifier(num_classes=num_colors).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
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
        # 只取最大值 one-hot
        color_idx = probs.argmax()
        color_onehot = torch.zeros_like(torch.tensor(probs))
        color_onehot[color_idx] = 1.0
        return color_onehot.tolist()