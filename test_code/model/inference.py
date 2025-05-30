import torch
from PIL import Image
from torchvision import transforms as T
import os
import sys
import numpy as np
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from attr_extractor import AttributeExtractor
from model import CompatibilityModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 路徑設定 ===
top_image_path = os.path.join(project_root, "tshirt.jpg")
bottom_dir = os.path.join(project_root, "user_bottom")
attr_ckpt = os.path.join(project_root, "label_train", "saved_models", "best_tagger.pth")
compat_ckpt = os.path.join(project_root, "model", "compatibility_model.pth")

# === 載入模型 ===
attr_extractor = AttributeExtractor(attr_ckpt, device)
compat_model = CompatibilityModel().to(device)
compat_model.load_state_dict(torch.load(compat_ckpt, map_location=device))
compat_model.eval()

# === 圖片預處理 ===
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])
def get_label_emb(image_path):
    with torch.no_grad():
        label_emb = attr_extractor.extract(image_path)
    return torch.tensor(label_emb, device=device).unsqueeze(0)

def get_image_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def recommend_bottom(top_img_path, bottom_dir, top_k=5):
    top_emb = get_label_emb(top_img_path)
    top_img_tensor = get_image_tensor(top_img_path)
    scores = []
    candidates = []
    for fname in os.listdir(bottom_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        b_path = os.path.join(bottom_dir, fname)
        bottom_emb = get_label_emb(b_path)
        bottom_img_tensor = get_image_tensor(b_path)
        with torch.no_grad():
            score = compat_model(top_img_tensor, bottom_img_tensor, top_emb, bottom_emb)
            scores.append(score.item())
            candidates.append(fname)
    # 排序並輸出 top_k
    idxs = np.argsort(scores)[::-1][:top_k]
    print("推薦下身：")
    for i in idxs:
        print(f"{candidates[i]}  score: {scores[i]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=str, default=os.path.join(project_root, "tshirt.jpg"), help="Path to the top image")
    parser.add_argument('--wardrobe', type=str, default=os.path.join(project_root, "user_bottom"), help="Directory of bottom images")
    parser.add_argument('--topk', type=int, default=3, help="Number of recommendations to show")
    args = parser.parse_args()

    recommend_bottom(args.top, args.wardrobe, top_k=args.topk)
