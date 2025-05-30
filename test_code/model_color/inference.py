import torch
import os
import sys
import numpy as np
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model import AttributeMatrixCompatibility
from color_util import ColorExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 路徑設定 ===
attr_ckpt = os.path.join(project_root, "label_train", "saved_models", "best_tagger.pth")
compat_ckpt = os.path.join(project_root, "model_color", "color_matrix.pth")
color_ckpt = os.path.join(project_root, "color_label", "color_classifier.pt")

# === 載入模型 ===
color_extractor = ColorExtractor(color_ckpt, device, num_colors=12)
num_labels = 12  # 只考慮顏色標籤
compat_model = AttributeMatrixCompatibility(num_labels=num_labels).to(device)
compat_model.load_state_dict(torch.load(compat_ckpt, map_location=device))
compat_model.eval()

# === 圖片預處理 ===
def get_label_bin(image_path, threshold=0.2):
    probs = color_extractor.extract(image_path)
    return (torch.tensor(probs) > threshold).float().unsqueeze(0).to(device)  # (1, 12)

def get_full_label(image_path, threshold=0.2):
    color_onehot = torch.tensor(color_extractor.extract(image_path)).float()
    full_label = color_onehot.unsqueeze(0).to(device)  # (1, num_labels)
    return full_label


def recommend_bottom(top_img_path, bottom_dir, top_k=5):
    top_bin = get_full_label(top_img_path)  # (1, 12)
    scores = []
    candidates = []
    for fname in os.listdir(bottom_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        b_path = os.path.join(bottom_dir, fname)
        bottom_bin = get_full_label(b_path)  # (1, 12)
        with torch.no_grad():
            score = compat_model(top_bin, bottom_bin)  # (1,)
            scores.append(score.item())
            candidates.append(fname)
    idxs = np.argsort(scores)[::-1][:top_k]
    print("推薦下身：")
    for i in idxs:
        print(f"{candidates[i]}  score: {scores[i]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=str, required=True, help="Path to the top image")
    parser.add_argument('--wardrobe', type=str, required=True, help="Directory of bottom images")
    parser.add_argument('--topk', type=int, default=3, help="Number of recommendations to show")
    args = parser.parse_args()

    recommend_bottom(args.top, args.wardrobe, top_k=args.topk)