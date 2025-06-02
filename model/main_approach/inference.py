import torch
import os
import sys
import numpy as np
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from attr_extractor import AttributeExtractor
from model import AttributeMLPCompatibility
from color_util import ColorExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 路徑設定 ===
attr_ckpt = os.path.join(project_root, "model", "attr_label", "best_tagger.pth")
compat_ckpt = os.path.join(project_root, "main_approach", "compatibility_mlp_main_v2.pth")
color_ckpt = os.path.join(project_root, "model", "color_label", "color_classifier.pt")

# === 載入模型 ===
attr_extractor = AttributeExtractor(attr_ckpt, device)
color_extractor = ColorExtractor(color_ckpt, device, num_colors=12)
num_labels = len(attr_extractor.attr_names) + 12
compat_model = AttributeMLPCompatibility(num_labels=num_labels).to(device)
compat_model.load_state_dict(torch.load(compat_ckpt, map_location=device, weights_only=False))
compat_model.eval()


def get_label_bin(image_path, threshold=0.02):
    probs = attr_extractor.extract(image_path)
    return (torch.tensor(probs) > threshold).float().unsqueeze(0).to(device)

def get_full_label(image_path, threshold=0.02):
    attr_probs = attr_extractor.extract(image_path)
    attr_bin = (torch.tensor(attr_probs) > threshold).float()
    color_onehot = torch.tensor(color_extractor.extract(image_path)).float()
    full_label = torch.cat([attr_bin, color_onehot]).unsqueeze(0).to(device)
    return full_label

def recommend_bottom(top_img_path, bottom_dir, top_k=5):
    top_bin = get_full_label(top_img_path)
    scores = []
    candidates = []
    for fname in os.listdir(bottom_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        b_path = os.path.join(bottom_dir, fname)
        bottom_bin = get_full_label(b_path)
        with torch.no_grad():
            score = compat_model(top_bin, bottom_bin).item()
        scores.append(score)
        candidates.append(fname)
    # Get top-k results based on scores
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_k_results = [(candidates[i], scores[i]) for i in top_indices]
    
    # Get all results with positive scores
    positive_indices = [i for i, score in enumerate(scores) if score > 0]
    positive_results = [(candidates[i], scores[i]) for i in positive_indices]
    
    # Sort positive results by score (highest first)
    positive_results.sort(key=lambda x: x[1], reverse=True)
    
    return top_k_results, positive_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=str, required=True, help='Path to top image')
    parser.add_argument('--bottom_dir', type=str, required=True, help='Directory of bottom images')
    parser.add_argument('--top_k', type=int, default=5, help='Number of recommendations')
    parser.add_argument('--mode', type=str, choices=['topk', 'positive'], default='positive', help='Choose result mode: topk or positive')
    args = parser.parse_args()
    top_k_results, positive_results = recommend_bottom(args.top, args.bottom_dir, args.top_k)
    if args.mode == 'topk':
        for fname, score in top_k_results:
            print(f"Top-{args.top_k} Recommendation - {fname}: {score:.4f}")
    elif args.mode == 'positive':
        for fname, score in positive_results:
            print(f"Positive Result - {fname}: {score:.4f}")
