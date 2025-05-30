import os
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from model import CompatibilityNet
from label_train.model import AttributeResNet
from color_label.model import ColorClassifier

# ===== 參數 =====
top_image_path = 'static/net_top/9.png'
bottom_dir = 'static/net_bottom'
attr_model_path = 'label_train/saved_models/best_tagger_top200.pth'
color_model_path = 'color_label/color_classifier.pt'
compat_model_path = 'checkpoints/model2_best.pth'
bot_cache = './bot_feats_cache.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_labels = 200
num_colors = 12
top_k = 5
# =================

# Transform
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load models
attr_model = AttributeResNet(num_labels=num_labels).to(device).eval()
attr_model.load_state_dict(torch.load(attr_model_path, map_location=device))

color_model = ColorClassifier(num_classes=num_colors).to(device).eval()
color_model.load_state_dict(torch.load(color_model_path, map_location=device))

with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224).to(device)
    in_dim = attr_model(dummy).shape[1] + color_model(dummy).shape[1]
compat_model = CompatibilityNet(in_dim=in_dim).to(device).eval()
compat_model.load_state_dict(torch.load(compat_model_path, map_location=device)['model2_state_dict'])

# Feature extraction
def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        f = torch.cat([attr_model(img), color_model(img)], dim=1)
    return f

if os.path.exists(bot_cache):
    cache = torch.load(bot_cache)
    bot_paths, bot_feats = cache['paths'], cache['feats']
else:
    bot_paths, bot_feats = [], []
    for f in os.listdir(bottom_dir):
        p = os.path.join(bottom_dir, f)
        bot_paths.append(p)
        bot_feats.append(extract_features(p).cpu())
    bot_feats = torch.cat(bot_feats, dim=0)
    torch.save({'paths': bot_paths, 'feats': bot_feats}, bot_cache)

# Inference
top_feat = extract_features(top_image_path).repeat(len(bot_paths), 1)
with torch.no_grad():
    scores = torch.sigmoid(compat_model(top_feat.to(device), bot_feats.to(device))).squeeze(1).cpu()
ranked = sorted(zip(bot_paths, scores.tolist()), key=lambda x: -x[1])

# Results
print(f'\nTop {top_k} Recommendations:')
for i, (path, score) in enumerate(ranked[:top_k]):
    print(f'{i+1}. {path}  Score: {score:.4f}')
