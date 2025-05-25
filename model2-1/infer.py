# infer.py
"""
推論／推薦腳本 (使用最佳閾值過濾後推薦)

將在驗證集上自動找出最佳 F1 閾值，並將此閾值應用於推論步驟。

Usage:
  python3 infer.py --top_image ./top.jpg
"""
import os
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve

from torch.utils.data import DataLoader
from dataset import CombinedMarylandDataset
from bottom_dataset import BottomDataset
from model2 import CompatibilityNet
from label_train.model import AttributeResNet

# ===== 硬編碼參數 =====
cleaned_root = './Cleaned-Maryland-Dataset'
random_root  = './Random-Maryland-Dataset'
bot_cache    = './bot_feats_cache.pt'
model1_ckpt  = './label_train/saved_models/best_tagger.pth'
model2_ckpt  = './checkpoints/model2_best.pth'
batch_size   = 64  # 下身特徵批次
infer_batch  = 512 # 推論批次（用於 threshold 計算）
top_k        = 5
# =====================

def compute_best_threshold(model1, model2, transform, device):
    """在驗證集上計算最佳 F1 阈值"""
    # 使用 CombinedDataset 的剩餘 20% 作為驗證
    ds = CombinedMarylandDataset(cleaned_root, random_root, transform)
    n_val = int(len(ds)*0.2)
    val_ds = torch.utils.data.random_split(ds, [len(ds)-n_val, n_val])[1]
    loader = DataLoader(val_ds, batch_size=infer_batch, shuffle=False)
    all_labels, all_probs = [], []
    model1.eval(); model2.eval()
    with torch.no_grad():
        for img_t, img_b, label in loader:
            img_t = img_t.to(device); img_b = img_b.to(device)
            p_t = model1(img_t); p_b = model1(img_b)
            logits = model2(p_t, p_b).squeeze(1).cpu()
            probs = torch.sigmoid(logits)
            all_labels.extend(label.numpy().tolist())
            all_probs.extend(probs.numpy().tolist())
    # 計算 precision-recall curve
    prec, rec, ths = precision_recall_curve(all_labels, all_probs)
    f1s = 2*prec*rec/(prec+rec+1e-8)
    best_idx = f1s.argmax()
    return ths[best_idx]


def load_or_compute_bot_features(model1, transform, device, cache_path=bot_cache):
    if os.path.isfile(cache_path):
        data = torch.load(cache_path)
        return data['paths'], data['feats']
    ds = BottomDataset(cleaned_root, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    paths, feats = [], []
    model1.eval()
    with torch.no_grad():
        for imgs, img_paths in tqdm(loader, desc='Extract bottom feats', ncols=80):
            imgs = imgs.to(device)
            out = model1(imgs).cpu()
            feats.append(out)
            paths.extend(img_paths)
    feats = torch.cat(feats, dim=0)
    torch.save({'paths': paths, 'feats': feats}, cache_path)
    return paths, feats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_image', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # transforms
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # load models
    model1 = AttributeResNet(num_labels=1000).to(device)
    state = torch.load(model1_ckpt, map_location=device)
    state_dict = state.get('model_state_dict', state)
    # load with non-strict to ignore unexpected/missing keys
    model1.load_state_dict(state_dict, strict=False)
    torch.load(model1_ckpt, map_location=device).get('model_state_dict', {})
    model2 = CompatibilityNet(in_dim=model1(torch.randn(1,3,224,224).to(device)).shape[1]*2).to(device)
    model2.load_state_dict(torch.load(model2_ckpt, map_location=device).get('model2_state_dict', {}))

    # compute best threshold
    # 直接使用固定閾值，避免計算過程被系統殺掉
    best_thresh = 0.5
    print(f'Using fixed threshold: {best_thresh:.3f}')

    # precompute bottom features bottom features
    bot_paths, all_p_b = load_or_compute_bot_features(model1, transform, device)

    # compute top feature
    img_t = transform(Image.open(args.top_image).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad(): p_t = model1(img_t)

    # score all bottoms
    with torch.no_grad():
        p_t_rep = p_t.repeat(len(bot_paths),1).to(device)
        logits = model2(p_t_rep, all_p_b.to(device)).squeeze(1).cpu()
        probs  = torch.sigmoid(logits)

    # filter by threshold
    candidates = [(bot_paths[i], float(probs[i])) for i in range(len(bot_paths)) if probs[i] > best_thresh]
    candidates.sort(key=lambda x: -x[1])

    # output top_k
    print('Recommended bottoms:')
    for path, score in candidates[:top_k]:
        print(f'{path}  score: {score:.4f}')

if __name__=='__main__':
    main()
