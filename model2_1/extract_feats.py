# extract_feats.py
"""
极端分组抽特征脚本，防止一次性杀掉
1) Cleaned 数据集分组抽特征
2) Random 数据集分组抽特征
3) 合并所有 partial .pt -> train_feats.pt

无需命令列参数，硬编码路径与 batch_size。
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

from dataset import CombinedMarylandDataset  # 仅为拿 transform
from label_train.model import AttributeResNet

# ===== 配置 =====
CLEAN_ROOT   = './Cleaned-Maryland-Dataset'
RANDOM_ROOT  = './Random-Maryland-Dataset'
MODEL1_CKPT  = './label_train/saved_models/best_tagger.pth'
BATCH_SIZE   = 8        # 再调小一点
NUM_WORKERS  = 0
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHUNK_SIZE   = 100      # 每次只处理 100 对
OUTPUT_DIR   = './feat_chunks'
FINAL_OUTPUT = 'train_feats.pt'
# ===============

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_for_root(root_dir, label_value):
    # 拿 transform
    tmp_ds = CombinedMarylandDataset(pos_root=CLEAN_ROOT,
                                     neg_root=RANDOM_ROOT,
                                     transform=None)
    transform = tmp_ds.transform

    # 收集这一 root_dir 下所有 (top,bottom,label)
    pairs = [(t,b,label_value) 
             for t,b,lab in tmp_ds.pairs 
             if lab==label_value]

    # 分组
    chunks = [pairs[i:i+CHUNK_SIZE] 
              for i in range(0, len(pairs), CHUNK_SIZE)]

    model1 = AttributeResNet(num_labels=1000).to(DEVICE).eval()
    state = torch.load(MODEL1_CKPT, map_location=DEVICE)
    model1.load_state_dict(state.get('model_state_dict', state), strict=False)

    for idx, chunk in enumerate(chunks):
        class ChunkDataset(Dataset):
            def __init__(self, data, transform):
                self.data = data
                self.transform = transform
            def __len__(self):
                return len(self.data)
            def __getitem__(self, i):
                t_path, b_path, lbl = self.data[i]
                img_t = self.transform(Image.open(t_path).convert('RGB'))
                img_b = self.transform(Image.open(b_path).convert('RGB'))
                return img_t, img_b, torch.tensor(lbl, dtype=torch.float)

        ds = ChunkDataset(chunk, transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS)

        feats_list, labels_list = [], []
        for img_t, img_b, lbl in tqdm(loader, desc=f'Extracting {"CLEAN" if label_value==1 else "RAND"} chunk {idx+1}/{len(chunks)}', ncols=80):
            img_t = img_t.to(DEVICE); img_b = img_b.to(DEVICE)
            with torch.no_grad():
                ft_t = model1(img_t).cpu()
                ft_b = model1(img_b).cpu()
            feat = torch.cat([ft_t, ft_b], dim=1)
            feats_list.append(feat)
            labels_list.append(lbl)  # 直接使用 lbl tensor

        feats = torch.cat(feats_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        chunk_path = os.path.join(OUTPUT_DIR, f"{'clean' if label_value==1 else 'rand'}_{idx}.pt")
        torch.save({'feats': feats, 'labels': labels}, chunk_path)
        print(f"Saved chunk {chunk_path}: {feats.shape[0]} samples")


def merge_chunks():
    files = sorted(os.listdir(OUTPUT_DIR))
    all_feats, all_labels = [], []
    for fn in files:
        data = torch.load(os.path.join(OUTPUT_DIR, fn))
        all_feats.append(data['feats'])
        all_labels.append(data['labels'])
    feats  = torch.cat(all_feats,  dim=0)
    labels = torch.cat(all_labels, dim=0)
    torch.save({'feats': feats, 'labels': labels}, FINAL_OUTPUT)
    print(f"Merged {len(files)} chunks into {FINAL_OUTPUT}: {feats.shape[0]} samples")

if __name__=='__main__':
    extract_for_root(CLEAN_ROOT, label_value=1)
    extract_for_root(RANDOM_ROOT, label_value=0)
    merge_chunks()
