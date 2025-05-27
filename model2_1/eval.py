# eval.py
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import torchvision.transforms as T

from dataset import CombinedMarylandDataset
from label_train.model import AttributeResNet
from model2 import CompatibilityNet

def main():
    # ——— 硬编码参数 ———
    cleaned_root = './Cleaned-Maryland-Dataset'
    random_root  = './Random-Maryland-Dataset'
    model1_ckpt   = './label_train/saved_models/best_tagger.pth'
    model2_ckpt   = './checkpoints/model2_best.pth'
    batch_size    = 32
    val_ratio     = 0.2
    # ————————————

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transform
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # dataset split
    ds = CombinedMarylandDataset(cleaned_root, random_root, transform)
    n_val = int(len(ds) * val_ratio)
    val_ds = torch.utils.data.random_split(ds, [len(ds)-n_val, n_val])[1]
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # load Model1
    model1 = AttributeResNet(1000).to(device).eval()
    ck = torch.load(model1_ckpt, map_location=device)
    model1.load_state_dict(ck.get('model_state_dict', ck))

    # load Model2
    dummy = torch.randn(1,3,224,224).to(device)
    D = model1(dummy).shape[1]
    model2 = CompatibilityNet(in_dim=2*D).to(device).eval()
    ck2 = torch.load(model2_ckpt, map_location=device)
    model2.load_state_dict(ck2.get('model2_state_dict', ck2))

    # collect preds & labels
    all_preds, all_labels = [], []
    with torch.no_grad():
        for t, b, lbl in loader:
            t, b = t.to(device), b.to(device)
            p_t = model1(t); p_b = model1(b)
            logits = model2(p_t, p_b).squeeze(1).cpu()
            probs  = torch.sigmoid(logits)
            all_preds.extend(probs.numpy().tolist())
            all_labels.extend(lbl.tolist())

    # metrics
    acc = accuracy_score(all_labels, [p>0.5 for p in all_preds])
    auc = roc_auc_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, [p>0.5 for p in all_preds])
    print(f'Val Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}')
    print('Confusion Matrix:\n', cm)

if __name__=='__main__':
    main()
