import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import AttributeMatrixCompatibility
from dataset import OutfitPairDataset
from color_util import ColorExtractor

def train_compatibility():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    color_extractor = ColorExtractor('../color_label/color_classifier.pt', device, num_colors=12)
    num_labels = 12  # 只考慮顏色標籤
    model = AttributeMatrixCompatibility(num_labels=num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    dataset = OutfitPairDataset('../Cleaned-Maryland-Dataset', '../Random-Maryland-Dataset', color_extractor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model.train()
    for epoch in range(10):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            top_label = batch['top_label_emb'].to(device)
            bottom_label = batch['bottom_label_emb'].to(device)
            pair_label = batch['pair_label'].to(device)
            optimizer.zero_grad()
            score = model(top_label, bottom_label)
            loss = criterion(score, pair_label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))
        print(f"Epoch {epoch+1} average loss: {total_loss / len(dataloader)}")
    torch.save(model.state_dict(), 'color_matrix.pth')
    print("Model saved to compatibility_matrix.pth")

if __name__ == '__main__':
    train_compatibility()