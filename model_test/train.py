import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import AttributeMatrixCompatibility
from dataset import OutfitPairDataset
from attr_extractor import AttributeExtractor

def train_compatibility():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attr_extractor = AttributeExtractor('../label_train/saved_models/best_tagger.pth', device)
    model = AttributeMatrixCompatibility().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    dataset = OutfitPairDataset('../Cleaned-Maryland-Dataset', '../Random-Maryland-Dataset', attr_extractor)
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
    torch.save(model.state_dict(), 'compatibility_matrix.pth')
    print("Model saved to compatibility_matrix.pth")

if __name__ == '__main__':
    train_compatibility()