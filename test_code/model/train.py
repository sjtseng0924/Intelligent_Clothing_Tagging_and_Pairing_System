import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from attr_extractor import AttributeExtractor
from dataset import OutfitPairDataset
from model import CompatibilityModel

def train_compatibility():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    attr_extractor = AttributeExtractor('../label_train/saved_models/best_tagger.pth', device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = OutfitPairDataset('../Cleaned-Maryland-Dataset', '../Random-Maryland-Dataset', attr_extractor, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    model = CompatibilityModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(10):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch in pbar:
            top_img = batch['top_img'].to(device)
            bottom_img = batch['bottom_img'].to(device)
            top_label_emb = batch['top_label_emb'].to(device)
            bottom_label_emb = batch['bottom_label_emb'].to(device)
            pair_label = batch['pair_label'].to(device)

            optimizer.zero_grad()
            output = model(top_img, bottom_img, top_label_emb, bottom_label_emb)
            loss = criterion(output, pair_label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))
        
        print(f"Epoch {epoch+1} average loss: {total_loss / len(dataloader)}")

    # Save the trained model
    torch.save(model.state_dict(), 'compatibility_model.pth')
    print("Model saved to compatibility_model.pth")

if __name__ == '__main__':
    train_compatibility()
