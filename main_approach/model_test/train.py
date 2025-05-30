import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import AttributeMatrixCompatibility
from dataset import OutfitPairDataset
from attr_extractor import AttributeExtractor
from color_util import ColorExtractor

def train_compatibility():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attr_extractor = AttributeExtractor('../label_train/saved_models/best_tagger.pth', device)
    color_extractor = ColorExtractor('../color_label/color_classifier.pt', device, num_colors=12)
    num_labels = len(attr_extractor.attr_names) + 12  # N + C
    model = AttributeMatrixCompatibility(num_labels=num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Split dataset 8:1:1
    full_dataset = OutfitPairDataset('../dataset/Cleaned-Maryland-Dataset', '../dataset/Unmatch-Dataset2', attr_extractor, color_extractor)
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    valid_len = int(0.1 * total_len)
    test_len = total_len - train_len - valid_len
    train_set, valid_set, test_set = random_split(full_dataset, [train_len, valid_len, test_len])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    model.train()
    for epoch in range(10):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
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
        print(f"Epoch {epoch+1} average train loss: {total_loss / len(train_loader)}")

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                top_label = batch['top_label_emb'].to(device)
                bottom_label = batch['bottom_label_emb'].to(device)
                pair_label = batch['pair_label'].to(device)
                score = model(top_label, bottom_label)
                loss = criterion(score, pair_label)
                valid_loss += loss.item()
        print(f"Epoch {epoch+1} average valid loss: {valid_loss / len(valid_loader)}")
        model.train()

    torch.save(model.state_dict(), 'compatibility_matrix_newdataset.pth')
    print("Model saved to compatibility_matrix_newdataset.pth")

    # Test
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            top_label = batch['top_label_emb'].to(device)
            bottom_label = batch['bottom_label_emb'].to(device)
            pair_label = batch['pair_label'].to(device)
            score = model(top_label, bottom_label)
            loss = criterion(score, pair_label)
            test_loss += loss.item()
    print(f"Test average loss: {test_loss / len(test_loader)}")

if __name__ == '__main__':
    train_compatibility()