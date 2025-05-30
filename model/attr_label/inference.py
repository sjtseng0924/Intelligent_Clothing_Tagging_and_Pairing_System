import torch
import torchvision.transforms as T
from PIL import Image
from model import AttributeResNet
import os

model = AttributeResNet(num_labels=1000)
model.load_state_dict(torch.load('best_tagger.pth', map_location='cpu'))
model.eval()

# 載入屬性名稱
attr_names = []
with open('../list_attr_cloth.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()[2:]  # 跳過第一行數量與第二行欄名
    for line in lines:
        parts = line.strip().split()
        name = ' '.join(parts[:-1])  # 合併前面所有欄位當作名稱
        attr_names.append(name)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def predict_image(image_path, topk=10):
    if not os.path.exists(image_path):
        print(f"找不到圖片: {image_path}")
        return

    print(f"\n載入圖片: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    print("預測中...")
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).squeeze()

    topk_indices = probs.topk(topk).indices
    topk_scores = probs[topk_indices]

    print(f"\n 預測屬性 (Top {topk}):")
    for i, idx in enumerate(topk_indices):
        print(f"{attr_names[idx]} (score: {topk_scores[i]:.2f})")

if __name__ == '__main__':
    image_path = r'../test/top.jpg'  
    predict_image(image_path, topk=100)
