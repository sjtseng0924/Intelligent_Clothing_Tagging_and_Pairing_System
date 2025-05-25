import torch
from PIL import Image
from torchvision import transforms as T
import os
import sys
import ast
import numpy as np

# 加入專案根目錄（模型和其他模組）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from label_train.model import AttributeResNet
from pair_train.train import Top2BottomModel  # 假設你把搭配模型放這裡

# === 設定參數 ===
image_path = os.path.join(project_root, "tshirt.jpg")
attribute_model_path = os.path.join(project_root, "label_train", "saved_models", "best_tagger.pth")
pairing_model_path = os.path.join(project_root, "pair_train", "top2bottom.pth")
num_labels = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. 載入 AttributeResNet 模型（產生 label） ===
attribute_model = AttributeResNet(num_labels=num_labels)
attribute_model.load_state_dict(torch.load(attribute_model_path, map_location=device))
attribute_model.eval().to(device)

# === 2. 載入 Top2Bottom 搭配模型 ===
pairing_model = Top2BottomModel(input_dim=num_labels, output_dim=num_labels)
pairing_model.load_state_dict(torch.load(pairing_model_path, map_location=device))
pairing_model.eval().to(device)

# === 3. 圖片預處理 ===
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

# 獲得衣服的label
def get_top_label(image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = attribute_model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().tolist()
    return probs

# 預測搭配的label
def predict_bottom_label(top_label):
    top_tensor = torch.tensor(top_label, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = pairing_model(top_tensor).squeeze().cpu().tolist()
    return pred

# 與使用者的褲子去搭配
def compare_with_user_bottoms(predicted_label, bottom_dir, top_k=10):
    # 取得 predicted_label 中前 top_k 大值的索引
    top_indices = np.argsort(predicted_label)[-top_k:]

    min_distance = float('inf')
    best_match_image = None

    for filename in os.listdir(bottom_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(bottom_dir, filename)
        bottom_label = get_top_label(img_path)

        # 只取前 top_k 對應的值來比較
        pred_vec = np.array(predicted_label)[top_indices]
        bottom_vec = np.array(bottom_label)[top_indices]

        distance = np.linalg.norm(pred_vec - bottom_vec)
        print(f"📸 {filename} 差異距離（top-{top_k}）: {distance:.4f}")

        if distance < min_distance:
            min_distance = distance
            best_match_image = filename

    return best_match_image, min_distance

# === 主程式執行 ===
top_label = get_top_label(image_path)
predicted_bottom_label = predict_bottom_label(top_label)

# 顯示結果
# print("🔎 上衣 label：", top_label)
# print("🎯 預測的褲子 label：", predicted_bottom_label)

user_bottom_dir = os.path.join(project_root, "user_bottom")
best_match, best_distance = compare_with_user_bottoms(predicted_bottom_label, user_bottom_dir, top_k=10)

print(f"\n✅ 最推薦的使用者褲子是: {best_match}（距離: {best_distance:.4f}）")
