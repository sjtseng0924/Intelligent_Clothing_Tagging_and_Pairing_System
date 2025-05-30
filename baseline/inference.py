import torch
from PIL import Image
from torchvision import transforms as T
import os
import sys
import ast
import numpy as np
import json

# 加入專案根目錄（模型和其他模組）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from label_train.model import AttributeResNet
from pair_train.train import Top2BottomModel  # 假設你把搭配模型放這裡

# === 設定參數 ===
image_path = os.path.join(project_root, "baseline","Top", "8933008.jpg")
attribute_model_path = os.path.join(project_root, "label_train", "saved_models", "best_tagger.pth")
pairing_model_path = os.path.join(project_root, "baseline", "top2bottom.pth")
num_labels = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1. 載入 AttributeResNet 模型（產生 label） ===
attribute_model = AttributeResNet(num_labels=num_labels)
attribute_model.load_state_dict(torch.load(attribute_model_path, map_location=device))
attribute_model.eval().to(device)

# label removing
keep_indices_path = os.path.join(project_root, "baseline", "keep_indices.json")
with open(keep_indices_path, "r") as f:
    keep_indices = json.load(f)["valid_indices"]



# === 2. 載入 Top2Bottom 搭配模型 ===
pairing_model = Top2BottomModel(input_dim=554, output_dim=554)
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
    
    # 根據 valid_indices 篩選出有效的 label
    filtered_probs = [probs[i] for i in keep_indices]
    return filtered_probs

# 預測搭配的label
def predict_bottom_label(top_label):
    top_tensor = torch.tensor(top_label, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = pairing_model(top_tensor).squeeze().cpu().tolist()
    return pred

# 與使用者的褲子去搭配
def compare_with_user_bottoms(predicted_label, bottom_dir, top_k=10, show_top_n=20, top_filename=None):
    top_indices = np.argsort(predicted_label)[-top_k:]

    distances = []

    for filename in os.listdir(bottom_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(bottom_dir, filename)
        bottom_label = get_top_label(img_path)

        pred_vec = np.array(predicted_label)[top_indices]
        bottom_vec = np.array(bottom_label)[top_indices]

        distance = np.linalg.norm(pred_vec - bottom_vec)
        distances.append((filename, distance))

    # 排序並取前 N 名
    distances.sort(key=lambda x: x[1])
    top_matches = distances[:show_top_n]

    print(f"\n前 {show_top_n} 名推薦結果（依距離排序）：")
    for rank, (filename, distance) in enumerate(top_matches, 1):
        print(f"{rank:2d}. {filename} 距離: {distance:.4f}")

    # 尋找同名褲子排第幾名
    same_name_bottom = top_filename.replace("Top", "Bottom")
    match_index = next((i for i, (filename, _) in enumerate(distances) if filename == same_name_bottom), -1)

    if match_index >= 0:
        print(f"\n同檔名褲子 `{same_name_bottom}` 排名第 {match_index + 1} 名")
    else:
        print(f"\n找不到與 `{top_filename}` 同名的褲子 `{same_name_bottom}`")

    return distances[0]  # 回傳最佳匹配的褲子（距離最小）


# === 主程式執行 ===
# === 新增：Top 資料夾路徑 ===
top_dir = os.path.join(project_root, "baseline", "Top")
user_bottom_dir = os.path.join(project_root, "baseline", "Bottom")

# === 批次處理 Top 資料夾中所有圖片 ===
for top_filename in sorted(os.listdir(top_dir)):
    if not top_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(top_dir, top_filename)
    print(f"\n處理上衣：{top_filename}")

    try:
        # 1. 擷取上衣的特徵
        top_label = get_top_label(image_path)

        # 2. 預測該上衣應該搭配的下身特徵
        predicted_bottom_label = predict_bottom_label(top_label)

        # 3. 與使用者的所有 Bottom 做距離比較，找出最推薦的
        best_match_info = compare_with_user_bottoms(
            predicted_bottom_label,
            user_bottom_dir,
            top_k=10,
            show_top_n=5,
            top_filename=top_filename
        )

        best_match, best_distance = best_match_info
        print(f"推薦下身：{best_match}（距離: {best_distance:.4f}）")

    except Exception as e:
        print(f"發生錯誤：{e}")
