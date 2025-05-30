import torch
from PIL import Image
import torchvision.transforms as transforms
from model import ColorClassifier
from dataset_attr import ColorDataset  # 用來抓 class_names
import os

# === 參數設定 ===
data_path = "color_dataset"
class_names = sorted(os.listdir(data_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 載入模型 ===
model = ColorClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load("color_classifier.pt", map_location=device))
model.to(device)
model.eval()

# === 預處理 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === 預測函數 ===
def predict_image_colors(img_path, threshold=0.3):
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = model(input_tensor).squeeze().cpu().numpy()

    result = {}
    for i, prob in enumerate(probs):
        result[class_names[i]] = round(float(prob), 3)  # 四捨五入方便看

    print(f"\n預測結果（圖片：{img_path}）")
    for color, score in result.items():
        if score > threshold:
            print(f"{color}: {score}")
        else:
            print(f"   {color}: {score}")
    return result

# === 測試用範例 ===
if __name__ == "__main__":
    test_img = input("請輸入圖片路徑： ")
    predict_image_colors(test_img)
