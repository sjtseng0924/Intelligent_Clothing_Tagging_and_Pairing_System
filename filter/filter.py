import os
import shutil
import random
from PIL import Image
import torch
import torchvision.transforms as T
from model import AttributeResNet  # 你模型定義的地方

# 讀 list.txt
def load_bad_pairs(file_path='list.txt'):
    bad_pairs = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ',' in line:
                a1, a2 = line.strip().split(',')
                bad_pairs.add((a1.strip(), a2.strip()))
                bad_pairs.add((a2.strip(), a1.strip()))  # 加反向，方便判斷
    return bad_pairs

# 載入屬性名稱
def load_attr_names(attr_file='list_attr_cloth.txt'):
    attr_names = []
    with open(attr_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[2:]
        for line in lines:
            parts = line.strip().split()
            name = ' '.join(parts[:-1])
            attr_names.append(name)
    return attr_names

# 圖片轉換
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 預測圖片屬性，回傳屬性名稱list(分數>0.2)
def predict_attributes(model, image_path, attr_names, threshold=0.02, device='cpu'):
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze()
    attrs = [attr_names[i] for i, p in enumerate(probs) if p >= threshold]
    return attrs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttributeResNet(num_labels=1000)
    model.load_state_dict(torch.load('best_tagger.pth', map_location=device))
    model.eval()
    model.to(device)

    attr_names = load_attr_names()
    bad_pairs = load_bad_pairs('list_filter.txt')

    top_dir = 'Re-PolyVore/top'
    bottom_dir = 'Re-PolyVore/bottom'
    unmatch_dir = 'unmatching3_dataset'
    os.makedirs(unmatch_dir, exist_ok=True)

    # 遞迴抓取 top_dir 和 bottom_dir 底下所有圖片檔案
    def get_all_image_files(root_dir):
        image_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    image_files.append(os.path.join(dirpath, fname))
        return sorted(image_files)

    top_files = get_all_image_files(top_dir)
    bottom_files = get_all_image_files(bottom_dir)

    count = 0
    for top_file in top_files:
        top_path = top_file
        top_attrs = predict_attributes(model, top_path, attr_names, device=device)

        # 隨機選 10 個 bottom
        sampled_bottoms = random.sample(bottom_files, min(1, len(bottom_files)))

        for bottom_file in sampled_bottoms:
            bottom_path = bottom_file
            bottom_attrs = predict_attributes(model, bottom_path, attr_names, device=device)

            # 檢查是否有不協調屬性對
            found_bad = False
            for a1 in top_attrs:
                for a2 in bottom_attrs:
                    if (a1, a2) in bad_pairs:
                        found_bad = True
                        break
                if found_bad:
                    break

            if found_bad:
                # 建資料夾存放 top.jpg bottom.jpg
                count += 1
                pair_folder = os.path.join(unmatch_dir, f'unmatch_{count}')
                os.makedirs(pair_folder, exist_ok=True)
                shutil.copy(top_path, os.path.join(pair_folder, 'top.jpg'))
                shutil.copy(bottom_path, os.path.join(pair_folder, 'bottom.jpg'))
                print(f"不協調配對: {top_file} + {bottom_file} → {pair_folder}")
                break

if __name__ == '__main__':
    main()