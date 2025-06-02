from flask import Flask, render_template, request, jsonify, redirect
from werkzeug.utils import secure_filename
import os
import torch
import numpy as np

# === CLI 版模型工具 ===
from model.main_approach.attr_extractor import AttributeExtractor
from model.main_approach.color_util import ColorExtractor
from model.main_approach.model import AttributeMLPCompatibility

app = Flask(__name__)

# === 資料夾設定 ===
project_root = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.join(project_root, 'static', 'user_top')
bottom_dir = os.path.join(project_root, 'static', 'user_bottom')
net_bottom_dir = os.path.join(project_root, 'static', 'net_bottom')

# === 模型載入與初始化（與 inference.py 相同） ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
attr_ckpt = os.path.join(project_root, "model", "attr_label", "best_tagger.pth")
color_ckpt = os.path.join(project_root, "model", "color_label", "color_classifier.pt")
compat_ckpt = os.path.join(project_root, "model", "main_approach", "compatibility_mlp_main_v2.pth")

attr_extractor = AttributeExtractor(attr_ckpt, device)
color_extractor = ColorExtractor(color_ckpt, device, num_colors=12)
num_labels = len(attr_extractor.attr_names) + 12

pairing_model = AttributeMLPCompatibility(num_labels=num_labels)
pairing_model.load_state_dict(torch.load(compat_ckpt, map_location=device))
pairing_model.eval().to(device)

# === 功能方法 ===
def get_full_label(image_path, threshold=0.02):
    attr_probs = attr_extractor.extract(image_path)
    attr_bin = (torch.tensor(attr_probs) > threshold).float()
    color_onehot = torch.tensor(color_extractor.extract(image_path)).float()
    full_label = torch.cat([attr_bin, color_onehot]).unsqueeze(0).to(device)
    return full_label

def compute_top_k(query_label, candidate_dir, is_top_query=True, k=3):
    scores = []
    filenames = []

    for filename in os.listdir(candidate_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue
        img_path = os.path.join(candidate_dir, filename)
        candidate_label = get_full_label(img_path)
        with torch.no_grad():
            if is_top_query:
                score = pairing_model(query_label, candidate_label)
            else:
                score = pairing_model(candidate_label, query_label)
            scores.append(score.item())
            filenames.append(filename)

    # 排序取前 k 名
    top_indices = np.argsort(scores)[::-1][:k]
    top_filenames = [filenames[i] for i in top_indices]
    return top_filenames


# === 首頁 ===
@app.route('/')
def index():
    return render_template(
        'index.html',
        user_tops=os.listdir(top_dir),
        user_bottoms=os.listdir(bottom_dir),
        net_bottoms=os.listdir(net_bottom_dir)
    )

# === 推薦 API ===
@app.route('/recommend', methods=['POST'])
def recommend():
    image_name = request.form['image']
    image_type = request.form['type']

    if image_type != 'user_top':
        return jsonify({'error': '只支援上衣推薦下身'}), 400

    image_path = os.path.join(top_dir, image_name)
    top_label = get_full_label(image_path)

    best_users = compute_top_k(top_label, bottom_dir, is_top_query=True, k=3)
    best_nets = compute_top_k(top_label, net_bottom_dir, is_top_query=True, k=3)

    return jsonify({
        'user_paths': [f'user_bottom/{f}' for f in best_users],
        'net_paths': [f'net_bottom/{f}' for f in best_nets]
    })


# === 上傳衣物圖片 ===
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "未選擇檔案", 400
    file = request.files['file']
    category = request.form.get('category')
    if file.filename == '':
        return "檔名為空", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if category == 'user_top':
            save_path = os.path.join(top_dir, filename)
        elif category == 'user_bottom':
            save_path = os.path.join(bottom_dir, filename)
        else:
            return "無效的類別", 400
        file.save(save_path)
        return redirect('/')
    return "上傳失敗：不支援的檔案格式", 400

# === 啟動應用 ===
if __name__ == '__main__':
    app.run(debug=True)
