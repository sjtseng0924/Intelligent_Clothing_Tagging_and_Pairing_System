from flask import Flask, render_template, request, jsonify, redirect
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import numpy as np
from torchvision import transforms as T
from label_train.model import AttributeResNet
from model2_2.train import Top2BottomModel

app = Flask(__name__)

# === 資料夾設定 ===
project_root = os.path.dirname(os.path.abspath(__file__))
top_dir = os.path.join(project_root, 'static', 'user_top')
bottom_dir = os.path.join(project_root, 'static', 'user_bottom')
net_top_dir = os.path.join(project_root, 'static', 'net_top')
net_bottom_dir = os.path.join(project_root, 'static', 'net_bottom')

# === 模型與轉換 ===
num_labels = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attribute_model = AttributeResNet(num_labels=num_labels)
attribute_model.load_state_dict(torch.load(
    os.path.join(project_root, "label_train", "saved_models", "best_tagger.pth"),
    map_location=device
))
attribute_model.eval().to(device)

pairing_model = Top2BottomModel(input_dim=num_labels, output_dim=num_labels)
pairing_model.load_state_dict(torch.load(
    os.path.join(project_root, "model2_2", "best_model.pt"),
    map_location=device
))
pairing_model.eval().to(device)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

# === 功能方法 ===
def get_label(image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = attribute_model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().tolist()
    return probs

def predict_label(input_label):
    top_tensor = torch.tensor(input_label, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = pairing_model(top_tensor).squeeze().cpu().tolist()
    return pred

def compare_with_closet(predicted_label, search_dir, top_k=10):
    top_indices = np.argsort(predicted_label)[-top_k:]
    min_distance = float('inf')
    best_match_image = None

    for filename in os.listdir(search_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            continue
        img_path = os.path.join(search_dir, filename)
        label = get_label(img_path)
        pred_vec = np.array(predicted_label)[top_indices]
        label_vec = np.array(label)[top_indices]
        distance = np.linalg.norm(pred_vec - label_vec)
        if distance < min_distance:
            min_distance = distance
            best_match_image = filename

    return best_match_image

# === 首頁 ===
@app.route('/')
def index():
    return render_template(
        'index.html',
        user_tops=os.listdir(top_dir),
        user_bottoms=os.listdir(bottom_dir),
        net_tops=os.listdir(net_top_dir),
        net_bottoms=os.listdir(net_bottom_dir)
    )

# === 推薦 API ===
@app.route('/recommend', methods=['POST'])
def recommend():
    image_name = request.form['image']
    image_type = request.form['type']

    if image_type == 'user_top':
        image_path = os.path.join(top_dir, image_name)
        top_label = get_label(image_path)
        predicted_bottom_label = predict_label(top_label)

        best_user = compare_with_closet(predicted_bottom_label, bottom_dir)
        best_net = compare_with_closet(predicted_bottom_label, net_bottom_dir)

        return jsonify({
            'user_path': f'user_bottom/{best_user}',
            'net_path': f'net_bottom/{best_net}'
        })

    elif image_type == 'user_bottom':
        image_path = os.path.join(bottom_dir, image_name)
        bottom_label = get_label(image_path)
        predicted_top_label = predict_label(bottom_label)

        best_user = compare_with_closet(predicted_top_label, top_dir)
        best_net = compare_with_closet(predicted_top_label, net_top_dir)

        return jsonify({
            'user_path': f'user_top/{best_user}',
            'net_path': f'net_top/{best_net}'
        })

    return jsonify({'error': '不支援的選擇類型'}), 400

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
