# 👗 Intelligent Clothing Tagging and Pairing System

A deep learning-based system that automatically tags clothing images with attributes and recommends compatible outfit combinations based on style, color, and fabric. The project includes attribute prediction, color classification, outfit compatibility modeling, and a web interface for user interaction.

---

## 📝 Project Introduction

This project aims to build an intelligent system for clothing attribute tagging and outfit pairing. It leverages deep learning models to:

- **Tag clothing images** with fine-grained attributes (style, fabric, pattern, etc.)
- **Classify dominant colors** in clothing images
- **Recommend compatible outfit pairs** (e.g., top and bottom) based on learned compatibility
- **Provide a web interface** for users to upload images and receive recommendations

The system is modular, supporting both research/analysis scripts and a user-facing web application.

---

## 📁 Project Structure

```
Intelligent_Clothing_Tagging_and_Pairing_System/
│
├── app.py                      # Flask web backend
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── static/                     # Static files for web
│   ├── user_top/
│   ├── user_bottom/
│   └── net_bottom/
│
├── templates/                  # HTML templates for Flask
│   └── index.html
│
├── model/                      # Model code and checkpoints
│   ├── attr_label/             # Attribute tagging model
│   │   ├── model.py
│   │   ├── best_tagger.pth
│   │   └── list_attr_cloth.txt
│   ├── color_label/            # Color classifier
│   │   ├── color_classifier.pt
│   │   └── ...
│   ├── main_approach/          # Outfit compatibility model (main)
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── attr_extractor.py
│   │   ├── color_util.py
│   │   └── ...
│   └── baseline/               # Baseline model and scripts
│       ├── train.py
│       ├── inference.py
│       ├── pair_train/
│       ├── Top/
│       ├── Bottom/
│       └── ...
│
├── test_code/                  # Testing scripts
│   └── filter/
│       ├── filter.py
│       ├── list_filter.txt
│       └── ...
│
└── analysis/                   # Analysis and evaluation scripts
    ├── analysis.py
    ├── compare.py
    ├── rank.py
    └── sort.py
```

---

## 🧩 Main Components

- **Attribute Tagging:**  
  Uses a ResNet-based model to predict fine-grained clothing attributes from images.

- **Color Classification:**  
  Classifies dominant clothing colors using a separate model.

- **Outfit Compatibility:**  
  Trains an MLP model to score the compatibility of top-bottom pairs based on their attributes and colors.

- **Web Interface:**  
  Flask-based frontend for uploading images and receiving outfit recommendations.

- **Analysis & Evaluation:**  
  Scripts for ranking, comparing, and analyzing model performance.

---

## 🚀 How to Run Backend and Frontend

### 1. Backend (Flask API)

- Make sure all dependencies are installed (see requirements).
- Place your trained model files in the correct locations as referenced in `app.py`.
- Run the backend server:
  ```bash
  python app.py
  ```
- The Flask server will start (default: http://127.0.0.1:5000).

### 2. Frontend

- The frontend is served by Flask using HTML templates (e.g., `templates/index.html`).
- Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000) to use the web interface.

---

## ⚙️ Additional Usage

- **Model Training:**  
  See scripts in `model/attr_label/`, `model/color_label/`, and `model/main_approach/` for training and evaluation.

- **Dataset Preparation:**  
  Use scripts in `dataset/` to organize and preprocess data.

- **Analysis:**  
  Use scripts in `analysis/` to evaluate and compare model performance.

---

**Note:**  
If you modify the backend code, restart the Flask server to apply changes.
