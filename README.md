# Intelligent_Clothing_Tagging_and_Pairing_System
A deep learning-based system that automatically tags clothing images with attributes and recommends compatible outfit combinations based on style, color, and fabric.

---

## How to Run Backend and Frontend

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

**Note:**  
If you modify the backend code, restart the Flask server to apply changes.
