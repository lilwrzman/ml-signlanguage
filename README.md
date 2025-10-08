# 🧩 Sign Language Detection – YOLO11x + OpenCV

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![YOLO11x](https://img.shields.io/badge/YOLO11x-8.3.205-green?logo=ultralytics)
![Status](https://img.shields.io/badge/Setup%20Test-Passed-success?logo=anaconda)

### 👋 Overview
This project focuses on **real-time recognition of American Sign Language (ASL)** hand gestures using **YOLO11x** for object detection and **OpenCV** for video streaming and inference.  
It aims to translate hand signs (A–Z letters, space, period, and comma gestures) into readable text in real-time.

---

### 🏗️ Project Architecture
Camera → YOLO11x Detection → Letter Buffer → Word/Punctuation Logic → Text Output (GUI)

---

### 🧠 Features
- ASL gesture detection (A–Z + punctuation)
- Real-time webcam inference with OpenCV
- Custom-trained YOLO11x model
- Streamlit/Tkinter GUI for live translation
- ONNX/TorchScript export for lightweight deployment

---

### ⚙️ Tech Stack
| Component | Technology |
|------------|-------------|
| Model | YOLO11x (Ultralytics) |
| Framework | PyTorch |
| Vision | OpenCV |
| UI | Streamlit / Tkinter |
| Dataset | ASL Kaggle Dataset |
| Language | Python 3.10 |

---

### 📦 Setup Guide

**1. Clone the Repository**
```bash
git clone https://github.com/lilwrzman/ml-signlanguage.git
cd signlanguage-detection
```

**2. Create Conda Environment**
```bash
conda create -n signlanguage python=3.10 -y
conda activate signlanguage
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run Initial Test Notebook**
To ensure all datasets and training results are stored inside this project (not in your AppData folder):
```bash
yolo settings datasets_dir="./data"
yolo settings runs_dir="./outputs"
```

**5. Run Initial Test Notebook**
Open the notebook:
```bash
notebooks/init_test.ipynb
```

This notebook performs a basic verification of:
- YOLO11x installation and version
- GPU/CUDA availability
- OpenCV camera initialization test
- Sample image inference

Run all cells in init_test.ipynb.
If all tests pass (model loads, GPU detected, and camera preview works), your environment is ready for dataset preparation and training.

---

### 📂 Folder Structure
```graphql
ml-signlanguage/
│
├── config/
│   └── data.yaml # Dataset config file
│
├── data/
│   ├── processed/ # YOLO-ready images 
│   └── raw/ # Original datasets (Kaggle ASL)and labels
│
├── models/
│   ├── checkpoints/ # Pretrained and trained YOLO weights
│   └── exports/ # ONNX / TorchScript exported models
│
├── notebooks/
│   ├── init_test.ipynb # Init test
│   └── main.ipynb
│
├── outputs/
│   └── logs/ # Training logs and metrics visualization
│
├── src/
│   ├── detection.py # Real-time gesture detection (webcam)
│   ├── inference.py # Static image/video detection
│   ├── app.py # Streamlit/Tkinter live demo
│   └── utils/
│       ├── dataset_utils.py
│       ├── preprocessing.py
│       └── postprocessing.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

### 🚀 Roadmap
- Project Initialization
- Dataset Preparation
- YOLO Training and Evaluation
- Real-Time Detection
- Word & Punctuation Logic
- UI + Deployment

---

### 🧩 Author

Egy Dya Hermawan | Application Developer / ML Enthusiast

📍 Indonesia | 📧 [Email](mailto:egydya.edh12@gmail.com) | 🔗 [LinkedIn](https://www.linkedin.com/in/egydyahermawan/)