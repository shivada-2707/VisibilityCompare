# Visibility Project (RAW / CLAHE / AOD-Net) — YOLOv5

Short description: compare RAW, CLAHE and AOD-Net processed images using YOLOv5 and show results via Flask GUI.

## Run locally
1. create virtualenv: `python3 -m venv venv`
2. activate: `source venv/bin/activate`
3. install: `pip install -r requirements.txt`
4. run: `python app.py`

Add screenshots to `static/screenshots/` and reference them in this README.
# VisibilityCompare

This is my MCA project (RAW, CLAHE, and AOD-Net comparison).
# Visibility Project (RAW / CLAHE / AOD-Net) using YOLOv5  

## 🔍 About
This project compares object detection in foggy images using three approaches:
- **RAW** images  
- **CLAHE** enhanced images (traditional method)  
- **AOD-Net** dehazed images (deep learning method)
This project is a proof of deep learning method is better than traditional method

## ⚙️ Tech Stack
- Python, Flask  
- YOLOv5  
- OpenCV  
- PyTorch  

## ▶️ Run Locally
1. Clone the repo  
   ```bash
   git clone https://github.com/shivada-2707/VisibilityCompare.git
   cd VisibilityCompare

## Screenshots

### Home Page
![Home Page](static/screenshots/homepage.png)

### Comparison Page
![Comparison Page1](static/screenshots/comparisonpage1.png)
![Comparison Page2](static/screenshots/comparisonpage2.png)
![Comparison Page3](static/screenshots/comparisonpage3.png)
