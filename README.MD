# YOLO Object Detection API 

**Short Description:**  
This project provides an object detection API powered by a YOLO-based model, using FastAPI as the backend. The application supports both image URLs and direct image uploads from local machines, performs object detection, and returns detection results along with annotated images (bounding boxes).

---

## 🔧 Key Features
- Accept images via URL or direct file upload
- JSON-based inference API for programmatic usage (`/predict/`)
- Simple web interface (Jinja2) to submit URLs or upload images and view annotated results (`/` → form)
- Bounding boxes rendered with OpenCV; no file persistence (in-memory processing with data URI responses)

---

## 📁 Project Structure (Summary)
- `api/` — FastAPI application
  - `main.py` — entry point and templates/static configuration
  - `routers/predict_router.py` — API and web routes
- `models/predict.py` — image loading, model inference, image processing helpers
- `api/files/model/` — model files (.pt, .onnx, .tflite)
- `templates/` — Jinja2 HTML templates (home / results)
- `static/` — static assets (CSS/JS if needed)
- `requirements.txt` — required Python dependencies

---

## ⚙️ Requirements & Installation
- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt

---

## ▶️ Chạy server (local)
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 10000
```
Mở trình duyệt tới `http://localhost:10000/` để truy cập giao diện web, hoặc dùng Postman để gọi API.

---

## 📬 API & Web endpoints
1) POST `/predict/` (JSON) — programmatic
- Body JSON:
```json
{
  "image_urls": ["https://example.com/img1.jpg"],
  "conf_input": 0.24,
  "iou_input": 0.2,
  "portion": "Lon & Chai",
  "repuirements_count": 1
}
```
- Response: HJSON containing detection statistics (object counts per class, block information if applicable).


2) POST `/predict/detect_form` (form-data) — web form / Postman file upload
- Fields (form-data):
  - `image_urls` (text, mỗi URL 1 dòng)
  - `files` (file, hỗ trợ nhiều files)
  - `conf_input`, `iou_input`, `portion`, `repuirements_count`
- Response: HTML results page containing annotated images (embedded as data URIs) and JSON counts.

3) GET `/` — Web Form
- Web interface to input image URLs or upload image files.
---

## 🧪 Quick Testing with Postman / curl
- JSON request (URL flow):
```bash
curl -X POST "http://localhost:10000/predict/" -H "Content-Type: application/json" -d '{"image_urls":["https://example.com/img.jpg"]}'
```
- File upload (form):
```bash
curl -X POST "http://localhost:10000/predict/detect_form" -F "files=@/path/to/image.jpg" -F "conf_input=0.24"
```
