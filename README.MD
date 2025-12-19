# YOLO Object Detection API (No Copyright Statement) ✅

**Short Description:**
This project provides an object detection API powered by a YOLO-based model, using FastAPI as the backend. The application supports both image URLs and direct image uploads from local machines, performs object detection, and returns results along with annotated images (bounding boxes).

---

## 🔧 Tính năng chính
- Accept images via URL or direct file upload
- JSON-based inference API for programmatic usage (`/predict/`)
- Simple web interface (Jinja2) to submit URLs or upload images and view annotated results (`/` → form)
- Bounding boxes rendered with OpenCV; no file persistence (in-memory processing with data URI responses)

---

## 📁 Cấu trúc thư mục (tóm tắt)
- `api/` — FastAPI app
  - `main.py` — entrypoint và cấu hình templates/static
  - `routers/predict_router.py` — API và web routes
- `models/predict.py` — load ảnh, gọi model, helper xử lý ảnh
- `api/files/model/` — chứa các mô hình (.pt, .onnx, tflite)
- `templates/` — Jinja2 HTML templates (home / results)
- `static/` — tài nguyên tĩnh (CSS/JS nếu cần)
- `requirements.txt` — thư viện cần cài đặt

---

## ⚙️ Yêu cầu & cài đặt
- Python 3.10+
- Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

Lưu ý: lần chạy đầu tiên có thể chậm do tải mô hình; cần có kết nối internet nếu mô hình cần tải.

---

## 🛠 Cấu hình mô hình (quick)
Mở `utils_cf/config.py` để chỉnh:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_OD_PT = "api/files/model/pt/ADYL11m.pt"
MODEL_IS_ONNX = "api/files/model/onnx/AD_YL11s_Seg_1.80.onnx"
MODEL_BLOCK_PT = "api/files/model/pt/AD_Productblock_YL11.pt"
```
- Thay đường dẫn tới mô hình mong muốn nếu cần.
- Để ép chạy trên CPU, đặt `DEVICE = "cpu"`.

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
- Trả về: JSON chứa thống kê phát hiện (số lượng theo lớp, thông tin block nếu áp dụng).

2) POST `/predict/detect_form` (form-data) — web form / Postman file upload
- Fields (form-data):
  - `image_urls` (text, mỗi URL 1 dòng)
  - `files` (file, hỗ trợ nhiều files)
  - `conf_input`, `iou_input`, `portion`, `repuirements_count`
- Trả về: HTML trang kết quả có ảnh đã annotate (embedded data URI) và JSON counts.

3) GET `/` — form web để input URL / upload ảnh.

---

## 🧪 Test nhanh bằng Postman / curl
- JSON request (URL flow):
```bash
curl -X POST "http://localhost:10000/predict/" -H "Content-Type: application/json" -d '{"image_urls":["https://example.com/img.jpg"]}'
```
- File upload (form):
```bash
curl -X POST "http://localhost:10000/predict/detect_form" -F "files=@/path/to/image.jpg" -F "conf_input=0.24"
```
