# YOLO Object Detection API (No Copyright Statement) ✅

**Mô tả ngắn:**
Dự án cung cấp một API phát hiện vật thể dựa trên mô hình YOLO, dùng FastAPI làm backend. Ứng dụng hỗ trợ gửi URL ảnh hoặc upload ảnh từ máy, thực hiện phát hiện và trả về kết quả cộng hình ảnh đã đánh dấu (bounding boxes) — tất cả xử lý trong bộ nhớ, không lưu file lên đĩa.

---

## 🔧 Tính năng chính
- Nhận ảnh từ URL hoặc upload file qua form
- Trả về JSON kết quả cho API programmatic (`/predict/`)
- Giao diện web đơn giản (Jinja2) để submit URL/upload và xem ảnh đã annotate (`/` → form)
- Vẽ bounding boxes bằng OpenCV; không lưu ảnh lên đĩa (in-memory + trả về data URI)

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

---

## 💡 Gợi ý gỡ lỗi
- `Cannot download image` → kiểm tra URL (public & trả mã 200).
- `Cannot decode uploaded file` → kiểm tra file (khả năng OpenCV không đọc được; code có fallback dùng PIL).
- Nếu inference quá chậm, thử giảm `conf_input`/`iou_input` hoặc chạy trên máy có GPU.

---

Nếu bạn muốn tôi thêm: bản README tiếng Anh, Dockerfile mẫu, hoặc một Postman Collection mẫu để import — nói tôi biết mục nào bạn muốn tôi bổ sung.
