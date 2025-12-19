# API phát hiện vật thể (YOLOv11) 🔧

**Mô tả ngắn:**
API phát hiện vật thể sử dụng mô hình YOLO (PyTorch / ONNX) và FastAPI. Nhận đầu vào là danh sách URL ảnh, trả về các vật thể được phát hiện (số lượng, bounding box/segmentation theo mô hình).

---

## 🔎 Điểm nổi bật
- Hỗ trợ cả mô hình **PyTorch (.pt)** và **ONNX (.onnx)**
- API đơn giản: POST `/predict/` với payload chứa `image_urls`
- Hỗ trợ phân loại theo **portion** (ví dụ: "Lon & Chai", "Thùng Hàng", "Khối Hàng")

---

## 📁 Cấu trúc dự án (chỉ mục chính)
- `api/` — FastAPI app
  - `main.py` — entrypoint (app)
  - `routers/predict_router.py` — endpoint `/predict/`
  - `schemas/request_model.py` — model request (Pydantic)
- `models/predict.py` — logic load ảnh và gọi model (detect/segment)
- `api/files/model/` — chứa mô hình (.pt, .onnx, tflite)
- `utils_cf/config.py` — cấu hình đường dẫn mô hình, DEVICE
- `requirements.txt` — dependencies

---

## ⚙️ Yêu cầu
- Python >= 3.10  
- Gói (có trong `requirements.txt`): `fastapi`, `uvicorn`, `ultralytics`, `onnxruntime`, `opencv-python`, `numpy`, `requests`, `Pillow`, `loguru`, v.v.

Cài đặt:

```bash
pip install -r requirements.txt
```

---

## 🛠 Cấu hình nhanh
File `utils_cf/config.py` chứa:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_OD_PT = "api/files/model/pt/ADYL11m.pt"
MODEL_IS_ONNX = "api/files/model/onnx/AD_YL11s_Seg_1.80.onnx"
MODEL_BLOCK_PT = "api/files/model/pt/AD_Productblock_YL11.pt"
```
- Thay đường dẫn nếu muốn dùng mô hình khác.
- `DEVICE` tự phát hiện GPU (nếu torch có CUDA), bạn có thể ép về `"cpu"` khi cần.

---

## ▶️ Chạy API
Khởi chạy server (mặc định port 10000 trong cấu hình hiện tại):

```bash
uvicorn api.main:app --host 0.0.0.0 --port 10000
```
Truy cập gốc: `GET /` (trả về thông báo server đang chạy)

---

## 📨 Endpoint chính

**POST** `/predict/`  
Body (JSON):

```json
{
  "image_urls": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"],
  "conf_input": 0.24,
  "iou_input": 0.2,
  "portion": "Lon & Chai",
  "repuirements_count": 1
}
```
- `image_urls`: danh sách URL ảnh (bắt buộc)
- `conf_input`: confidence threshold (mặc định ~0.24)
- `iou_input`: IOU threshold (mặc định ~0.2)
- `portion`: tên lớp tổng hợp (ví dụ "Lon & Chai", "Thùng Hàng", ...)
- `repuirements_count`: số lượng yêu cầu để được xem là "Đạt"

Ví dụ trả về (cấu trúc mẫu):

```json
{
  "results": [
    { "API URL IMAGE": "https://.../img1.jpg", "Lon & Chai": 2, "OtherClass": 1 },
    { "Số Lượng": 2, "Kết Quả": "Đạt", "Lý Do": "" }
  ]
}
```

---

## 🧾 Mô hình kèm theo (ví dụ có trong repo)
- PT: `api/files/model/pt/ADYL11m.pt`, `AD_Productblock_YL11.pt`, ...
- ONNX: `api/files/model/onnx/AD_YL11s_Seg_1.80.onnx`, ...
- TFLite (saved models) nằm trong `api/files/model/tflite/...`

---

## 💡 Ghi chú & hướng dẫn gỡ lỗi
- Nếu lỗi tải ảnh: kiểm tra URL (mã trả về 200). Hàm `load_image_from_url` sẽ ném HTTPException nếu tải thất bại.
- Nếu muốn chạy hoàn toàn trên CPU: ép `DEVICE = "cpu"` trong `utils_cf/config.py`.
- Khi dùng ONNX, cần `onnxruntime` được cài đặt.
