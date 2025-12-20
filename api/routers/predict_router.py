from fastapi import APIRouter, HTTPException, Body, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from api.schemas.client import PredictRequest
from models.predict import Detected ,max_detect, detect_block, load_image_from_url, encode_image_to_base64, detect_block_image
from utils_cf import config
from ultralytics import YOLO
import json
import numpy as np
import cv2
from types import SimpleNamespace


### call model predict functions from models/predict.py
MODEL_OD_PT = config.MODEL_OD_PT 
MODEL_IS_ONNX = config.MODEL_IS_ONNX 
DEVICE = config.DEVICE
MODEL_BLOCK = config.MODEL_BLOCK_PT


### Define the API router & 
router = APIRouter(prefix="/predict", tags=["Predict"])

# Templates
templates = Jinja2Templates(directory="templates")

model_object_detection = YOLO(MODEL_OD_PT).to(DEVICE)
model_instance_segmentation = YOLO(MODEL_IS_ONNX, task="segment")
model_block = YOLO(model=MODEL_BLOCK, task="segment")


@router.post("/")
async def predict_images(request: PredictRequest = Body(...)):
    """
    Endpoint
    {
        "image_urls": ["https://example.com/image.jpg"]
    }
    """
    if not request.image_urls:
        raise HTTPException(status_code=400, detail="No image provided")

    final_results = []

    for url in request.image_urls:
        results_class = {
            "API URL IMAGE" : url
        }
        results_class.update(Detected(
            result=results_class,
            url=url,
            task="detect",
            model = model_object_detection,
            request=request
        ))
        results_class.update(Detected(
            result=results_class,
            url=url,
            task="segment",
            model=model_instance_segmentation,
            request=request
        ))
        if request.portion == "Thùng Hàng" or request.portion == "Khối Hàng":
            results_class.update(detect_block(block_model=model_block,
                                              product_model=model_instance_segmentation,
                                              url=url,
                                              request= request
                                        ))

        final_results.append(results_class)

    count = max_detect(results=final_results, portion=request.portion)


    if count>= request.repuirements_count:
        final_results.append({"Số Lượng" : count,"Kết Quả" : "Đạt", "Lý Do": ""})
    else:
        if count > 0:
            final_results.append({"Số Lượng" : count, "Kết Quả": "Không Đạt", "Lý do": "HÌNH ẢNH SẢN PHẨM KHÔNG ĐẠT YÊU CẦU - KHÔNG ĐỦ SỐ LƯỢNG"})
        else:
            final_results.append({"Số Lượng" : count, "Kết Quả": "Không Đạt","Lý do": "KHÔNG CÓ HÌNH ẢNH VẬT PHẨM TRƯNG BÀY"})

    return {
        "results":final_results
    }


# -----------------
# Web UI endpoints
# -----------------

def annotate_boxes(img, results, model):
    """Draw bounding boxes from model results onto image (in-place on a copy)."""
    out = img.copy()
    for r in results:
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = getattr(r.boxes, "conf", None)
        confs = confs.cpu().numpy() if confs is not None else [0]*len(cls_ids)

        for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            label = f"{model.names[int(cls_id)]} {conf:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            cv2.putText(out, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


@router.post("/detect_form", response_class=HTMLResponse)
async def detect_form(request: Request,
                      image_urls: str = Form(None),
                      files: list[UploadFile] = File(None),
                      conf_input: float = Form(0.24),
                      iou_input: float = Form(0.2),
                      portion: str = Form("Lon & Chai"),
                      repuirements_count: int = Form(1)):
    """Handle form submissions (URL list and file uploads). Returns a rendered HTML page with annotated images."""

    images_results = []

    # Normalize request-like object
    req = SimpleNamespace(conf_input=conf_input, iou_input=iou_input, portion=portion, repuirements_count=repuirements_count)

    # Process URLs
    if image_urls:
        urls = [u.strip() for u in image_urls.splitlines() if u.strip()]
        for url in urls:
            try:
                img = load_image_from_url(url)
            except HTTPException as e:
                images_results.append({"error": f"Cannot load URL: {url} - {str(e.detail)}", "url": url})
                continue

            # run detection and segmentation
            results_detect = model_object_detection(source=img, iou=req.iou_input, conf=req.conf_input)
            results_seg = model_instance_segmentation(source=img, iou=req.iou_input, conf=req.conf_input, device="cpu")

            # build counts
            counts = {}
            for r in results_detect:
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for cls_id in cls_ids:
                    name = str(model_object_detection.names[int(cls_id)])
                    counts[name] = counts.get(name, 0) + 1
            for r in results_seg:
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for cls_id in cls_ids:
                    name = str(model_instance_segmentation.names[int(cls_id)])
                    counts[name] = counts.get(name, 0) + 1

            # detect block data if required
            block_info = None
            if portion == "Thùng Hàng" or portion == "Khối Hàng":
                block_info = detect_block(block_model=model_block, product_model=model_instance_segmentation, url=url, request=req)

            # annotate image and encode
            annotated = annotate_boxes(img, results_detect, model_object_detection)
            data_uri = f"data:image/jpeg;base64,{encode_image_to_base64(annotated)}"

            images_results.append({"url": url, "image": data_uri, "counts": counts, "block": block_info})

    # Process uploaded files
    if files:
        for uploaded in files:
            content = await uploaded.read()
            if not content:
                images_results.append({"error": f"Empty file uploaded: {uploaded.filename}", "filename": uploaded.filename})
                continue

            arr = np.frombuffer(content, np.uint8)
            if arr.size == 0:
                images_results.append({"error": f"Cannot read uploaded file (empty buffer): {uploaded.filename}", "filename": uploaded.filename})
                continue

            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                # Try fallback with PIL
                try:
                    from PIL import Image
                    import io
                    pil_img = Image.open(io.BytesIO(content)).convert("RGB")
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    images_results.append({"error": f"Cannot decode uploaded file: {uploaded.filename} - {e}", "filename": uploaded.filename})
                    continue

            # Run detection
            try:
                results_detect = model_object_detection(source=img, iou=req.iou_input, conf=req.conf_input)
                results_seg = model_instance_segmentation(source=img, iou=req.iou_input, conf=req.conf_input, device="cpu")
            except Exception as e:
                images_results.append({"error": f"Model inference error for file: {uploaded.filename} - {e}", "filename": uploaded.filename})
                continue

            counts = {}
            for r in results_detect:
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for cls_id in cls_ids:
                    name = str(model_object_detection.names[int(cls_id)])
                    counts[name] = counts.get(name, 0) + 1
            for r in results_seg:
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for cls_id in cls_ids:
                    name = str(model_instance_segmentation.names[int(cls_id)])
                    counts[name] = counts.get(name, 0) + 1

            block_info = None
            if portion == "Thùng Hàng" or portion == "Khối Hàng":
                block_info = detect_block_image(block_model=model_block, product_model=model_instance_segmentation, img=img, request=req)

            annotated = annotate_boxes(img, results_detect, model_object_detection)
            data_uri = f"data:image/jpeg;base64,{encode_image_to_base64(annotated)}"

            images_results.append({"filename": uploaded.filename, "image": data_uri, "counts": counts, "block": block_info})

    for item in images_results:
        if "counts" in item and isinstance(item["counts"], dict):
            item["counts_json"] = json.dumps(
                item["counts"],
                indent=2,
                ensure_ascii=False
            )

    return templates.TemplateResponse("show_results.html", {"request": request, "images": images_results, "portion": portion})
