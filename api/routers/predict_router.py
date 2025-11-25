from fastapi import APIRouter, HTTPException, Body
from api.schemas.request_model import PredictRequest
from models.predict import Detected, max_detect
from utils_cf import config
from ultralytics import YOLO


### call model predict functions from models/predict.py
MODEL_OD_PT = config.MODEL_OD_PT 
MODEL_IS_ONNX = config.MODEL_IS_ONNX 
DEVICE = config.DEVICE


### Define the API router & 
router = APIRouter(prefix="/predict", tags=["Predict"])


model_object_detection = YOLO(MODEL_OD_PT).to(DEVICE)
model_instance_segmentation = YOLO(MODEL_IS_ONNX, task="segment")


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
        final_results.append(results_class)

    count = max_detect(results=final_results, portion=request.portion)

    for item in final_results:
        if count >= request.repuirements_count:
            item["Số Lượng"] = count
            item["Kết Quả"] = "Đạt"
        else:
            item["Số lượng"] = count
            item["Kết Quả"] = "Không Đạt"

            if count > 0:
                item["Lý Do"] = "HÌNH ẢNH SẢN PHẨM KHÔNG ĐẠT YÊU CẦU - KHÔNG ĐỦ SỐ LƯỢNG"
            else:
                item["Lý Do"] = "KHÔNG CÓ HÌNH ẢNH VẬT PHẨM TRƯNG BÀY"

    return {
        "results":final_results
    }
