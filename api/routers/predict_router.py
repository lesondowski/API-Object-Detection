from fastapi import APIRouter, HTTPException, Body
from api.schemas.request_model import PredictRequest
from models.predict import Detected, max_detect, detect_block
from utils_cf import config
from ultralytics import YOLO


### call model predict functions from models/predict.py
MODEL_OD_PT = config.MODEL_OD_PT 
MODEL_IS_ONNX = config.MODEL_IS_ONNX 
DEVICE = config.DEVICE
MODEL_BLOCK = config.MODEL_BLOCK_PT


### Define the API router & 
router = APIRouter(prefix="/predict", tags=["Predict"])


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
