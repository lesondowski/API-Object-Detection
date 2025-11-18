from fastapi import APIRouter, HTTPException, Body
from api.schemas.request_model import PredictRequest
from models.predict import load_image_from_url, encode_image_to_base64, Detected
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

        if results_class.get(request.type_product) >= request.repuirements_count:
            results_class.update(
                {
                    "Kết Quả": "Đạt"
                }
            )
        else:
            results_class.update(
                {
                    "Kết Quả": "Không Đạt"
                }
            )


        final_results.append(results_class)



    return {
        "results":final_results
    }