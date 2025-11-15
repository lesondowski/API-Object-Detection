from fastapi import APIRouter, HTTPException, Body
from api.schemas.request_model import PredictRequest
from models.predict import load_image_from_url
from utils_cf import config
from ultralytics import YOLO


### call model predict functions from models/predict.py
MODELS_OD = config.MODEL_POD
MODELS_SEG = config.MODEL_PIS
DEVICE = config.DEVICE

### Define the API router &
router = APIRouter(prefix="/predict", tags=["Predict"])

model_object_detection = YOLO(MODELS_OD).to(DEVICE)
model_instance_segmentation = YOLO(MODELS_SEG , task="segment")



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
            "API URL IMAGE": url
        }
        img = load_image_from_url(url)
        results_can = model_object_detection(
            source=img,
            iou = request.iou_input,
            conf = request.conf_input
        )   
        for r in results_can:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = str(model_object_detection.names[class_id])
                results_class[class_name] = results_class.get(class_name,0) + 1

        results_other = model_instance_segmentation(
                            source=img,
                            iou = request.iou_input,
                            conf = request.conf_input,
                            device="cpu"
                        )
        for r in results_other:
            for box in r.boxes:
                class_id = int(box.cls) 
                class_name = str(model_instance_segmentation.names[class_id])
                results_class[class_name] = results_class.get(class_name,0) + 1
        final_results.append(results_class)
    return {
        "results":final_results
    }
