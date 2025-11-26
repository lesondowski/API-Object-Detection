import numpy as np
import cv2
import base64
import requests
from fastapi import HTTPException
from utils_cf import config
from ultralytics import YOLO


MODEL_OD_PT = config.MODEL_OD_PT 
MODEL_IS_ONNX = config.MODEL_IS_ONNX 
DEVICE = config.DEVICE

MODEL_OD = YOLO(MODEL_OD_PT).to(DEVICE)
MODEL_IS = YOLO(MODEL_IS_ONNX, task="segment")




def load_image_from_url(url: str):
    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Cannot download image from {url}")
    arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def encode_image_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")

def Detected(result: dict, url: str, task: str, model, request):
    img = load_image_from_url(url)
    if task =="detect":
        result_can = model(
            source=img,
            iou = request.iou_input,
            conf = request.conf_input
        )
        for r in result_can:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = str(model.names[class_id])
                result[class_name] = result.get(class_name,0) + 1
    if task == "segment":
        results_other = model(
            source=img,
            iou = request.iou_input,
            conf = request.conf_input,
            device ="cpu"            
        )
        for r in results_other:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = str(model.names[class_id])
                result[class_name] = result.get(class_name,0) + 1
    return result
def max_detect(results: list, portion: str):
    values = [r_dict[portion] for r_dict in results if portion in r_dict]
    return max(values, default=0)


def detect_product(task:str, model_product, url:str, request, results:dict, model_block):
    img = load_image_from_url(url=url)

    if task == "count":
        results = model_product(
            source = img,
            iou = request.iou_input,
            conf= request.conf_input,
            device = "cpu",
            classes = [0]
        )
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = str(model_product.names[class_id])
                results[class_name] = results.get(class_name,0) + 1
        


