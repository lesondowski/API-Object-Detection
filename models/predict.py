import numpy as np
import cv2
import base64
import requests
from fastapi import HTTPException
from utils_cf import config
from ultralytics import YOLO
from collections import Counter



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
 

from collections import Counter

def safe_counter(counter_obj):
    return {int(k): int(v) for k, v in counter_obj.items()}

def get_max_class_count(model, results):
    boxes = results[0].boxes.xyxy.cpu().numpy()
    img = results[0].orig_img
    max_count = 0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]

        results_crop = model.predict(
            source=crop
        )

        cls_ids = results_crop[0].boxes.cls.cpu().numpy().astype(int)
        crop_count = Counter(cls_ids)
        crop_count = safe_counter(crop_count)

        max_count = max(max_count, crop_count.get(0, 0))
        
    return max_count


def detect_block(block_model, product_model, url, request):
    img = load_image_from_url(url=url)
    
    # Predict blocks
    results_block  = block_model.predict(
        source=img,
        conf = request.conf_input,
        iou = request.iou_input,
        classes  = [0]
    )
    cls_ids_block = results_block[0].boxes.cls.cpu().numpy().astype(int)
    counts_block = safe_counter(Counter(cls_ids_block))

    # Get max boxes per block
    max_boxes_per_block = get_max_class_count(model=product_model, results=results_block)

    # Predict products
    results_product = product_model.predict(
        source=img,
        conf = request.conf_input,
        iou = request.iou_input,
        classes=[0]
    )
    cls_ids_prd = results_product[0].boxes.cls.cpu().numpy().astype(int)
    counts_product = safe_counter(Counter(cls_ids_prd))

    # Build results dict safely
    results_dict = {
        "Số lượng khối": counts_block.get(0, 0),
        "Max thùng/khối": max_boxes_per_block,
        "Tổng Lượng thùng": counts_product.get(0, 0)
    } 
    return results_dict


def detect_block_image(block_model, product_model, img, request):
    """Same as `detect_block` but accepts an image (numpy array) instead of a URL."""
    # Predict blocks
    results_block  = block_model.predict(
        source=img,
        conf = request.conf_input,
        iou = request.iou_input,
        classes  = [0]
    )
    cls_ids_block = results_block[0].boxes.cls.cpu().numpy().astype(int)
    counts_block = safe_counter(Counter(cls_ids_block))

    # Get max boxes per block
    max_boxes_per_block = get_max_class_count(model=product_model, results=results_block)

    # Predict products
    results_product = product_model.predict(
        source=img,
        conf = request.conf_input,
        iou = request.iou_input,
        classes=[0]
    )
    cls_ids_prd = results_product[0].boxes.cls.cpu().numpy().astype(int)
    counts_product = safe_counter(Counter(cls_ids_prd))

    # Build results dict safely
    results_dict = {
        "Số lượng khối": counts_block.get(0, 0),
        "Max thùng/khối": max_boxes_per_block,
        "Tổng Lượng thùng": counts_product.get(0, 0)
    } 
    return results_dict
