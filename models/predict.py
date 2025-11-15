from ultralytics import YOLO
import numpy as np
import cv2
import base64
import requests
from collections import Counter
from fastapi import HTTPException
from utils_cf import config

MODELS_OD = config.MODEL_POD
MODELS_SEG = config.MODEL_PIS
DEVICE = config.DEVICE


model_instance_segmentation = YOLO(MODELS_SEG)

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


 