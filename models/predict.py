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


model_instance_segmentation = YOLO(MODELS_SEG).to(DEVICE)

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

def detect_objects(image):
    results = MODELS_OD.predict(
        source=image,
        conf=0.25,
        iou=0.35,
        save=False,
        show=False,
        verbose=False
    )
    class_counts = Counter()
    for r in results:
        classes = r.boxes.cls.cpu().numpy()
        for cls in classes:
            class_name = MODELS_OD.names[int(cls)]
            class_counts[class_name] += 1

    plotted = results[0].plot()  # Ảnh đã vẽ bounding box
    encoded_image = encode_image_to_base64(plotted)

    return dict(class_counts), encoded_image

def detect_instance_segmentation(model,image, class_index=0):

    results = model.predict(
        source=image,
        conf=0.25,
        iou=0.35,
        save=False,
        show=False,
        verbose=False,
        classes=class_index
    )
    class_counts = Counter()
    for r in results:
        classes = r.boxes.cls.cpu().numpy()
        for cls in classes:
            class_name = model.names[int(cls)]
            class_counts[class_name] += 1
    plotted = results[0].plot()
    encoded_image = encode_image_to_base64(plotted)
    return dict(class_counts), encoded_image




 