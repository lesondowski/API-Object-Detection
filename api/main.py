from fastapi import FastAPI
from src.api.routers import predict_router

app = FastAPI(
    title="API phát hiện vật thể bằng mô hình YOLO",
    version="1.0",
    description="YOLOv11"
)

app.include_router(predict_router.router)

@app.get("/")
def root():
    return {"message": "AUDIT YOLOV11 VERSION 1.O API is running"}

