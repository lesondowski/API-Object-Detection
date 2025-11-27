from fastapi import FastAPI
from api.routers import predict_router

app = FastAPI(
    title="API phát hiện vật thể bằng mô hình YOLO",
    version="1.0",
    description="YOLOv11"
)

app.include_router(predict_router.router)

@app.get("/")
def root():
    return {"message": "Uvicorn running on http://0.0.0.0:10000"}

