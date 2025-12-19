from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from api.routers import predict_router
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="API phát hiện vật thể bằng mô hình YOLO",
    version="1.0",
    description="YOLOv11"
)

# Mount static files (if needed for CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

app.include_router(predict_router.router)

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    """Render homepage with form to upload images or submit URLs."""
    return templates.TemplateResponse("home.html", {"request": request})

