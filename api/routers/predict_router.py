from fastapi import APIRouter, HTTPException, Body
from src.api.schemas.request_model import PredictRequest
from src.models.predict import load_image_from_url, detect_objects

router = APIRouter(prefix="/predict", tags=["Predict"])

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

    results = []
    for url in request.image_urls:
        img = load_image_from_url(url)
        class_counts, encoded = detect_objects(img)
        results.append({
            "class_counts": class_counts,
        })

    return {"results": results}
