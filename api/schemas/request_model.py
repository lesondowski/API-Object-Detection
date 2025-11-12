from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class PredictRequest(BaseModel):
    image_urls: Optional[List[HttpUrl]] = None  # Danh sách URL ảnh (nếu không upload file)
