from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class PredictRequest(BaseModel):
    image_urls: Optional[List[HttpUrl]] = None  # Danh sách URL ảnh
    conf_input:  Optional[float] = 0.24 ### Rank chỉ nên để từ 0.2 - 0.6, nếu các trường hợp bất khả kháng mới nên điều chỉnh 0.1
    iou_input:  Optional[float] = 0.2 ### thường để mặc định, nhưng chỉ nên setup rank từ 0.1 - 0.3
    

