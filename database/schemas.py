from pydantic import BaseModel
from typing_extensions import Optional

class ImageToTextBase(BaseModel):
    url : str
    original_text : Optional[str] = None
    translated_text : Optional[str] = None

class ImageToTextCreate(ImageToTextBase):
    pass

class ImageToTextResponse(ImageToTextBase):
    id: int