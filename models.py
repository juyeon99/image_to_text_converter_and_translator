from database import Base
from sqlalchemy import Column, Integer, String, Float

class ImageToText(Base):
    __tablename__ = 'imageToText'
    
    id = Column(Integer, primary_key=True)
    url = Column(String(512))
    original_text = Column(String(512))
    translated_text = Column(String(512))
