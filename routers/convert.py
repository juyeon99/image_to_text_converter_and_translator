from fastapi import APIRouter
from models.image_to_text import ImageToText
import models.image_to_text as image_to_text, database.schemas as schemas
from sqlalchemy.orm import Session
from fastapi import Request, Depends
from fastapi.templating import Jinja2Templates
from database.database import engine, session_local
from sqlalchemy.orm import Session
from services.convert import convert_image_to_text

image_to_text.Base.metadata.create_all(bind=engine)

def get_db():
    db = session_local()
    try:
        yield db
    finally:
        db.close()

templates = Jinja2Templates(directory="template")

router = APIRouter()

@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})

@router.post("/convert", response_model=schemas.ImageToTextResponse)
async def convert(
    image_to_text: schemas.ImageToTextCreate,
    db: Session=Depends(get_db)
):
    result = convert_image_to_text(image_to_text, db)
    return result

@router.get("/conversions", response_model=list[schemas.ImageToTextResponse])
async def find_all_conversions(db: Session=Depends(get_db)):
    all_conversions = db.query(ImageToText).all()
    return all_conversions