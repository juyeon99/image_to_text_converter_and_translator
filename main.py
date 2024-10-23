from flask import Flask, request, jsonify
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast
from PIL import Image
import requests
import io
from fastapi import FastAPI, Request, Depends, Form, status
from fastapi.templating import Jinja2Templates
import models, schemas
from database import engine, session_local
from sqlalchemy.orm import Session
from fastapi.responses import RedirectResponse

# 손글씨 이미지에서 텍스트 추출하여 텍스트의 언어를 탐지 후 텍스트를 한국어로 번역해서 출력하는 프로그램
# 1. Image to Text
# 2. Text Classification
# 3. Translation

templates = Jinja2Templates(directory="template")
app = FastAPI()

models.Base.metadata.create_all(bind=engine)

def get_db():
    db = session_local()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def home(request: Request, db: Session=Depends(get_db)):
    return templates.TemplateResponse("base.html", {"request":request})

# Load models once when the server starts
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

@app.post("/convert", response_model=schemas.ImageToTextResponse)
def convert(
    image_to_text: schemas.ImageToTextCreate,
    db: Session=Depends(get_db)
):
    image_url = image_to_text.url
    
    # Process the image
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Detect language
    lang_result = lang_detector(generated_text)
    detected_lang = lang_result[0]['label']
    
    # Translate to Korean
    tokenizer.src_lang = detected_lang + "_" + detected_lang.upper()

    if (detected_lang == 'en' or detected_lang == 'es' or detected_lang == 'fr' or detected_lang == 'ja' or detected_lang == 'pt'):
        tokenizer.src_lang = detected_lang + "_XX"
    
    encoded_lang = tokenizer(generated_text, return_tensors="pt")
    generated_tokens = translation_model.generate(
        **encoded_lang,
        forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"]
    )
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    db_imageToText = models.ImageToText(
        url = image_url,
        original_text = generated_text,
        translated_text = translated_text
    )
    
    db.add(db_imageToText)
    db.commit()
    
    return db_imageToText

@app.get("/conversions", response_model=list[schemas.ImageToTextResponse])
async def find_all_conversions(db: Session=Depends(get_db)):
    all_conversions = db.query(models.ImageToText).all()
    return all_conversions

if __name__ == '__main__':
    app.run(debug=True)