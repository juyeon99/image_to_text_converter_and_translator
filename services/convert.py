from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast
from PIL import Image
import requests
import models.image_to_text as img_to_txt, database.schemas as schemas
from sqlalchemy.orm import Session

# Load models once when the server starts
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
translation_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

def convert_image_to_text(
    image_to_text: schemas.ImageToTextCreate,
    db: Session
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
    
    db_imageToText = img_to_txt.ImageToText(
        url = image_url,
        original_text = generated_text,
        translated_text = translated_text
    )
    
    db.add(db_imageToText)
    db.commit()
    
    return db_imageToText