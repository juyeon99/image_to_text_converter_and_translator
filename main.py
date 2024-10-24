from fastapi import FastAPI
from routers import convert

# 손글씨 이미지에서 텍스트 추출하여 텍스트의 언어를 탐지 후 텍스트를 한국어로 번역해서 출력하는 프로그램
# 1. Image to Text
# 2. Text Classification
# 3. Translation

app = FastAPI()

app.include_router(convert.router)