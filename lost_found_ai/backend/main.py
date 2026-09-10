from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import mysql.connector
import os
import shutil
import cloudinary
import cloudinary.uploader
import math
import requests
import numpy as np
import onnxruntime as ort

from PIL import Image
from io import BytesIO

from backend.download_model import download_model, MODEL_PATH

# ==========================================
# MobileCLIP2-S0 ONNX 모델 준비
# ==========================================

download_model()

ai_session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("MobileCLIP2-S0 ONNX 모델 로딩 완료")

app = FastAPI()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)


# ==========================================
# CORS 설정
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 업로드 폴더
# ==========================================

UPLOAD_DIR = "uploads"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads"
)


# ==========================================
# MySQL 연결
# ==========================================

def get_db():
    return mysql.connector.connect(
        host="mysql-8e81ffd-sh251200.b.aivencloud.com",
        port=17785,
        user="avnadmin",
        password=os.getenv("DB_PASSWORD"),
        database="lost_found_db"
    )


# ==========================================
# AI 모델 설정 - MobileCLIP2 ONNX
# ==========================================

print("서버 시작 완료 - MobileCLIP2 ONNX 사용")


# ==========================================
# 이미지 특징 추출 함수
# ==========================================

def get_image_features(image_url):

    response = requests.get(
        image_url,
        timeout=30
    )

    response.raise_for_status()

    image = Image.open(
        BytesIO(response.content)
    ).convert("RGB")


    # MobileCLIP2-S0 입력 크기
    image = image.resize(
        (256, 256)
    )


    # 0~255 → 0~1
    image_array = np.array(
        image,
        dtype=np.float32
    ) / 255.0


    # HWC → CHW
    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )


    # 배치 차원 추가
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # ONNX 실행
    outputs = ai_session.run(
        None,
        {
            "pixel_values":
                image_array
        }
    )


    features = outputs[0][0]


    # L2 정규화
    norm = np.linalg.norm(
        features
    )

    if norm != 0:
        features = features / norm


    return features.tolist()

# ==========================================
# 코사인 유사도 계산 함수
# ==========================================

def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    norm_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ==========================================
# 기본 테스트
# ==========================================

@app.get("/")
def home():

    return {
        "message": "분실물 AI 매칭 서버가 정상적으로 실행 중입니다."
    }


# ==========================================
# 분실물 등록
# ==========================================

@app.post("/lost-items")
async def create_lost_item(

    item_name: str = Form(...),
    location: str = Form(...),
    lost_date: str = Form(...),
    description: str = Form(""),
    image: UploadFile = File(...)

):

    upload_result = cloudinary.uploader.upload(
        image.file,
        folder="lost_found_ai/lost"
    )

    file_path = upload_result["secure_url"]


    db = get_db()

    cursor = db.cursor()


    sql = """
        INSERT INTO lost_items
        (item_name, location, lost_date, description, image_path)
        VALUES (%s, %s, %s, %s, %s)
    """


    values = (
        item_name,
        location,
        lost_date,
        description,
        file_path
    )


    cursor.execute(
        sql,
        values
    )

    db.commit()


    new_id = cursor.lastrowid


    cursor.close()
    db.close()


    return {

        "success": True,

        "message": "분실물이 등록되었습니다.",

        "id": new_id,

        "image_path": file_path
    }


# ==========================================
# 분실물 목록
# ==========================================

@app.get("/lost-items")
def get_lost_items():

    db = get_db()

    cursor = db.cursor(
        dictionary=True
    )


    cursor.execute("""
        SELECT
            id,
            item_name,
            location,
            lost_date,
            description,
            image_path,
            created_at

        FROM lost_items

        ORDER BY id DESC
    """)


    items = cursor.fetchall()


    cursor.close()
    db.close()


    return items


# ==========================================
# 분실물 삭제
# ==========================================

@app.delete("/lost-items/{item_id}")
def delete_lost_item(item_id: int):

    db = get_db()

    cursor = db.cursor(
        dictionary=True
    )


    cursor.execute("""
        SELECT image_path
        FROM lost_items
        WHERE id = %s
    """, (item_id,))


    item = cursor.fetchone()


    if item is None:

        cursor.close()
        db.close()

        raise HTTPException(
            status_code=404,
            detail="해당 분실물을 찾을 수 없습니다."
        )


    image_path = item["image_path"]


    if image_path and os.path.exists(image_path):

        os.remove(image_path)


    cursor.execute("""
        DELETE FROM lost_items
        WHERE id = %s
    """, (item_id,))


    db.commit()


    cursor.close()
    db.close()


    return {

        "success": True,

        "message": "분실물이 삭제되었습니다."
    }


# ==========================================
# 습득물 등록
# ==========================================

@app.post("/found-items")
async def create_found_item(

    item_name: str = Form(...),
    location: str = Form(...),
    found_date: str = Form(...),
    description: str = Form(""),
    image: UploadFile = File(...)

):

    upload_result = cloudinary.uploader.upload(
        image.file,
        folder="lost_found_ai/found"
    )

    file_path = upload_result["secure_url"]

    db = get_db()

    cursor = db.cursor()


    sql = """
        INSERT INTO found_items
        (item_name, location, found_date, description, image_path)
        VALUES (%s, %s, %s, %s, %s)
    """


    values = (
        item_name,
        location,
        found_date,
        description,
        file_path
    )


    cursor.execute(
        sql,
        values
    )

    db.commit()


    new_id = cursor.lastrowid


    cursor.close()
    db.close()


    return {

        "success": True,

        "message": "습득물이 등록되었습니다.",

        "id": new_id,

        "image_path": file_path
    }


# ==========================================
# 습득물 목록
# ==========================================

@app.get("/found-items")
def get_found_items():

    db = get_db()

    cursor = db.cursor(
        dictionary=True
    )


    cursor.execute("""
        SELECT
            id,
            item_name,
            location,
            found_date,
            description,
            image_path,
            created_at

        FROM found_items

        ORDER BY id DESC
    """)


    items = cursor.fetchall()


    cursor.close()
    db.close()


    return items


# ==========================================
# 습득물 삭제
# ==========================================

@app.delete("/found-items/{item_id}")
def delete_found_item(item_id: int):

    db = get_db()

    cursor = db.cursor(
        dictionary=True
    )


    cursor.execute("""
        SELECT image_path
        FROM found_items
        WHERE id = %s
    """, (item_id,))


    item = cursor.fetchone()


    if item is None:

        cursor.close()
        db.close()

        raise HTTPException(
            status_code=404,
            detail="해당 습득물을 찾을 수 없습니다."
        )


    image_path = item["image_path"]


    if image_path and os.path.exists(image_path):

        os.remove(image_path)


    cursor.execute("""
        DELETE FROM found_items
        WHERE id = %s
    """, (item_id,))


    db.commit()


    cursor.close()
    db.close()


    return {

        "success": True,

        "message": "습득물이 삭제되었습니다."
    }


# ==========================================
# ⭐ AI 분실물 매칭
# ==========================================

@app.get("/match/{lost_id}")
def match_lost_item(lost_id: int):

    print(f"AI 매칭 시작 - 분실물 ID: {lost_id}")

    db = get_db()

    cursor = db.cursor(
        dictionary=True
    )


    # --------------------------------------
    # 1. 분실물 가져오기
    # --------------------------------------

    cursor.execute("""
        SELECT *
        FROM lost_items
        WHERE id = %s
    """, (lost_id,))

    lost_item = cursor.fetchone()


    if lost_item is None:

        cursor.close()
        db.close()

        raise HTTPException(
            status_code=404,
            detail="해당 분실물을 찾을 수 없습니다."
        )


    # --------------------------------------
    # 2. 분실물 이미지 확인
    # --------------------------------------

    lost_image_url = lost_item["image_path"]


    if not lost_image_url:

        cursor.close()
        db.close()

        raise HTTPException(
            status_code=400,
            detail="분실물 이미지가 없습니다."
        )


    # Cloudinary 이미지 URL인지 확인
    if not (
        lost_image_url.startswith("http://") or
        lost_image_url.startswith("https://")
    ):

        cursor.close()
        db.close()

        raise HTTPException(
            status_code=400,
            detail="기존 로컬 이미지입니다. Cloudinary로 등록한 새 분실물로 테스트해주세요."
        )


    # --------------------------------------
    # 3. 분실물 AI 특징 추출
    # --------------------------------------

    try:

        print("분실물 이미지 AI 분석 중...")

        lost_features = get_image_features(
            lost_image_url
        )

    except Exception as e:

        cursor.close()
        db.close()

        print(f"분실물 AI 분석 실패: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"분실물 AI 분석 중 오류가 발생했습니다: {str(e)}"
        )


    # --------------------------------------
    # 4. 모든 습득물 가져오기
    # --------------------------------------

    cursor.execute("""
        SELECT *
        FROM found_items
        ORDER BY id DESC
    """)

    found_items = cursor.fetchall()


    results = []


    # --------------------------------------
    # 5. 습득물 하나씩 AI 비교
    # --------------------------------------

    for found_item in found_items:

        found_image_url = found_item["image_path"]


        if not found_image_url:
            continue


        # 예전 uploads/... 이미지는 제외
        if not (
            found_image_url.startswith("http://") or
            found_image_url.startswith("https://")
        ):
            continue


        try:

            print(
                f"습득물 {found_item['id']} AI 분석 중..."
            )


            found_features = get_image_features(
                found_image_url
            )


            # ----------------------------------
            # 코사인 유사도 계산
            # ----------------------------------

            similarity = cosine_similarity(
                lost_features,
                found_features
            )


            similarity_percent = (
                similarity * 100
            )


            results.append({

                "id":
                    found_item["id"],

                "item_name":
                    found_item["item_name"],

                "location":
                    found_item["location"],

                "found_date":
                    str(found_item["found_date"]),

                "description":
                    found_item["description"],

                "image_path":
                    found_item["image_path"],

                "similarity":
                    round(
                        similarity_percent,
                        2
                    )
            })


        except Exception as e:

            print(
                f"습득물 {found_item['id']} AI 분석 실패: {e}"
            )

            continue


    # --------------------------------------
    # 6. 유사도 높은 순 정렬
    # --------------------------------------

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )


    # --------------------------------------
    # 7. 상위 10개만 반환
    # --------------------------------------

    results = results[:10]


    cursor.close()
    db.close()


    print("AI 매칭 완료!")


    return {

        "success": True,

        "lost_item": {

            "id":
                lost_item["id"],

            "item_name":
                lost_item["item_name"],

            "location":
                lost_item["location"],

            "lost_date":
                str(lost_item["lost_date"]),

            "description":
                lost_item["description"],

            "image_path":
                lost_item["image_path"]
        },

        "matches":
            results
    }
