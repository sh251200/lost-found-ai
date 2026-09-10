from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import mysql.connector
import os
import shutil

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


app = FastAPI()


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
# AI 모델 설정
# ==========================================

# Render 무료 서버 테스트를 위해
# 서버 시작 시 CLIP 모델을 자동 로딩하지 않음

model = None
processor = None

print("서버 시작 완료 - AI 모델 로딩 보류")

# ==========================================
# 이미지 특징 추출 함수
# ==========================================

def get_image_features(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        features = model.get_image_features(
            **inputs
        )

        # transformers 버전에 따라
        # 반환 형태가 다른 경우 처리
        if hasattr(features, "pooler_output"):
            features = features.pooler_output

        elif hasattr(features, "image_embeds"):
            features = features.image_embeds

    # 벡터 정규화
    features = features / features.norm(
        dim=-1,
        keepdim=True
    )

    return features


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

    file_name = image.filename

    file_path = os.path.join(
        UPLOAD_DIR,
        file_name
    )

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            image.file,
            buffer
        )


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

    file_name = image.filename

    file_path = os.path.join(
        UPLOAD_DIR,
        file_name
    )


    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            image.file,
            buffer
        )


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

    print(
        f"AI 매칭 시작 - 분실물 ID: {lost_id}"
    )


    # --------------------------------------
    # 1. 분실물 가져오기
    # --------------------------------------

    db = get_db()

    cursor = db.cursor(
        dictionary=True
    )


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
    # 2. 분실물 이미지 경로
    # --------------------------------------

    lost_image_path = lost_item["image_path"]


    if not lost_image_path:

        cursor.close()
        db.close()

        raise HTTPException(
            status_code=400,
            detail="분실물 이미지가 없습니다."
        )


    if not os.path.exists(lost_image_path):

        cursor.close()
        db.close()

        raise HTTPException(
            status_code=404,
            detail="분실물 이미지 파일을 찾을 수 없습니다."
        )


    # --------------------------------------
    # 3. 분실물 AI 특징 추출
    # --------------------------------------

    print("분실물 이미지 분석 중...")


    lost_features = get_image_features(
        lost_image_path
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

        found_image_path = found_item["image_path"]


        if not found_image_path:
            continue


        if not os.path.exists(found_image_path):
            continue


        try:

            print(
                f"습득물 {found_item['id']} 분석 중..."
            )


            found_features = get_image_features(
                found_image_path
            )


            # ----------------------------------
            # 코사인 유사도 계산
            # ----------------------------------

            similarity = torch.sum(
                lost_features * found_features
            ).item()


            # 0~1 → 0~100%
            similarity_percent = (
                similarity * 100
            )


            results.append({

                "id": found_item["id"],

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
                f"습득물 {found_item['id']} 분석 실패: {e}"
            )


    # --------------------------------------
    # 6. 유사도 높은 순으로 정렬
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


    print(
        "AI 매칭 완료!"
    )


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
