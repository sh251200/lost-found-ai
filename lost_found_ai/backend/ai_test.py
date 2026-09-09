from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


print("AI 모델을 불러오는 중입니다...")


# CLIP AI 모델 불러오기
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)


print("AI 모델 로딩 완료!")


# 테스트할 이미지
image_path = "uploads/test.jpg"


# 이미지 열기
image = Image.open(image_path).convert("RGB")


# 이미지 전처리
inputs = processor(
    images=image,
    return_tensors="pt"
)


# 이미지 특징 추출
with torch.no_grad():

    image_features = model.get_image_features(
        **inputs
    )

    # 현재 transformers 버전에 맞춰
    # 실제 Tensor만 가져오기
    if hasattr(image_features, "pooler_output"):
        image_features = image_features.pooler_output

    elif hasattr(image_features, "image_embeds"):
        image_features = image_features.image_embeds


print("이미지 분석 완료!")

print(
    "이미지 특징 벡터 크기:",
    image_features.shape
)