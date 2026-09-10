import os
import requests


MODEL_URL = (
    "https://huggingface.co/plhery/mobileclip2-onnx/"
    "resolve/main/onnx/s0/vision_model.onnx"
)

MODEL_DIR = "backend/models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mobileclip_s0.onnx"
)


def download_model():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    if os.path.exists(MODEL_PATH):
        print("ONNX 모델이 이미 존재합니다.")
        return

    print("MobileCLIP2-S0 모델 다운로드 시작...")

    response = requests.get(
        MODEL_URL,
        stream=True,
        timeout=300
    )

    response.raise_for_status()

    with open(MODEL_PATH, "wb") as file:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:
                file.write(chunk)

    print("MobileCLIP2-S0 모델 다운로드 완료!")


if __name__ == "__main__":
    download_model()
