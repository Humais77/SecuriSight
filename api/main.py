from pathlib import Path
from io import BytesIO
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv(
    "SECURISIGHT_MODEL_PATH",
    str(PROJECT_ROOT / "outputs" / "vit-real-vs-ai")
))
UI_PATH = PROJECT_ROOT / "compare_ui" / "index.html"

app = FastAPI(title="SecuriSight Local Image Comparison API", version="1.0.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = None
model = None


def load_model():
    global processor, model
    if processor is not None and model is not None:
        return

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model folder not found: {MODEL_PATH}. "
            "Place your trained model in outputs/vit-real-vs-ai "
            "or set SECURISIGHT_MODEL_PATH."
        )

    # local_files_only=True guarantees that model files are read from disk,
    # not downloaded from Hugging Face or another online service.
    processor = AutoImageProcessor.from_pretrained(
        str(MODEL_PATH), local_files_only=True
    )
    model = AutoModelForImageClassification.from_pretrained(
        str(MODEL_PATH), local_files_only=True
    )
    model.to(device)
    model.eval()


def classify_image(image_bytes: bytes):
    try:
        image = Image.open(BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="One of the uploaded files is not a valid image.")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

    # This matches the training labels used for this SecuriSight model:
    # index 0 = REAL, index 1 = AI_GENERATED.
    if len(probs) != 2:
        raise HTTPException(
            status_code=500,
            detail=f"Expected a 2-class model, but found {len(probs)} output classes."
        )

    predicted_index = int(max(range(len(probs)), key=lambda i: probs[i]))
    label = "REAL" if predicted_index == 0 else "AI_GENERATED"

    return {
        "label": label,
        "real_probability": round(float(probs[0]), 6),
        "ai_generated_probability": round(float(probs[1]), 6),
        "confidence": round(float(probs[predicted_index]), 6)
    }


@app.get("/")
def home():
    if not UI_PATH.exists():
        raise HTTPException(status_code=404, detail=f"UI file not found: {UI_PATH}")
    return FileResponse(UI_PATH)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "device": str(device),
        "local_only": True
    }


@app.post("/api/compare")
async def compare_images(
    original: UploadFile = File(...),
    compressed: UploadFile = File(...)
):
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if original.content_type not in allowed or compressed.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Upload JPG, PNG, or WEBP images only.")

    original_bytes = await original.read()
    compressed_bytes = await compressed.read()

    max_bytes = 15 * 1024 * 1024
    if not original_bytes or not compressed_bytes:
        raise HTTPException(status_code=400, detail="Both image files are required.")
    if len(original_bytes) > max_bytes or len(compressed_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail="Each image must be 15 MB or smaller.")

    try:
        load_model()
        original_result = classify_image(original_bytes)
        compressed_result = classify_image(compressed_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    original_size = len(original_bytes)
    compressed_size = len(compressed_bytes)
    reduction = ((original_size - compressed_size) / original_size * 100) if original_size else 0

    return {
        "original": original_result,
        "compressed": compressed_result,
        "files": {
            "original_name": original.filename,
            "compressed_name": compressed.filename,
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "size_reduction_percent": round(reduction, 2)
        },
        "device": str(device)
    }
