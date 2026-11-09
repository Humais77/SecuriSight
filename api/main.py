from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model folders. You can override these paths using environment variables.
MODEL_PATHS = {
    "baseline": Path(os.getenv(
        "SECURISIGHT_BASELINE_MODEL_PATH",
        str(PROJECT_ROOT / "outputs" / "vit-real-vs-ai")
    )),
    "finetuned": Path(os.getenv(
        "SECURISIGHT_FINETUNED_MODEL_PATH",
        str(PROJECT_ROOT / "outputs" / "vit-real-vs-ai-v2")
    )),
    "compression": Path(os.getenv(
        "SECURISIGHT_COMPRESSION_MODEL_PATH",
        str(PROJECT_ROOT / "outputs" / "vit-compression-robust")
    )),
}

UI_PATH = PROJECT_ROOT / "compare_ui" / "index.html"

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

JPEG_QUALITIES = (20, 40, 60, 80, 100)
PREVIEW_MAX_SIZE = (560, 560)

MODEL_NAMES = {
    "baseline": "Original ViT (v1)",
    "finetuned": "Fine-tuned ViT (v2)",
    "compression": "Compression-aware ViT",
}

# Used only if a saved model has generic LABEL_0 / LABEL_1 labels.
# Verify these mappings against your training scripts and saved checkpoints.
FALLBACK_LABELS = {
    "baseline": {
        0: "REAL",
        1: "AI_GENERATED",
    },
    "finetuned": {
        0: "REAL",
        1: "AI_GENERATED",
    },
    "compression": {
        0: "AI_GENERATED",
        1: "REAL",
    },
}

app = FastAPI(
    title="SecuriSight Three-Model Image Analysis API",
    version="3.0.0",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models: dict[str, dict[str, Any] | None] = {
    "baseline": None,
    "finetuned": None,
    "compression": None,
}


def _normalise_label(raw_label: Any) -> str | None:
    value = str(raw_label).upper().replace("-", "_").replace(" ", "_")

    if "AI" in value or "FAKE" in value or "GENERATED" in value:
        return "AI_GENERATED"

    if "REAL" in value or "AUTHENTIC" in value:
        return "REAL"

    return None


def _load_one_model(kind: str, model_path: Path) -> dict[str, Any]:
    if not model_path.is_dir():
        raise RuntimeError(
            f"{kind} model folder not found: {model_path}"
        )

    if not (model_path / "config.json").is_file():
        raise RuntimeError(
            f"config.json is missing from model folder: {model_path}"
        )

    processor = AutoImageProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    classifier = AutoModelForImageClassification.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    if classifier.config.num_labels != 2:
        raise RuntimeError(
            f"{kind} must have exactly two output classes. "
            f"Found: {classifier.config.num_labels}"
        )

    configured = classifier.config.id2label or {}
    labels: dict[int, str] = {}

    for index in range(classifier.config.num_labels):
        raw_label = configured.get(
            index,
            configured.get(str(index), f"LABEL_{index}")
        )

        normalised = _normalise_label(raw_label)

        if normalised:
            labels[index] = normalised
        else:
            labels[index] = FALLBACK_LABELS[kind].get(
                index,
                f"UNKNOWN_{index}"
            )

    if set(labels.values()) != {"REAL", "AI_GENERATED"}:
        raise RuntimeError(
            f"Invalid label mapping for {kind}: {labels}. "
            "Check the model's saved id2label configuration and fallback mapping."
        )

    classifier.to(device)
    classifier.eval()

    return {
        "processor": processor,
        "model": classifier,
        "labels": labels,
        "path": str(model_path),
        "name": MODEL_NAMES[kind],
    }


def load_models() -> None:
    """Load all three local models once."""

    for kind, model_path in MODEL_PATHS.items():
        if models[kind] is None:
            models[kind] = _load_one_model(kind, model_path)


def _open_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:

            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Image dimensions are too large. "
                        f"Maximum is {MAX_IMAGE_PIXELS:,} pixels."
                    ),
                )

            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()

            return image

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid or supported image.",
        )


def _predict(image: Image.Image, kind: str) -> dict[str, Any]:
    bundle = models[kind]

    if bundle is None:
        raise RuntimeError(f"{kind} model is not loaded.")

    processor = bundle["processor"]
    classifier = bundle["model"]
    labels = bundle["labels"]

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        logits = classifier(**inputs).logits

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )[0].detach().cpu().tolist()

    real_probability = 0.0
    ai_probability = 0.0

    for index, probability in enumerate(probabilities):
        if labels[index] == "REAL":
            real_probability = float(probability)

        elif labels[index] == "AI_GENERATED":
            ai_probability = float(probability)

    predicted_label = (
        "REAL"
        if real_probability >= ai_probability
        else "AI_GENERATED"
    )

    confidence = max(real_probability, ai_probability)

    return {
        "label": predicted_label,
        "real_probability": round(real_probability, 6),
        "ai_generated_probability": round(ai_probability, 6),
        "confidence": round(confidence, 6),
    }


def _predict_all(image: Image.Image) -> dict[str, Any]:
    """Run one image through all three models."""

    return {
        kind: {
            "model_name": MODEL_NAMES[kind],
            **_predict(image, kind),
        }
        for kind in MODEL_NAMES
    }


def _preview_data_url(image: Image.Image) -> str:
    preview = image.copy()
    preview.thumbnail(PREVIEW_MAX_SIZE)

    buffer = io.BytesIO()

    preview.save(
        buffer,
        format="JPEG",
        quality=82,
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"


async def _read_upload(upload: UploadFile) -> bytes:
    allowed = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if upload.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Upload JPG, PNG, or WEBP images only.",
        )

    data = await upload.read(MAX_UPLOAD_BYTES + 1)

    if not data:
        raise HTTPException(
            status_code=400,
            detail="An image file is required.",
        )

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be 15 MB or smaller.",
        )

    return data


@app.get("/")
def home():
    if not UI_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"UI file not found: {UI_PATH}",
        )

    return FileResponse(UI_PATH)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models": {
            kind: {
                "name": MODEL_NAMES[kind],
                "loaded": models[kind] is not None,
                "path": str(MODEL_PATHS[kind]),
            }
            for kind in MODEL_NAMES
        },
        "device": str(device),
        "local_only": True,
    }


@app.post("/api/load-models")
def api_load_models():
    try:
        load_models()

        return {
            "status": "loaded",
            "models": {
                kind: {
                    "name": MODEL_NAMES[kind],
                    "loaded": models[kind] is not None,
                }
                for kind in MODEL_NAMES
            },
            "device": str(device),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post("/api/predict")
async def predict_single_image(image: UploadFile = File(...)):
    """
    Main image detection:
    Run the uploaded image through all three ViT models.
    """

    image_bytes = await _read_upload(image)

    try:
        load_models()

        pil_image = _open_image(image_bytes)

        results = _predict_all(pil_image)

        return {
            "filename": image.filename,
            "results": results,
            "device": str(device),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post("/api/compression-analysis")
async def compression_analysis(image: UploadFile = File(...)):
    """
    Compression lab:
    Generate the original plus five JPEG quality versions.
    Run all three models on every version.
    """

    image_bytes = await _read_upload(image)

    try:
        load_models()

        original = _open_image(image_bytes)

        variants: list[tuple[str, Image.Image, int]] = [
            ("Original", original, len(image_bytes))
        ]

        for quality in JPEG_QUALITIES:
            buffer = io.BytesIO()

            original.save(
                buffer,
                format="JPEG",
                quality=quality,
                optimize=True,
            )

            jpeg_bytes = buffer.getvalue()

            compressed_image = _open_image(jpeg_bytes)

            variants.append((
                f"JPEG Q{quality}",
                compressed_image,
                len(jpeg_bytes),
            ))

        output = []
        original_size = len(image_bytes)

        for name, variant_image, variant_size in variants:

            results = _predict_all(variant_image)

            reduction = (
                (original_size - variant_size) / original_size * 100
                if original_size
                else 0
            )

            output.append({
                "name": name,
                "preview": _preview_data_url(variant_image),
                "size_bytes": variant_size,
                "size_reduction_percent": round(reduction, 2),
                "results": results,
            })

        return {
            "filename": image.filename,
            "variants": output,
            "device": str(device),
            "note": (
                "JPEG Q100 is a re-encoded JPEG, not the untouched source file. "
                "All three models were run on every version."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )