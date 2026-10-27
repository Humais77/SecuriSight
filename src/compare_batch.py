
from pathlib import Path
import csv
import json

import numpy as np
import torch
from PIL import Image
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "vit-real-vs-ai"

ORIGINAL_DIR = ROOT / "test_images" / "original"
COMPRESSED_DIR = ROOT / "test_images" / "compressed"
LABELS_CSV = ROOT / "test_images" / "labels.csv"

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_CSV = OUTPUT_DIR / "batch_comparison.csv"
SUMMARY_JSON = OUTPUT_DIR / "batch_summary.json"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_image(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def normalize_label(label):
    label = str(label).strip().upper()
    label = label.replace("-", "_").replace(" ", "_")

    if label in {"REAL", "HUMAN", "AUTHENTIC"}:
        return "REAL"

    if label in {
        "AI_GENERATED", "FAKE", "GENERATED", "SYNTHETIC", "AI"
    }:
        return "AI_GENERATED"

    return label


def load_ground_truth():
    if not LABELS_CSV.exists():
        return {}

    labels = {}

    with LABELS_CSV.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or not {
            "image", "label"
        }.issubset(reader.fieldnames):
            raise ValueError(
                "labels.csv must contain image,label columns."
            )

        for row in reader:
            name = Path(row["image"].strip()).stem
            labels[name] = normalize_label(row["label"])

    return labels


def get_model_label(model, index):
    config_labels = model.config.id2label

    return str(
        config_labels.get(
            index,
            config_labels.get(str(index), f"LABEL_{index}")
        )
    ).upper()


def predict(image, model, processor, device):
    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)

    scores = {}

    for index, probability in enumerate(probabilities):
        label = get_model_label(model, index)
        scores[label] = float(probability.item())

    best_index = int(probabilities.argmax())
    prediction = get_model_label(model, best_index)
    confidence = float(probabilities[best_index].item())

    return prediction, confidence, scores


def class_score(scores, target):
    """Support common alternative names for model classes."""
    aliases = {
        "REAL": {"REAL", "HUMAN", "AUTHENTIC", "LABEL_0"},
        "AI_GENERATED": {
            "AI_GENERATED", "AI", "FAKE", "GENERATED",
            "SYNTHETIC", "LABEL_1"
        },
    }

    for label, score in scores.items():
        normalized = normalize_label(label)

        if normalized == target:
            return score

        if label in aliases[target]:
            return score

    return 0.0


def is_correct(prediction, truth):
    if not truth:
        return ""

    # Ground-truth labels must use the same class names as the model.
    return normalize_label(prediction) == truth


def safe_accuracy(rows, key):
    labelled = [
        row for row in rows
        if row["ground_truth"] in {"REAL", "AI_GENERATED"}
    ]

    if not labelled:
        return None

    correct = sum(
        row[key] == row["ground_truth"]
        for row in labelled
    )

    return {
        "correct": correct,
        "total": len(labelled),
        "accuracy_percent": round(
            correct / len(labelled) * 100, 2
        ),
    }


def main():
    if not MODEL_DIR.exists():
        raise SystemExit(
            f"Trained model not found: {MODEL_DIR}\n"
            "Run src/train.py first."
        )

    for folder in (ORIGINAL_DIR, COMPRESSED_DIR):
        if not folder.exists():
            raise SystemExit(f"Folder not found: {folder}")

    originals = {
        p.stem: p
        for p in ORIGINAL_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }

    compressed = {
        p.stem: p
        for p in COMPRESSED_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }

    common = sorted(set(originals) & set(compressed))

    if len(originals) != 20 or len(compressed) != 20:
        raise SystemExit(
            f"Expected 20 images in each folder. "
            f"Found {len(originals)} originals and "
            f"{len(compressed)} compressed."
        )

    if len(common) != 20:
        raise SystemExit(
            f"Expected 20 matching filename pairs, found {len(common)}."
        )

    print("Loading ViT model...")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_DIR
    ).to(device).eval()

    print(f"Device: {device}")
    print(f"Model labels: {model.config.id2label}")

    ground_truth = load_ground_truth()
    rows = []

    for name in common:
        original_path = originals[name]
        compressed_path = compressed[name]

        original = load_image(original_path)
        compressed_img = load_image(compressed_path)

        original_bytes = original_path.stat().st_size
        compressed_bytes = compressed_path.stat().st_size

        reduction = (
            (original_bytes - compressed_bytes)
            / original_bytes * 100
            if original_bytes else 0.0
        )

        ratio = (
            original_bytes / compressed_bytes
            if compressed_bytes else 0.0
        )

        aligned = compressed_img.resize(
            original.size, Image.Resampling.LANCZOS
        )

        a = np.asarray(original, dtype=np.uint8)
        b = np.asarray(aligned, dtype=np.uint8)

        if np.array_equal(a, b):
            psnr = float("inf")
        else:
            psnr = float(
                peak_signal_noise_ratio(a, b, data_range=255)
            )

        ssim = float(
            structural_similarity(
                a, b, channel_axis=2, data_range=255
            )
        )

        orig_pred, orig_conf, orig_scores = predict(
            original, model, processor, device
        )

        comp_pred, comp_conf, comp_scores = predict(
            compressed_img, model, processor, device
        )

        truth = ground_truth.get(name, "")

        orig_real = class_score(orig_scores, "REAL")
        orig_ai = class_score(orig_scores, "AI_GENERATED")
        comp_real = class_score(comp_scores, "REAL")
        comp_ai = class_score(comp_scores, "AI_GENERATED")

        rows.append({
            "image": name,
            "ground_truth": truth,

            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "size_reduction_percent": round(reduction, 3),
            "compression_ratio": round(ratio, 4),

            "psnr_db": (
                round(psnr, 4) if np.isfinite(psnr) else "Infinity"
            ),
            "ssim": round(ssim, 6),

            "original_prediction": orig_pred,
            "original_confidence_percent": round(orig_conf * 100, 3),
            "original_real_score_percent": round(orig_real * 100, 3),
            "original_ai_score_percent": round(orig_ai * 100, 3),

            "compressed_prediction": comp_pred,
            "compressed_confidence_percent": round(comp_conf * 100, 3),
            "compressed_real_score_percent": round(comp_real * 100, 3),
            "compressed_ai_score_percent": round(comp_ai * 100, 3),

            "confidence_change_percentage_points": round(
                (comp_conf - orig_conf) * 100, 3
            ),
            "real_score_change_percentage_points": round(
                (comp_real - orig_real) * 100, 3
            ),
            "ai_score_change_percentage_points": round(
                (comp_ai - orig_ai) * 100, 3
            ),

            "prediction_consistent": orig_pred == comp_pred,
            "original_correct": is_correct(orig_pred, truth),
            "compressed_correct": is_correct(comp_pred, truth),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    consistent = sum(
        row["prediction_consistent"] for row in rows
    )

    original_accuracy = safe_accuracy(rows, "ground_truth")
    # Calculate accuracy explicitly to avoid comparing against predictions.
    original_accuracy = safe_accuracy(
        [
            {
                **row,
                "original_prediction": row["original_prediction"],
            }
            for row in rows
        ],
        "original_prediction",
    )
    compressed_accuracy = safe_accuracy(
        rows, "compressed_prediction"
    )

    summary = {
        "image_pairs_tested": len(rows),
        "prediction_consistency_percent": round(
            consistent / len(rows) * 100, 2
        ),
        "consistent_pairs": consistent,
        "average_size_reduction_percent": round(
            float(np.mean([
                r["size_reduction_percent"] for r in rows
            ])), 3
        ),
        "average_compression_ratio": round(
            float(np.mean([
                r["compression_ratio"] for r in rows
            ])), 4
        ),
        "average_ssim": round(
            float(np.mean([r["ssim"] for r in rows])), 6
        ),
        "average_psnr_db": (
            round(float(np.mean([
                float(r["psnr_db"]) for r in rows
                if r["psnr_db"] != "Infinity"
            ])), 4)
            if any(r["psnr_db"] != "Infinity" for r in rows)
            else "Infinity"
        ),
        "original_accuracy": original_accuracy,
        "compressed_accuracy": compressed_accuracy,
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    # Detailed per-pair terminal report.
    print("\n" + "=" * 70)
    print("SECURISIGHT - ViT IMAGE COMPRESSION ANALYSIS")
    print("=" * 70)

    for i, row in enumerate(rows, start=1):
        print(f"\nIMAGE PAIR {i}: {row['image']}")
        print("-" * 60)

        print("ORIGINAL IMAGE")
        print(f"  Prediction  : {row['original_prediction']}")
        print(f"  Confidence  : {row['original_confidence_percent']:.2f}%")
        print(f"  REAL score  : {row['original_real_score_percent']:.2f}%")
        print(f"  AI score    : {row['original_ai_score_percent']:.2f}%")
        print(f"  File size   : {row['original_bytes']:,} bytes")

        print("\nCOMPRESSED IMAGE")
        print(f"  Prediction  : {row['compressed_prediction']}")
        print(f"  Confidence  : {row['compressed_confidence_percent']:.2f}%")
        print(f"  REAL score  : {row['compressed_real_score_percent']:.2f}%")
        print(f"  AI score    : {row['compressed_ai_score_percent']:.2f}%")
        print(f"  File size   : {row['compressed_bytes']:,} bytes")

        print("\nCOMPRESSION METRICS")
        print(f"  Size reduction : {row['size_reduction_percent']:.2f}%")
        print(f"  Size ratio     : {row['compression_ratio']:.2f}:1")
        print(f"  PSNR           : {row['psnr_db']} dB")
        print(f"  SSIM           : {row['ssim']:.4f}")

        print("\nPREDICTION CHANGE")
        print(
            f"  Confidence change : "
            f"{row['confidence_change_percentage_points']:+.2f} points"
        )
        print(
            f"  REAL score change : "
            f"{row['real_score_change_percentage_points']:+.2f} points"
        )
        print(
            f"  AI score change   : "
            f"{row['ai_score_change_percentage_points']:+.2f} points"
        )
        print(
            f"  Prediction stable : "
            f"{'YES' if row['prediction_consistent'] else 'NO'}"
        )

        print(f"  Ground truth      : {row['ground_truth'] or 'Not provided'}")
        print(f"  Original correct  : {row['original_correct']}")
        print(f"  Compressed correct: {row['compressed_correct']}")

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Pairs tested           : {len(rows)}")
    print(
        f"Prediction consistency: {consistent}/{len(rows)} "
        f"({summary['prediction_consistency_percent']:.2f}%)"
    )
    print(
        f"Average size reduction: "
        f"{summary['average_size_reduction_percent']:.2f}%"
    )
    print(
        f"Average compression ratio: "
        f"{summary['average_compression_ratio']:.2f}:1"
    )
    print(f"Average SSIM           : {summary['average_ssim']:.4f}")
    print(f"Average PSNR           : {summary['average_psnr_db']} dB")

    if original_accuracy:
        print(
            f"Original accuracy      : "
            f"{original_accuracy['accuracy_percent']:.2f}%"
        )
        print(
            f"Compressed accuracy    : "
            f"{compressed_accuracy['accuracy_percent']:.2f}%"
        )
    else:
        print("Accuracy               : Not calculated; labels are missing.")

    print(f"\nCSV report     : {OUTPUT_CSV}")
    print(f"Summary report : {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
