from pathlib import Path
import argparse, csv, math, torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from transformers import AutoImageProcessor, AutoModelForImageClassification

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "vit-real-vs-ai"

def load_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")

def predict(image, model, processor, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    idx = int(probs.argmax())
    label = model.config.id2label.get(str(idx), model.config.id2label.get(idx, str(idx)))
    return label, float(probs[idx])

def main():
    parser = argparse.ArgumentParser(description="Compare an original image with its compressed version.")
    parser.add_argument("original", help="Path to original/reference image")
    parser.add_argument("compressed", help="Path to compressed image")
    parser.add_argument("--csv", default="outputs/comparison.csv", help="Output CSV path")
    args = parser.parse_args()
    if not MODEL_DIR.exists():
        raise SystemExit("Trained model not found. Train it first with: python src/train.py")

    orig = load_rgb(args.original)
    comp = load_rgb(args.compressed)
    orig_size = Path(args.original).stat().st_size
    comp_size = Path(args.compressed).stat().st_size
    size_reduction = (orig_size - comp_size) / orig_size * 100 if orig_size else 0
    ratio = orig_size / comp_size if comp_size else float("inf")

    # Pixel metrics require the same dimensions. Resize compressed image to reference dimensions for comparison.
    comp_aligned = comp.resize(orig.size, Image.Resampling.LANCZOS)
    a = np.asarray(orig, dtype=np.uint8)
    b = np.asarray(comp_aligned, dtype=np.uint8)
    psnr = float(peak_signal_noise_ratio(a, b, data_range=255))
    ssim = float(structural_similarity(a, b, channel_axis=2, data_range=255))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR).to(device).eval()
    orig_label, orig_conf = predict(orig, model, processor, device)
    comp_label, comp_conf = predict(comp, model, processor, device)
    result = {
        "original_path": args.original, "compressed_path": args.compressed,
        "original_size_bytes": orig_size, "compressed_size_bytes": comp_size,
        "size_reduction_percent": round(size_reduction, 3),
        "compression_ratio_original_to_compressed": round(ratio, 4),
        "psnr_db_after_resize": round(psnr, 4), "ssim_after_resize": round(ssim, 6),
        "original_prediction": orig_label, "original_confidence_percent": round(orig_conf*100, 3),
        "compressed_prediction": comp_label, "compressed_confidence_percent": round(comp_conf*100, 3),
        "prediction_consistent": orig_label == comp_label,
        "note": "PSNR/SSIM are computed after resizing compressed image to original dimensions. Size reduction is not the same as visual quality loss."
    }
    print("\nCOMPARISON REPORT")
    for k, v in result.items(): print(f"{k}: {v}")
    out = ROOT / args.csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader(); writer.writerow(result)
    print(f"\nCSV saved to: {out}")

if __name__ == "__main__":
    main()
