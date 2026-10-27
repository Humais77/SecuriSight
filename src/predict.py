from pathlib import Path
import argparse, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "vit-real-vs-ai"

def predict(path, model, processor, device):
    with Image.open(path) as im:
        image = im.convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    idx = int(probs.argmax())
    label = model.config.id2label.get(str(idx), model.config.id2label.get(idx, str(idx)))
    return label, float(probs[idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image")
    args = parser.parse_args()
    if not MODEL_DIR.exists():
        raise SystemExit("Trained model not found. Run: python src/train.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR).to(device).eval()
    label, confidence = predict(args.image, model, processor, device)
    print(f"Image: {args.image}")
    print(f"Prediction: {label}")
    print(f"Model confidence: {confidence*100:.2f}%")
    print("Note: confidence is not a guarantee of authenticity.")

if __name__ == "__main__":
    main()
