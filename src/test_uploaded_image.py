from pathlib import Path
import argparse
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "vit-real-vs-ai-v2"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(str(MODEL_PATH))
    model = AutoModelForImageClassification.from_pretrained(str(MODEL_PATH))
    model.to(device)
    model.eval()

    image_path = Path(args.image)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    predicted_index = probabilities.argmax().item()
    predicted_label = model.config.id2label[predicted_index]

    print("\n========== SECURISIGHT RESULT ==========")
    print(f"Image: {image_path.name}")
    print(f"Prediction: {predicted_label}")
    print(f"REAL: {probabilities[0].item() * 100:.2f}%")
    print(f"AI GENERATED: {probabilities[1].item() * 100:.2f}%")
    print("========================================\n")

if __name__ == "__main__":
    main()