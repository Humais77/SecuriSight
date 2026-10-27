from pathlib import Path
import json, random, argparse, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "vit-real-vs-ai"
CHECKPOINT = "google/vit-base-patch16-224-in21k"
LABELS = {"real": 0, "ai_generated": 1}
ID2LABEL = {0: "REAL", 1: "AI_GENERATED"}
LABEL2ID = {"REAL": 0, "AI_GENERATED": 1}

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class ImageFolderDataset(Dataset):
    def __init__(self, folder, processor):
        self.processor = processor
        self.samples = []
        for name, label in LABELS.items():
            class_dir = folder / name
            if class_dir.exists():
                self.samples += [(p, label) for p in class_dir.iterdir()
                                 if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".webp",".bmp"}]
        self.samples.sort(key=lambda x: str(x[0]))
        if not self.samples:
            raise RuntimeError(f"No images found under {folder}. Run src/prepare_dataset.py first.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
            pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixels, torch.tensor(label, dtype=torch.long)

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    losses = []
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(pixel_values=x).logits
            losses.append(loss_fn(logits, y).item())
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_ai_generated": float(p),
        "recall_ai_generated": float(r),
        "f1_ai_generated": float(f1),
        "classification_report": classification_report(y_true, y_pred, target_names=["REAL", "AI_GENERATED"], output_dict=True, zero_division=0),
        "confusion_matrix_labels_order": ["REAL", "AI_GENERATED"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0,1]).tolist(),
        "n_images": len(y_true)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-train-images", type=int, default=0, help="0 uses all; use 200 for a quick smoke test")
    parser.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    args = parser.parse_args()
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("CPU training can be slow. For a faster experiment, use Google Colab/Kaggle GPU.")
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
    model = AutoModelForImageClassification.from_pretrained(
        CHECKPOINT, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    ).to(device)
    train_ds = ImageFolderDataset(DATA / "train", processor)
    val_ds = ImageFolderDataset(DATA / "val", processor)
    test_ds = ImageFolderDataset(DATA / "test", processor)
    if args.max_train_images > 0:
        train_ds.samples = train_ds.samples[:args.max_train_images]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type=="cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=(device.type=="cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))
    best_val = float("inf")
    OUT.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x, y in progress:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
                logits = model(pixel_values=x).logits
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch+1, "train_loss": total_loss / max(1, len(train_loader)), "val": val_metrics}
        history.append(row)
        print(f"Epoch {epoch+1}: train_loss={row['train_loss']:.4f}, "
              f"val_accuracy={val_metrics['accuracy']:.4f}, val_loss={val_metrics['loss']:.4f}")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            model.save_pretrained(OUT)
            processor.save_pretrained(OUT)
            (OUT / "labels.json").write_text(json.dumps(ID2LABEL, indent=2), encoding="utf-8")
    # Reload best validation checkpoint before final test evaluation.
    model = AutoModelForImageClassification.from_pretrained(OUT).to(device)
    test_metrics = evaluate(model, test_loader, device)
    report = {"checkpoint": CHECKPOINT, "device": str(device), "epochs": args.epochs,
              "batch_size": args.batch_size, "learning_rate": args.lr,
              "history": history, "test": test_metrics}
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nFINAL TEST METRICS")
    print(json.dumps({k:v for k,v in test_metrics.items() if k not in ("classification_report",)}, indent=2))
    print(f"\nSaved model and metrics to: {OUT}")

if __name__ == "__main__":
    main()
