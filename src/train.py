from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "vit-real-vs-ai"
CHECKPOINT = "google/vit-base-patch16-224-in21k"
LABELS = {"real": 0, "ai_generated": 1}
ID2LABEL = {0: "REAL", 1: "AI_GENERATED"}
LABEL2ID = {"REAL": 0, "AI_GENERATED": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str):
    """Return (device, backend_name). AMD on Windows may use torch-directml."""
    if requested == "cpu":
        return torch.device("cpu"), "CPU"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False. CUDA requires a supported NVIDIA GPU.")
        return torch.device("cuda"), "CUDA"
    if requested == "directml":
        try:
            import torch_directml
            return torch_directml.device(), "DirectML"
        except Exception as exc:
            raise RuntimeError(
                "DirectML was requested but torch-directml could not be imported. "
                "Install a compatible torch-directml/PyTorch combination first."
            ) from exc
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), "CUDA"
        try:
            import torch_directml
            return torch_directml.device(), "DirectML"
        except Exception:
            return torch.device("cpu"), "CPU"
    raise ValueError(f"Unknown device: {requested}")


class ImageFolderDataset(Dataset):
    def __init__(self, folder: Path, processor):
        self.processor = processor
        self.samples: list[tuple[Path, int]] = []
        for class_name, label in LABELS.items():
            class_dir = folder / class_name
            if class_dir.exists():
                for path in class_dir.rglob("*"):
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                        self.samples.append((path, label))
        self.samples.sort(key=lambda item: str(item[0]).lower())
        if not self.samples:
            raise RuntimeError(
                f"No images found under {folder}. Run src/prepare_dataset.py first."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        except Exception as exc:
            raise RuntimeError(f"Could not read image {path}: {exc}") from exc
        return pixels, torch.tensor(label, dtype=torch.long)


def evaluate(model, loader, device, backend: str):
    model.eval()
    y_true, y_pred = [], []
    losses = []
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(pixel_values=x).logits
            losses.append(float(loss_fn(logits, y).item()))
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_ai_generated": float(p),
        "recall_ai_generated": float(r),
        "f1_ai_generated": float(f1),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["REAL", "AI_GENERATED"],
            output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels_order": ["REAL", "AI_GENERATED"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n_images": len(y_true),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT for REAL vs AI_GENERATED images.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows initially.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-images", type=int, default=0, help="0 uses all training images.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "directml"], default="auto")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience based on validation loss.")
    parser.add_argument("--resume", action="store_true", help="Resume from outputs/vit-real-vs-ai if possible.")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("--epochs and --batch-size must be positive, and --lr must be > 0")

    seed_everything(args.seed)
    device, backend = choose_device(args.device)
    print(f"Device backend: {backend}")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if backend == "CPU":
        print("WARNING: training is using CPU, not the AMD GPU.")
    elif backend == "DirectML":
        print("DirectML selected. Some operations may be slower or unsupported depending on installed versions.")
    elif backend == "CUDA":
        print("CUDA selected (NVIDIA GPU).")

    processor_source = str(OUT) if args.resume and (OUT / "preprocessor_config.json").exists() else CHECKPOINT
    model_source = str(OUT) if args.resume and (OUT / "config.json").exists() else CHECKPOINT
    processor = AutoImageProcessor.from_pretrained(processor_source)
    if model_source == str(OUT):
        model = AutoModelForImageClassification.from_pretrained(OUT)
        # Keep semantic class mapping consistent with the prepared dataset.
        model.config.id2label = ID2LABEL
        model.config.label2id = LABEL2ID
    else:
        model = AutoModelForImageClassification.from_pretrained(
            CHECKPOINT,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    model = model.to(device)

    train_ds = ImageFolderDataset(DATA / "train", processor)
    val_ds = ImageFolderDataset(DATA / "val", processor)
    test_ds = ImageFolderDataset(DATA / "test", processor)
    if args.max_train_images > 0:
        train_ds.samples = train_ds.samples[:args.max_train_images]

    use_pin_memory = backend == "CUDA"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=use_pin_memory
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        seen_batches = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for x, y in progress:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            # CUDA AMP can improve throughput. Leave it off for DirectML/CPU compatibility.
            if backend == "CUDA":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(pixel_values=x).logits
                    loss = loss_fn(logits, y)
            else:
                logits = model(pixel_values=x).logits
                loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
            seen_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(1, seen_batches)
        val_metrics = evaluate(model, val_loader, device, backend)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val": val_metrics}
        history.append(row)
        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
            f"val_accuracy={val_metrics['accuracy']:.4f}, val_loss={val_metrics['loss']:.4f}"
        )

        if val_metrics["loss"] is not None and val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            model.save_pretrained(OUT)
            processor.save_pretrained(OUT)
            (OUT / "labels.json").write_text(json.dumps(ID2LABEL, indent=2), encoding="utf-8")
            torch.save(optimizer.state_dict(), OUT / "optimizer.pt")
            (OUT / "training_state.json").write_text(
                json.dumps({"best_epoch": best_epoch, "best_val_loss": best_val}, indent=2),
                encoding="utf-8"
            )
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"Early stopping: validation loss did not improve for {args.patience} epoch(s).")
                break

    # Evaluate the best validation checkpoint, not the last epoch.
    model = AutoModelForImageClassification.from_pretrained(OUT).to(device)
    test_metrics = evaluate(model, test_loader, device, backend)
    report = {
        "checkpoint": CHECKPOINT,
        "device_backend": backend,
        "device": str(device),
        "torch_version": torch.__version__,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - started, 2),
        "train_images": len(train_ds),
        "validation_images": len(val_ds),
        "test_images": len(test_ds),
        "history": history,
        "test": test_metrics,
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nFINAL TEST METRICS")
    print(json.dumps({k: v for k, v in test_metrics.items() if k != "classification_report"}, indent=2))
    print(f"\nSaved best model and metrics to: {OUT}")


if __name__ == "__main__":
    main()
