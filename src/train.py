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

# Your existing, previously trained model.
OLD_MODEL = ROOT / "outputs" / "vit-real-vs-ai"

# New model output. This keeps the old model safe.
DEFAULT_OUTPUT = ROOT / "outputs" / "vit-real-vs-ai-v2"

CHECKPOINT = "google/vit-base-patch16-224-in21k"

LABELS = {
    "real": 0,
    "ai_generated": 1,
}

ID2LABEL = {
    0: "REAL",
    1: "AI_GENERATED",
}

LABEL2ID = {
    "REAL": 0,
    "AI_GENERATED": 1,
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tif", ".tiff",
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str):
    """Return the selected device and its backend name."""

    if requested == "cpu":
        return torch.device("cpu"), "CPU"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is unavailable. "
                "Select a GPU runtime in Colab."
            )

        return torch.device("cuda"), "CUDA"

    if requested == "directml":
        try:
            import torch_directml

            return torch_directml.device(), "DirectML"

        except Exception as exc:
            raise RuntimeError(
                "DirectML could not be imported. "
                "Install a compatible torch-directml version."
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

            if not class_dir.exists():
                continue

            for path in class_dir.rglob("*"):
                if (
                    path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    self.samples.append((path, label))

        self.samples.sort(
            key=lambda item: str(item[0]).lower()
        )

        if not self.samples:
            raise RuntimeError(
                f"No images found under {folder}. "
                "Run prepare_datasets.py first."
            )

        # Ensure both classes are present in every split.
        class_counts = {
            "real": sum(label == 0 for _, label in self.samples),
            "ai_generated": sum(label == 1 for _, label in self.samples),
        }

        if not all(class_counts.values()):
            raise RuntimeError(
                f"Both classes must be present in {folder}. "
                f"Found: {class_counts}"
            )

        print(f"{folder.name}: {class_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            with Image.open(path) as image:
                image = image.convert("RGB")

                pixels = self.processor(
                    images=image,
                    return_tensors="pt",
                )["pixel_values"].squeeze(0)

        except Exception as exc:
            raise RuntimeError(
                f"Could not read image {path}: {exc}"
            ) from exc

        return pixels, torch.tensor(label, dtype=torch.long)


def evaluate(model, loader, device):

    model.eval()

    y_true = []
    y_pred = []
    losses = []

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():

        for x, y in tqdm(
            loader,
            desc="Evaluating",
            leave=False,
        ):
            x = x.to(device)
            y = y.to(device)

            logits = model(pixel_values=x).logits

            loss = loss_fn(logits, y)

            losses.append(float(loss.item()))

            y_true.extend(
                y.detach().cpu().tolist()
            )

            y_pred.extend(
                logits.argmax(dim=1).detach().cpu().tolist()
            )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    return {
        "loss": float(np.mean(losses)) if losses else None,

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision_ai_generated": float(precision),

        "recall_ai_generated": float(recall),

        "f1_ai_generated": float(f1),

        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["REAL", "AI_GENERATED"],
            output_dict=True,
            zero_division=0,
        ),

        "confusion_matrix_labels_order": [
            "REAL",
            "AI_GENERATED",
        ],

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),

        "n_images": len(y_true),
    }


def load_model_and_processor(
    resume: bool,
    resume_from: Path | None,
):
    """
    Load the existing trained model when resuming.
    Otherwise initialize a new model from the ViT checkpoint.
    """

    if resume:
        model_path = (
            resume_from.resolve()
            if resume_from
            else OLD_MODEL.resolve()
        )

        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"Model configuration not found: {model_path}"
            )

        if not (
            (model_path / "model.safetensors").is_file()
            or (model_path / "pytorch_model.bin").is_file()
        ):
            raise FileNotFoundError(
                f"Model weights not found in {model_path}"
            )

        if not (
            model_path / "preprocessor_config.json"
        ).is_file():
            raise FileNotFoundError(
                f"Image processor configuration not found: {model_path}"
            )

        print(f"\nLoading existing model: {model_path}")

        processor = AutoImageProcessor.from_pretrained(
            str(model_path)
        )

        model = AutoModelForImageClassification.from_pretrained(
            str(model_path)
        )

        if model.config.num_labels != 2:
            raise ValueError(
                f"Expected 2 classes, found {model.config.num_labels}."
            )

        # Check that the previous model uses the same class order.
        actual_mapping = {
            int(key): str(value).upper()
            for key, value in model.config.id2label.items()
        }

        if actual_mapping != ID2LABEL:
            raise ValueError(
                "The old model's label mapping does not match "
                "the expected mapping.\n"
                f"Expected: {ID2LABEL}\n"
                f"Found: {actual_mapping}\n"
                "Do not continue until the class mapping is verified."
            )

        print("Existing model loaded successfully.")

    else:
        print(f"\nInitializing base model: {CHECKPOINT}")

        processor = AutoImageProcessor.from_pretrained(
            CHECKPOINT
        )

        model = AutoModelForImageClassification.from_pretrained(
            CHECKPOINT,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID

    return model, processor


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune ViT for REAL vs AI_GENERATED images."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-train-images",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "directml"],
        default="auto",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load the existing trained model instead of starting fresh.",
    )

    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to the existing model checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where the updated model will be saved.",
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error(
            "--epochs and --batch-size must be positive, "
            "and --lr must be greater than zero."
        )

    if args.resume_from and not args.resume:
        parser.error("--resume-from requires --resume.")

    if args.resume and not args.resume_from and not OLD_MODEL.exists():
        parser.error(
            f"Existing model not found: {OLD_MODEL}"
        )

    if args.max_train_images < 0:
        parser.error("--max-train-images cannot be negative.")

    seed_everything(args.seed)

    device, backend = choose_device(args.device)

    output_dir = args.output_dir.expanduser().resolve()

    if args.resume:
        source_dir = (
            args.resume_from.expanduser().resolve()
            if args.resume_from
            else OLD_MODEL.resolve()
        )

        if source_dir == output_dir:
            raise ValueError(
                "For safety, the output directory must be different "
                "from the starting checkpoint directory. "
                "Use a new directory such as outputs/vit-real-vs-ai-v2."
            )

    print("\nTRAINING CONFIGURATION")
    print("----------------------")
    print("Device backend:", backend)
    print("Device:", device)
    print("PyTorch:", torch.__version__)
    print("Resume:", args.resume)
    print("Output directory:", output_dir)

    if backend == "CUDA":
        print("CUDA GPU:", torch.cuda.get_device_name(0))

    elif backend == "CPU":
        print("WARNING: Training is using CPU.")

    elif backend == "DirectML":
        print("DirectML selected.")

    model, processor = load_model_and_processor(
        resume=args.resume,
        resume_from=args.resume_from,
    )

    model = model.to(device)

    # Load prepared datasets.
    train_ds = ImageFolderDataset(
        DATA / "train",
        processor,
    )

    val_ds = ImageFolderDataset(
        DATA / "val",
        processor,
    )

    test_ds = ImageFolderDataset(
        DATA / "test",
        processor,
    )

    if args.max_train_images > 0:
        train_ds.samples = train_ds.samples[
            :args.max_train_images
        ]

    if not train_ds.samples:
        raise RuntimeError("No training images available.")

    use_pin_memory = backend == "CUDA"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_pin_memory,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.CrossEntropyLoss()

    # Automatic mixed precision for CUDA.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(backend == "CUDA"),
    )

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    started = time.time()

    for epoch in range(args.epochs):

        model.train()

        total_loss = 0.0
        seen_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        for x, y in progress:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            if backend == "CUDA":

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = model(pixel_values=x).logits
                    loss = loss_fn(logits, y)

            else:
                logits = model(pixel_values=x).logits
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().item())
            seen_batches += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = total_loss / max(1, seen_batches)

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val": val_metrics,
        }

        history.append(row)

        print(
            f"\nEpoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_accuracy={val_metrics['accuracy']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}"
        )

        # Save only the best validation checkpoint.
        if (
            val_metrics["loss"] is not None
            and val_metrics["loss"] < best_val
        ):

            best_val = val_metrics["loss"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

            (
                output_dir / "labels.json"
            ).write_text(
                json.dumps(ID2LABEL, indent=2),
                encoding="utf-8",
            )

            torch.save(
                optimizer.state_dict(),
                output_dir / "optimizer.pt",
            )

            (
                output_dir / "training_state.json"
            ).write_text(
                json.dumps({
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                    "starting_checkpoint": (
                        str(source_dir)
                        if args.resume
                        else CHECKPOINT
                    ),
                }, indent=2),
                encoding="utf-8",
            )

            print("Best model checkpoint saved.")

        else:

            epochs_without_improvement += 1

            if (
                args.patience > 0
                and epochs_without_improvement >= args.patience
            ):
                print(
                    "Early stopping: validation loss did not improve "
                    f"for {args.patience} epoch(s)."
                )
                break

    if best_epoch == 0:
        raise RuntimeError(
            "No best checkpoint was saved. Check validation data and loss."
        )

    # Evaluate the best validation checkpoint.
    print("\nLoading best checkpoint for final test evaluation.")

    model = AutoModelForImageClassification.from_pretrained(
        output_dir
    ).to(device)

    test_metrics = evaluate(
        model,
        test_loader,
        device,
    )

    report = {
        "starting_checkpoint": (
            str(source_dir) if args.resume else CHECKPOINT
        ),
        "output_checkpoint": str(output_dir),
        "device_backend": backend,
        "device": str(device),
        "torch_version": torch.__version__,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "elapsed_seconds": round(
            time.time() - started,
            2,
        ),
        "train_images": len(train_ds),
        "validation_images": len(val_ds),
        "test_images": len(test_ds),
        "history": history,
        "test": test_metrics,
    }

    (
        output_dir / "metrics.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL TEST METRICS")

    print(
        json.dumps(
            {
                key: value
                for key, value in test_metrics.items()
                if key != "classification_report"
            },
            indent=2,
        )
    )

    print("\nClassification report:")
    print(
        json.dumps(
            test_metrics["classification_report"],
            indent=2,
        )
    )

    print(
        f"\nSaved updated model and metrics to: {output_dir}"
    )


if __name__ == "__main__":
    main()