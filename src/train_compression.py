
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import argparse
import json


CHECKPOINT = "google/vit-base-patch16-224-in21k"

LABEL2ID = {
    "ai_generated": 0,
    "real": 1,
}

ID2LABEL = {
    0: "AI_GENERATED",
    1: "REAL",
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


class ImageDataset(Dataset):
    def __init__(self, root, processor):
        self.processor = processor
        self.samples = []

        for class_name, label in LABEL2ID.items():
            folder = Path(root) / class_name

            for path in folder.iterdir():
                if path.is_file():
                    self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                )

            pixel_values = inputs["pixel_values"].squeeze(0)

            return pixel_values, label

        except Exception as error:
            raise RuntimeError(f"Cannot read image {path}: {error}")


def run_epoch(model, loader, optimizer, device, training):
    model.train(training)

    total_loss = 0.0
    predictions = []
    targets = []

    progress = tqdm(loader, leave=False)

    for pixel_values, labels in progress:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(
                pixel_values=pixel_values,
                labels=labels
            )

            loss = outputs.loss

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

        predicted = outputs.logits.argmax(dim=1)

        predictions.extend(predicted.detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = total_loss / max(len(loader), 1)
    accuracy = accuracy_score(targets, predictions)

    return average_loss, accuracy, targets, predictions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/compression_dataset"
    )

    parser.add_argument(
        "--output",
        default="outputs/vit-compression-robust"
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=3)

    args = parser.parse_args()

    data_root = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(CHECKPOINT)

    train_dataset = ImageDataset(
        data_root / "train",
        processor
    )

    val_dataset = ImageDataset(
        data_root / "val",
        processor
    )

    test_dataset = ImageDataset(
        data_root / "test",
        processor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = AutoModelForImageClassification.from_pretrained(
        CHECKPOINT,
        num_labels=2,
        id2label=ID2LABEL,
        label2id={
            "AI_GENERATED": 0,
            "REAL": 1,
        },
        ignore_mismatched_sizes=True,
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate
    )

    best_val_accuracy = -1
    patience_counter = 0

    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc, _, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training=True
        )

        val_loss, val_acc, _, _ = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            training=False
        )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
        })

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0

            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

            print("Best model saved.")

        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    # Reload the best validation checkpoint before testing.
    model = AutoModelForImageClassification.from_pretrained(
        output_dir
    ).to(device)

    test_loss, test_acc, targets, predictions = run_epoch(
        model,
        test_loader,
        optimizer,
        device,
        training=False
    )

    print("\nFinal test results")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    print(
        classification_report(
            targets,
            predictions,
            labels=[0, 1],
            target_names=["AI_GENERATED", "REAL"],
            zero_division=0
        )
    )

    with open(output_dir / "training_history.json", "w") as file:
        json.dump(history, file, indent=4)

    with open(output_dir / "test_results.json", "w") as file:
        json.dump({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "best_validation_accuracy": best_val_accuracy,
        }, file, indent=4)

    print(f"\nModel saved at: {output_dir}")


if __name__ == "__main__":
    main()
