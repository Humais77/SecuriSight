
from pathlib import Path
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import pandas as pd
import json
import argparse
from tqdm import tqdm


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


class ManifestDataset(Dataset):
    def __init__(self, dataframe, root, processor):
        self.df = dataframe.reset_index(drop=True)
        self.root = Path(root)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        path = self.root / row["image_path"]

        with Image.open(path) as image:
            image = image.convert("RGB")

            inputs = self.processor(
                images=image,
                return_tensors="pt"
            )

        label = 0 if row["label"] == "AI_GENERATED" else 1

        return (
            inputs["pixel_values"].squeeze(0),
            label,
            index
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/compression_dataset"
    )

    parser.add_argument(
        "--model",
        default="outputs/vit-compression-robust"
    )

    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()

    root = Path(args.data)
    model_path = Path(args.model)

    device = get_device()

    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(model_path)

    model = AutoModelForImageClassification.from_pretrained(
        model_path
    ).to(device)

    model.eval()

    df = pd.read_csv(root / "manifest.csv")
    df = df[df["split"] == "test"].copy()

    dataset = ManifestDataset(df, root, processor)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    predictions = [None] * len(df)
    confidence_scores = [None] * len(df)

    with torch.no_grad():
        for pixels, labels, indices in tqdm(loader):
            pixels = pixels.to(device)

            logits = model(pixel_values=pixels).logits

            probabilities = torch.softmax(logits, dim=1)

            confidence, predicted = probabilities.max(dim=1)

            for i, index in enumerate(indices.tolist()):
                predictions[index] = int(predicted[i].cpu())
                confidence_scores[index] = float(
                    confidence[i].cpu()
                )

    df["prediction"] = predictions
    df["confidence"] = confidence_scores

    df["actual_id"] = df["label"].map({
        "AI_GENERATED": 0,
        "REAL": 1
    })

    df["correct"] = df["prediction"] == df["actual_id"]

    results = []

    for variant, group in df.groupby("variant"):
        actual = group["actual_id"]
        predicted = group["prediction"]

        results.append({
            "variant": variant,
            "images": len(group),
            "accuracy": accuracy_score(actual, predicted),
            "precision": precision_score(
                actual, predicted, zero_division=0
            ),
            "recall": recall_score(
                actual, predicted, zero_division=0
            ),
            "f1_score": f1_score(
                actual, predicted, zero_division=0
            ),
            "average_confidence": group["confidence"].mean(),
        })

    results_df = pd.DataFrame(results)

    print("\nCompression evaluation results:")
    print(results_df.to_string(index=False))

    output_dir = model_path / "compression_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        output_dir / "quality_metrics.csv",
        index=False
    )

    df.to_csv(
        output_dir / "individual_predictions.csv",
        index=False
    )

    # Check whether all versions of an original get the same prediction.
    consistency = (
        df.groupby("original_id")["prediction"]
        .nunique()
    )

    consistency_score = float(
        (consistency == 1).mean()
    ) if len(consistency) else 0.0

    summary = {
        "prediction_consistency": consistency_score,
        "original_images_evaluated": int(len(consistency)),
        "variant_metrics": results,
    }

    with open(output_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=4)

    print(
        "\nPrediction consistency: "
        f"{consistency_score * 100:.2f}%"
    )

    print(f"\nResults saved in: {output_dir}")


if __name__ == "__main__":
    main()
