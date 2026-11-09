from pathlib import Path
import csv
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = ROOT / "dataset_source"
OUTPUT_CSV = ROOT / "data" / "labels.csv"

CLASS_FOLDERS = {
    "real": "REAL",
    "ai_generated": "AI_GENERATED",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tif", ".tiff"
}


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    unreadable = []

    for folder_name, label in CLASS_FOLDERS.items():
        class_folder = DATASET_ROOT / folder_name

        if not class_folder.exists():
            print(f"WARNING: Folder not found: {class_folder}")
            continue

        for image_path in class_folder.rglob("*"):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()

                rows.append({
                    "filename": image_path.name,
                    "label": label,
                    "path": str(image_path.resolve())
                })

            except (OSError, UnidentifiedImageError, ValueError) as exc:
                unreadable.append((str(image_path), str(exc)))

    rows.sort(key=lambda row: row["path"].lower())

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "label", "path"]
        )
        writer.writeheader()
        writer.writerows(rows)

    real_count = sum(row["label"] == "REAL" for row in rows)
    ai_count = sum(row["label"] == "AI_GENERATED" for row in rows)

    print("\nDataset label generation complete.")
    print(f"REAL images:       {real_count}")
    print(f"AI_GENERATED:      {ai_count}")
    print(f"Total valid images: {len(rows)}")
    print(f"Unreadable images:  {len(unreadable)}")
    print(f"CSV saved to:       {OUTPUT_CSV}")

    if unreadable:
        print("\nFirst 20 unreadable images:")
        for path, error in unreadable[:20]:
            print(path, "=>", error)


if __name__ == "__main__":
    main()