
from pathlib import Path
from PIL import Image, ImageFile
import argparse
import hashlib
import csv
import random
import shutil
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True

QUALITIES = [20, 40, 60, 80, 100]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_images(folder):
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )


def split_images(images, seed):
    images = images.copy()
    random.Random(seed).shuffle(images)

    total = len(images)
    train_count = int(total * SPLIT_RATIOS["train"])
    val_count = int(total * SPLIT_RATIOS["val"])

    if total >= 3:
        train_count = max(1, train_count)
        val_count = max(1, val_count)

        if train_count + val_count >= total:
            val_count = max(1, total - train_count - 1)

    return {
        "train": images[:train_count],
        "val": images[train_count:train_count + val_count],
        "test": images[train_count + val_count:],
    }


def open_as_rgb(path):
    with Image.open(path) as image:
        image.load()

        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            return background

        if image.mode == "P":
            image = image.convert("RGBA")
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            return background

        return image.convert("RGB")


def save_image_variants(source, destination, original_id):
    destination.mkdir(parents=True, exist_ok=True)

    # Preserve the original file without changing its contents.
    original_name = f"{original_id}__original{source.suffix.lower()}"
    original_path = destination / original_name
    shutil.copy2(source, original_path)

    records = [{
        "filename": original_name,
        "variant": "original",
        "quality": "",
        "path": original_path,
    }]

    # Generate JPEG versions.
    image = open_as_rgb(source)

    for quality in QUALITIES:
        filename = f"{original_id}__q{quality}.jpg"
        output_path = destination / filename

        image.save(
            output_path,
            format="JPEG",
            quality=quality,
            optimize=True
        )

        records.append({
            "filename": filename,
            "variant": f"Q{quality}",
            "quality": quality,
            "path": output_path,
        })

    return records


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="dataset_source",
        help="Source dataset folder"
    )

    parser.add_argument(
        "--output",
        default="data/compression_dataset",
        help="Output dataset folder"
    )

    parser.add_argument("--real-dir", default="real")
    parser.add_argument("--ai-dir", default="ai_generated")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    source_root = Path(args.source)
    output_root = Path(args.output)

    classes = {
        "real": source_root / args.real_dir,
        "ai_generated": source_root / args.ai_dir,
    }

    for label, folder in classes.items():
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing {label} folder: {folder}"
            )

    # Avoid mixing files from an older generated dataset.
    if output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    seen_hashes = set()

    for label, folder in classes.items():
        images = get_images(folder)

        print(f"\n{label}: found {len(images)} images")

        # Remove exact duplicate files before splitting.
        unique_images = []

        for image_path in images:
            try:
                file_hash = sha256_file(image_path)

                if file_hash in seen_hashes:
                    print(f"Skipping duplicate: {image_path.name}")
                    continue

                seen_hashes.add(file_hash)
                unique_images.append(image_path)

            except Exception as error:
                print(f"Skipping unreadable file {image_path}: {error}")

        print(f"{label}: {len(unique_images)} unique images")

        split_groups = split_images(unique_images, args.seed)

        for split_name, split_images_list in split_groups.items():
            print(
                f"{label} / {split_name}: "
                f"{len(split_images_list)} original images"
            )

            for index, source_image in enumerate(split_images_list):
                original_id = (
                    f"{label}_{index:06d}_"
                    f"{sha256_file(source_image)[:10]}"
                )

                destination = (
                    output_root / split_name / label
                )

                try:
                    variants = save_image_variants(
                        source_image,
                        destination,
                        original_id
                    )

                    for variant in variants:
                        relative_path = variant["path"].relative_to(
                            output_root
                        )

                        manifest.append({
                            "filename": variant["filename"],
                            "label": label.upper(),
                            "image_path": relative_path.as_posix(),
                            "variant": variant["variant"],
                            "quality": variant["quality"],
                            "original_id": original_id,
                            "split": split_name,
                            "source_path": str(source_image.resolve()),
                        })

                except Exception as error:
                    print(
                        f"Failed to process {source_image.name}: {error}"
                    )

    manifest_path = output_root / "manifest.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "label",
                "image_path",
                "variant",
                "quality",
                "original_id",
                "split",
                "source_path",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest)

    summary = {
        "total_images": len(manifest),
        "classes": list(classes.keys()),
        "qualities": QUALITIES,
        "splits": SPLIT_RATIOS,
    }

    with open(output_root / "dataset_summary.json", "w") as file:
        json.dump(summary, file, indent=4)

    print("\nDataset generation completed.")
    print(f"Total generated files: {len(manifest)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
