from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LABEL_MAP = {
    "real": "real",
    "genuine": "real",
    "authentic": "real",
    "0": "real",
    "ai_generated": "ai_generated",
    "ai-generated": "ai_generated",
    "ai generated": "ai_generated",
    "fake": "ai_generated",
    "synthetic": "ai_generated",
    "1": "ai_generated",
}
SPLITS = ("train", "val", "test")


def normalize_label(value: str) -> str | None:
    return LABEL_MAP.get(value.strip().lower())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(roots: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    seen_paths: set[str] = set()
    for root in roots:
        if not root.exists():
            print(f"WARNING: source directory does not exist: {root}", file=sys.stderr)
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                resolved = str(path.resolve()).lower()
                if resolved not in seen_paths:
                    found.append(path)
                    seen_paths.add(resolved)
    return sorted(found, key=lambda p: str(p).lower())


def read_label_csv(csv_path: Path, filename_col: str, label_col: str) -> tuple[dict[str, str], list[dict]]:
    labels: dict[str, str] = {}
    errors: list[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV is empty or has no header row.")
        columns = {name.strip().lower(): name for name in reader.fieldnames}
        if filename_col.lower() not in columns or label_col.lower() not in columns:
            raise ValueError(
                f"CSV must contain columns '{filename_col}' and '{label_col}'. "
                f"Found columns: {reader.fieldnames}"
            )
        actual_filename_col = columns[filename_col.lower()]
        actual_label_col = columns[label_col.lower()]
        for row_number, row in enumerate(reader, start=2):
            filename = (row.get(actual_filename_col) or "").strip().replace("\\", "/")
            raw_label = (row.get(actual_label_col) or "").strip()
            if not filename:
                errors.append({"row": row_number, "reason": "empty filename"})
                continue
            label = normalize_label(raw_label)
            if label is None:
                errors.append({
                    "row": row_number,
                    "filename": filename,
                    "label": raw_label,
                    "reason": "unrecognized label; use REAL or AI_GENERATED (or supported aliases)",
                })
                continue
            # CSV contains filenames only. If a path was accidentally included, use its basename.
            key = Path(filename).name.lower()
            previous = labels.get(key)
            if previous and previous != label:
                errors.append({
                    "row": row_number,
                    "filename": filename,
                    "reason": f"conflicting labels for same filename: {previous} vs {label}",
                })
                continue
            labels[key] = label
    return labels, errors


def split_groups(groups_by_label: dict[str, list[dict]], seed: int,
                 train_ratio: float, val_ratio: float, test_ratio: float) -> list[dict]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must add up to 1.0")
    output: list[dict] = []
    rng = random.Random(seed)
    for label, groups in groups_by_label.items():
        groups = list(groups)
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # With enough groups, ensure each split receives examples where possible.
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))
        elif n == 2:
            n_train, n_val = 1, 1
        elif n == 1:
            n_train, n_val = 1, 0
        assignments = (
            ["train"] * n_train
            + ["val"] * n_val
            + ["test"] * max(0, n - n_train - n_val)
        )
        for group, split in zip(groups, assignments):
            for item in group["items"]:
                item["split"] = split
            output.append(group)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a CSV-labelled real-vs-AI image dataset, searching nested folders."
    )
    parser.add_argument("--csv", required=True, type=Path, help="CSV with filename,label columns")
    parser.add_argument(
        "--images-dir", required=True, action="append", type=Path,
        help="Image root directory; repeat this option to include multiple roots"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--filename-col", default="filename")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true",
                        help="Delete the output directory before writing it")
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    roots = [p.resolve() for p in args.images_dir]
    output_dir = args.output_dir.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    labels, csv_errors = read_label_csv(csv_path, args.filename_col, args.label_col)
    if csv_errors:
        print(f"CSV warnings/errors: {len(csv_errors)}")
        for error in csv_errors[:30]:
            print("  -", error)
    if not labels:
        raise RuntimeError("No valid filename/label rows were found in the CSV.")

    files = image_files(roots)
    print(f"Indexed {len(files)} image files across {len(roots)} root(s).")
    # Match by basename. Every matching path receives the CSV label.
    # If the CSV filename has no extension, also match by filename stem.
    matched_by_csv_name: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        matched_by_csv_name[path.name.lower()].append(path)
        matched_by_csv_name.setdefault(f"__stem__:{path.stem.lower()}", []).append(path)

    matches: list[dict] = []
    missing: list[str] = []
    for filename_key, label in labels.items():
        candidates = matched_by_csv_name.get(filename_key, [])
        if not candidates and Path(filename_key).suffix == "":
            candidates = matched_by_csv_name.get(f"__stem__:{filename_key}", [])
        # De-duplicate path candidates (stem index and exact index can overlap).
        candidates = sorted({p.resolve() for p in candidates}, key=lambda p: str(p).lower())
        if not candidates:
            missing.append(filename_key)
            continue
        for path in candidates:
            matches.append({"path": path, "label": label, "csv_filename": filename_key})

    # Hash content to avoid exact duplicates leaking across train/val/test.
    # If byte-identical content is assigned different labels, stop and ask for correction.
    hash_groups: dict[str, dict] = {}
    unreadable: list[dict] = []
    for item in matches:
        path = item["path"]
        try:
            with Image.open(path) as image:
                image.verify()
            content_hash = sha256_file(path)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            unreadable.append({"path": str(path), "error": str(exc)})
            continue

        if content_hash not in hash_groups:
            hash_groups[content_hash] = {
                "hash": content_hash,
                "label": item["label"],
                "items": [],
            }
        elif hash_groups[content_hash]["label"] != item["label"]:
            raise RuntimeError(
                "The same byte-identical image has conflicting labels:\n"
                f"  existing label: {hash_groups[content_hash]['label']}\n"
                f"  conflicting file: {path}\n"
                f"Correct the CSV labels before training."
            )
        hash_groups[content_hash]["items"].append({
            "path": path,
            "label": item["label"],
            "csv_filename": item["csv_filename"],
            "hash": content_hash,
        })

    groups_by_label: dict[str, list[dict]] = {"real": [], "ai_generated": []}
    for group in hash_groups.values():
        groups_by_label[group["label"]].append(group)

    assigned_groups = split_groups(
        groups_by_label, args.seed, args.train, args.val, args.test
    )

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    copied_hashes: set[str] = set()
    counts = {split: {"real": 0, "ai_generated": 0} for split in SPLITS}
    duplicate_files_skipped = 0

    for group in assigned_groups:
        split = group["items"][0]["split"] if group["items"] else "train"
        # Exact duplicates have the same hash; copy only one representative.
        representative = sorted(group["items"], key=lambda x: str(x["path"]).lower())[0]
        content_hash = group["hash"]
        if content_hash in copied_hashes:
            duplicate_files_skipped += len(group["items"])
            continue
        copied_hashes.add(content_hash)

        label = group["label"]
        source = representative["path"]
        # Unique deterministic name avoids overwrites when basenames repeat.
        output_name = f"{content_hash[:12]}_{source.stem}{source.suffix.lower()}"
        destination = output_dir / split / label / output_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        counts[split][label] += 1

        manifest.append({
            "split": split,
            "label": label.upper(),
            "source_path": str(source),
            "output_path": str(destination),
            "source_filename": source.name,
            "sha256": content_hash,
            "duplicate_paths_for_same_content": [str(x["path"]) for x in group["items"]],
        })

    report = {
        "csv": str(csv_path),
        "source_roots": [str(x) for x in roots],
        "output_dir": str(output_dir),
        "seed": args.seed,
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "csv_label_names": sorted(labels.keys()),
        "matched_file_paths": len(matches),
        "unique_image_contents_written": len(manifest),
        "exact_duplicate_file_paths_not_copied": duplicate_files_skipped,
        "missing_csv_filenames": missing,
        "csv_errors": csv_errors,
        "unreadable_images": unreadable,
        "counts": counts,
        "manifest": manifest,
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "label", "source_path", "output_path", "source_filename", "sha256"],
        )
        writer.writeheader()
        for row in manifest:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    print("\nDataset preparation complete.")
    print(json.dumps({
        "counts": counts,
        "matched_file_paths": len(matches),
        "unique_images_written": len(manifest),
        "missing_filenames": len(missing),
        "unreadable_images": len(unreadable),
        "report": str(output_dir / "dataset_report.json"),
        "manifest": str(output_dir / "manifest.csv"),
    }, indent=2))
    if missing:
        print("\nMissing filenames (first 30):")
        for filename in missing[:30]:
            print("  -", filename)
    if unreadable:
        print("\nUnreadable files (first 10):")
        for item in unreadable[:10]:
            print("  -", item["path"], "=>", item["error"])
    if sum(sum(v.values()) for v in counts.values()) < 3:
        print("WARNING: Fewer than 3 unique images were prepared; training/evaluation splits may be invalid.")


if __name__ == "__main__":
    main()
