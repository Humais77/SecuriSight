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

from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tif", ".tiff"
}

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


def image_files(roots: list[Path]) -> list[Path]:
    """Index supported images recursively from the supplied source folders."""
    found: list[Path] = []
    seen_paths: set[str] = set()

    for root in roots:
        if not root.exists():
            print(
                f"WARNING: Source directory does not exist: {root}",
                file=sys.stderr
            )
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            resolved = path.resolve()
            key = str(resolved).lower()

            if key not in seen_paths:
                found.append(resolved)
                seen_paths.add(key)

    return sorted(found, key=lambda p: str(p).lower())


def read_label_csv(
    csv_path: Path,
    filename_col: str,
    label_col: str,
    path_col: str = "path",
) -> tuple[list[dict], list[dict], bool]:
    """
    Read either:
      Old format: filename,label
      New format: filename,label,path

    Returns:
      records, errors, has_path_column
    """
    records: list[dict] = []
    errors: list[dict] = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError(f"CSV is empty or has no header: {csv_path}")

        columns = {
            name.strip().lower(): name
            for name in reader.fieldnames
            if name
        }

        filename_key = filename_col.strip().lower()
        label_key = label_col.strip().lower()
        path_key = path_col.strip().lower()

        if filename_key not in columns:
            raise ValueError(
                f"Missing '{filename_col}' column in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        if label_key not in columns:
            raise ValueError(
                f"Missing '{label_col}' column in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        has_path_column = path_key in columns

        for row_number, row in enumerate(reader, start=2):
            filename = (
                row.get(columns[filename_key]) or ""
            ).strip()

            raw_label = (
                row.get(columns[label_key]) or ""
            ).strip()

            raw_path = ""

            if has_path_column:
                raw_path = (
                    row.get(columns[path_key]) or ""
                ).strip()

            label = normalize_label(raw_label)

            if not filename or label is None:
                errors.append({
                    "csv": str(csv_path),
                    "row": row_number,
                    "filename": filename,
                    "label": raw_label,
                    "reason": "Missing filename or invalid label",
                })
                continue

            image_path = None

            if raw_path:
                image_path = Path(raw_path).expanduser()

                if not image_path.is_absolute():
                    # First try a path relative to the CSV file.
                    csv_relative_path = (
                        csv_path.parent / image_path
                    )

                    if csv_relative_path.is_file():
                        image_path = csv_relative_path
                    else:
                        image_path = Path.cwd() / image_path

                image_path = image_path.resolve()

                if not image_path.is_file():
                    errors.append({
                        "csv": str(csv_path),
                        "row": row_number,
                        "filename": filename,
                        "path": raw_path,
                        "reason": "Image file not found",
                    })
                    continue

            records.append({
                "path": image_path,
                "filename": filename,
                "label": label,
                "csv": str(csv_path),
                "csv_row": row_number,
            })

    return records, errors, has_path_column


def split_groups(
    groups_by_label: dict[str, list[dict]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> list[dict]:

    total_ratio = train_ratio + val_ratio + test_ratio

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            "Train, validation and test ratios must add up to 1.0."
        )

    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split ratios cannot be negative.")

    output: list[dict] = []
    rng = random.Random(seed)

    for label, original_groups in groups_by_label.items():
        groups = list(original_groups)
        rng.shuffle(groups)

        n = len(groups)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Ensure all three splits receive at least one group
        # when there are at least three unique groups.
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))

        elif n == 2:
            n_train = 1
            n_val = 1

        elif n == 1:
            n_train = 1
            n_val = 0

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
        description=(
            "Prepare a real-vs-AI image dataset using old filename-only "
            "CSVs or new CSVs containing full image paths."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        action="append",
        type=Path,
        help=(
            "CSV file. Repeat this argument to combine multiple CSVs, "
            "for example an old CSV and a new CSV."
        ),
    )

    parser.add_argument(
        "--images-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Image root directory for old filename-only CSVs. "
            "Repeat to include multiple directories."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument("--filename-col", default="filename")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--path-col", default="path")

    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before writing the new dataset.",
    )

    args = parser.parse_args()

    csv_paths = [p.expanduser().resolve() for p in args.csv]
    roots = [p.expanduser().resolve() for p in args.images_dir]
    output_dir = args.output_dir.expanduser().resolve()

    for csv_path in csv_paths:
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}"
            )

    # Read all CSV files. This supports combining your previous
    # filename-only CSV with a new full-path CSV.
    all_records: list[dict] = []
    csv_errors: list[dict] = []
    legacy_records: list[dict] = []

    for csv_path in csv_paths:
        records, errors, has_path_column = read_label_csv(
            csv_path=csv_path,
            filename_col=args.filename_col,
            label_col=args.label_col,
            path_col=args.path_col,
        )

        all_records.extend(records)
        csv_errors.extend(errors)

        if not has_path_column:
            legacy_records.extend(records)

    print(f"CSV files loaded: {len(csv_paths)}")
    print(f"Valid CSV records: {len(all_records)}")

    if csv_errors:
        print(f"CSV warnings/errors: {len(csv_errors)}")

        for error in csv_errors[:30]:
            print("WARNING:", error)

    if not all_records:
        raise RuntimeError(
            "No valid records were found in the supplied CSV files."
        )

    # Only index folders when at least one CSV uses the old format.
    indexed_files: list[Path] = []

    if legacy_records:
        if not roots:
            raise ValueError(
                "At least one --images-dir is required because "
                "a CSV does not contain a path column."
            )

        indexed_files = image_files(roots)

        print(
            f"Indexed {len(indexed_files)} images across "
            f"{len(roots)} source folder(s)."
        )

    # Build filename and stem indexes for legacy CSV records.
    matched_by_name: dict[str, list[Path]] = defaultdict(list)
    matched_by_stem: dict[str, list[Path]] = defaultdict(list)

    for path in indexed_files:
        matched_by_name[path.name.lower()].append(path)
        matched_by_stem[path.stem.lower()].append(path)

    matches: list[dict] = []
    missing: list[dict] = []
    ambiguous_matches: list[dict] = []

    for item in all_records:
        # New CSV: use the exact path. No basename matching.
        if item["path"] is not None:
            matches.append({
                "path": item["path"],
                "label": item["label"],
                "csv_filename": item["filename"],
                "csv": item["csv"],
                "csv_row": item["csv_row"],
            })
            continue

        # Old CSV: search by filename or, if extension is missing, stem.
        filename = item["filename"].strip()
        filename_key = filename.lower()

        candidates = matched_by_name.get(filename_key, [])

        if not candidates and Path(filename).suffix == "":
            candidates = matched_by_stem.get(filename_key, [])

        candidates = sorted(
            {p.resolve() for p in candidates},
            key=lambda p: str(p).lower(),
        )

        if not candidates:
            missing.append({
                "csv": item["csv"],
                "row": item["csv_row"],
                "filename": filename,
            })
            continue

        if len(candidates) > 1:
            ambiguous_matches.append({
                "filename": filename,
                "candidate_paths": [str(p) for p in candidates],
                "csv": item["csv"],
                "row": item["csv_row"],
            })

        for path in candidates:
            matches.append({
                "path": path,
                "label": item["label"],
                "csv_filename": filename,
                "csv": item["csv"],
                "csv_row": item["csv_row"],
            })

    if not matches:
        raise RuntimeError(
            "No image paths could be matched. Check your CSV paths "
            "and --images-dir arguments."
        )

    print(f"Matched image paths: {len(matches)}")

    if missing:
        print(f"Missing old-CSV filenames: {len(missing)}")

    if ambiguous_matches:
        print(
            f"WARNING: {len(ambiguous_matches)} old-CSV filenames "
            "matched multiple image paths."
        )
        print(
            "All matching paths will be included. Check these entries "
            "in dataset_report.json."
        )

    # Validate images and group byte-identical files by SHA256.
    hash_groups: dict[str, dict] = {}
    unreadable: list[dict] = []

    for item in matches:
        path = item["path"]

        try:
            with Image.open(path) as image:
                image.verify()

            content_hash = sha256_file(path)

        except (
            OSError,
            UnidentifiedImageError,
            ValueError,
        ) as exc:
            unreadable.append({
                "path": str(path),
                "error": str(exc),
            })
            continue

        if content_hash not in hash_groups:
            hash_groups[content_hash] = {
                "hash": content_hash,
                "label": item["label"],
                "items": [],
            }

        elif hash_groups[content_hash]["label"] != item["label"]:
            raise RuntimeError(
                "The same byte-identical image has conflicting labels.\n"
                f"Existing label: {hash_groups[content_hash]['label']}\n"
                f"Conflicting file: {path}\n"
                "Correct the CSV labels before training."
            )

        hash_groups[content_hash]["items"].append({
            "path": path,
            "label": item["label"],
            "csv_filename": item["csv_filename"],
            "csv": item["csv"],
            "csv_row": item["csv_row"],
            "hash": content_hash,
        })

    groups_by_label: dict[str, list[dict]] = {
        "real": [],
        "ai_generated": [],
    }

    for group in hash_groups.values():
        groups_by_label[group["label"]].append(group)

    print("\nUnique image contents by class:")

    for label, groups in groups_by_label.items():
        print(f"{label}: {len(groups)}")

    if not groups_by_label["real"] or not groups_by_label["ai_generated"]:
        raise RuntimeError(
            "Both classes must contain at least one valid unique image."
        )

    assigned_groups = split_groups(
        groups_by_label=groups_by_label,
        seed=args.seed,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
    )

    # Clean only the selected output directory.
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    counts = {
        split: {
            "real": 0,
            "ai_generated": 0,
        }
        for split in SPLITS
    }

    duplicate_files_skipped = 0

    for group in assigned_groups:
        if not group["items"]:
            continue

        split = group["items"][0]["split"]

        # One representative is copied for each exact-content group.
        representative = sorted(
            group["items"],
            key=lambda x: str(x["path"]).lower(),
        )[0]

        content_hash = group["hash"]
        label = group["label"]
        source = representative["path"]

        output_name = (
            f"{content_hash[:12]}_"
            f"{source.stem}{source.suffix.lower()}"
        )

        destination = (
            output_dir / split / label / output_name
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(source, destination)

        counts[split][label] += 1

        duplicate_files_skipped += max(
            0,
            len(group["items"]) - 1,
        )

        manifest.append({
            "split": split,
            "label": label.upper(),
            "source_path": str(source),
            "output_path": str(destination),
            "source_filename": source.name,
            "sha256": content_hash,
            "duplicate_paths_for_same_content": [
                str(x["path"])
                for x in group["items"]
            ],
        })

    report = {
        "csv_files": [str(p) for p in csv_paths],
        "source_roots": [str(p) for p in roots],
        "output_dir": str(output_dir),
        "seed": args.seed,
        "ratios": {
            "train": args.train,
            "val": args.val,
            "test": args.test,
        },
        "matched_file_paths": len(matches),
        "unique_image_contents_written": len(manifest),
        "exact_duplicate_file_paths_not_copied": duplicate_files_skipped,
        "missing_old_csv_filenames": missing,
        "ambiguous_old_csv_matches": ambiguous_matches,
        "csv_errors": csv_errors,
        "unreadable_images": unreadable,
        "counts": counts,
        "manifest": manifest,
    }

    report_path = output_dir / "dataset_report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    manifest_path = output_dir / "manifest.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "split",
            "label",
            "source_path",
            "output_path",
            "source_filename",
            "sha256",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in manifest:
            writer.writerow({
                key: row[key]
                for key in fieldnames
            })

    total_written = sum(
        sum(split_counts.values())
        for split_counts in counts.values()
    )

    print("\nDataset preparation complete.")
    print(json.dumps({
        "counts": counts,
        "matched_file_paths": len(matches),
        "unique_images_written": len(manifest),
        "exact_duplicates_not_copied": duplicate_files_skipped,
        "missing_old_csv_filenames": len(missing),
        "ambiguous_old_csv_matches": len(ambiguous_matches),
        "unreadable_images": len(unreadable),
        "total_images_written": total_written,
        "report": str(report_path),
        "manifest": str(manifest_path),
    }, indent=2))

    if missing:
        print("\nMissing filenames (first 20):")
        for item in missing[:20]:
            print(" -", item)

    if unreadable:
        print("\nUnreadable images (first 10):")
        for item in unreadable[:10]:
            print(" -", item["path"], "=>", item["error"])

    if total_written < 3:
        print(
            "\nWARNING: Fewer than 3 unique images were prepared. "
            "Your training and evaluation splits may be invalid."
        )


if __name__ == "__main__":
    main()