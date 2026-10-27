from pathlib import Path
import random, shutil, json, hashlib
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEED = 42
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
JPEG_QUALITIES = [90, 70, 40, 20]
MAX_PER_CLASS = 0  # 0 = use all images; set e.g. 500 for a small trial

def image_files(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)

def stable_key(path):
    # Source paths are used only for deterministic splitting; copied images get unique names.
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]

def main():
    random.seed(SEED)
    classes = ["real", "ai_generated"]
    sources = {}
    for cls in classes:
        folder = RAW / cls
        files = image_files(folder)
        if MAX_PER_CLASS:
            files = files[:MAX_PER_CLASS]
        if len(files) < 10:
            raise SystemExit(
                f"Need at least 10 images in {folder}; found {len(files)}. "
                "Add real images to data/raw/real and AI-generated images to data/raw/ai_generated."
            )
        random.shuffle(files)
        n = len(files)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])
        sources[cls] = {
            "train": files[:n_train],
            "val": files[n_train:n_train+n_val],
            "test": files[n_train+n_val:]
        }

    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = []
    for split, class_map in ((s, {c: sources[c][s] for c in classes}) for s in SPLITS):
        for cls, files in class_map.items():
            dest = OUT / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for idx, src in enumerate(tqdm(files, desc=f"{split}/{cls}")):
                try:
                    with Image.open(src) as im:
                        im = im.convert("RGB")
                        stem = f"{stable_key(src)}_{idx:06d}"
                        # Keep original image.
                        original_path = dest / f"{stem}_original.jpg"
                        im.save(original_path, "JPEG", quality=95, optimize=True)
                        manifest.append({
                            "split": split, "label": cls, "source": str(src.relative_to(RAW)),
                            "output": str(original_path.relative_to(ROOT)), "variant": "original",
                            "jpeg_quality": 95
                        })
                        # Generate controlled JPEG variants. The source identity stays in the same split.
                        for quality in JPEG_QUALITIES:
                            out_path = dest / f"{stem}_q{quality}.jpg"
                            im.save(out_path, "JPEG", quality=quality, optimize=True)
                            manifest.append({
                                "split": split, "label": cls, "source": str(src.relative_to(RAW)),
                                "output": str(out_path.relative_to(ROOT)), "variant": "compressed",
                                "jpeg_quality": quality
                            })
                except Exception as exc:
                    print(f"Skipping unreadable image {src}: {exc}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nDataset prepared.")
    print(f"Processed images: {sum(1 for p in OUT.rglob('*.jpg'))}")
    print(f"Manifest: {OUT / 'manifest.json'}")
    print("IMPORTANT: all variants of each source image remain within the same split.")

if __name__ == "__main__":
    main()
