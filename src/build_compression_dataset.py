from pathlib import Path
from PIL import Image, ImageOps
import argparse, csv, hashlib, json, random, shutil

LABELS = ("real", "ai_generated")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
QUALITIES = (20, 40, 60, 80, 100)
RATIOS = {"train": .70, "val": .15, "test": .15}

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def collect(source):
    result, hashes = {}, {}
    for label in LABELS:
        folder = source / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")
        unique = {}
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in EXTS:
                unique.setdefault(digest(p), p)
        if not unique:
            raise RuntimeError(f"No supported images in {folder}")
        result[label] = list(unique.values())
        hashes[label] = set(unique)
    conflict = hashes["real"] & hashes["ai_generated"]
    if conflict:
        raise RuntimeError("Exact byte-identical images exist in both classes. Resolve label conflicts before building.")
    return result

def make_variant(src, dst, quality):
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, "white")
            bg.paste(rgba, mask=rgba.getchannel("A"))
            im = bg
        else:
            im = im.convert("RGB")
        im.save(dst, "JPEG", quality=quality, optimize=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Contains real/ and ai_generated/")
    ap.add_argument("--output", required=True, help="Dedicated generated-data folder; deleted before rebuild")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    src, out = Path(a.source).expanduser().resolve(), Path(a.output).expanduser().resolve()
    if src == out or src in out.parents or out in src.parents:
        raise ValueError("Source and output must be separate, non-nested folders.")
    if out.exists():
        print(f"Deleting existing generated output: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    source_files = collect(src)
    manifest, summary = [], {}
    for label in LABELS:
        items = source_files[label][:]
        random.Random(a.seed).shuffle(items)
        n = len(items)
        ntrain, nval = int(n * .70), int(n * .15)
        splits = {"train": items[:ntrain], "val": items[ntrain:ntrain+nval], "test": items[ntrain+nval:]}
        summary[label] = {"unique_sources": n, "splits": {k: len(v) for k, v in splits.items()}}
        for split, paths in splits.items():
            dest_dir = out / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            for source in paths:
                oid = digest(source)[:16]
                original_name = f"{oid}_original{source.suffix.lower()}"
                original_dest = dest_dir / original_name
                shutil.copy2(source, original_dest)
                common = {"label": label, "original_id": oid, "split": split, "source_path": str(source)}
                manifest.append({**common, "filename": original_name, "image_path": original_dest.relative_to(out).as_posix(), "variant": "original", "quality": ""})
                for q in QUALITIES:
                    name = f"{oid}_jpeg_q{q}.jpg"
                    dest = dest_dir / name
                    make_variant(source, dest, q)
                    manifest.append({**common, "filename": name, "image_path": dest.relative_to(out).as_posix(), "variant": f"jpeg_q{q}", "quality": q})
    fields = ["filename", "label", "image_path", "variant", "quality", "original_id", "split", "source_path"]
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(manifest)
    summary.update(total_source_images=sum(summary[x]["unique_sources"] for x in LABELS),
                   total_generated_files=len(manifest), qualities=list(QUALITIES),
                   split_ratios=RATIOS, seed=a.seed)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Dataset created: {out}")

if __name__ == "__main__":
    main()
