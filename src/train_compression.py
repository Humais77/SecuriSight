from pathlib import Path
from PIL import Image
import argparse, json, random, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

LABEL2ID = {"ai_generated": 0, "real": 1}
ID2LABEL = {0: "AI_GENERATED", 1: "REAL"}
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class FolderImages(Dataset):
    def __init__(self, root, split, processor):
        self.items, self.processor = [], processor
        for label, idx in LABEL2ID.items():
            folder = Path(root) / split / label
            if not folder.is_dir(): raise FileNotFoundError(folder)
            self.items += [(p, idx) for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in EXTS]
        if not self.items: raise RuntimeError(f"No images found in {root}/{split}")
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im:
            x = self.processor(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return x, torch.tensor(label, dtype=torch.long)

def collate(batch):
    x, y = zip(*batch)
    return torch.stack(x), torch.stack(y)

def epoch(model, loader, device, optimizer=None, scaler=None, scheduler=None):
    training = optimizer is not None
    model.train(training); losses=[]; pred_all=[]; true_all=[]
    for pixels, labels in tqdm(loader, leave=False):
        pixels, labels = pixels.to(device), labels.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                out = model(pixel_values=pixels, labels=labels)
            if training:
                if scaler.is_enabled():
                    scaler.scale(out.loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    out.loss.backward(); optimizer.step()
                if scheduler: scheduler.step()
        losses.append(out.loss.detach().item())
        pred_all.extend(out.logits.argmax(1).detach().cpu().tolist())
        true_all.extend(labels.detach().cpu().tolist())
    return {"loss": float(np.mean(losses)), "accuracy": float(accuracy_score(true_all, pred_all)),
            "macro_f1": float(f1_score(true_all, pred_all, average="macro", zero_division=0))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=3); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default="google/vit-base-patch16-224-in21k")
    a=ap.parse_args(); seed_all(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type=="cuda" else "")
    outdir=Path(a.output); outdir.mkdir(parents=True, exist_ok=True)
    processor=AutoImageProcessor.from_pretrained(a.checkpoint)
    model=AutoModelForImageClassification.from_pretrained(
        a.checkpoint, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True).to(device)
    train_ds=FolderImages(a.data, "train", processor); val_ds=FolderImages(a.data, "val", processor)
    train_dl=DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                        pin_memory=device.type=="cuda", collate_fn=collate)
    val_dl=DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                      pin_memory=device.type=="cuda", collate_fn=collate)
    opt=torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    steps=len(train_dl)*a.epochs
    sched=get_linear_schedule_with_warmup(opt, int(.05*steps), steps)
    scaler=torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    best=-1; stale=0; history=[]
    for ep in range(1, a.epochs+1):
        t=time.time()
        tr=epoch(model, train_dl, device, opt, scaler, sched)
        va=epoch(model, val_dl, device)
        rec={"epoch":ep, "train":tr, "validation":va, "seconds":round(time.time()-t,1)}
        history.append(rec); print(json.dumps(rec, indent=2))
        if va["macro_f1"] > best:
            best=va["macro_f1"]; stale=0
            model.save_pretrained(outdir); processor.save_pretrained(outdir)
            print(f"Saved best checkpoint; validation macro-F1={best:.4f}")
        else:
            stale+=1
            if stale>=a.patience: print("Early stopping."); break
    (outdir/"training_history.json").write_text(json.dumps(history, indent=2))
    print("Best model saved to", outdir)

if __name__=="__main__": main()
