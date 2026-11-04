from pathlib import Path
from PIL import Image
import argparse, csv, json
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm.auto import tqdm

LABEL2ID={"ai_generated":0, "real":1}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=["train","val","test"], default="test")
    ap.add_argument("--batch-size", type=int, default=16); ap.add_argument("--output", default="")
    a=ap.parse_args(); root=Path(a.data); model_dir=Path(a.model)
    with (root/"manifest.csv").open(encoding="utf-8") as f:
        rows=[r for r in csv.DictReader(f) if r["split"]==a.split]
    if not rows: raise RuntimeError(f"No rows for split {a.split}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor=AutoImageProcessor.from_pretrained(model_dir)
    model=AutoModelForImageClassification.from_pretrained(model_dir).to(device).eval()

    # --- Label mapping sanity check ---
    expected_labels = {
        0: "AI_GENERATED",
        1: "REAL"
    }
    actual_labels = model.config.id2label
    if actual_labels != expected_labels:
        raise ValueError(
            f"Model label mapping mismatch: {actual_labels}"
        )
    # --- End label mapping check ---

    yt=[]; yp=[]; confs=[]; by=defaultdict(lambda: {"y":[],"p":[],"c":[]}); groups=defaultdict(list)
    for start in tqdm(range(0,len(rows),a.batch_size)):
        batch=rows[start:start+a.batch_size]; imgs=[]
        for row in batch:
            with Image.open(root/row["image_path"]) as im: imgs.append(im.convert("RGB").copy())
        px=processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode():
            probs=torch.softmax(model(pixel_values=px).logits, dim=1)
            confidence,pred=probs.max(dim=1)
        for row,p,c in zip(batch,pred.cpu().tolist(),confidence.cpu().tolist()):
            y=LABEL2ID[row["label"]]; yt.append(y); yp.append(p); confs.append(c)
            v=by[row["variant"]]; v["y"].append(y); v["p"].append(p); v["c"].append(c)
            groups[row["original_id"]].append(p)
    precision,recall,f1,_=precision_recall_fscore_support(yt,yp,average="macro",zero_division=0)
    result={"split":a.split,"number_of_images":len(rows),
      "overall":{"accuracy":float(accuracy_score(yt,yp)),"macro_precision":float(precision),
        "macro_recall":float(recall),"macro_f1":float(f1),"average_max_probability":float(np.mean(confs)),
        "classification_report":classification_report(yt,yp,labels=[0,1],
          target_names=["AI_GENERATED","REAL"],output_dict=True,zero_division=0)},
      "by_variant":{},
      "original_group_consistency":{"groups":len(groups),
        "all_variants_same_prediction_percent":float(100*np.mean([len(set(v))==1 for v in groups.values()]))}}
    for name,v in by.items():
        p,r,f,_=precision_recall_fscore_support(v["y"],v["p"],average="macro",zero_division=0)
        result["by_variant"][name]={"count":len(v["y"]),"accuracy":float(accuracy_score(v["y"],v["p"])),
          "macro_precision":float(p),"macro_recall":float(r),"macro_f1":float(f),
          "average_max_probability":float(np.mean(v["c"]))}
    print(json.dumps(result,indent=2))
    target=Path(a.output) if a.output else model_dir/f"{a.split}_evaluation.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("Saved:",target)

if __name__=="__main__": main()