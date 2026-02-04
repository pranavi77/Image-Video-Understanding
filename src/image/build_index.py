# src/build_index.py  (SigLIP - CORRECT ARCHITECTURE)

# Fix OpenMP conflict on Mac (MUST BE FIRST!)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import numpy as np
import torch
import faiss
from datasets import load_dataset
from transformers import AutoModel, AutoImageProcessor

DATA_DIR = "data/10k/test"
OUT_INDEX = "image.index"
OUT_META  = "meta.json"

MODEL = "google/siglip-base-patch16-224"
BATCH = 16
MAXN  = 500 

print("Loading dataset...", flush=True)
ds = load_dataset("imagefolder", data_dir=DATA_DIR, split="train")
print("Total images:", len(ds), flush=True)

ds = ds.select(range(min(MAXN, len(ds))))
print("Using:", len(ds), flush=True)

paths = [ex.get("path", getattr(ex["image"], "filename", "")) for ex in ds]

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Device:", device, flush=True)

print("Loading SigLIP...", flush=True)
model = AutoModel.from_pretrained(MODEL).to(device).eval()
image_processor = AutoImageProcessor.from_pretrained(MODEL)
print("SigLIP loaded.", flush=True)

embs = []
print("Encoding...", flush=True)

with torch.no_grad():
    for i in range(0, len(ds), BATCH):
        batch = ds[i : i + BATCH]
        imgs = batch["image"]

        # Process images
        inp = image_processor(images=imgs, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}

        # SigLIP: vision_model returns pooled features directly
        vision_out = model.vision_model(**inp)
        x = vision_out.pooler_output  # This is already the final embedding: (B, 768)
        
        # Normalize
        x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
        
        embs.append(x.cpu().numpy().astype("float32"))

        print(f"Encoded {min(i+BATCH, len(ds))}/{len(ds)}", flush=True)

embs = np.vstack(embs)  # (N, D)

index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)
faiss.write_index(index, OUT_INDEX)

with open(OUT_META, "w") as f:
    json.dump(paths, f)

print("=" * 60)
print("✅ DONE!")
print("=" * 60)
print(f"Saved: {OUT_INDEX}, {OUT_META}")
print(f"Indexed: N={len(paths)}, D={embs.shape[1]}")
print(f"\nNow run: python src/query_index.py")