# src/siglip_embed_noindex.py
# SigLIP direct search (NO FAISS). Re-encodes images each run.

import os
import shutil
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoImageProcessor, T5TokenizerFast

def get_device():
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/10k/test")
    ap.add_argument("--out_dir", default="search_results_noindex")
    ap.add_argument("--query", default="a car driving on a road")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max_images", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--model_name", default="google/siglip-base-patch16-224")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")

    print("Loading dataset...", flush=True)
    ds = load_dataset("imagefolder", data_dir=args.data_dir, split="train")
    ds = ds.select(range(min(args.max_images, len(ds))))
    print("Using:", len(ds), flush=True)

    paths = [ex.get("path", getattr(ex["image"], "filename", "")) for ex in ds]

    device = get_device()
    print("Device:", device, flush=True)

    print("Loading SigLIP...", flush=True)
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)
    print("SigLIP loaded.", flush=True)

    # Encode images
    print("Encoding images...", flush=True)
    embs = []
    with torch.no_grad():
        for i in range(0, len(ds), args.batch_size):
            batch = ds[i : i + args.batch_size]
            imgs = batch["image"]

            inp = image_processor(images=imgs, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}

            # Get vision embeddings using vision_model (like build_index.py)
            vision_out = model.vision_model(**inp)
            x = vision_out.pooler_output  # Extract the actual tensor (B, D)
            
            # Normalize
            x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
            embs.append(x.cpu())

            if i == 0:
                print("First batch done.", flush=True)

    image_embs = torch.cat(embs, dim=0)                    # (N, D)
    print("Image embs:", tuple(image_embs.shape), flush=True)

    # Encode query
    print("Query:", args.query, flush=True)
    with torch.no_grad():
        t = tokenizer([args.query], return_tensors="pt", padding=True, truncation=True)
        t = {k: v.to(device) for k, v in t.items()}
        
        # Get text embeddings using text_model (matching your query_index.py)
        text_out = model.text_model(**t)
        q = text_out.pooler_output  # Extract the actual tensor (1, D)
        
        # Normalize
        q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
        q = q.cpu()

    # Search (cosine via dot)
    scores = (image_embs @ q.T).squeeze(1)
    k = min(args.topk, len(scores))
    topk_scores, topk_idx = torch.topk(scores, k)

    print("Top-k indices:", topk_idx.tolist(), flush=True)
    print("Top-k scores:", [float(x) for x in topk_scores.tolist()], flush=True)

    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    for f in os.listdir(args.out_dir):
        fp = os.path.join(args.out_dir, f)
        if os.path.isfile(fp):
            try: os.remove(fp)
            except: pass

    for rank, idx in enumerate(topk_idx.tolist(), start=1):
        src = paths[idx]
        if src and os.path.exists(src):
            filename = os.path.basename(src)
            dst = os.path.join(args.out_dir, f"{rank:02d}_score_{float(scores[idx]):.4f}_{filename}")
            shutil.copy(src, dst)
            print("Saved:", dst, flush=True)

    print("✅ DONE", flush=True)

if __name__ == "__main__":
    main()