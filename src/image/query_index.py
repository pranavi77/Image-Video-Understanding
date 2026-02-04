import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import shutil
import numpy as np
import torch
import faiss
from transformers import AutoModel, T5TokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

INDEX_PATH = os.path.join(ROOT, "image.index")
META_PATH  = os.path.join(ROOT, "meta.json")
OUT_DIR    = os.path.join(ROOT, "search_results")

MODEL = "google/siglip-base-patch16-224"
TOPK_DEFAULT = 5


def get_device():
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def encode_text(model, tokenizer, device, query: str) -> np.ndarray:
    """Encode text query to embedding vector"""
    with torch.no_grad():
        # Tokenize text (T5 tokenizer returns input_ids only by default)
        inp = tokenizer(query, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}

        # Get attention mask if not present (T5 quirk)
        if "attention_mask" not in inp:
            inp["attention_mask"] = torch.ones_like(inp["input_ids"])

        # SigLIP: text_model returns pooled features directly
        text_out = model.text_model(**inp)
        q = text_out.pooler_output  # Final embedding: (1, 768)
        
        # Normalize
        q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
        
        return q.cpu().numpy().astype("float32")


def main():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError(
            "Missing image.index/meta.json. Run build_index.py first."
        )

    print("=" * 60)
    print("SigLIP Image Search")
    print("=" * 60)
    
    # Load index
    index = faiss.read_index(INDEX_PATH)
    paths = json.load(open(META_PATH))
    
    print(f"✅ Loaded index: {index.ntotal} vectors, dim={index.d}")
    print(f"✅ Loaded metadata: {len(paths)} paths")

    # Load model
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Loading model: {MODEL}")
    
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    tokenizer = T5TokenizerFast.from_pretrained(MODEL)
    
    print("✅ Model loaded")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("Type a query (or 'exit' to quit)")
    print("=" * 60)
    
    while True:
        query = input("\nQuery> ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue

        k_str = input(f"Top-K [default {TOPK_DEFAULT}]> ").strip()
        topk = int(k_str) if k_str else TOPK_DEFAULT
        topk = min(topk, index.ntotal)

        # Search
        print(f"\n🔍 Searching for: '{query}' (top-{topk})")
        qvec = encode_text(model, tokenizer, device, query)
        scores, idxs = index.search(qvec, topk)

        # Clear old results
        for f in os.listdir(OUT_DIR):
            fp = os.path.join(OUT_DIR, f)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except:
                    pass

        # Display and save results
        print("\n" + "-" * 60)
        print("Results:")
        print("-" * 60)
        
        saved = 0
        for rank, (s, i) in enumerate(zip(scores[0], idxs[0]), start=1):
            i = int(i)
            src = paths[i]
            print(f"{rank:02d}. score={float(s):.4f}  {src}")

            if src and os.path.exists(src):
                filename = os.path.basename(src)
                dst = os.path.join(OUT_DIR, f"{rank:02d}_score_{float(s):.4f}_{filename}")
                shutil.copy(src, dst)
                saved += 1
        
        print("-" * 60)
        print(f"✅ Saved {saved}/{topk} images to: {OUT_DIR}/")

    print("\n👋 Done!")


if __name__ == "__main__":
    main()