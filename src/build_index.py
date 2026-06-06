import pandas as pd
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

RAW       = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Medical RAG — Building FAISS Vector Index")
print("=" * 60)

# ── Load Papers ────────────────────────────────────────────
df = pd.read_csv(RAW / "pubmed_papers.csv")
df = df.dropna(subset=["abstract", "title"])
df["text"] = df["title"] + " " + df["abstract"]
print(f"\nPapers loaded : {len(df)}")

# ── Embedding Model ────────────────────────────────────────
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded ✓")

# ── Create Embeddings ──────────────────────────────────────
print("\nCreating embeddings (1-2 min)...")
texts      = df["text"].tolist()
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"Embeddings shape: {embeddings.shape}")

# ── Build FAISS Index ──────────────────────────────────────
print("\nBuilding FAISS index...")
dim   = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity)
index.add(embeddings.astype(np.float32))
print(f"FAISS index built ✓")
print(f"Total vectors : {index.ntotal}")

# ── Save Everything ────────────────────────────────────────
faiss.write_index(index, str(PROCESSED / "pubmed.index"))

# Metadata save karo
metadata = df[["pmid","title","abstract","journal","year","authors","url"]].to_dict("records")
with open(PROCESSED / "metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

# Embedding model name save karo
with open(PROCESSED / "config.pkl", "wb") as f:
    pickle.dump({"model_name": "all-MiniLM-L6-v2", "dim": dim}, f)

print("\n" + "=" * 60)
print("Files saved:")
print(f"  data/processed/pubmed.index   — FAISS vector index")
print(f"  data/processed/metadata.pkl   — paper metadata")
print(f"  data/processed/config.pkl     — config")
print(f"\nIndex size : {index.ntotal} vectors")
print(f"Dimensions : {dim}")
print("=" * 60)

# ── Test Search ────────────────────────────────────────────
print("\nTest search: 'diabetes treatment outcomes'")
query      = "diabetes treatment outcomes"
q_emb      = model.encode([query], convert_to_numpy=True)
faiss.normalize_L2(q_emb)
scores, ids = index.search(q_emb.astype(np.float32), k=3)

print("\nTop 3 relevant papers:")
for i, (score, idx) in enumerate(zip(scores[0], ids[0]), 1):
    paper = metadata[idx]
    print(f"\n{i}. [{score:.3f}] {paper['title'][:70]}...")
    print(f"   Journal: {paper['journal'][:50]}")
    print(f"   Year: {paper['year']}")

print("\nNext → src/rag_pipeline.py")