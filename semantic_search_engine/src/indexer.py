"""
indexer.py — FAISS index builder and persistence layer.

Supports two index types:
  • Flat  (IndexFlatIP)  — exact cosine search, best for <10k docs.
  • IVF   (IndexIVFFlat) — approximate search, scales to millions of docs.

Usage:
    from src.indexer import build_index, save_index, load_index, save_documents, load_documents
"""
import os
import pickle
import numpy as np
import faiss

from src.config import IVF_NLIST, FAISS_INDEX_PATH, DOCS_PATH


# ─── Index Building ───────────────────────────────────────────────────────────

def build_flat_index(embeddings: np.ndarray) -> faiss.Index:
    """Exact inner-product index (cosine on normalized vectors). Fast for <10k docs."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def build_ivf_index(embeddings: np.ndarray, nlist: int = IVF_NLIST) -> faiss.Index:
    """
    Approximate IVF index — partitions the vector space into `nlist` Voronoi cells.
    At query time, only nearby cells are searched → much faster at scale.

    Requires at least `nlist` training vectors.
    """
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dim)                        # Coarse quantizer
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    embeddings_f32 = embeddings.astype(np.float32)
    index.train(embeddings_f32)                               # Learn cluster centroids
    index.add(embeddings_f32)
    index.nprobe = max(1, nlist // 10)                        # Cells to visit at query time
    return index


def build_index(embeddings: np.ndarray, index_type: str = "IVF") -> faiss.Index:
    """
    Build either a Flat or IVF index depending on corpus size and `index_type`.

    Args:
        embeddings:  numpy array of shape (N, D), already L2-normalised.
        index_type:  "Flat" or "IVF".
    """
    n = embeddings.shape[0]

    # Fall back to Flat when corpus is too small to train IVF
    if index_type == "IVF" and n >= IVF_NLIST:
        print(f"  Building IVF index (nlist={IVF_NLIST}) for {n:,} vectors …")
        return build_ivf_index(embeddings)
    else:
        if index_type == "IVF":
            print(f"  Corpus too small for IVF ({n} < {IVF_NLIST}). Falling back to Flat index.")
        else:
            print(f"  Building Flat index for {n:,} vectors …")
        return build_flat_index(embeddings)


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_index(index: faiss.Index, path: str = FAISS_INDEX_PATH) -> None:
    """Persist a FAISS index to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    print(f"  ✅ FAISS index saved → {path}")


def load_index(path: str = FAISS_INDEX_PATH) -> faiss.Index:
    """Load a FAISS index from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No FAISS index found at '{path}'. Run --mode build first.")
    index = faiss.read_index(path)
    print(f"  ✅ FAISS index loaded ← {path}  ({index.ntotal:,} vectors)")
    return index


def save_documents(documents: list, path: str = DOCS_PATH) -> None:
    """Pickle the document corpus alongside the index."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(documents, f)
    print(f"  ✅ Documents saved → {path}  ({len(documents):,} passages)")


def load_documents(path: str = DOCS_PATH) -> list:
    """Load the pickled document corpus."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No documents found at '{path}'. Run --mode build first.")
    with open(path, "rb") as f:
        documents = pickle.load(f)
    print(f"  ✅ Documents loaded ← {path}  ({len(documents):,} passages)")
    return documents


def index_exists(index_path: str = FAISS_INDEX_PATH, docs_path: str = DOCS_PATH) -> bool:
    """Return True if both index and documents are saved to disk."""
    return os.path.exists(index_path) and os.path.exists(docs_path)
