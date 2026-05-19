"""
retriever.py — Bi-Encoder retriever backed by FAISS.

Encodes documents and queries using a Sentence Transformer (all-MiniLM-L6-v2).
Supports building an in-memory index from scratch OR loading a pre-built index
from disk (via indexer.py) so you never wait for re-encoding.
"""
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.config import BI_ENCODER_MODEL, DEVICE, BATCH_SIZE
from src.config  import INDEX_TYPE  # noqa: F401
from src.indexer import build_index as _build_index


class BiEncoderRetriever:
    """First-stage retriever: fast embedding + FAISS similarity search."""

    def __init__(self):
        print(f"  Loading bi-encoder: {BI_ENCODER_MODEL} on {DEVICE} …")
        self.model = SentenceTransformer(BI_ENCODER_MODEL, device=DEVICE)
        self.index: faiss.Index = None
        self.documents: list = []
        self.dimension: int = self.model.get_embedding_dimension()

    # ─── Encoding ─────────────────────────────────────────────────────────────

    def encode(self, texts: list, show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode a list of texts into L2-normalised embeddings.

        Returns:
            numpy array of shape (len(texts), dimension), dtype float32.
        """
        return self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,   # Cosine via dot product
        ).astype(np.float32)

    # ─── Index Building ───────────────────────────────────────────────────────

    def build_index(self, documents: list, index_type: str = "Flat") -> faiss.Index:
        """
        Encode `documents` and build a FAISS index in memory.

        Args:
            documents:   List of passage strings.
            index_type:  "Flat" or "IVF" — passed to indexer.build_index().
        """
        self.documents = documents
        print(f"  Encoding {len(documents):,} documents …")
        embeddings = self.encode(documents, show_progress_bar=True)
        self.index = _build_index(embeddings, index_type=index_type)
        return self.index

    def load_index(self, index: faiss.Index, documents: list) -> None:
        """
        Attach an externally built/loaded FAISS index and corpus.
        Use this when the index was persisted to disk via indexer.py.
        """
        self.index = index
        self.documents = documents

    # ─── Searching ────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 100) -> list:
        """
        Retrieve the top-k most similar documents for a query.

        Returns:
            List of (document_text, similarity_score) sorted by score desc.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        k = min(k, len(self.documents))
        query_embedding = self.encode([query])
        distances, indices = self.index.search(query_embedding, k)

        results = []
        seen = set()
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            if doc not in seen:
                seen.add(doc)
                results.append((doc, float(distances[0][i])))
        return results

    def search_timed(self, query: str, k: int = 100) -> tuple:
        """
        Same as search() but also returns latency in milliseconds.

        Returns:
            (results, latency_ms)
        """
        t0 = time.perf_counter()
        results = self.search(query, k=k)
        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms
