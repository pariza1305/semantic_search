"""
reranker.py — Cross-Encoder reranker (second retrieval stage).

Takes top-K candidates from the bi-encoder and scores each (query, doc) pair
jointly — giving full self-attention between query and document tokens.
This is slower but far more accurate than bi-encoder similarity.
"""
import time
import numpy as np
from sentence_transformers import CrossEncoder

from src.config import CROSS_ENCODER_MODEL, DEVICE, K1, K2


class CrossEncoderReranker:
    """Second-stage reranker using a cross-encoder for precise relevance scoring."""

    def __init__(self):
        """Load the cross-encoder model."""
        print(f"  Loading cross-encoder: {CROSS_ENCODER_MODEL} on {DEVICE} …")
        self.model = CrossEncoder(CROSS_ENCODER_MODEL, device=DEVICE)

    # ─── Reranking ────────────────────────────────────────────────────────────

    def rerank(self, query: str, candidates: list, top_k: int = K2) -> list:
        """
        Rerank candidate documents using the cross-encoder.

        Args:
            query:      User search query string.
            candidates: List of (document_text, bi_encoder_score) from stage 1.
            top_k:      Number of documents to return after reranking.

        Returns:
            List of (document_text, relevance_score) sorted by score desc.
        """
        if not candidates:
            return []

        # Take at most K1 candidates to cap cross-encoder latency
        documents = [doc for doc, _ in candidates[:K1]]

        # Build (query, doc) pairs and run cross-encoder
        pairs  = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Sort by relevance score descending
        reranked = sorted(zip(documents, scores.tolist()), key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def rerank_timed(self, query: str, candidates: list, top_k: int = K2) -> tuple:
        """
        Same as rerank() but also returns latency in milliseconds.

        Returns:
            (reranked_results, latency_ms)
        """
        t0 = time.perf_counter()
        results = self.rerank(query, candidates, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms