"""
evaluator.py — Information retrieval evaluation metrics and pipeline runner.

Metrics implemented:
    • Precision@K
    • Recall@K
    • MRR@K  (Mean Reciprocal Rank)
    • NDCG@K (Normalized Discounted Cumulative Gain)  ← key IR metric

Usage:
    from src.evaluator import Evaluator
    ev = Evaluator(qrels)
    metrics = ev.evaluate_queries(results_dict)
"""
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional
from tqdm import tqdm

from src.config import NDCG_K, MRR_K


class Evaluator:
    """Compute standard IR metrics against MS MARCO-style qrels."""

    def __init__(self, qrels: Dict[str, Set[str]]):
        """
        Args:
            qrels: Dict mapping query_id → set of relevant document/passage ids.
        """
        self.qrels = qrels

    # ─── Core Metrics ─────────────────────────────────────────────────────────

    def precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Fraction of retrieved@K that are relevant."""
        if k == 0:
            return 0.0
        hits = sum(1 for doc in retrieved[:k] if doc in relevant)
        return hits / k

    def recall_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Fraction of relevant documents found in top-K retrieved."""
        if not relevant:
            return 0.0
        hits = sum(1 for doc in retrieved[:k] if doc in relevant)
        return hits / len(relevant)

    def mrr_at_k(self, retrieved: List[str], relevant: Set[str], k: int = MRR_K) -> float:
        """
        Mean Reciprocal Rank @K.
        Returns 1/rank of the first relevant document within top-K, or 0.
        """
        for i, doc in enumerate(retrieved[:k]):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0

    def ndcg_at_k(self, retrieved: List[str], relevant: Set[str], k: int = NDCG_K) -> float:
        """
        Normalized Discounted Cumulative Gain @K.

        DCG  = Σ rel_i / log2(i+2)   for i in 0..K-1
        IDCG = DCG of perfect ranking (all relevant docs first)
        NDCG = DCG / IDCG

        Assumes binary relevance (relevant=1, not relevant=0).
        """
        if not relevant:
            return 0.0

        # DCG: score actual ranking
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            if doc in relevant:
                dcg += 1.0 / math.log2(i + 2)   # i+2 because log2(1)=0

        # IDCG: score ideal ranking (all relevant docs at top)
        n_rel = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))

        return dcg / idcg if idcg > 0 else 0.0

    # ─── Batch Evaluation ─────────────────────────────────────────────────────

    def evaluate_queries(
        self,
        results: Dict[str, List[str]],
        k_precision: int = 5,
        k_recall: int = 10,
        k_mrr: int = MRR_K,
        k_ndcg: int = NDCG_K,
    ) -> Dict[str, float]:
        """
        Evaluate a dict of {qid: [ranked doc ids]} against stored qrels.

        Returns:
            Dict of averaged metric values.
        """
        metrics: Dict[str, List[float]] = {
            f"precision@{k_precision}": [],
            f"recall@{k_recall}": [],
            f"mrr@{k_mrr}": [],
            f"ndcg@{k_ndcg}": [],
        }

        for qid, retrieved in results.items():
            relevant = self.qrels.get(qid, set())
            metrics[f"precision@{k_precision}"].append(
                self.precision_at_k(retrieved, relevant, k_precision)
            )
            metrics[f"recall@{k_recall}"].append(
                self.recall_at_k(retrieved, relevant, k_recall)
            )
            metrics[f"mrr@{k_mrr}"].append(
                self.mrr_at_k(retrieved, relevant, k_mrr)
            )
            metrics[f"ndcg@{k_ndcg}"].append(
                self.ndcg_at_k(retrieved, relevant, k_ndcg)
            )

        return {name: float(np.mean(vals)) for name, vals in metrics.items()}

    # ─── Pipeline Runner ──────────────────────────────────────────────────────

    def run_pipeline_evaluation(
        self,
        queries: Dict[str, str],
        retriever,
        reranker=None,
        k1: int = 100,
        k2: int = 10,
        max_queries: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run end-to-end evaluation comparing bi-encoder alone vs bi+cross-encoder.

        Args:
            queries:     Dict qid → query text.
            retriever:   BiEncoderRetriever instance (index must already be built).
            reranker:    CrossEncoderReranker (optional; if None, only stage-1 is evaluated).
            k1:          Number of candidates from bi-encoder.
            k2:          Final top-K after reranking.
            max_queries: Limit number of queries for faster runs (None = all).

        Returns:
            pd.DataFrame with one row per stage showing all metrics.
        """
        query_items = list(queries.items())
        if max_queries:
            query_items = query_items[:max_queries]

        bi_results:    Dict[str, List[str]] = {}
        cross_results: Dict[str, List[str]] = {}

        print(f"  Evaluating {len(query_items):,} queries …")
        for qid, query_text in tqdm(query_items, desc="  Evaluating"):
            # Stage 1: bi-encoder retrieval
            candidates = retriever.search(query_text, k=k1)
            bi_retrieved = [doc for doc, _ in candidates]
            bi_results[qid] = bi_retrieved

            # Stage 2: cross-encoder reranking (optional)
            if reranker is not None:
                reranked = reranker.rerank(query_text, candidates, top_k=k2)
                cross_results[qid] = [doc for doc, _ in reranked]

        # Compute metrics
        bi_metrics    = self.evaluate_queries(bi_results)
        rows = [{"Stage": "Bi-Encoder only", **bi_metrics}]

        if reranker is not None:
            cross_metrics = self.evaluate_queries(cross_results)
            rows.append({"Stage": "Bi-Encoder + Cross-Encoder", **cross_metrics})

        df = pd.DataFrame(rows).set_index("Stage")

        # Compute improvement delta
        if reranker is not None and len(df) == 2:
            delta = df.iloc[1] - df.iloc[0]
            delta.name = "Δ Improvement"
            df = pd.concat([df, delta.to_frame().T])

        return df
