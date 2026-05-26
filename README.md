# Two-Stage Semantic Search Engine

> **Bi-Encoder → Cross-Encoder retrieval** — understands the *meaning* of your query, not just keywords.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA%20%7C%20CPU-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-2.x-orange?style=flat)](https://sbert.net)
[![FAISS](https://img.shields.io/badge/FAISS-1.7%2B-blue?style=flat)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)

---

## 📌 Overview

This project implements a **production-style two-stage semantic search engine** — the same architecture used inside Google Search, Bing, and enterprise RAG pipelines.

The key insight from the *Attention Is All You Need* paper (Vaswani et al., 2017) is that **self-attention** allows a model to weigh the relevance of every token against every other token. This project uses that mechanism in two complementary ways:

| Stage | Model | Speed | Accuracy |
|-------|-------|-------|----------|
| **Bi-Encoder** (retriever) | `all-MiniLM-L6-v2` | ⚡ Very fast (FAISS) | Good |
| **Cross-Encoder** (reranker) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 🐢 Slower (full attention) | Excellent |

### Why Two Stages?

```
Query ──▶ [Bi-Encoder]  ──▶ top-100 candidates (fast, ~5ms)
               │
               ▼
         [FAISS Index]
               │
               ▼
        top-100 candidates ──▶ [Cross-Encoder]  ──▶ top-10 results (precise, ~300ms)
```

- **Stage 1** reduces millions of documents to a small candidate set instantly.
- **Stage 2** re-reads query + each candidate *together*, giving it full self-attention context — catching nuances that Stage 1 misses.

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone <your-repo-url>
cd semantic_search_engine

# Activate your virtual environment
source search-env/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Run the demo (no downloads needed)

```bash
python run_pipeline.py --mode demo
```

Sample output:
```
🔍  Query: "What is a transformer model?"
  Stage 1 ── Bi-Encoder retrieved 20 candidates  (4.2 ms)
  Stage 2 ── Cross-Encoder reranked to 10 results  (312 ms)

📄 Results:
   1. [Score: +8.02]  The Transformer model uses self-attention mechanisms…
   2. [Score: +3.56]  BERT is a Transformer-based model pre-trained on…
   3. [Score: +1.94]  Attention is All You Need introduced the Transformer…
```

### 3. Launch the web interface

```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🛠️ CLI Reference

```bash
# Quick demo on 20 built-in documents
python run_pipeline.py --mode demo

# Index 50,000 MS MARCO passages (downloads ~200 MB, takes ~2 min on M2)
python run_pipeline.py --mode build --max-docs 50000

# Search against the saved index
python run_pipeline.py --mode search --query "How does attention work?"

# Evaluate on MS MARCO dev queries (requires data/raw/queries + qrels files)
python run_pipeline.py --mode eval --max-queries 200

# Options
python run_pipeline.py --help
```

---

## 📁 Project Structure

```
semantic_search_engine/
├── app/
│   └── app.py               # Streamlit web interface
├── data/
│   ├── raw/                 # MS MARCO queries & qrels (download separately)
│   └── processed/           # FAISS index + corpus pickle (auto-generated)
├── notebooks/               # Evaluation plots & exploration
├── src/
│   ├── __init__.py
│   ├── config.py            # All hyperparameters & paths
│   ├── data_loader.py       # MS MARCO streaming + PDF/TXT loader
│   ├── evaluator.py         # Precision@K, Recall@K, MRR@K, NDCG@K
│   ├── indexer.py           # FAISS index building & persistence
│   ├── retriever.py         # Bi-Encoder + FAISS search
│   └── reranker.py          # Cross-Encoder reranking
├── tests/                   # Unit tests
├── requirements.txt
├── run_pipeline.py          # Main CLI entry point
└── README.md
```

---

## 📊 Evaluation Results

> Results below are on the **MS MARCO Dev Small** set (6,980 queries).
> Run `python run_pipeline.py --mode eval` after building the index to reproduce.

| Stage | MRR@10 | NDCG@10 | Precision@5 | Recall@10 |
|-------|--------|---------|-------------|-----------|
| Bi-Encoder only | — | — | — | — |
| Bi-Encoder + Cross-Encoder | — | — | — | — |
| **Δ Improvement** | — | — | — | — |

*Fill in after running evaluation.*

---

## ⚙️ Configuration

All parameters live in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BI_ENCODER_MODEL` | `all-MiniLM-L6-v2` | Stage-1 embedding model |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Stage-2 reranker |
| `K1` | `100` | Bi-encoder candidate count |
| `K2` | `10` | Final results after reranking |
| `MAX_DOCS` | `50,000` | MS MARCO passages to index |
| `INDEX_TYPE` | `IVF` | `Flat` (exact) or `IVF` (approximate) |
| `DEVICE` | `mps` (auto) | `mps` / `cuda` / `cpu` |
| `BATCH_SIZE` | `32` | Encoding batch size |

---

## 🔗 Architecture (from the Paper)

The Cross-Encoder uses the **full Transformer encoder stack** from *Attention Is All You Need*:

```
Input: [CLS] query [SEP] document [SEP]
          │
    ┌─────▼──────────────────────────┐
    │  Multi-Head Self-Attention ×6  │  ← query and doc tokens attend to each other
    │  Feed-Forward Network          │
    └─────┬──────────────────────────┘
          │
    [CLS] representation
          │
    Linear → Relevance Score (logit)
```

This full joint encoding is why the Cross-Encoder is much more accurate than the Bi-Encoder (which encodes query and document *independently*).

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt). Key dependencies:

- `torch >= 2.0` (MPS support on Apple Silicon)
- `sentence-transformers >= 2.2`
- `faiss-cpu >= 1.7.4`
- `streamlit >= 1.28`
- `datasets >= 2.14` (for MS MARCO streaming)
- `PyMuPDF >= 1.23` (for PDF parsing in the UI)
- `plotly >= 5.17` (for score chart)

---

## 🏗️ Roadmap

- [x] Bi-Encoder retriever with FAISS
- [x] Cross-Encoder reranker
- [x] IVF index for scale (50k+ docs)
- [x] Index persistence (save/load)
- [x] MS MARCO data loader
- [x] Full evaluation pipeline (MRR, NDCG, P@K, R@K)
- [x] Streamlit UI with file upload
- [x] Professional CLI with argparse
- [ ] Streamlit Cloud deployment
- [ ] Docker container
- [ ] Async query handling

---

## 📖 References

- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
- Johnson et al. (2019). *Billion-scale similarity search with GPUs* (FAISS). [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)
- Nguyen et al. (2016). *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset*. [arXiv:1611.09268](https://arxiv.org/abs/1611.09268)
