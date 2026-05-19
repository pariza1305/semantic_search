# Configuration parameters for the semantic search system
import torch
import os

# ─── Model Configuration ────────────────────────────────────────────────────
BI_ENCODER_MODEL = "all-MiniLM-L6-v2"                       # Fast embedding model (384-dim)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Precision reranker

# ─── Index Configuration ─────────────────────────────────────────────────────
INDEX_TYPE = "IVF"          # "Flat" = exact brute-force | "IVF" = approximate (faster at scale)
IVF_NLIST = 100             # Number of IVF clusters (rule of thumb: sqrt(N))
FAISS_INDEX_PATH = os.path.join("data", "processed", "faiss.index")
DOCS_PATH        = os.path.join("data", "processed", "documents.pkl")

# ─── Retrieval Parameters ────────────────────────────────────────────────────
K1 = 100    # Candidates from first stage (bi-encoder)
K2 = 10     # Final results after reranking (cross-encoder)

# ─── Dataset Parameters ──────────────────────────────────────────────────────
MAX_DOCS = 50_000           # Number of MS MARCO passages to index
MS_MARCO_SPLIT = "train"    # Split to stream passages from
PAPER_PATH = "Attention is all you need paper.pdf"  # Default demo corpus

# ─── Evaluation Parameters ───────────────────────────────────────────────────
NDCG_K = 10                 # NDCG cut-off
MRR_K  = 10                 # MRR cut-off
TEST_QUERIES_FILE = os.path.join("data", "raw", "queries.dev.small.tsv")
QRELS_FILE        = os.path.join("data", "raw", "qrels.dev.small.tsv")

# ─── System Configuration ────────────────────────────────────────────────────
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"  # GPU on Mac M2
BATCH_SIZE = 32             # Encoding batch size