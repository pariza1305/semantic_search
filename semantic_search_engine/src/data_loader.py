"""
data_loader.py — Data ingestion utilities.

Functions:
    load_msmarco_passages   — stream MS MARCO passages via HuggingFace datasets
    load_msmarco_eval_data  — load dev queries + qrels for evaluation
    load_custom_texts       — parse plain .txt or .pdf files (for UI upload)
"""
import os
import csv
from typing import Dict, List, Set, Tuple

from tqdm import tqdm

from src.config import MAX_DOCS, MS_MARCO_SPLIT, TEST_QUERIES_FILE, QRELS_FILE, PAPER_PATH


# ─── Research Paper Loader (default demo corpus) ─────────────────────────────

def load_paper(paper_path: str = PAPER_PATH, chunk_size: int = 120) -> Tuple[List[str], str]:
    """
    Parse the 'Attention Is All You Need' PDF into overlapping text chunks.
    Used as the default demo corpus so users can query the actual paper.

    Args:
        paper_path: Path to the PDF (relative to project root or absolute).
        chunk_size: Approximate words per chunk (120 ≈ 2-3 sentences).

    Returns:
        (chunks, source_name)
    """
    import re

    # Resolve path relative to project root if not absolute
    if not os.path.isabs(paper_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        paper_path = os.path.join(project_root, paper_path)

    if not os.path.exists(paper_path):
        print(f"  ⚠️  Paper not found at '{paper_path}'. Falling back to demo documents.")
        return DEMO_DOCUMENTS, "Built-in demo corpus"

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("  ⚠️  PyMuPDF not installed. Run: pip install PyMuPDF")
        return DEMO_DOCUMENTS, "Built-in demo corpus"

    doc = fitz.open(paper_path)
    raw_pages = [page.get_text() for page in doc]
    doc.close()

    # Clean each page: remove page numbers, URLs, and excessive whitespace
    cleaned_pages = []
    for text in raw_pages:
        # Remove lines that are just numbers (page numbers)
        lines = [l for l in text.splitlines() if not re.match(r'^\s*\d+\s*$', l)]
        # Remove arxiv / conference header lines (short lines at top)
        text = "\n".join(lines)
        # Collapse multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        if text:
            cleaned_pages.append(text)

    # Split into overlapping chunks for better retrieval
    all_words = " ".join(cleaned_pages).split()
    chunks: List[str] = []
    stride = max(1, chunk_size - 20)   # 20-word overlap between chunks
    for i in range(0, len(all_words), stride):
        chunk = " ".join(all_words[i : i + chunk_size]).strip()
        if len(chunk.split()) >= 30:   # skip very short trailing chunks
            chunks.append(chunk)

    source_name = os.path.basename(paper_path)
    print(f"  ✅ Loaded {len(chunks):,} chunks from '{source_name}'")
    return chunks, source_name


# ─── MS MARCO Passages ────────────────────────────────────────────────────────

def load_msmarco_passages(max_docs: int = MAX_DOCS, split: str = MS_MARCO_SPLIT) -> List[str]:
    """
    Stream up to `max_docs` passages from the MS MARCO v2.1 dataset.

    The dataset is streamed (not downloaded in full), so RAM usage stays low.
    First call will cache the dataset locally via HuggingFace.

    Returns:
        List[str] of passage texts.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print(f"  Streaming MS MARCO (split={split}, max={max_docs:,}) …")
    dataset = load_dataset("ms_marco", "v2.1", split=split, streaming=True, trust_remote_code=True)

    passages: List[str] = []
    with tqdm(total=max_docs, desc="  Loading passages", unit="doc") as pbar:
        for example in dataset:
            if len(passages) >= max_docs:
                break
            # MS MARCO v2.1: each example has a 'passages' dict with list of texts
            raw_passages = example.get("passages", {})
            passage_texts = raw_passages.get("passage_text", [])
            for text in passage_texts:
                text = text.strip()
                if text and len(passages) < max_docs:
                    passages.append(text)
                    pbar.update(1)

    print(f"  ✅ Loaded {len(passages):,} passages from MS MARCO")
    return passages


# ─── MS MARCO Evaluation Data ─────────────────────────────────────────────────

def load_msmarco_eval_queries(path: str = TEST_QUERIES_FILE) -> Dict[str, str]:
    """
    Load MS MARCO dev queries from a TSV file.

    File format (tab-separated, no header):
        qid\tquery_text

    Returns:
        Dict mapping qid (str) → query_text (str).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Queries file not found: '{path}'\n"
            "Download from: https://msmarco.blob.core.windows.net/msmarcoranking/queries.dev.small.tsv.gz"
        )
    queries: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                qid, query_text = row[0].strip(), row[1].strip()
                queries[qid] = query_text
    print(f"  ✅ Loaded {len(queries):,} eval queries from {path}")
    return queries


def load_qrels(path: str = QRELS_FILE) -> Dict[str, Set[str]]:
    """
    Load MS MARCO qrels (relevance judgments) from a TSV file.

    File format (tab-separated):
        qid\t0\tpid\trelevance

    Returns:
        Dict mapping qid (str) → Set of relevant passage ids (str).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Qrels file not found: '{path}'\n"
            "Download from: https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.small.tsv.gz"
        )
    qrels: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 4:
                qid, _, pid, relevance = row[0], row[1], row[2], row[3]
                if int(relevance) > 0:
                    qrels.setdefault(qid, set()).add(pid)
    print(f"  ✅ Loaded qrels for {len(qrels):,} queries from {path}")
    return qrels


# ─── Custom Corpus (for UI Upload) ───────────────────────────────────────────

def load_custom_texts(file_path: str) -> Tuple[List[str], str]:
    """
    Parse a plain .txt or .pdf file and split into passages.

    Args:
        file_path: Absolute path to a .txt or .pdf file.

    Returns:
        (passages, source_name) where passages is a list of non-empty text chunks.
    """
    ext = os.path.splitext(file_path)[1].lower()
    source_name = os.path.basename(file_path)

    if ext == ".txt":
        passages = _load_txt(file_path)
    elif ext == ".pdf":
        passages = _load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Use .txt or .pdf")

    # Split into ~200-word chunks for better retrieval granularity
    passages = _chunk_text(passages, chunk_size=200)
    print(f"  ✅ Loaded {len(passages):,} passages from '{source_name}'")
    return passages, source_name


def _load_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [f.read()]


def _load_pdf(path: str) -> List[str]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Run: pip install PyMuPDF  (needed for PDF parsing)")

    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages


def _chunk_text(texts: List[str], chunk_size: int = 200) -> List[str]:
    """Split texts into chunks of approximately `chunk_size` words."""
    chunks: List[str] = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
    return chunks


# ─── Demo Corpus ──────────────────────────────────────────────────────────────

DEMO_DOCUMENTS = [
    "The Transformer model uses self-attention mechanisms to process sequences in parallel.",
    "Attention is All You Need is a seminal paper in deep learning that introduced the Transformer architecture.",
    "Semantic search understands the meaning behind queries, not just keywords or exact matches.",
    "FAISS is a library by Meta AI for efficient similarity search and clustering of dense vectors.",
    "Cross-encoders are more accurate but slower than bi-encoders for information retrieval tasks.",
    "Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
    "Positional encodings inject sequence order information into Transformers which have no recurrence.",
    "BERT is a Transformer-based model pre-trained on masked language modeling and next sentence prediction.",
    "Retrieval-augmented generation (RAG) combines dense retrieval with large language models for better answers.",
    "Dense retrieval uses neural networks to embed queries and documents into a shared vector space.",
    "Re-ranking is a two-stage approach: fast bi-encoder retrieval followed by precise cross-encoder scoring.",
    "The MS MARCO dataset is a standard benchmark for passage ranking and question answering evaluation.",
    "Sentence-BERT (SBERT) modifies BERT to produce semantically meaningful sentence embeddings efficiently.",
    "Cosine similarity measures the angle between two embedding vectors and is equivalent to dot product on normalized vectors.",
    "Hugging Face provides thousands of pre-trained Transformer models via the transformers and sentence-transformers libraries.",
    "The encoder-decoder Transformer uses cross-attention to allow the decoder to attend to encoder representations.",
    "FAISS IVF (Inverted File) index clusters vectors into groups for approximate nearest neighbor search at scale.",
    "Bi-encoders encode query and document independently, enabling pre-computation of document embeddings.",
    "Knowledge distillation trains a small student model to mimic the output distribution of a larger teacher model.",
    "The dot product between query and key vectors in attention computes compatibility scores before softmax normalization.",
]
