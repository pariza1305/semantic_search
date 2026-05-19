"""
run_pipeline.py — CLI entry point for the Semantic Search Engine.

Modes:
    demo    Run a quick demo on 20 built-in documents (no downloads needed).
    build   Stream MAX_DOCS MS MARCO passages, encode, and save index to disk.
    search  Load saved index and search for a query.
    eval    Run evaluation on MS MARCO dev queries and print metrics table.

Examples:
    python run_pipeline.py --mode demo
    python run_pipeline.py --mode build --max-docs 50000
    python run_pipeline.py --mode search --query "What is self-attention?"
    python run_pipeline.py --mode eval --max-queries 200
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config      import K1, K2, FAISS_INDEX_PATH, DOCS_PATH, INDEX_TYPE
from src.retriever   import BiEncoderRetriever
from src.reranker    import CrossEncoderReranker
from src.indexer     import save_index, load_index, save_documents, load_documents, index_exists
from src.data_loader import (
    load_msmarco_passages,
    load_msmarco_eval_queries,
    load_qrels,
    DEMO_DOCUMENTS,
)
from src.evaluator   import Evaluator


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _print_results(results: list, query: str) -> None:
    print(f"\n📄 Results for: \"{query}\"")
    print("─" * 70)
    for i, (doc, score) in enumerate(results, 1):
        snippet = doc[:120] + ("…" if len(doc) > 120 else "")
        print(f"  {i:>2}. [Score: {score:+.4f}]  {snippet}")
    print()


def _run_search(retriever: BiEncoderRetriever, reranker: CrossEncoderReranker, query: str) -> None:
    print(f"\n🔍  Query: \"{query}\"")

    # Stage 1
    t0 = time.perf_counter()
    candidates = retriever.search(query, k=K1)
    t1 = time.perf_counter()
    print(f"  Stage 1 ── Bi-Encoder retrieved {len(candidates):>3} candidates  "
          f"({(t1 - t0)*1000:.1f} ms)")

    # Stage 2
    results, lat = reranker.rerank_timed(query, candidates, top_k=K2)
    print(f"  Stage 2 ── Cross-Encoder reranked to {len(results):>3} results    "
          f"({lat:.1f} ms)")

    _print_results(results, query)


# ─── Modes ────────────────────────────────────────────────────────────────────

def mode_demo(args) -> None:
    print("\n🚀  DEMO MODE — using built-in corpus (no download required)\n")
    retriever = BiEncoderRetriever()
    retriever.build_index(DEMO_DOCUMENTS, index_type="Flat")

    reranker = CrossEncoderReranker()

    # ── Preset showcase queries ──────────────────────────────────────────
    if not args.skip_presets:
        preset_queries = [
            "What is a transformer model?",
            "How does FAISS handle large-scale vector search?",
            "What is the difference between bi-encoder and cross-encoder?",
        ]
        for query in preset_queries:
            _run_search(retriever, reranker, query)

    # ── Interactive loop: type your own queries ──────────────────────────
    print("\n" + "─" * 70)
    print("  💬  Interactive mode — type your own queries (Ctrl+C or 'quit' to exit)")
    print("─" * 70 + "\n")
    while True:
        try:
            query = input("  🔍  Your query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  👋  Exiting demo. Bye!\n")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("\n  👋  Exiting demo. Bye!\n")
            break
        _run_search(retriever, reranker, query)


def mode_build(args) -> None:
    max_docs   = args.max_docs
    index_type = args.index_type

    print(f"\n🔨  BUILD MODE — loading {max_docs:,} MS MARCO passages\n")
    passages = load_msmarco_passages(max_docs=max_docs)

    retriever = BiEncoderRetriever()
    retriever.build_index(passages, index_type=index_type)

    save_index(retriever.index)
    save_documents(passages)
    print(f"\n✅  Index and corpus saved. Run --mode search to query.\n")


def mode_search(args) -> None:
    if not index_exists():
        print("❌  No saved index found. Run --mode build first.")
        sys.exit(1)

    print("\n🔍  SEARCH MODE — loading index from disk\n")
    index     = load_index()
    documents = load_documents()

    retriever = BiEncoderRetriever()
    retriever.load_index(index, documents)

    reranker = CrossEncoderReranker()

    query = args.query or input("Enter query: ")
    _run_search(retriever, reranker, query)


def mode_eval(args) -> None:
    if not index_exists():
        print("❌  No saved index found. Run --mode build first.")
        sys.exit(1)

    print("\n📊  EVAL MODE — loading index and dev queries\n")
    index     = load_index()
    documents = load_documents()

    retriever = BiEncoderRetriever()
    retriever.load_index(index, documents)

    reranker  = CrossEncoderReranker()
    queries   = load_msmarco_eval_queries()
    qrels     = load_qrels()
    evaluator = Evaluator(qrels)

    df = evaluator.run_pipeline_evaluation(
        queries   = queries,
        retriever = retriever,
        reranker  = reranker,
        k1        = K1,
        k2        = K2,
        max_queries = args.max_queries,
    )

    print("\n" + "═" * 70)
    print("  EVALUATION RESULTS")
    print("═" * 70)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("═" * 70 + "\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-Stage Semantic Search Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "build", "search", "eval"],
        default="demo",
        help="Operation mode (default: demo)",
    )
    parser.add_argument("--query",        type=str,  default=None,  help="Query string for --mode search")
    parser.add_argument("--max-docs",     type=int,  default=50_000, help="Max MS MARCO passages to index (default: 50000)")
    parser.add_argument("--max-queries",  type=int,  default=None,  help="Limit eval queries for faster runs")
    parser.add_argument("--index-type",   type=str,  default=INDEX_TYPE, choices=["Flat", "IVF"],
                        help="FAISS index type for --mode build (default: IVF)")
    parser.add_argument("--skip-presets", action="store_true",
                        help="Skip the 3 preset demo queries and go straight to interactive mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dispatch = {
        "demo":   mode_demo,
        "build":  mode_build,
        "search": mode_search,
        "eval":   mode_eval,
    }
    dispatch[args.mode](args)
