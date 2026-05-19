"""
app/app.py — Semantic Search Engine · Streamlit Web Interface
Clear, guided layout for first-time visitors.
"""
import sys, os, time, tempfile
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever   import BiEncoderRetriever
from src.reranker    import CrossEncoderReranker
from src.data_loader import DEMO_DOCUMENTS, load_custom_texts, load_paper
from src.config      import BI_ENCODER_MODEL, CROSS_ENCODER_MODEL, K1, K2, DEVICE

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0b0f1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #0d1225 !important;
    border-right: 1px solid rgba(99,179,237,0.1) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e0 !important; }

/* Search input */
.stTextInput > div > div > input {
    background: #111827 !important;
    border: 2px solid rgba(99,179,237,0.35) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-size: 1.05rem !important;
    padding: 0.8rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #63b3ed !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.18) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.5rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(99,179,237,0.25) !important;
}

/* Step cards */
.step-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.step-card {
    flex: 1;
    min-width: 180px;
    background: linear-gradient(135deg, #111827, #0f172a);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.step-icon  { font-size: 2rem; margin-bottom: 0.4rem; }
.step-num   { font-size: 0.68rem; letter-spacing: 0.1em; color: #63b3ed; text-transform: uppercase; font-weight: 600; margin-bottom: 0.2rem; }
.step-title { font-size: 0.9rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.3rem; }
.step-desc  { font-size: 0.78rem; color: #718096; line-height: 1.5; }
.step-arrow { font-size: 1.5rem; color: #4a5568; display: flex; align-items: center; padding-top: 1.5rem; }

/* Result card */
.result-card {
    background: #111827;
    border: 1px solid rgba(99,179,237,0.12);
    border-left: 4px solid;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    animation: fadeUp 0.3s ease forwards;
    opacity: 0;
}
.result-card:hover { border-color: rgba(99,179,237,0.35) !important; }
@keyframes fadeUp {
    from { opacity:0; transform:translateY(8px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem; }
.rank-badge {
    background: rgba(99,179,237,0.12);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.score-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #a0aec0;
}
.relevance-label {
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 10px;
    margin-left: auto;
}
.result-text { font-size: 0.88rem; color: #e2e8f0; line-height: 1.65; }
.score-bar-bg   { background: rgba(255,255,255,0.06); border-radius: 4px; height: 3px; margin-top: 0.7rem; }
.score-bar-fill { height: 100%; border-radius: 4px; }

/* Timing */
.timing-row { display: flex; gap: 0.7rem; margin: 0.8rem 0 1.2rem; flex-wrap: wrap; }
.timing-box {
    background: rgba(99,179,237,0.06);
    border: 1px solid rgba(99,179,237,0.14);
    border-radius: 10px;
    padding: 0.55rem 1rem;
    text-align: center; min-width: 110px;
}
.timing-val { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 600; color: #63b3ed; }
.timing-lbl { font-size: 0.68rem; color: #718096; text-transform: uppercase; letter-spacing: 0.06em; }

/* Section heading */
.section-heading {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* Chip buttons */
.chip-wrap > div > button {
    background: rgba(99,179,237,0.07) !important;
    border: 1px solid rgba(99,179,237,0.22) !important;
    border-radius: 20px !important;
    color: #63b3ed !important;
    font-size: 0.8rem !important;
    padding: 0.3rem 0.9rem !important;
    font-weight: 400 !important;
}
.chip-wrap > div > button:hover {
    background: rgba(99,179,237,0.15) !important;
    transform: none !important;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 2.5rem 1rem;
    color: #4a5568;
}

hr { border-color: rgba(99,179,237,0.08) !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
for k, v in {
    "query": "", "results": None,
    "bi_lat": None, "cross_lat": None,
    "uploaded_docs": [], "upload_name": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Models (cached) ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_models():
    r  = BiEncoderRetriever()
    rr = CrossEncoderReranker()
    return r, rr

@st.cache_resource(show_spinner=False)
def _load_paper_corpus():
    """Load and cache the research paper corpus once at startup."""
    chunks, name = load_paper()
    return chunks, name

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Semantic Search")
    st.markdown("---")

    # ── What is this? ──
    st.markdown("### What does this app do?")
    st.markdown("""
<div style='font-size:0.82rem; color:#a0aec0; line-height:1.8;'>
You type a <b style='color:#e2e8f0;'>question or topic</b>, and this app
finds the most relevant passages from the document corpus — ranked by meaning,
not just word overlap.<br><br>
<b style='color:#e2e8f0;'>Example:</b> Ask <i>"How does attention work?"</i>
and it returns passages like <i>"Multi-head attention allows the model to focus
on different parts of the input…"</i> — even if none of those words appear in your query.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Under the Hood")
    st.markdown("""
<div style='font-size:0.8rem; color:#a0aec0; line-height:1.9;'>
<b style='color:#63b3ed;'>Step 1 — Bi-Encoder</b><br>
Converts your query + all documents into numerical vectors. Fast, but approximate.<br><br>
<b style='color:#9f7aea;'>Step 2 — Cross-Encoder</b><br>
Re-reads query + top candidates <i>together</i> for a precise relevance score. Slower, but very accurate.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📂 Document Corpus")
    corpus_choice = st.radio(
        "Which documents do you want to search?",
        ["📖 Built-in demo (20 AI/ML passages)", "📄 Upload my own file (PDF or TXT)"],
        index=0,
    )

    uploaded_docs = []
    upload_name   = ""
    if "Upload" in corpus_choice:
        st.markdown("<div style='font-size:0.78rem; color:#718096;'>Drag and drop a file or click Browse</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["pdf", "txt"], label_visibility="collapsed")
        if uploaded_file:
            with st.spinner("Reading & indexing your file …"):
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    passages, src = load_custom_texts(tmp_path)
                    st.session_state.uploaded_docs = passages
                    st.session_state.upload_name   = src
                    st.success(f"✅ Indexed {len(passages):,} passages from '{src}'")
                except Exception as e:
                    st.error(f"Could not read file: {e}")
                finally:
                    os.unlink(tmp_path)

    st.markdown("---")
    st.markdown("""
<div style='font-size:0.72rem; color:#4a5568; line-height:1.6;'>
Based on the paper<br>
<i>Attention Is All You Need</i><br>
Vaswani et al., NeurIPS 2017
</div>
""", unsafe_allow_html=True)

# ─── Main Page ────────────────────────────────────────────────────────────────

# Title
st.markdown("""
<h1 style='font-size:1.9rem; font-weight:700; margin:0 0 0.2rem;
background:linear-gradient(90deg,#63b3ed,#9f7aea);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>
⚡ Semantic Search Engine
</h1>
<p style='color:#718096; font-size:0.92rem; margin:0 0 1.5rem;'>
Finds documents by <b style="color:#a0aec0;">meaning</b>, not just keyword matching
</p>
""", unsafe_allow_html=True)

# ── How it works (3 steps) ───────────────────────────────────────────────────
st.markdown('<div class="section-heading">📖 How it works</div>', unsafe_allow_html=True)
st.markdown("""
<div class="step-row">
  <div class="step-card">
    <div class="step-icon">✍️</div>
    <div class="step-num">Step 1</div>
    <div class="step-title">Type your question</div>
    <div class="step-desc">Ask anything in plain English. No special syntax needed.</div>
  </div>
  <div class="step-arrow">→</div>
  <div class="step-card">
    <div class="step-icon">🧠</div>
    <div class="step-num">Step 2 — Bi-Encoder</div>
    <div class="step-title">Fast retrieval</div>
    <div class="step-desc">Converts query + documents into vectors. Finds top-100 candidates in milliseconds using FAISS.</div>
  </div>
  <div class="step-arrow">→</div>
  <div class="step-card">
    <div class="step-icon">🎯</div>
    <div class="step-num">Step 3 — Cross-Encoder</div>
    <div class="step-title">Precise re-ranking</div>
    <div class="step-desc">Re-reads your query with each candidate together to find the truly most relevant results.</div>
  </div>
  <div class="step-arrow">→</div>
  <div class="step-card">
    <div class="step-icon">📄</div>
    <div class="step-num">Result</div>
    <div class="step-title">Ranked results</div>
    <div class="step-desc">Top results sorted by relevance score, with timing breakdown.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Load models & build index ────────────────────────────────────────────────
with st.spinner("Loading AI models … (first run only, ~10 sec)"):
    retriever, reranker = _load_models()

use_upload   = "Upload" in corpus_choice and bool(st.session_state.uploaded_docs)

if use_upload:
    active_docs  = st.session_state.uploaded_docs
    corpus_label = st.session_state.upload_name
else:
    # Default: load the research paper
    with st.spinner("Loading 'Attention Is All You Need' paper …"):
        active_docs, corpus_label = _load_paper_corpus()

corpus_key = f"idx_{hash(tuple(active_docs[:5]))}"
if st.session_state.get("_corpus_key") != corpus_key:
    with st.spinner(f"Building search index for {len(active_docs):,} passages …"):
        retriever.build_index(active_docs, index_type="Flat")
    st.session_state["_corpus_key"] = corpus_key

# ── Corpus info bar ──────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("📚 Documents indexed", f"{len(active_docs):,}")
col2.metric("🗂️ Current corpus",    corpus_label[:30] + ("…" if len(corpus_label) > 30 else ""))
col3.metric("💻 Running on",        DEVICE.upper())

st.markdown("---")

# ── Search ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">🔍 Search</div>', unsafe_allow_html=True)

# Example chips — paper-specific questions
EXAMPLES = [
    "What is the attention mechanism?",
    "Why did they remove recurrence?",
    "What is multi-head attention?",
    "How does positional encoding work?",
    "What are the results on WMT translation?",
]
st.markdown("**Try an example query:**")
cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    with cols[i]:
        st.markdown('<div class="chip-wrap">', unsafe_allow_html=True)
        if st.button(ex, key=f"chip_{i}"):
            st.session_state.query = ex
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

# Query input
query_val = st.text_input(
    "Or type your own question below:",
    value=st.session_state.query,
    placeholder="e.g. How does attention mechanism work in neural networks?",
)

col_btn, col_k = st.columns([5, 1])
with col_btn:
    search_clicked = st.button("🚀  Search", use_container_width=True)
with col_k:
    top_k = st.slider("Results", 1, min(15, len(active_docs)), min(5, len(active_docs)))

# ── Execute search ────────────────────────────────────────────────────────────
if search_clicked:
    if not query_val.strip():
        st.warning("⚠️ Please type a query first.")
    else:
        with st.spinner("Stage 1 — finding candidates …"):
            candidates, bi_lat = retriever.search_timed(query_val.strip(), k=min(K1, len(active_docs)))
        with st.spinner("Stage 2 — re-ranking for accuracy …"):
            results, cross_lat = reranker.rerank_timed(query_val.strip(), candidates, top_k=top_k)
        st.session_state.results   = results
        st.session_state.bi_lat    = bi_lat
        st.session_state.cross_lat = cross_lat
        st.session_state.query     = query_val.strip()

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    results   = st.session_state.results
    bi_lat    = st.session_state.bi_lat
    cross_lat = st.session_state.cross_lat
    total_lat = (bi_lat or 0) + (cross_lat or 0)

    st.markdown("---")
    st.markdown(f"### Results for: *\"{st.session_state.query}\"*")

    # Timing explanation
    st.markdown(f"""
<div class="timing-row">
  <div class="timing-box">
    <div class="timing-val">{bi_lat:.0f} ms</div>
    <div class="timing-lbl">Step 2 · Bi-Encoder</div>
  </div>
  <div class="timing-box">
    <div class="timing-val">{cross_lat:.0f} ms</div>
    <div class="timing-lbl">Step 3 · Cross-Encoder</div>
  </div>
  <div class="timing-box">
    <div class="timing-val">{total_lat:.0f} ms</div>
    <div class="timing-lbl">Total time</div>
  </div>
  <div class="timing-box">
    <div class="timing-val">{len(results)}</div>
    <div class="timing-lbl">Results shown</div>
  </div>
</div>
<div style='font-size:0.78rem;color:#4a5568;margin-bottom:1rem;'>
  💡 <b>How to read scores:</b> Positive scores = relevant match. The higher the score, the better the match.
  Negative scores mean the document is not a good match for your query.
</div>
""", unsafe_allow_html=True)

    # Tabs
    tab_cards, tab_chart = st.tabs(["📄 Results",  "📊 Score Chart"])

    scores    = [s for _, s in results]
    max_score = max(scores) if scores else 1
    min_score = min(scores) if scores else 0
    rng       = max(max_score - min_score, 0.01)

    COLORS = ["#f6ad55","#63b3ed","#9f7aea","#68d391","#fc8181",
              "#b794f4","#4fd1c5","#f687b3","#76e4f7","#fbd38d",
              "#a0aec0","#a0aec0","#a0aec0","#a0aec0","#a0aec0"]

    with tab_cards:
        for i, (doc, score) in enumerate(results):
            color   = COLORS[i % len(COLORS)]
            bar_pct = int(((score - min_score) / rng) * 100)
            delay   = i * 0.04

            # Relevance label
            if score > 5:
                rel_color, rel_label = "#68d391", "✅ Strong match"
            elif score > 0:
                rel_color, rel_label = "#f6ad55", "🟡 Partial match"
            else:
                rel_color, rel_label = "#fc8181", "❌ Weak match"

            st.markdown(f"""
<div class="result-card" style="border-left-color:{color}; animation-delay:{delay:.2f}s;">
  <div class="result-header">
    <span class="rank-badge" style="color:{color};">#{i+1}</span>
    <span class="score-badge">Score: {score:+.2f}</span>
    <span class="relevance-label" style="background:{rel_color}22;color:{rel_color};">{rel_label}</span>
  </div>
  <div class="result-text">{doc}</div>
  <div class="score-bar-bg">
    <div class="score-bar-fill" style="width:{bar_pct}%;background:{color};opacity:0.7;"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    with tab_chart:
        try:
            import plotly.graph_objects as go
            labels = [f"#{i+1} {doc[:50]}…" if len(doc)>50 else f"#{i+1} {doc}" for i,(doc,_) in enumerate(results)]
            fig = go.Figure(go.Bar(
                x=scores, y=labels, orientation="h",
                marker=dict(color=COLORS[:len(results)], opacity=0.85),
                hovertemplate="<b>%{y}</b><br>Score: %{x:.2f}<extra></extra>",
            ))
            fig.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_dash="dash")
            fig.add_annotation(x=0, y=-0.5, text="0 = relevance threshold", showarrow=False,
                               font=dict(color="#718096", size=10), yanchor="top")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#cbd5e0", size=12),
                margin=dict(l=10, r=20, t=30, b=30),
                xaxis=dict(title="Relevance Score (positive = good match)", gridcolor="rgba(99,179,237,0.1)", color="#718096"),
                yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)", color="#cbd5e0"),
                height=max(300, len(results)*45),
                title=dict(text="Cross-Encoder Relevance Scores", font=dict(color="#e2e8f0", size=13)),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Run `pip install plotly` to see the chart.")

else:
    st.markdown("---")
    st.markdown("#### What you are searching through")
    st.markdown(
        f"The corpus is **{corpus_label}** split into **{len(active_docs):,} passages**. "
        "When you search, the engine finds which passages best answer your question."
    )
    st.markdown("**Sample passages from the corpus (showing first 6):**")
    preview_docs = active_docs[:6]
    col_a, col_b = st.columns(2)
    for i, doc in enumerate(preview_docs):
        col = col_a if i % 2 == 0 else col_b
        with col:
            snippet = doc[:250] + ("..." if len(doc) > 250 else "")
            st.info(f"**Passage {i+1}:** {snippet}")
    st.caption(f"Showing 6 of {len(active_docs):,} passages. Type a question above to search all of them.")
