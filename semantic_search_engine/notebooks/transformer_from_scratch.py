"""
Transformer From Scratch — "Attention Is All You Need" (Vaswani et al., 2017)
=============================================================================
This script implements the core Transformer components from the paper
using only PyTorch (no HuggingFace). Each section maps directly to a
specific equation or figure in the paper.

Run:
    python notebooks/transformer_from_scratch.py

Or copy cells into a Jupyter notebook for interactive use.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.facecolor'] = '#0d1117'
matplotlib.rcParams['axes.facecolor']   = '#161b22'
matplotlib.rcParams['text.color']       = '#e6edf3'
matplotlib.rcParams['axes.labelcolor']  = '#e6edf3'
matplotlib.rcParams['xtick.color']      = '#8b949e'
matplotlib.rcParams['ytick.color']      = '#8b949e'

print("=" * 65)
print("  Transformer From Scratch — Attention Is All You Need")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Scaled Dot-Product Attention (Paper §3.2.1, Equation 1)
# ─────────────────────────────────────────────────────────────────────────────
# Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
#
# Q = Queries  (what am I looking for?)
# K = Keys     (what do I have?)
# V = Values   (what do I return?)
# d_k = key dimension (we divide by √d_k to prevent vanishing gradients
#        in softmax when d_k is large)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Paper Eq. 1: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        Q: (batch, heads, seq_len, d_k)
        K: (batch, heads, seq_len, d_k)
        V: (batch, heads, seq_len, d_v)
        mask: optional boolean mask (True = ignore)
    Returns:
        output: (batch, heads, seq_len, d_v)
        weights: attention weights for visualization
    """
    d_k = Q.size(-1)

    # Step 1: Dot product of Q and K (compatibility scores)
    scores = torch.matmul(Q, K.transpose(-2, -1))   # (batch, heads, seq, seq)

    # Step 2: Scale by 1/√d_k  ← crucial! Without this, softmax saturates
    scores = scores / math.sqrt(d_k)

    # Step 3: Optional mask (for decoder causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Softmax → attention weights (rows sum to 1)
    weights = F.softmax(scores, dim=-1)

    # Step 5: Weighted sum of values
    output = torch.matmul(weights, V)

    return output, weights


# Quick test
print("\n── Section 1: Scaled Dot-Product Attention ──")
batch, seq_len, d_k = 1, 5, 64
Q = torch.randn(batch, 1, seq_len, d_k)
K = torch.randn(batch, 1, seq_len, d_k)
V = torch.randn(batch, 1, seq_len, d_k)
out, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"  Input  Q shape : {Q.shape}")
print(f"  Output shape   : {out.shape}")
print(f"  Attn weights   : {attn_weights.shape}  (rows sum to 1: {attn_weights[0,0,0].sum().item():.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Multi-Head Attention (Paper §3.2.2, Equation 2)
# ─────────────────────────────────────────────────────────────────────────────
# Instead of one big attention, run h smaller attention "heads" in parallel,
# then concatenate. Each head can focus on different parts of the sequence.
#
# MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O
# where head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)

class MultiHeadAttention(nn.Module):
    """Paper §3.2.2 Multi-Head Attention with h=8 heads."""

    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads   # dimension per head

        # Learned linear projections for Q, K, V (one per head, batched)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        # Output projection that combines all heads
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        """Reshape (batch, seq, d_model) → (batch, heads, seq, d_k)."""
        batch, seq, d_model = x.size()
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        batch = Q.size(0)

        # Project and split into heads
        Q = self.split_heads(self.W_Q(Q))  # (batch, heads, seq, d_k)
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        # Run attention on all heads simultaneously
        attn_out, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Merge heads back: (batch, heads, seq, d_k) → (batch, seq, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)

        # Final linear projection
        return self.W_O(attn_out)


print("\n── Section 2: Multi-Head Attention ──")
d_model, num_heads, seq_len = 512, 8, 10
mha = MultiHeadAttention(d_model, num_heads)
x   = torch.randn(2, seq_len, d_model)
out = mha(x, x, x)   # Self-attention: Q=K=V=x
print(f"  Input  shape : {x.shape}   (batch=2, seq=10, d_model=512)")
print(f"  Output shape : {out.shape}")
print(f"  Heads={num_heads}, d_k per head={d_model//num_heads}")
print(f"  Attention weights shape: {mha.attn_weights.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Positional Encoding (Paper §3.5, Equation 3&4)
# ─────────────────────────────────────────────────────────────────────────────
# Transformers have NO recurrence — they process all tokens in parallel.
# So we inject position info using sin/cos waves of different frequencies.
#
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

class PositionalEncoding(nn.Module):
    """Paper §3.5 sinusoidal positional encoding."""

    def __init__(self, d_model=512, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the PE matrix once (not learned)
        pe  = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()
        # Frequencies: 1/10000^(2i/d_model)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)   # even dims → sin
        pe[:, 1::2] = torch.cos(pos * div)   # odd  dims → cos

        pe = pe.unsqueeze(0)                  # (1, max_seq, d_model)
        self.register_buffer('pe', pe)        # not a trainable parameter

    def forward(self, x):
        # Add positional signal to token embeddings
        return self.dropout(x + self.pe[:, :x.size(1)])


print("\n── Section 3: Positional Encoding ──")
pe_module = PositionalEncoding(d_model=512)
x_embed   = torch.randn(2, 20, 512)
x_with_pe = pe_module(x_embed)
print(f"  Embedding shape    : {x_embed.shape}")
print(f"  After PE shape     : {x_with_pe.shape}  (same — we just ADD positions)")
print(f"  PE is NOT learned  : {not any(p.requires_grad for p in pe_module.parameters())}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Feed-Forward Network (Paper §3.3)
# ─────────────────────────────────────────────────────────────────────────────
# FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2
# Applied position-wise (independently to each token).
# d_ff = 2048 in the paper (4× d_model).

class FeedForward(nn.Module):
    """Paper §3.3 position-wise FFN."""

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Encoder Layer (Paper §3.1, Figure 1 left)
# ─────────────────────────────────────────────────────────────────────────────
# Each encoder layer:
#   x = LayerNorm(x + MultiHeadAttention(x, x, x))   ← residual + norm
#   x = LayerNorm(x + FeedForward(x))                 ← residual + norm

class EncoderLayer(nn.Module):
    """One Transformer encoder layer (Paper Figure 1, left side)."""

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads)
        self.ff         = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Sub-layer 1: Self-attention with residual connection
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Sub-layer 2: Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Full Mini Transformer Encoder (N stacked layers)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """Stack of N encoder layers — the encoder from the paper."""

    def __init__(self, vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_seq=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_seq, dropout)
        self.layers    = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids, mask=None):
        # Token embedding + positional encoding
        x = self.embedding(token_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_enc(x)
        # Pass through N encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


print("\n── Section 5 & 6: Full Transformer Encoder ──")
encoder = TransformerEncoder(
    vocab_size  = 1000,
    d_model     = 128,   # smaller for demo
    num_heads   = 4,
    num_layers  = 2,
    d_ff        = 512,
)
token_ids = torch.randint(0, 1000, (2, 15))   # batch=2, seq_len=15
out = encoder(token_ids)
print(f"  Input token_ids : {token_ids.shape}  (batch=2, seq_len=15)")
print(f"  Encoder output  : {out.shape}        (batch=2, seq_len=15, d_model=128)")
n_params = sum(p.numel() for p in encoder.parameters())
print(f"  Total parameters: {n_params:,}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Visualize Attention Weights
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Section 7: Attention Weight Visualization ──")
words  = ["The", "transformer", "uses", "self", "-", "attention", "to", "understand", "meaning", "."]
tokens = torch.arange(len(words)).unsqueeze(0)   # fake token ids

small_encoder = TransformerEncoder(vocab_size=100, d_model=64, num_heads=2, num_layers=1, d_ff=128)
_ = small_encoder(tokens)

# Get attention weights from the first layer, head 0
attn = small_encoder.layers[0].self_attn.attn_weights
attn_head0 = attn[0, 0].detach().numpy()   # (seq, seq)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Attention Is All You Need — Multi-Head Attention Visualization",
             color='#e6edf3', fontsize=13, fontweight='bold')

# Head 0
im0 = axes[0].imshow(attn_head0, cmap='Blues', aspect='auto')
axes[0].set_title("Head 0 attention weights", color='#e6edf3')
axes[0].set_xticks(range(len(words))); axes[0].set_xticklabels(words, rotation=45, ha='right', fontsize=9)
axes[0].set_yticks(range(len(words))); axes[0].set_yticklabels(words, fontsize=9)
axes[0].set_xlabel("Attending TO (Keys)", color='#8b949e')
axes[0].set_ylabel("Query token", color='#8b949e')
plt.colorbar(im0, ax=axes[0])

# Head 1
attn_head1 = attn[0, 1].detach().numpy()
im1 = axes[1].imshow(attn_head1, cmap='Purples', aspect='auto')
axes[1].set_title("Head 1 attention weights", color='#e6edf3')
axes[1].set_xticks(range(len(words))); axes[1].set_xticklabels(words, rotation=45, ha='right', fontsize=9)
axes[1].set_yticks(range(len(words))); axes[1].set_yticklabels(words, fontsize=9)
axes[1].set_xlabel("Attending TO (Keys)", color='#8b949e')
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig("notebooks/attention_visualization.png", dpi=120, bbox_inches='tight')
plt.show()
print("  ✅ Attention heatmap saved → notebooks/attention_visualization.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Connection to This Project
# ─────────────────────────────────────────────────────────────────────────────

print("""
── Section 8: How This Connects to Your Semantic Search Engine ──

  The models you use (all-MiniLM-L6-v2 and cross-encoder/ms-marco-MiniLM-L-6-v2)
  are both pre-trained Transformer encoders — EXACTLY the architecture above,
  but with 6 layers and d_model=384, trained on hundreds of millions of sentences.

  Bi-Encoder (Stage 1):
    query  → TransformerEncoder → 384-dim vector ─┐
                                                    ├─ cosine similarity → FAISS
    doc    → TransformerEncoder → 384-dim vector ─┘

  Cross-Encoder (Stage 2):
    [CLS] query [SEP] doc [SEP] → TransformerEncoder → linear → relevance score
    ↑ query and doc tokens attend to EACH OTHER (full attention)
    ↑ this is why cross-encoders are more accurate

  Key equations from the paper used in production:
    Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V  ← in every forward pass
    PE(pos,2i)       = sin(pos/10000^(2i/d))  ← handles sequence order
    FFN(x)           = max(0, xW₁+b₁)W₂+b₂  ← in every encoder layer
""")

print("=" * 65)
print("  ✅  All components implemented from scratch successfully!")
print("=" * 65)
