import nbformat as nbf

nb = nbf.v4.new_notebook()
text_cells = [
    "# Transformer Architecture From Scratch\n\nThis notebook demonstrates the core components of the Transformer architecture from the *Attention Is All You Need* (2017) paper.",
    "## 1. Setup and Imports",
    "## 2. Scaled Dot-Product Attention\n\nThe core of the Transformer. The formula is:\n\n$$Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V$$\n\nLet's test our from-scratch implementation.",
    "## 3. The Full Transformer Encoder\n\nNow we put it all together: Multi-Head Attention, Position-wise Feed Forward, Positional Encoding, and Residual Connections."
]

code_cells = [
    "import sys\nimport os\nimport torch\n\n# Add the src directory to the path so we can import our custom module\nsys.path.append(os.path.abspath(\"..\"))\nfrom src.transformer import ScaledDotProductAttention, MultiHeadAttention, TransformerEncoder",
    "d_k = 64\nseq_len = 10\nbatch_size = 2\nnum_heads = 1\n\n# Create dummy queries, keys, and values\nq = torch.randn(batch_size, num_heads, seq_len, d_k)\nk = torch.randn(batch_size, num_heads, seq_len, d_k)\nv = torch.randn(batch_size, num_heads, seq_len, d_k)\n\nattention_layer = ScaledDotProductAttention(d_k)\noutput, weights = attention_layer(q, k, v)\n\nprint(f\"Input shape: {q.shape}\")\nprint(f\"Output shape: {output.shape}\")\nprint(f\"Attention weights shape: {weights.shape}\")\n\n# Verify that attention weights sum to 1.0 along the sequence dimension\nprint(f\"\\nSum of attention weights (should be ~1.0): {weights[0, 0, 0, :].sum().item():.4f}\")",
    "d_model = 384\nvocab_size = 30000\nmax_seq_len = 512\n\n# Initialize our custom Encoder\nencoder = TransformerEncoder(\n    vocab_size=vocab_size,\n    d_model=d_model,\n    num_heads=6,\n    num_layers=6,\n    d_ff=1536,\n    max_seq_len=max_seq_len\n)\n\n# Create some dummy input token IDs (e.g., representing a sentence)\ndummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))\n\nprint(f\"Input tokens shape: {dummy_input_ids.shape}\")\n\n# Pass through the encoder\nencoder_output = encoder(dummy_input_ids)\n\nprint(f\"Encoder output shape: {encoder_output.shape}\")\nprint(\"Notice the output shape is (batch_size, seq_len, d_model), perfectly matching the original paper!\")"
]

nb.cells = [
    nbf.v4.new_markdown_cell(text_cells[0]),
    nbf.v4.new_markdown_cell(text_cells[1]),
    nbf.v4.new_code_cell(code_cells[0]),
    nbf.v4.new_markdown_cell(text_cells[2]),
    nbf.v4.new_code_cell(code_cells[1]),
    nbf.v4.new_markdown_cell(text_cells[3]),
    nbf.v4.new_code_cell(code_cells[2]),
]

with open("notebooks/Transformer_From_Scratch.ipynb", "w") as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
