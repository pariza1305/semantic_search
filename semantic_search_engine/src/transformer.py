import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Computes the scaled dot-product attention: softmax(Q * K^T / sqrt(d_k)) * V
    """
    def __init__(self, d_k: int):
        super().__init__()
        self.scale_factor = math.sqrt(d_k)

    def forward(self, q, k, v, mask=None):
        # q, k, v shape: (batch_size, num_heads, seq_len, d_k)
        
        # 1. Compute dot product between query and key
        # k.transpose(-2, -1) swaps the last two dimensions to allow matrix multiplication
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale_factor
        
        # 2. Apply mask (optional, e.g., for padding or preventing future lookahead)
        if mask is not None:
            # Mask should be broadcastable to the scores tensor
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 3. Apply softmax to get attention probabilities
        attention_weights = F.softmax(scores, dim=-1)
        
        # 4. Multiply by values
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention block from 'Attention Is All You Need'
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Final linear projection after concatenating heads
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Linear projections and reshape for multi-head
        # Output shape: (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        if mask is not None:
            # Expand mask for num_heads dimension
            mask = mask.unsqueeze(1)
            
        # 2. Apply Scaled Dot-Product Attention
        x, attn_weights = self.attention(q, k, v, mask)
        
        # 3. Concatenate heads
        # Shape: (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. Final linear projection
        output = self.W_o(x)
        
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    """
    The two-layer Feed-Forward network applied to each position separately and identically.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens in the sequence.
    """
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        
        # Create a matrix of shape (max_seq_len, d_model) initialized with zeros
        pe = torch.zeros(max_seq_len, d_model)
        
        # Create a column vector of shape (max_seq_len, 1) with positions (0, 1, 2, ...)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create a row vector with the divisors
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as a buffer so it's not a trainable parameter but is saved with the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input x
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class EncoderLayer(nn.Module):
    """
    A single block of the Transformer Encoder.
    Contains Multi-Head Attention, Add & Norm, Feed Forward, and Add & Norm.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Multi-Head Self-Attention
        # Query, Key, and Value are all the input x
        attn_output, _ = self.self_attn(x, x, x, mask)
        
        # 2. Add & Norm (Residual connection)
        x = self.layer_norm1(x + self.dropout1(attn_output))
        
        # 3. Position-wise Feed Forward
        ff_output = self.feed_forward(x)
        
        # 4. Add & Norm (Residual connection)
        x = self.layer_norm2(x + self.dropout2(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    """
    The full Transformer Encoder stack.
    """
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding layer
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Stack of N EncoderLayers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x is the input sequence of token IDs of shape (batch_size, seq_len)
        
        # 1. Embed tokens
        x = self.embedding(x)
        
        # 2. Add Positional Encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 3. Pass through N Encoder layers
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

class CustomTransformerEmbedder(nn.Module):
    """
    A wrapper class that uses our from-scratch TransformerEncoder to produce
    a single fixed-size vector for a document (using mean pooling), similar
    to what SentenceTransformers does.
    """
    def __init__(self, vocab_size=30522, d_model=384, num_heads=6, num_layers=6, d_ff=1536, max_seq_len=512):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len)
        
    def forward(self, input_ids, attention_mask=None):
        # Output shape: (batch_size, seq_len, d_model)
        token_embeddings = self.encoder(input_ids, attention_mask)
        
        # Mean Pooling - Take attention mask into account for correct averaging
        if attention_mask is not None:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = torch.mean(token_embeddings, dim=1)
            
        # L2 Normalize
        pooled_output = F.normalize(pooled_output, p=2, dim=1)
        return pooled_output

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True):
        """
        Mock implementation of the SentenceTransformer encode API.
        Since this model is completely untrained, it just generates random tokens
        to pass through the architecture to show it works mechanically.
        """
        self.eval()
        all_embeddings = []
        
        # Simple mock tokenization (just map characters to random IDs for demonstration)
        # In a real model, you'd use a tokenizer like BERT's WordPiece.
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Create fake token IDs just to demonstrate the forward pass works
                # Shape: (batch_size, sequence_length=32)
                batch_ids = torch.randint(0, 30000, (len(batch_texts), 32))
                batch_mask = torch.ones((len(batch_texts), 32))
                
                embeddings = self.forward(batch_ids, batch_mask)
                all_embeddings.append(embeddings)
                
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return all_embeddings.cpu().numpy()
        return all_embeddings
