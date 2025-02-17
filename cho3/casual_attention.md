# Causal Self-Attention with PyTorch

## Overview
This repository contains an implementation of **Causal Self-Attention** using PyTorch. This type of attention mechanism ensures that each token in a sequence only attends to itself and previous tokens, making it suitable for autoregressive models like GPT.

## Features
- Implements causal self-attention with multi-head attention.
- Uses a precomputed lower-triangular mask to enforce causality.
- Supports dropout for better generalization.
- Efficient computation using matrix operations.

## Installation
Ensure you have PyTorch installed. You can install it via:
```bash
pip install torch
```

## Implementation from Scratch
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.embed_dim % self.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, 512, 512)))  # Precompute max-length mask

    def forward(self, x):
        B, T, C = x.shape  # Batch, Time, Channels
        qkv = self.qkv_proj(x).chunk(3, dim=-1)  # Split into query, key, value
        
        q, k, v = [t.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]
        
        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = attn_weights.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # Apply causal mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(attn_output)

# Example usage:
batch_size, seq_len, embed_dim, num_heads = 2, 10, 64, 8
x = torch.randn(batch_size, seq_len, embed_dim)
attn = CausalSelfAttention(embed_dim, num_heads)
out = attn(x)
print(out.shape)  # Should be (batch_size, seq_len, embed_dim)
```

## Usage
```python
import torch
from causal_attention import CausalSelfAttention

batch_size, seq_len, embed_dim, num_heads = 2, 10, 64, 8
x = torch.randn(batch_size, seq_len, embed_dim)
attn = CausalSelfAttention(embed_dim, num_heads)
out = attn(x)
print(out.shape)  # Should be (batch_size, seq_len, embed_dim)
```

## Explanation
1. **Query, Key, and Value Projections**: Input embeddings are projected into query, key, and value tensors.
2. **Attention Calculation**: Scaled dot-product attention is applied with a causal mask.
3. **Causal Masking**: Ensures that each token attends only to past tokens.
4. **Output Projection**: The attended values are projected back to the embedding space.

## License
This project is released under the MIT License.

