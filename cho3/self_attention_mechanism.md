# Self-Attention from Scratch in PyTorch

## Introduction
This repository contains a simple implementation of the **Self-Attention Mechanism** in PyTorch. Self-attention is a key component of **Transformer architectures**, allowing models to focus on relevant parts of input sequences when making predictions.

## 📌 Steps in Self-Attention
1. **Project input (`X`)** into Query (`Q`), Key (`K`), and Value (`V`) matrices using learnable weights.
2. Compute the **attention scores** using the scaled dot-product formula:
   
   \[
   \text{scores} = \frac{QK^T}{\sqrt{d_k}}
   \]
   
3. Apply **softmax** to obtain attention weights.
4. Multiply attention weights by `V` to get the final output.

---

## 🛠 Implementation

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size

        # Learnable weight matrices for Query, Key, and Value
        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_k = nn.Linear(embed_size, embed_size, bias=False)
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)
        
    def forward(self, X):
        # Step 1: Compute Q, K, V matrices
        Q = self.W_q(X)  # (batch_size, seq_len, embed_size)
        K = self.W_k(X)  # (batch_size, seq_len, embed_size)
        V = self.W_v(X)  # (batch_size, seq_len, embed_size)

        # Step 2: Compute attention scores
        d_k = self.embed_size ** 0.5  # Scaling factor
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k  # (batch_size, seq_len, seq_len)

        # Step 3: Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Step 4: Compute the weighted sum of values
        output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, embed_size)
        
        return output, attention_weights  # Return attention weights for visualization
```

---

## 🔧 Usage

```python
# Define input dimensions
batch_size, seq_len, embed_size = 2, 5, 8  # Example dimensions
X = torch.rand(batch_size, seq_len, embed_size)  # Random input

# Initialize and run self-attention module
self_attention = SelfAttention(embed_size)
output, attention_weights = self_attention(X)

print("Self-Attention Output:", output.shape)  # Expected: (batch_size, seq_len, embed_size)
print("Attention Weights:", attention_weights.shape)  # Expected: (batch_size, seq_len, seq_len)
```

---

## 🔥 Key Features
- Implements **Self-Attention** from scratch in PyTorch.
- Uses **Query, Key, and Value projections** for attention computation.
- **Softmax normalization** ensures proper attention weight distribution.
- Suitable for use in **Transformer models and NLP tasks**.

---

## 🚀 Future Improvements
- Extend to **Multi-Head Self-Attention**.
- Implement **Positional Encoding** for sequence information.
- Add **Masking** for handling padding tokens in NLP tasks.

---

## 📜 References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar

---

💡 **Feel free to contribute or modify this implementation for your projects!** 🚀

