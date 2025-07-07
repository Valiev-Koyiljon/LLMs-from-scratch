## 🔍 What is Multi-Head Self-Attention?

Multi-head self-attention enhances the model’s ability to focus on different parts of a sequence by running **multiple self-attention operations (heads)** in parallel. Each head learns to attend to different representation subspaces.

### 💡 Why use multiple heads?

* One head may focus on **syntax**, another on **semantics**.
* Encourages diversity in learned attention patterns.

### 🧮 Steps in Multi-Head Attention

1. Linearly project the input into multiple sets of Q, K, V (one for each head).
2. Compute **scaled dot-product attention** for each head.
3. Concatenate all head outputs.
4. Apply a final linear projection.

---

## ✅ Multi-Head Self-Attention: PyTorch Implementation

```python
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Linear layers for Q, K, V (shared across heads)
        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)

        # Output linear layer after concatenation
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, X):
        batch_size, seq_len, _ = X.shape

        # Linear projections
        Q = self.W_q(X)  # (B, T, E)
        K = self.W_k(X)
        V = self.W_v(X)

        # Split into heads: (B, T, H, D) -> (B, H, T, D)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5
        attention_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, V)  # (B, H, T, D)

        # Concatenate heads and apply final linear projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)
        out = self.fc_out(out)

        return out, attention_weights
```

---

## 📄 Updated README for Multi-Head Attention

````markdown
# Multi-Head Self-Attention from Scratch in PyTorch

## 🧠 Introduction

This repository implements **Self-Attention** and **Multi-Head Self-Attention** mechanisms from scratch using **PyTorch**.

Self-attention is the core operation behind Transformer architectures, allowing models to weigh the importance of different parts of a sequence.

Multi-head attention improves this by enabling the model to capture **multiple representation subspaces** in parallel.

---

## 📌 Key Concepts

### ✔️ Self-Attention Steps

1. Project input `X` into `Q`, `K`, and `V`.
2. Compute attention scores:  
   \[
   \text{scores} = \frac{QK^\top}{\sqrt{d_k}}
   \]
3. Apply `softmax` to scores.
4. Multiply weights with `V` to get the output.

### ✔️ Multi-Head Self-Attention

- Multiple heads process the input in parallel.
- Outputs are concatenated and linearly projected.
- Allows learning from different representation subspaces.

---

## 🛠 Implementations

### **🔹 Self-Attention (Single-Head)**

See [`self_attention.py`](./self_attention.py)

### **🔸 Multi-Head Self-Attention**

See [`multihead_attention.py`](./multihead_attention.py)

```python
from multihead_attention import MultiHeadSelfAttention

batch_size, seq_len, embed_size, num_heads = 2, 5, 8, 2
X = torch.rand(batch_size, seq_len, embed_size)
mha = MultiHeadSelfAttention(embed_size, num_heads)
output, attn_weights = mha(X)
````

---

## 🔍 Visualizing Attention Weights

```python
from visualize import visualize_attention
visualize_attention(attn_weights[0][0].detach().numpy())  # First head of first batch
```

---

## 🚀 Features

* Pure PyTorch implementation
* Clear visualization tools using `matplotlib` and `seaborn`
* Easy to extend to full Transformer blocks

---

## 🧩 Future Work

* Add **Positional Encoding**
* Build a **Transformer Encoder Block**
* Support **Masking** and **Causal Attention**
* Integrate with **language modeling tasks**

---

## 📚 References

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

💡 Pull requests are welcome! Feel free to fork and experiment.

