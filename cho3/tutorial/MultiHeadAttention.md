````markdown
# 🧠 Multi-Head Self-Attention from Scratch in PyTorch

This repository implements **Self-Attention** and **Multi-Head Self-Attention** mechanisms using **PyTorch**, core components of Transformer-based architectures. These attention mechanisms empower models to dynamically focus on relevant parts of sequences for each prediction step.

---

## 🔍 What is Multi-Head Self-Attention?

Multi-head self-attention enhances a model’s capacity to learn richer representations by computing attention multiple times in **parallel** — each with different learnable projections. Every "head" can focus on different aspects of the sequence.

### 💡 Why Use Multiple Heads?

- One head might capture **syntactic** relationships.
- Another might focus on **semantic** meanings.
- Promotes diversity in learned attention patterns.

---

## 🧮 Steps in Multi-Head Self-Attention

1. Linearly project the input into multiple sets of Queries (`Q`), Keys (`K`), and Values (`V`).
2. Compute **scaled dot-product attention** independently in each head.
3. Concatenate all head outputs.
4. Apply a final linear projection to combine the heads' information.

---

## ✅ Multi-Head Self-Attention Implementation (PyTorch)

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

        # Learnable projection matrices for all heads
        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)

        # Output linear layer
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, X):
        batch_size, seq_len, _ = X.shape

        # Step 1: Compute Q, K, V
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        # Step 2: Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 3: Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5
        attention_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, V)

        # Step 4: Concatenate heads and apply final projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)
        out = self.fc_out(out)

        return out, attention_weights
````

---

## 🧪 Example Usage

```python
from multihead_attention import MultiHeadSelfAttention
import torch

batch_size, seq_len, embed_size, num_heads = 2, 5, 8, 2
X = torch.rand(batch_size, seq_len, embed_size)

mha = MultiHeadSelfAttention(embed_size, num_heads)
output, attn_weights = mha(X)

print("Output shape:", output.shape)  # (2, 5, 8)
print("Attention shape:", attn_weights.shape)  # (2, 2, 5, 5)
```

---

## 📊 Visualizing Attention Weights

You can visualize the attention matrix from any head using `matplotlib` and `seaborn`.

```python
from visualize import visualize_attention

# Visualize attention weights from head 0 of batch 0
visualize_attention(attn_weights[0][0].detach().numpy())
```

---

## 🚀 Features

* Implements **Multi-Head Self-Attention** from scratch
* Includes both **single-head** and **multi-head** attention
* Clean, modular PyTorch code for easy experimentation
* Attention weight visualization utility

---

## 🧩 TODO / Future Work

* ✅ Add **Positional Encoding**
* 🔜 Implement **Transformer Encoder Block**
* 🔜 Support **Masking and Causal Attention**
* 🔜 Apply to **NLP/vision downstream tasks**

---

## 📚 References

* [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
* [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

---

💡 Pull requests, suggestions, and forks are welcome! Feel free to use and expand this repo in your own projects. 🌟

```

