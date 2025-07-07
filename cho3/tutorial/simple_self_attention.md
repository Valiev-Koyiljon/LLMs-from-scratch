# Simplified Self-Attention Mechanism

## Overview

This document explains a **very simplified** variant of self-attention, which does **not** contain any trainable weights. It serves purely as an **illustration** of the core concepts behind attention mechanisms before introducing the full self-attention mechanism used in transformers.

## Understanding the Simplified Self-Attention Mechanism

We are given an **input sequence** consisting of token embeddings:

```
x(1), x(2), ..., x(T)
```

For example, given the sentence:

> "Your journey starts with one step"

- x(1) represents "Your"
- x(2) represents "journey"
- x(3) represents "starts", and so on.

The goal is to compute **context vectors** z(i) for each input token. A **context vector** is a new representation of the input token that incorporates information from the entire sequence.

## How It Works

For each input x(i), we compute a corresponding **context vector** z(i), which is a **weighted sum** of all input vectors:

```
z(i) = Σ(j=1 to T) α(ij) * x(j)
```

where:

- α(ij) is an **attention weight** that determines how much importance we give to x(j) when computing z(i).
- The weights α(ij) sum up to 1 for each i:

```
Σ(j=1 to T) α(ij) = 1
```

This means that each z(i) is a combination of all input tokens, but with different levels of importance for each token.

To make this more concrete, let's consider computing the **second** context vector z(2), which corresponds to the word "journey":

```
z(2) = Σ(j=1 to T) α(2j) * x(j)
```

- The weights α(2j) determine how much each input x(j) contributes to z(2).
- If α(21) is **high**, then "Your" contributes significantly to "journey".
- If α(24) is **low**, then "with" does not contribute much to "journey".
- z(2) is a new representation of "journey" that incorporates relevant information from all words in the sentence.

## Key Takeaways

1. **Context vectors capture dependencies:** Instead of just using x(i), we compute z(i), which considers all input tokens.
2. **Weights determine importance:** The attention weights α(ij) decide how much influence each word has when computing the context vector.
3. **No trainable parameters (yet):** This is a simplified, **non-trainable** version of self-attention, used for conceptual understanding.


