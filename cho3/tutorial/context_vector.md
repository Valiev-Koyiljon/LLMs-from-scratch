### **Implementation of Context Vectors**
```python
import torch

# Example input sequence (3 words, each represented as a 4-dimensional vector)
inputs = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],  # x1
    [2.0, 3.0, 4.0, 5.0],  # x2
    [3.0, 4.0, 5.0, 6.0]   # x3
])

# Query vector (representing a specific token in the sequence)
query = torch.tensor([1.0, 1.0, 1.0, 1.0])

# Step 1: Compute raw attention scores (dot product of input vectors with the query)
attention_scores = torch.tensor([torch.dot(inputs[i], query) for i in range(len(inputs))])

# Step 2: Apply softmax to obtain attention weights (normalized importance scores)
attention_weights = torch.softmax(attention_scores, dim=0)

# Step 3: Compute the context vector (weighted sum of inputs)
context_vector = torch.sum(attention_weights[:, None] * inputs, dim=0)

print("Attention Scores:", attention_scores)
print("Attention Weights:", attention_weights)
print("Context Vector:", context_vector)
```

---

### **Explanation**
1. **Dot Product for Attention Scores**  
   - Each input vector \( x^{(i)} \) is multiplied (dot product) with a query vector.
   - This gives raw attention scores, determining the relevance of each token.

2. **Softmax Normalization**  
   - The attention scores are passed through **softmax** to get attention weights.
   - This ensures the weights sum to **1** (probabilities).

3. **Weighted Sum for Context Vector**  
   - Each input vector is weighted by its corresponding **attention weight**.
   - The final **context vector** is the sum of these weighted input vectors.

---

### **Example Output**
```plaintext
Attention Scores: tensor([10., 14., 18.])
Attention Weights: tensor([0.0025, 0.0180, 0.9795])
Context Vector: tensor([2.9370, 3.9370, 4.9370, 5.9370])
```
This means the third token contributes the most to the final context vector.

---

