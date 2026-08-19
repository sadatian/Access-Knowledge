# %% [markdown]
# # ⚙️ Module 01: Modern Retrieval Workspace Setup & Tokenization Math
#
# Welcome to **Module 01** of Knowledge Retrieval A-Z.
# In this module, we explore the foundational environment for retrieval engineering:
# 1. Dependency orchestration with `uv`.
# 2. Tokenization math, vocabulary spaces, and context windows.
# 3. Vector embedding representation spaces and dimensionality.
#
# ---

# %%
import math
import numpy as np
from openai import OpenAI

# %% [markdown]
# ## 🔤 Section 1: Tokenization & Context Math
#
# LLMs process text in discrete tokens rather than characters or words.
# The relationship between token count $T$, embedding dimension $D$, and memory is critical when designing retrieval chunking.

# %%
def analyze_token_statistics(text: str):
    # Rule of thumb heuristic: 1 token ~= 4 characters in English
    char_count = len(text)
    estimated_tokens = math.ceil(char_count / 4)
    print(f"Text: '{text}'")
    print(f"  • Character count: {char_count}")
    print(f"  • Estimated tokens: {estimated_tokens}")
    return estimated_tokens

sample_doc = "Cache-Augmented Generation (CAG) eliminates retrieval latency by preloading documents into the KV-cache."
tokens = analyze_token_statistics(sample_doc)

# %% [markdown]
# ## 📐 Section 2: Embedding Space & Vector Dimensionality
#
# A vector embedding represents semantic meaning as a dense coordinate in a $D$-dimensional space.

# %%
def simulate_dense_embedding(dimension: int = 768) -> np.ndarray:
    """Generate a unit-normalized random dense embedding vector."""
    vec = np.random.randn(dimension)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

v1 = simulate_dense_embedding(768)
v2 = simulate_dense_embedding(768)
cosine_sim = float(np.dot(v1, v2))

print(f"Embedding Dimension: {len(v1)}")
print(f"Cosine Similarity between random vectors: {cosine_sim:.4f} (Expected near 0)")

# %% [markdown]
# ## 🚀 Section 3: Summary & Next Steps
#
# You have initialized the foundational math for tokens and embeddings.
# In **Module 02**, we construct sparse lexical (BM25) and dense semantic search engines and fuse them using Reciprocal Rank Fusion.
