# %% [markdown]
# # Module 03: Vector Indexing & Algorithms (HNSW / IVF)
#
# When vector collections grow beyond millions of items, brute-force exact nearest neighbor search ($O(N)$) becomes too slow for real-time inference.
#
# Approximate Nearest Neighbor (ANN) algorithms trade a tiny fraction of recall for orders-of-magnitude speedups.
# This tutorial explores:
# 1. **Exact Flat KNN Search** ($O(N)$)
# 2. **Inverted File Indexing (IVF / Voronoi Cells)**
# 3. **Hierarchical Navigable Small World (HNSW) Graphs**
#
# ---

# %%
import time
import numpy as np
from typing import List, Tuple

# %% [markdown]
# ## Section 1: Exact Flat KNN Search

# %%
def exact_knn(query_vec: np.ndarray, index_matrix: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact cosine similarities across all indexed vectors."""
    # Dot product of normalized vectors equals cosine similarity
    similarities = np.dot(index_matrix, query_vec)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return top_indices, similarities[top_indices]

# Generate synthetic 10,000 vectors of dimension 128
N, D = 10000, 128
np.random.seed(42)
raw_matrix = np.random.randn(N, D)
index_matrix = raw_matrix / np.linalg.norm(raw_matrix, axis=1, keepdims=True)

query_raw = np.random.randn(D)
query_vec = query_raw / np.linalg.norm(query_raw)

start_t = time.perf_counter()
indices, scores = exact_knn(query_vec, index_matrix, top_k=5)
flat_latency_ms = (time.perf_counter() - start_t) * 1000

print(f"Exact Flat Search completed in {flat_latency_ms:.2f} ms")
print(f"Top 5 Closest Vector IDs: {indices}")
print(f"Top 5 Cosine Scores: {scores}")

# %% [markdown]
# ## Section 2: HNSW Conceptual Graph Architecture
#
# HNSW constructs a multi-layer graph where:
# - **Top Layers:** Long-range skip connections across distant clusters.
# - **Bottom Layer ($L_0$):** Dense local neighborhood graph for fine-grained convergence.
#
# Average search complexity: $O(\log N)$.

# %%
# collapse_input
class HNSWSimulator:
    def __init__(self, num_vectors: int, max_layers: int = 4):
        self.num_vectors = num_vectors
        self.max_layers = max_layers
        self.layers = {layer: int(num_vectors * (0.1 ** layer)) for layer in range(max_layers)}

    def describe(self):
        print("\nHNSW Multi-Layer Hierarchy Simulation:")
        for layer, node_count in sorted(self.layers.items(), reverse=True):
            print(f"  Layer {layer}: {node_count:>6} nodes (Skip connections & cluster routing)")

hnsw = HNSWSimulator(num_vectors=100000)
hnsw.describe()
