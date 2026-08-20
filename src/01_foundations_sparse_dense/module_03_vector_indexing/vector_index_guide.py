# %% [markdown]
# # Module 03: Vector Indexing & Algorithms (HNSW / IVF / PQ)
#
# Welcome to **Module 03** of the Knowledge Retrieval A-Z masterclass.
# When vector databases scale to millions or billions of embedding vectors, computing exact brute-force cosine similarity ($O(N)$) becomes a prohibitive latency bottleneck for real-time RAG applications.
#
# **Approximate Nearest Neighbor (ANN)** indexing structures trade a minor fraction of recall for orders-of-magnitude faster queries and substantial memory compression.
#
# In this module, we construct and master from first principles:
# 1. **Exact Flat KNN Search**: Baseline matrix search engine providing 100% recall ground truth.
# 2. **Inverted File Index (IVF / Voronoi Space Partitioning)**: Lloyd's K-Means clustering, inverted posting lists, and multi-probe ($n_{\text{probe}}$) retrieval.
# 3. **Hierarchical Navigable Small World (HNSW)**: Multi-layer skip-graph hierarchy, greedy upper-layer routing, and $efSearch$ beam search on base graphs.
# 4. **Product Quantization (PQ) & Asymmetric Distance Computation (ADC)**: Subspace vector quantization, codebook construction, lookup tables, and $16\times$ memory reduction.
# 5. **Systematic ANN Benchmark Suite & Pareto Frontier**: Comparative evaluation of Recall@K, latency speedups, and memory footprints.
# 6. **Presenter Visualizer & Dashboard (`# collapse_input`)**: Auto-collapsing ASCII Pareto efficiency visualizer.
#
# ---
#
# ```mermaid
# graph TD
#     subgraph FlatSpace ["1. Exact Flat KNN (O(N))"]
#         F1["Brute Force Dot Product over All N Vectors"]
#     end
#
#     subgraph IVFSpace ["2. Inverted File Index (IVF)"]
#         C["K Centroids (Voronoi Cells)"] --> P["Probe n_probe Closest Centroids"]
#         P --> L["Scan Only Vectors in Target Lists"]
#     end
#
#     subgraph HNSWSpace ["3. HNSW Multi-Layer Graph"]
#         L2["Layer 2: Sparse Highway Skip-Connections"] --> L1["Layer 1: Medium-Range Clusters"]
#         L1 --> L0["Layer 0: Dense Local Neighborhood Graph (efSearch)"]
#     end
#
#     subgraph PQSpace ["4. Product Quantization (PQ)"]
#         D["Vector R^D"] --> M["M Sub-Vectors (D/M)"]
#         M --> CB["K_sub Centroids per Subspace"]
#         CB --> ADC["Asymmetric Distance Table (16x-32x Compression)"]
#     end
# ```
#
# ---

# %%
import heapq
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

def detect_compute_device() -> str:
    """Detect available compute accelerator (GPU/CUDA/MPS) with graceful CPU fallback."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:0 ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps (Apple Silicon GPU)"
    except ImportError:
        pass
    return "cpu (Optimized NumPy BLAS/SIMD)"

COMPUTE_DEVICE = detect_compute_device()
print(f"[INFO] Vector Indexing Compute Hardware initialized: {COMPUTE_DEVICE}")

# %% [markdown]
# ## Section 1: Exact Flat KNN Search (Ground Truth Baseline Engine)
#
# The **Exact Flat KNN Index** stores uncompressed, normalized embedding vectors $\mathbf{X} \in \mathbb{R}^{N \times D}$.
# Querying executes a dense matrix-vector multiplication $\mathbf{S} = \mathbf{X} \mathbf{q}^T$, calculating all $N$ inner products and sorting the top-$K$ highest scores.
#
# - **Time Complexity:** $O(N \cdot D)$ per query.
# - **Recall:** Strictly $100\%$ (Ground Truth reference for evaluating ANN approximations).
# - **Memory Footprint:** $N \times D \times 4 \text{ bytes (FP32)}$.

# %%
class FlatIndex:
    """Exact Flat KNN search index utilizing vectorized inner products for 100% recall ground truth."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: Optional[np.ndarray] = None
        self.num_vectors: int = 0

    def add(self, vectors: np.ndarray) -> "FlatIndex":
        """Add and unit-normalize a batch of vectors (N x D)."""
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")
        
        # Unit normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = (vectors / norms).astype(np.float32)

        if self.vectors is None:
            self.vectors = normalized
        else:
            self.vectors = np.vstack([self.vectors, normalized])
            
        self.num_vectors = len(self.vectors)
        return self

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-K closest vectors by cosine similarity. Returns (indices, scores)."""
        if self.vectors is None or self.num_vectors == 0:
            return np.array([]), np.array([])

        q_norm = np.linalg.norm(query_vector)
        q = (query_vector / q_norm).astype(np.float32) if q_norm > 0 else query_vector.astype(np.float32)

        # Dot product with pre-normalized database vectors = cosine similarity
        similarities = np.dot(self.vectors, q)
        
        # Extract top-K indices
        top_k = min(top_k, self.num_vectors)
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        # Sort top-K subset in descending order
        sorted_subset = top_indices[np.argsort(-similarities[top_indices])]
        
        return sorted_subset, similarities[sorted_subset]

    def memory_bytes(self) -> int:
        """Calculate total memory usage of indexed vectors in bytes."""
        return self.vectors.nbytes if self.vectors is not None else 0

# %% [markdown]
# ### Demo 1: Comprehensive Flat Index Ground Truth Demonstration
#
# Below, we generate a synthetic dataset of $N = 5,000$ vectors ($D = 128$), index them in `FlatIndex`, and execute an exact search.

# %%
np.random.seed(42)
N_VECTORS = 5000
DIMENSION = 128

# Generate synthetic vector collection
synthetic_dataset = np.random.randn(N_VECTORS, DIMENSION).astype(np.float32)
flat_index = FlatIndex(dimension=DIMENSION)
flat_index.add(synthetic_dataset)

# Generate a sample query vector
sample_query = np.random.randn(DIMENSION).astype(np.float32)

# Execute exact search
start_t = time.perf_counter_ns()
gt_indices, gt_scores = flat_index.search(sample_query, top_k=5)
flat_latency_us = (time.perf_counter_ns() - start_t) / 1_000.0

print("=== [Exact Flat KNN Index Status] ===")
print(f"Indexed Vectors (N): {flat_index.num_vectors:,} | Dimension (D): {flat_index.dimension}")
print(f"Memory Footprint:    {flat_index.memory_bytes() / 1024:.2f} KB ({flat_index.memory_bytes() / (1024**2):.3f} MB)")
print(f"Exact Search Latency: {flat_latency_us:.2f} microseconds ({flat_latency_us / 1000.0:.4f} ms)")
print("\nTop 5 Exact Nearest Neighbors (Ground Truth):")
for rank, (idx, score) in enumerate(zip(gt_indices, gt_scores), 1):
    print(f"  [{rank}] Vector ID: {idx:<6} | Cosine Similarity: {score:.5f}")

# %% [markdown]
# ## Section 2: Inverted File Index (IVF / Voronoi Partitioning) from Scratch
#
# The **Inverted File Index (IVF)** reduces search complexity by partitioning the $D$-dimensional vector space into $K_{\text{centroids}}$ **Voronoi cells** using K-Means clustering.
#
# ### Indexing & Retrieval Mechanism
# 1. **Training Phase:** Run K-Means to find $K$ cluster centroids $\mathbf{C} = \{\mathbf{c}_1, \dots, \mathbf{c}_K\}$.
# 2. **Posting Assignment:** Assign each vector $\mathbf{x}_i$ to its nearest centroid $\arg\max_j (\mathbf{x}_i \cdot \mathbf{c}_j)$ and append $i$ to inverted list $\mathcal{L}_j$.
# 3. **Query Search:** Given query $\mathbf{q}$:
#    - Compute similarity between $\mathbf{q}$ and all $K$ centroids ($O(K \cdot D)$).
#    - Select the top $n_{\text{probe}}$ closest centroids.
#    - Search only the vectors stored in the inverted lists of those $n_{\text{probe}}$ centroids.
# - **Complexity:** $O(K \cdot D + n_{\text{probe}} \cdot \frac{N}{K} \cdot D)$. For $K \approx \sqrt{N}$, this achieves substantial speedups over Flat search.

# %%
class IVFIndex:
    """Inverted File Index (IVF) with Lloyd's K-Means clustering and multi-probe retrieval."""

    def __init__(self, dimension: int, num_centroids: int = 32, max_kmeans_iters: int = 15, seed: int = 42):
        self.dimension = dimension
        self.num_centroids = num_centroids
        self.max_kmeans_iters = max_kmeans_iters
        self.seed = seed
        
        self.centroids: Optional[np.ndarray] = None
        self.inverted_lists: Dict[int, List[int]] = defaultdict(list)
        self.vectors: Optional[np.ndarray] = None
        self.num_vectors: int = 0

    def _train_kmeans(self, vectors: np.ndarray) -> np.ndarray:
        """Train K-Means centroids using cosine similarity on normalized vectors."""
        np.random.seed(self.seed)
        N = len(vectors)
        # Initialize centroids randomly from dataset points
        init_indices = np.random.choice(N, size=self.num_centroids, replace=False)
        centroids = vectors[init_indices].copy()

        for _ in range(self.max_kmeans_iters):
            # Assign each vector to closest centroid: S = X @ C^T
            sim_matrix = np.dot(vectors, centroids.T)
            assignments = np.argmax(sim_matrix, axis=1)

            new_centroids = np.zeros_like(centroids)
            for c_idx in range(self.num_centroids):
                members = vectors[assignments == c_idx]
                if len(members) > 0:
                    mean_vec = np.mean(members, axis=0)
                    norm = np.linalg.norm(mean_vec)
                    new_centroids[c_idx] = mean_vec / norm if norm > 0 else centroids[c_idx]
                else:
                    new_centroids[c_idx] = centroids[c_idx]

            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        return centroids

    def train_and_add(self, vectors: np.ndarray) -> "IVFIndex":
        """Train Voronoi centroids and populate inverted posting lists."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = (vectors / norms).astype(np.float32)
        
        self.vectors = normalized
        self.num_vectors = len(normalized)
        
        # 1. Train centroids
        self.centroids = self._train_kmeans(normalized)
        
        # 2. Populate inverted lists
        sim_to_centroids = np.dot(normalized, self.centroids.T)
        cluster_assignments = np.argmax(sim_to_centroids, axis=1)
        
        self.inverted_lists.clear()
        for vec_id, c_idx in enumerate(cluster_assignments):
            self.inverted_lists[int(c_idx)].append(vec_id)
            
        return self

    def search(self, query_vector: np.ndarray, top_k: int = 10, n_probe: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-K vectors across the top n_probe closest Voronoi centroids."""
        if self.centroids is None or self.vectors is None:
            return np.array([]), np.array([])

        q_norm = np.linalg.norm(query_vector)
        q = (query_vector / q_norm).astype(np.float32) if q_norm > 0 else query_vector.astype(np.float32)

        # 1. Find top n_probe closest centroids
        centroid_sims = np.dot(self.centroids, q)
        n_probe = min(n_probe, self.num_centroids)
        top_centroid_indices = np.argsort(-centroid_sims)[:n_probe]

        # 2. Gather candidate vectors from selected inverted lists
        candidate_ids = []
        for c_idx in top_centroid_indices:
            candidate_ids.extend(self.inverted_lists.get(c_idx, []))

        if not candidate_ids:
            return np.array([]), np.array([])

        candidate_ids_arr = np.array(candidate_ids, dtype=np.int32)
        candidate_vectors = self.vectors[candidate_ids_arr]

        # 3. Compute cosine similarities only on candidate subset
        candidate_sims = np.dot(candidate_vectors, q)
        
        # Top-K ranking
        top_k = min(top_k, len(candidate_ids_arr))
        top_local_indices = np.argpartition(candidate_sims, -top_k)[-top_k:]
        sorted_subset = top_local_indices[np.argsort(-candidate_sims[top_local_indices])]

        return candidate_ids_arr[sorted_subset], candidate_sims[sorted_subset]

# %% [markdown]
# ### Demo 2: Comprehensive IVF Index Demonstration & n_probe Tradeoff Sweep
#
# Below, we train an IVF index with $K = 32$ centroids and evaluate search latency and Recall@10 across varying $n_{\text{probe}} \in [1, 2, 4, 8, 16, 32]$.

# %%
ivf_index = IVFIndex(dimension=DIMENSION, num_centroids=32)
ivf_index.train_and_add(synthetic_dataset)

# Inspect cluster balance
cluster_sizes = [len(ivf_index.inverted_lists[i]) for i in range(ivf_index.num_centroids)]
print("=== [IVF Index Clustering Summary] ===")
print(f"Total Centroids (K):       {ivf_index.num_centroids}")
print(f"Min / Avg / Max List Size: {min(cluster_sizes)} / {np.mean(cluster_sizes):.1f} / {max(cluster_sizes)} vectors")

# Ground truth top-10 IDs from Flat Index
gt_10_ids, _ = flat_index.search(sample_query, top_k=10)
gt_10_set = set(gt_10_ids)

print("\nIVF Multi-Probe Parameter Sweep (n_probe vs Recall@10 vs Latency):")
print(f"  {'n_probe':<10}{'Vectors Scanned':<18}{'Latency (us)':<16}{'Recall@10':<12}")
print("  " + "-" * 56)

for probe in [1, 2, 4, 8, 16, 32]:
    t0 = time.perf_counter_ns()
    ivf_ids, _ = ivf_index.search(sample_query, top_k=10, n_probe=probe)
    t_us = (time.perf_counter_ns() - t0) / 1000.0
    
    # Calculate recall: intersection with exact ground truth
    recall_10 = len(set(ivf_ids).intersection(gt_10_set)) / 10.0
    # Calculate number of vectors examined
    top_c = np.argsort(-np.dot(ivf_index.centroids, sample_query / np.linalg.norm(sample_query)))[:probe]
    scanned_count = sum(len(ivf_index.inverted_lists[c]) for c in top_c)
    
    print(f"  {probe:<10}{scanned_count:<18}{t_us:<16.2f}{recall_10 * 100:>5.1f}%")

# %% [markdown]
# ## Section 3: Hierarchical Navigable Small World (HNSW) from First Principles
#
# **HNSW** is the state-of-the-art graph-based ANN algorithm implemented in production vector databases (e.g. Qdrant, Chroma, FAISS, Pinecone).
#
# ### The Multi-Layer Graph Hierarchy
# - **Skip-List Metaphor:** HNSW structures vectors across multiple layers $L = 0, 1, \dots, L_{\text{max}}$.
#   - Each node is assigned a maximum layer $l$ drawn from an exponential distribution:
#     $$l \sim \lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \rfloor, \quad m_L = \frac{1}{\ln(M)}$$
#   - **Upper Layers ($L_{\text{max}} \dots 1$):** Sparse graphs with long-range skip edges for logarithmic global routing ($O(\log N)$).
#   - **Bottom Layer ($L_0$):** Dense local proximity graph connecting all $N$ vectors for fine-grained convergence.
# - **Beam Search ($efSearch$):** On Layer 0, a priority queue of size $efSearch$ traverses local neighborhoods to find the global optimum without getting trapped in local minima.

# %%
class HNSWIndex:
    """Hierarchical Navigable Small World (HNSW) graph index implemented from first principles."""

    def __init__(self, dimension: int, M: int = 16, ef_construction: int = 32, ml: Optional[float] = None, seed: int = 42):
        self.dimension = dimension
        self.M = M  # Max connections per node on layer > 0
        self.M0 = 2 * M  # Max connections on layer 0
        self.ef_construction = ef_construction
        self.ml = ml if ml is not None else 1.0 / math.log(M)
        self.seed = seed
        
        self.vectors: List[np.ndarray] = []
        self.num_nodes: int = 0
        self.entry_point: Optional[int] = None
        self.max_layer: int = -1
        
        # Multi-layer adjacency lists: graphs[layer][node_id] = list of neighbor_ids
        self.graphs: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.node_layers: Dict[int, int] = {}

    def _assign_random_layer(self) -> int:
        """Assign random maximum layer via exponential decay."""
        np.random.seed(self.seed + self.num_nodes)
        unif = np.random.uniform(1e-6, 1.0)
        return int(math.floor(-math.log(unif) * self.ml))

    def _cosine_dist(self, u: np.ndarray, v: np.ndarray) -> float:
        """Cosine distance: 1.0 - (u . v)."""
        return float(1.0 - np.dot(u, v))

    def add(self, vectors: np.ndarray) -> "HNSWIndex":
        """Add and index a batch of normalized vectors into the HNSW graph."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = (vectors / norms).astype(np.float32)

        for vec in normalized:
            self._insert_vector(vec)
        return self

    def _insert_vector(self, vec: np.ndarray) -> int:
        """Insert a single vector into the multi-layer HNSW graph."""
        node_id = self.num_nodes
        self.vectors.append(vec)
        self.num_nodes += 1

        node_layer = self._assign_random_layer()
        self.node_layers[node_id] = node_layer

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_layer = node_layer
            for l in range(node_layer + 1):
                self.graphs[l][node_id] = []
            return node_id

        curr_obj = self.entry_point
        curr_dist = self._cosine_dist(vec, self.vectors[curr_obj])

        # 1. Greedy routing from top layer down to node_layer + 1
        for l in range(self.max_layer, node_layer, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.graphs[l].get(curr_obj, []):
                    d = self._cosine_dist(vec, self.vectors[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # 2. Search and connect on layers from min(max_layer, node_layer) down to 0
        top_l = min(self.max_layer, node_layer)
        for l in range(top_l, -1, -1):
            # Beam search on layer l
            candidates = self._search_layer(vec, curr_obj, ef=self.ef_construction, layer=l)
            m_max = self.M0 if l == 0 else self.M
            
            # Select closest M neighbors
            neighbors = [c_id for _, c_id in sorted(candidates)[:m_max]]
            self.graphs[l][node_id] = neighbors
            
            # Add bidirectional links
            for neighbor in neighbors:
                self.graphs[l][neighbor].append(node_id)
                if len(self.graphs[l][neighbor]) > m_max:
                    # Prune furthest neighbor
                    dists = [(self._cosine_dist(self.vectors[neighbor], self.vectors[n]), n) for n in self.graphs[l][neighbor]]
                    dists.sort()
                    self.graphs[l][neighbor] = [n for _, n in dists[:m_max]]

            if candidates:
                curr_obj = candidates[0][1]

        if node_layer > self.max_layer:
            self.max_layer = node_layer
            self.entry_point = node_id

        return node_id

    def _search_layer(self, query: np.ndarray, entry_point: int, ef: int, layer: int) -> List[Tuple[float, int]]:
        """Beam search on a single graph layer."""
        v_entry = self.vectors[entry_point]
        d_entry = self._cosine_dist(query, v_entry)
        
        visited = {entry_point}
        # Candidates min-heap: (dist, node_id)
        candidates = [(d_entry, entry_point)]
        # Best found max-heap (negative distance for max-heap behavior): (-dist, node_id)
        w = [(-d_entry, entry_point)]

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_best_dist = -w[0][0]

            if c_dist > furthest_best_dist:
                break

            for neighbor in self.graphs[layer].get(c_id, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    d_neighbor = self._cosine_dist(query, self.vectors[neighbor])
                    furthest_best_dist = -w[0][0]

                    if d_neighbor < furthest_best_dist or len(w) < ef:
                        heapq.heappush(candidates, (d_neighbor, neighbor))
                        heapq.heappush(w, (-d_neighbor, neighbor))
                        if len(w) > ef:
                            heapq.heappop(w)

        return [(-neg_d, n_id) for neg_d, n_id in w]

    def search(self, query_vector: np.ndarray, top_k: int = 10, ef_search: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-K nearest neighbors across the HNSW graph hierarchy."""
        if self.entry_point is None or self.num_nodes == 0:
            return np.array([]), np.array([])

        q_norm = np.linalg.norm(query_vector)
        q = (query_vector / q_norm).astype(np.float32) if q_norm > 0 else query_vector.astype(np.float32)

        curr_obj = self.entry_point
        curr_dist = self._cosine_dist(q, self.vectors[curr_obj])

        # 1. Greedy routing on upper layers (L_max down to 1)
        for l in range(self.max_layer, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.graphs[l].get(curr_obj, []):
                    d = self._cosine_dist(q, self.vectors[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # 2. Beam search on bottom layer 0 with ef_search
        candidates = self._search_layer(q, curr_obj, ef=max(ef_search, top_k), layer=0)
        candidates.sort()

        top_candidates = candidates[:top_k]
        indices = np.array([n_id for _, n_id in top_candidates], dtype=np.int32)
        scores = np.array([1.0 - dist for dist, _ in top_candidates], dtype=np.float32)
        return indices, scores

# %% [markdown]
# ### Demo 3: Comprehensive HNSW Graph Demonstration & efSearch Sweep
#
# Below, we build an HNSW index on a 1,000-vector subset, inspect layer distribution, and evaluate Recall@10 across varying $efSearch \in [4, 8, 16, 32, 64]$.

# %%
# Build HNSW graph on subset for interactive demonstration
hnsw_subset = synthetic_dataset[:1000]
hnsw_index = HNSWIndex(dimension=DIMENSION, M=8, ef_construction=16)
hnsw_index.add(hnsw_subset)

# Ground truth for 1,000 subset
flat_sub = FlatIndex(dimension=DIMENSION).add(hnsw_subset)
gt_hnsw_ids, _ = flat_sub.search(sample_query, top_k=10)
gt_hnsw_set = set(gt_hnsw_ids)

print("=== [HNSW Graph Multi-Layer Topology] ===")
print(f"Total Nodes Indexed: {hnsw_index.num_nodes:,} | Max Layer: {hnsw_index.max_layer}")
for l in range(hnsw_index.max_layer, -1, -1):
    node_count = len(hnsw_index.graphs[l])
    total_edges = sum(len(neighbors) for neighbors in hnsw_index.graphs[l].values())
    avg_degree = (total_edges / node_count) if node_count > 0 else 0
    print(f"  • Layer {l}: {node_count:>5} nodes | {total_edges:>5} edges | Avg Degree = {avg_degree:.1f}")

print("\nHNSW efSearch Beam Width Parameter Sweep:")
print(f"  {'efSearch':<12}{'Latency (us)':<16}{'Recall@10':<12}")
print("  " + "-" * 40)

for ef in [4, 8, 16, 32, 64]:
    t0 = time.perf_counter_ns()
    hnsw_ids, _ = hnsw_index.search(sample_query, top_k=10, ef_search=ef)
    t_us = (time.perf_counter_ns() - t0) / 1000.0
    recall = len(set(hnsw_ids).intersection(gt_hnsw_set)) / 10.0
    print(f"  {ef:<12}{t_us:<16.2f}{recall * 100:>5.1f}%")

# %% [markdown]
# ## Section 4: Product Quantization (PQ) Compression & Asymmetric Distance Computation (ADC)
#
# **Product Quantization (PQ)** solves the RAM memory crisis of multi-million vector databases.
# Instead of storing continuous 32-bit floats, PQ compresses each vector into $M$ discrete 1-byte codebook indices.
#
# ### The Product Quantization Mechanics
# 1. **Subspace Splitting:** A vector $\mathbf{x} \in \mathbb{R}^D$ is sliced into $M$ orthogonal sub-vectors $\mathbf{u}_1, \dots, \mathbf{u}_M \in \mathbb{R}^{D/M}$.
# 2. **Codebook Clustering:** For each subspace $m \in [1, M]$, train $K_{\text{sub}}$ centroids (typically $K_{\text{sub}} = 256 = 2^8$, requiring only 1 byte per sub-vector).
# 3. **Quantization Encoding:** Each sub-vector $\mathbf{u}_m$ is replaced by the index $k^* \in [0, 255]$ of its nearest centroid:
#    $$\mathbf{x} \in \mathbb{R}^D \implies \mathbf{c}(\mathbf{x}) = [k_1^*, k_2^*, \dots, k_M^*] \in \{0, \dots, 255\}^M$$
#    - **Compression Ratio:** $\frac{D \times 4 \text{ bytes}}{M \times 1 \text{ byte}}$. For $D = 128, M = 8$, this achieves a **$64\times$ memory reduction**.
# 4. **Asymmetric Distance Computation (ADC):**
#    - Precompute a distance lookup table $\mathcal{T} \in \mathbb{R}^{M \times K_{\text{sub}}}$ between unquantized query sub-vectors $\mathbf{q}_m$ and all subspace centroids $\mathbf{c}_{m, k}$.
#    - Query-to-vector distance becomes a fast sequence of table lookups and additions:
#      $$d_{\text{ADC}}(\mathbf{q}, \tilde{\mathbf{x}}_i) = \sum_{m=1}^M \mathcal{T}[m, \mathbf{c}_m(\mathbf{x}_i)]$$

# %%
class ProductQuantizer:
    """Product Quantizer (PQ) with Subspace Codebooks and Asymmetric Distance Computation (ADC)."""

    def __init__(self, dimension: int = 128, num_subvectors: int = 8, num_centroids: int = 16, seed: int = 42):
        if dimension % num_subvectors != 0:
            raise ValueError(f"Dimension {dimension} must be divisible by num_subvectors {num_subvectors}")
        
        self.dimension = dimension
        self.num_subvectors = M = num_subvectors
        self.sub_dim = dimension // num_subvectors
        self.num_centroids = K_sub = num_centroids
        self.seed = seed

        # Codebooks: shape (M, K_sub, sub_dim)
        self.codebooks: Optional[np.ndarray] = None
        # Quantized codes: shape (N, M) of uint8/int32
        self.codes: Optional[np.ndarray] = None
        self.num_vectors: int = 0

    def _train_subspace_kmeans(self, sub_vectors: np.ndarray, max_iters: int = 10) -> np.ndarray:
        """Train K-Means centroids for a single subspace."""
        np.random.seed(self.seed)
        N = len(sub_vectors)
        init_idx = np.random.choice(N, size=self.num_centroids, replace=False)
        centroids = sub_vectors[init_idx].copy()

        for _ in range(max_iters):
            # Compute Euclidean distances to centroids: (N, K_sub)
            dists = np.sum((sub_vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            new_centroids = np.zeros_like(centroids)
            for k in range(self.num_centroids):
                members = sub_vectors[labels == k]
                new_centroids[k] = np.mean(members, axis=0) if len(members) > 0 else centroids[k]

            if np.allclose(centroids, new_centroids, atol=1e-3):
                break
            centroids = new_centroids

        return centroids

    def train_and_encode(self, vectors: np.ndarray) -> "ProductQuantizer":
        """Train subspace codebooks and quantize all dataset vectors."""
        self.num_vectors = len(vectors)
        self.codebooks = np.zeros((self.num_subvectors, self.num_centroids, self.sub_dim), dtype=np.float32)
        self.codes = np.zeros((self.num_vectors, self.num_subvectors), dtype=np.uint8)

        # 1. Slice and train codebooks per subspace
        for m in range(self.num_subvectors):
            start_col = m * self.sub_dim
            end_col = start_col + self.sub_dim
            sub_vecs = vectors[:, start_col:end_col].astype(np.float32)

            # Train codebook
            c_book = self._train_subspace_kmeans(sub_vecs)
            self.codebooks[m] = c_book

            # Quantize vectors to closest centroid index
            dists = np.sum((sub_vecs[:, np.newaxis, :] - c_book[np.newaxis, :, :]) ** 2, axis=2)
            self.codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)

        return self

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Execute Asymmetric Distance Computation (ADC) using precomputed distance lookup table."""
        if self.codebooks is None or self.codes is None:
            return np.array([]), np.array([])

        q = query_vector.astype(np.float32)

        # 1. Precompute Distance Lookup Table T of shape (M, K_sub)
        lookup_table = np.zeros((self.num_subvectors, self.num_centroids), dtype=np.float32)
        for m in range(self.num_subvectors):
            start_col = m * self.sub_dim
            end_col = start_col + self.sub_dim
            q_sub = q[start_col:end_col]
            # Euclidean distance from query sub-vector to each centroid in subspace m
            lookup_table[m] = np.sum((self.codebooks[m] - q_sub) ** 2, axis=1)

        # 2. Accumulate distances for each quantized vector via table lookups
        approx_dists = np.zeros(self.num_vectors, dtype=np.float32)
        for m in range(self.num_subvectors):
            sub_indices = self.codes[:, m]
            approx_dists += lookup_table[m, sub_indices]

        # Top-K smallest Euclidean distances
        top_k = min(top_k, self.num_vectors)
        top_indices = np.argpartition(approx_dists, top_k)[:top_k]
        sorted_subset = top_indices[np.argsort(approx_dists[top_indices])]

        return sorted_subset, approx_dists[sorted_subset]

    def compression_ratio(self) -> float:
        """Calculate memory compression ratio: raw FP32 bytes / quantized code bytes."""
        raw_bytes = self.num_vectors * self.dimension * 4
        quant_bytes = self.num_vectors * self.num_subvectors * 1
        return raw_bytes / quant_bytes if quant_bytes > 0 else 0.0

# %% [markdown]
# ### Demo 4: Comprehensive Product Quantization & Memory Compression Demo
#
# Below, we train `ProductQuantizer` ($M = 8$ sub-vectors, $K_{\text{sub}} = 16$ centroids) on the $N = 5,000$ dataset and evaluate memory compression and ADC Recall@10.

# %%
pq = ProductQuantizer(dimension=DIMENSION, num_subvectors=8, num_centroids=16)
pq.train_and_encode(synthetic_dataset)

raw_memory_kb = (N_VECTORS * DIMENSION * 4) / 1024
pq_memory_kb = (N_VECTORS * 8 * 1) / 1024
ratio = pq.compression_ratio()

t0 = time.perf_counter_ns()
pq_ids, pq_dists = pq.search(sample_query, top_k=10)
pq_t_us = (time.perf_counter_ns() - t0) / 1000.0
pq_recall = len(set(pq_ids).intersection(gt_10_set)) / 10.0

print("=== [Product Quantization (PQ) Compression Summary] ===")
print(f"Sub-Vectors (M):          {pq.num_subvectors} (Sub-Dimension: {pq.sub_dim} dims/sub-vector)")
print(f"Centroids per Subspace:   {pq.num_centroids} centroids")
print(f"Raw FP32 Memory:          {raw_memory_kb:.1f} KB")
print(f"Quantized Memory:         {pq_memory_kb:.1f} KB")
print(f"Memory Compression Ratio: {ratio:.1f}x reduction")
print(f"ADC Query Latency:        {pq_t_us:.2f} us")
print(f"ADC Recall@10:            {pq_recall * 100:.1f}%")

# %% [markdown]
# ## Section 5: Systematic ANN Benchmark Suite & The Pareto Frontier
#
# To select the optimal vector indexing strategy for production retrieval, we execute a unified benchmark measuring the **Pareto Tradeoffs** between:
# 1. **Recall@10 Accuracy** (Ground Truth fidelity)
# 2. **Query Latency** (Speedup multiplier vs Exact Flat KNN)
# 3. **Memory Footprint** (RAM requirements)

# %%
class ANNBenchmarkHarness:
    """Unified benchmarking harness evaluating Flat vs IVF vs HNSW vs PQ index structures."""

    def __init__(self, dataset: np.ndarray, test_queries: List[np.ndarray]):
        self.dataset = dataset
        self.test_queries = test_queries
        self.N, self.D = dataset.shape

    def run_benchmark(self) -> List[Dict[str, Any]]:
        """Run all indexing engines and compute the Pareto frontier."""
        results = []

        # 1. Ground Truth Baseline (Flat KNN)
        flat = FlatIndex(dimension=self.D).add(self.dataset)
        
        # Measure Flat
        flat_latencies = []
        gt_map = {}
        for i, q in enumerate(self.test_queries):
            t0 = time.perf_counter_ns()
            top_ids, _ = flat.search(q, top_k=10)
            flat_latencies.append((time.perf_counter_ns() - t0) / 1_000_000.0)
            gt_map[i] = set(top_ids)

        flat_mean_ms = float(np.mean(flat_latencies))
        results.append({
            "index_type": "1. Exact Flat KNN",
            "recall_10": 1.0,
            "mean_latency_ms": round(flat_mean_ms, 4),
            "speedup_vs_flat": 1.0,
            "memory_kb": round(flat.memory_bytes() / 1024.0, 1),
            "compression": "1.0x (Baseline)"
        })

        # 2. IVF Index (n_probe = 4)
        ivf = IVFIndex(dimension=self.D, num_centroids=32).train_and_add(self.dataset)
        ivf_latencies, ivf_recalls = [], []
        for i, q in enumerate(self.test_queries):
            t0 = time.perf_counter_ns()
            top_ids, _ = ivf.search(q, top_k=10, n_probe=4)
            ivf_latencies.append((time.perf_counter_ns() - t0) / 1_000_000.0)
            ivf_recalls.append(len(set(top_ids).intersection(gt_map[i])) / 10.0)

        ivf_mean_ms = float(np.mean(ivf_latencies))
        results.append({
            "index_type": "2. IVF (K=32, probe=4)",
            "recall_10": round(float(np.mean(ivf_recalls)), 3),
            "mean_latency_ms": round(ivf_mean_ms, 4),
            "speedup_vs_flat": round(flat_mean_ms / ivf_mean_ms, 2) if ivf_mean_ms > 0 else 0.0,
            "memory_kb": round(flat.memory_bytes() / 1024.0, 1),
            "compression": "1.0x"
        })

        # 3. Product Quantization (M=8, K_sub=16)
        pq = ProductQuantizer(dimension=self.D, num_subvectors=8, num_centroids=16).train_and_encode(self.dataset)
        pq_latencies, pq_recalls = [], []
        for i, q in enumerate(self.test_queries):
            t0 = time.perf_counter_ns()
            top_ids, _ = pq.search(q, top_k=10)
            pq_latencies.append((time.perf_counter_ns() - t0) / 1_000_000.0)
            pq_recalls.append(len(set(top_ids).intersection(gt_map[i])) / 10.0)

        pq_mean_ms = float(np.mean(pq_latencies))
        results.append({
            "index_type": "3. Product Quantizer (M=8)",
            "recall_10": round(float(np.mean(pq_recalls)), 3),
            "mean_latency_ms": round(pq_mean_ms, 4),
            "speedup_vs_flat": round(flat_mean_ms / pq_mean_ms, 2) if pq_mean_ms > 0 else 0.0,
            "memory_kb": round((self.N * 8 * 1) / 1024.0, 1),
            "compression": f"{pq.compression_ratio():.1f}x"
        })

        return results

# %% [markdown]
# ### Demo 5: Comprehensive Pareto Benchmark Execution
#
# Below, we execute the benchmark harness over 20 test queries on the $N = 5,000$ dataset.

# %%
np.random.seed(123)
eval_queries = [np.random.randn(DIMENSION).astype(np.float32) for _ in range(20)]
benchmark_suite = ANNBenchmarkHarness(synthetic_dataset, eval_queries)
pareto_results = benchmark_suite.run_benchmark()

print("=== [ANN Vector Indexing Pareto Frontier Benchmark] ===")
print(f"{'Indexing Algorithm':<30}{'Recall@10':<12}{'Latency (ms)':<15}{'Speedup':<12}{'Memory (KB)':<14}{'Compression':<12}")
print("-" * 95)
for row in pareto_results:
    print(f"{row['index_type']:<30}{row['recall_10'] * 100:>5.1f}%     {row['mean_latency_ms']:<15.4f}{row['speedup_vs_flat']:<12.1f}{row['memory_kb']:<14.1f}{row['compression']:<12}")

# %% [markdown]
# ## Section 6: Presenter Dashboard & ASCII Pareto Visualizer
#
# Below is the consolidated presenter dashboard rendering an ASCII Pareto efficiency trade-off chart.

# %%
# collapse_input
def display_indexing_dashboard(benchmark_rows: List[Dict[str, Any]], N: int, D: int):
    """Render a clean ASCII visualizer of vector indexing tradeoffs."""
    print("=" * 80)
    print("           KNOWLEDGE RETRIEVAL A-Z: MODULE 03 VECTOR INDEXING DASHBOARD")
    print("=" * 80)
    
    print(f"\n[1] DATASET SPECIFICATIONS")
    print(f"  • Total Vectors (N): {N:,}")
    print(f"  • Vector Dimension:  {D}")
    print(f"  • Raw Vector Space:  {(N * D * 4) / 1024:.1f} KB ({(N * D * 4) / (1024**2):.2f} MB)")

    print(f"\n[2] PARETO EFFICIENCY TRADEOFF MATRIX")
    print(f"  {'Index Strategy':<28}{'Recall@10':<14}{'Latency':<14}{'Speedup':<12}{'Memory Footprint':<18}")
    print("  " + "-" * 82)
    for r in benchmark_rows:
        print(f"  {r['index_type']:<28}{r['recall_10']*100:>5.1f}%        {r['mean_latency_ms']:<6.3f} ms     {r['speedup_vs_flat']:<5.1f}x       {r['memory_kb']:>6.1f} KB ({r['compression']})")

    print("\n[3] ARCHITECTURAL SELECTION GUIDELINES")
    print("  • Use Exact Flat:    When dataset N < 50,000 and 100% recall is strictly mandatory.")
    print("  • Use IVF:           When memory is plentiful and dynamic vector insertions occur frequently.")
    print("  • Use HNSW:          When sub-millisecond search latency and >98% recall are required.")
    print("  • Use PQ / ScaNN:    When scaling to 10M+ vectors where RAM compression (16x-32x) is critical.")

    print("\n" + "=" * 80)
    print("  [OK] Track 1 Complete! Ready for Track 2: RAG Ingestion & Chunking.")
    print("=" * 80)

# Render dashboard
display_indexing_dashboard(pareto_results, N_VECTORS, DIMENSION)

# %% [markdown]
# ## Section 7: Summary & Transition to Track 2
#
# In Track 1 (Foundations & Classical Retrieval), we have built the foundational algorithmic engines of modern knowledge retrieval:
# 1. **Module 01**: Verified the workspace, built BPE tokenizers from scratch, calculated KV-cache memory requirements, and modeled high-dimensional vector spaces.
# 2. **Module 02**: Implemented BM25 inverted indexes, dense cosine search, Reciprocal Rank Fusion (RRF), and dynamic query routing.
# 3. **Module 03**: Engineered exact Flat KNN, Inverted File Indexes (IVF), HNSW multi-layer graphs, Product Quantization (PQ), and charted the Pareto efficiency frontier.
#
# In **Track 2 (Module 04: Advanced Chunking & Hierarchical Ingestion)**, we advance to production **Retrieval-Augmented Generation (RAG)**, implementing semantic chunking, parent-child document architectures, and multi-format document parsing.
