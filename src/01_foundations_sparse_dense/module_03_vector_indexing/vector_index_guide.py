# %% [markdown]
# # Module 03: Vector Indexing & Algorithms (FAISS GPU / HNSW / IVF / PQ)
#
# Welcome to **Module 03** of the Knowledge Retrieval A-Z masterclass.
# When vector databases scale to millions or billions of embedding vectors, computing exact brute-force cosine similarity ($O(N)$) becomes a prohibitive latency bottleneck for real-time RAG applications.
#
# **Approximate Nearest Neighbor (ANN)** indexing structures trade a minor fraction of recall for orders-of-magnitude faster queries and substantial memory compression.
#
# In this module, we construct and master production vector indexing utilizing the industry-standard **FAISS (Facebook AI Similarity Search)** library with **GPU Acceleration**:
# 1. **FAISS Exact Flat KNN (GPU-Accelerated)**: Baseline index providing 100% recall ground truth on CUDA GPU.
# 2. **FAISS Inverted File Index (IVF / Voronoi Space Partitioning)**: K-Means clustering, inverted posting lists, and multi-probe ($n_{\text{probe}}$) retrieval.
# 3. **FAISS Hierarchical Navigable Small World (HNSW)**: Multi-layer skip-graph hierarchy, greedy upper-layer routing, and $efSearch$ beam search on base graphs.
# 4. **FAISS Product Quantization (PQ) & Asymmetric Distance Computation (ADC)**: Subspace vector quantization, codebook construction, lookup tables, and $64\times$ memory reduction.
# 5. **Systematic ANN Benchmark Suite & Pareto Frontier**: Comparative evaluation of Recall@K, latency speedups, and memory footprints across GPU and CPU.
# 6. **Presenter Visualizer & Dashboard (`# collapse_input`)**: Auto-collapsing ASCII Pareto efficiency visualizer.
#
# ---
#
# ```mermaid
# graph TD
#     subgraph FlatSpace ["1. FAISS Exact Flat (faiss.IndexFlatIP / GPU)"]
#         F1["GPU Brute Force Dot Product over All N Vectors (RTX 4080)"]
#     end
#
#     subgraph IVFSpace ["2. FAISS Inverted File Index (faiss.IndexIVFFlat / GPU)"]
#         C["K Centroids (Voronoi Cells)"] --> P["Probe n_probe Closest Centroids"]
#         P --> L["Scan Only Vectors in Target Lists"]
#     end
#
#     subgraph HNSWSpace ["3. FAISS HNSW Graph (faiss.IndexHNSWFlat)"]
#         L2["Layer 2: Sparse Highway Skip-Connections"] --> L1["Layer 1: Medium-Range Clusters"]
#         L1 --> L0["Layer 0: Dense Local Proximity Graph (efSearch)"]
#     end
#
#     subgraph PQSpace ["4. FAISS Product Quantization (faiss.IndexPQ)"]
#         D["Vector R^D"] --> M["M Sub-Vectors (D/M)"]
#         M --> CB["K_sub Centroids per Subspace"]
#         CB --> ADC["Asymmetric Distance Table (64x Memory Reduction)"]
#     end
# ```
#
# ---

# %%
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch

# Hardware Accelerator Detection & FAISS GPU Initialization
def init_faiss_gpu_resources() -> Tuple[Optional[faiss.StandardGpuResources], str]:
    """Initialize FAISS GPU resources if CUDA is available."""
    if torch.cuda.is_available():
        gpu_res = faiss.StandardGpuResources()
        device_name = torch.cuda.get_device_name(0)
        status_msg = f"CUDA GPU: {device_name} (FAISS GPU Enabled)"
    else:
        gpu_res = None
        status_msg = "CPU (FAISS CPU Mode)"
    return gpu_res, status_msg

GPU_RESOURCES, COMPUTE_DEVICE_STATUS = init_faiss_gpu_resources()
print(f"[INFO] Vector Indexing Compute Hardware: {COMPUTE_DEVICE_STATUS}")

# %% [markdown]
# ## Section 1: FAISS Exact Flat KNN Search with GPU Acceleration
#
# The **FAISS IndexFlatIP** (Inner Product) index stores uncompressed, normalized embedding vectors $\mathbf{X} \in \mathbb{R}^{N \times D}$.
# When transferred to GPU (`faiss.index_cpu_to_gpu`), queries execute via massive parallel CUDA kernels:
#
# - **Time Complexity:** $O(N \cdot D)$ per query.
# - **Recall:** Strictly $100\%$ (Ground Truth reference for evaluating ANN approximations).
# - **Memory Footprint:** $N \times D \times 4 \text{ bytes (FP32)}$.

# %%
class FAISSFlatEngine:
    """Production Exact Flat KNN search engine utilizing FAISS with GPU acceleration."""

    def __init__(self, dimension: int, use_gpu: bool = True):
        self.dimension = dimension
        self.use_gpu = use_gpu and (GPU_RESOURCES is not None)
        
        # Base CPU Index (Inner Product for Cosine Similarity on normalized vectors)
        self.cpu_index = faiss.IndexFlatIP(dimension)
        
        # Transfer to GPU if available
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(GPU_RESOURCES, 0, self.cpu_index)
        else:
            self.index = self.cpu_index

    def add(self, vectors: np.ndarray) -> "FAISSFlatEngine":
        """Add and normalize a batch of vectors (N x D)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = (vectors / norms).astype(np.float32)
        self.index.add(normalized)
        return self

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-K closest vectors by cosine similarity. Returns (indices, scores)."""
        q_norm = np.linalg.norm(query_vector)
        q = (query_vector / q_norm).astype(np.float32).reshape(1, -1) if q_norm > 0 else query_vector.astype(np.float32).reshape(1, -1)
        
        scores, indices = self.index.search(q, top_k)
        return indices[0], scores[0]

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

# %% [markdown]
# ### Demo 1: Comprehensive FAISS Flat Index Ground Truth Demonstration
#
# Below, we generate a synthetic dataset of $N = 10,000$ vectors ($D = 128$), index them in `FAISSFlatEngine` on GPU, and execute an exact search.

# %%
np.random.seed(42)
N_VECTORS = 10000
DIMENSION = 128

# Generate synthetic vector collection
synthetic_dataset = np.random.randn(N_VECTORS, DIMENSION).astype(np.float32)
faiss_flat = FAISSFlatEngine(dimension=DIMENSION, use_gpu=True)
faiss_flat.add(synthetic_dataset)

# Generate a sample query vector
sample_query = np.random.randn(DIMENSION).astype(np.float32)

# Execute exact search
start_t = time.perf_counter_ns()
gt_indices, gt_scores = faiss_flat.search(sample_query, top_k=5)
flat_latency_us = (time.perf_counter_ns() - start_t) / 1_000.0

print("=== [FAISS Exact Flat KNN Index Status] ===")
print(f"Indexed Vectors (N): {faiss_flat.ntotal:,} | Dimension (D): {faiss_flat.dimension}")
print(f"Index Acceleration:  {'GPU (CUDA RTX 4080)' if faiss_flat.use_gpu else 'CPU'}")
print(f"Memory Footprint:    {(N_VECTORS * DIMENSION * 4) / 1024:.2f} KB ({(N_VECTORS * DIMENSION * 4) / (1024**2):.3f} MB)")
print(f"Exact Search Latency: {flat_latency_us:.2f} microseconds ({flat_latency_us / 1000.0:.4f} ms)")
print("\nTop 5 Exact Nearest Neighbors (Ground Truth):")
for rank, (idx, score) in enumerate(zip(gt_indices, gt_scores), 1):
    print(f"  [{rank}] Vector ID: {idx:<6} | Cosine Similarity: {score:.5f}")

# %% [markdown]
# ## Section 2: FAISS Inverted File Index (IVF / Voronoi Partitioning) with GPU
#
# The **FAISS IndexIVFFlat** index partitions the vector space into $n_{\text{list}}$ **Voronoi cells** using K-Means clustering.
#
# ### Indexing & Retrieval Mechanism
# 1. **Training Phase:** Run K-Means to compute $n_{\text{list}}$ cluster centroids.
# 2. **Posting Assignment:** Quantize each vector to its nearest centroid list.
# 3. **Query Search:** Given query $\mathbf{q}$, probe only the top $n_{\text{probe}}$ closest Voronoi centroids.

# %%
class FAISSIVFEngine:
    """Production Inverted File Index (IVF) utilizing FAISS with GPU acceleration."""

    def __init__(self, dimension: int, nlist: int = 64, use_gpu: bool = True):
        self.dimension = dimension
        self.nlist = nlist
        self.use_gpu = use_gpu and (GPU_RESOURCES is not None)
        
        # Quantizer: Inner Product
        self.quantizer = faiss.IndexFlatIP(dimension)
        self.cpu_index = faiss.IndexIVFFlat(self.quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(GPU_RESOURCES, 0, self.cpu_index)
        else:
            self.index = self.cpu_index

    def train_and_add(self, vectors: np.ndarray) -> "FAISSIVFEngine":
        """Train Voronoi centroids and populate inverted posting lists."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = (vectors / norms).astype(np.float32)
        
        self.index.train(normalized)
        self.index.add(normalized)
        return self

    def search(self, query_vector: np.ndarray, top_k: int = 10, nprobe: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-K vectors across the top nprobe closest Voronoi centroids."""
        q_norm = np.linalg.norm(query_vector)
        q = (query_vector / q_norm).astype(np.float32).reshape(1, -1) if q_norm > 0 else query_vector.astype(np.float32).reshape(1, -1)
        
        self.index.nprobe = nprobe
        scores, indices = self.index.search(q, top_k)
        return indices[0], scores[0]

# %% [markdown]
# ### Demo 2: Comprehensive FAISS IVF Index Demonstration & n_probe Sweep
#
# Below, we train `FAISSIVFEngine` on GPU ($n_{\text{list}} = 64$) and evaluate Recall@10 across varying $n_{\text{probe}} \in [1, 2, 4, 8, 16, 32, 64]$.

# %%
faiss_ivf = FAISSIVFEngine(dimension=DIMENSION, nlist=64, use_gpu=True)
faiss_ivf.train_and_add(synthetic_dataset)

# Ground truth top-10 IDs from Flat Index
gt_10_ids, _ = faiss_flat.search(sample_query, top_k=10)
gt_10_set = set(gt_10_ids)

print("=== [FAISS IVF Multi-Probe Parameter Sweep (GPU)] ===")
print(f"Total Voronoi Centroids (nlist): {faiss_ivf.nlist}")
print(f"  {'n_probe':<10}{'Latency (us)':<16}{'Recall@10':<12}")
print("  " + "-" * 38)

for probe in [1, 2, 4, 8, 16, 32, 64]:
    t0 = time.perf_counter_ns()
    ivf_ids, _ = faiss_ivf.search(sample_query, top_k=10, nprobe=probe)
    t_us = (time.perf_counter_ns() - t0) / 1000.0
    
    recall_10 = len(set(ivf_ids).intersection(gt_10_set)) / 10.0
    print(f"  {probe:<10}{t_us:<16.2f}{recall_10 * 100:>5.1f}%")

# %% [markdown]
# ## Section 3: FAISS Hierarchical Navigable Small World (HNSW)
#
# **FAISS IndexHNSWFlat** constructs a multi-layer proximity graph with logarithmic query complexity ($O(\log N)$) and high recall ($>98\%$).

# %%
class FAISSHNSWEngine:
    """Production HNSW graph index utilizing FAISS."""

    def __init__(self, dimension: int, M: int = 32, ef_construction: int = 40):
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        
        # HNSW Index
        self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction

    def add(self, vectors: np.ndarray) -> "FAISSHNSWEngine":
        """Add and index vectors into the HNSW graph."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = (vectors / norms).astype(np.float32)
        self.index.add(normalized)
        return self

    def search(self, query_vector: np.ndarray, top_k: int = 10, ef_search: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-K nearest neighbors with configurable efSearch beam width."""
        q_norm = np.linalg.norm(query_vector)
        q = (query_vector / q_norm).astype(np.float32).reshape(1, -1) if q_norm > 0 else query_vector.astype(np.float32).reshape(1, -1)
        
        self.index.hnsw.efSearch = ef_search
        scores, indices = self.index.search(q, top_k)
        return indices[0], scores[0]

# %% [markdown]
# ### Demo 3: Comprehensive FAISS HNSW Graph Demonstration
#
# Below, we build `FAISSHNSWEngine` and evaluate Recall@10 across varying beam widths $efSearch \in [8, 16, 32, 64, 128]$.

# %%
faiss_hnsw = FAISSHNSWEngine(dimension=DIMENSION, M=32, ef_construction=40)
faiss_hnsw.add(synthetic_dataset)

print("=== [FAISS HNSW efSearch Parameter Sweep] ===")
print(f"Total Nodes Indexed: {faiss_hnsw.index.ntotal:,} | Node Degree M: {faiss_hnsw.M}")
print(f"  {'efSearch':<12}{'Latency (us)':<16}{'Recall@10':<12}")
print("  " + "-" * 40)

for ef in [8, 16, 32, 64, 128]:
    t0 = time.perf_counter_ns()
    hnsw_ids, _ = faiss_hnsw.search(sample_query, top_k=10, ef_search=ef)
    t_us = (time.perf_counter_ns() - t0) / 1000.0
    recall = len(set(hnsw_ids).intersection(gt_10_set)) / 10.0
    print(f"  {ef:<12}{t_us:<16.2f}{recall * 100:>5.1f}%")

# %% [markdown]
# ## Section 4: FAISS Product Quantization (PQ) Compression Mechanics
#
# **FAISS IndexPQ** decomposes continuous $D$-dimensional vectors into $M$ discrete 1-byte codebook indices, compressing $D = 128$ vectors down to $M = 8$ bytes (**$64\times$ memory reduction**).

# %%
class FAISSPQEngine:
    """Production Product Quantization (PQ) index utilizing FAISS."""

    def __init__(self, dimension: int = 128, M: int = 8, nbits: int = 8):
        self.dimension = dimension
        self.M = M
        self.nbits = nbits
        self.index = faiss.IndexPQ(dimension, M, nbits, faiss.METRIC_INNER_PRODUCT)

    def train_and_add(self, vectors: np.ndarray) -> "FAISSPQEngine":
        """Train codebooks and quantize dataset vectors."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = (vectors / norms).astype(np.float32)
        
        self.index.train(normalized)
        self.index.add(normalized)
        return self

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Execute Asymmetric Distance Computation (ADC) query."""
        q_norm = np.linalg.norm(query_vector)
        q = (query_vector / q_norm).astype(np.float32).reshape(1, -1) if q_norm > 0 else query_vector.astype(np.float32).reshape(1, -1)
        
        scores, indices = self.index.search(q, top_k)
        return indices[0], scores[0]

    def compression_ratio(self) -> float:
        """Calculate memory compression ratio: raw FP32 / quantized codes."""
        raw_bytes = self.dimension * 4
        quant_bytes = self.M * 1
        return raw_bytes / quant_bytes

# %% [markdown]
# ### Demo 4: Comprehensive FAISS PQ Demonstration
#
# Below, we train `FAISSPQEngine` and evaluate memory savings and search latency.

# %%
faiss_pq = FAISSPQEngine(dimension=DIMENSION, M=8, nbits=8)
faiss_pq.train_and_add(synthetic_dataset)

raw_mem_kb = (N_VECTORS * DIMENSION * 4) / 1024
pq_mem_kb = (N_VECTORS * 8 * 1) / 1024
comp_ratio = faiss_pq.compression_ratio()

t0 = time.perf_counter_ns()
pq_ids, pq_scores = faiss_pq.search(sample_query, top_k=10)
pq_t_us = (time.perf_counter_ns() - t0) / 1000.0
pq_recall = len(set(pq_ids).intersection(gt_10_set)) / 10.0

print("=== [FAISS Product Quantization (PQ) Summary] ===")
print(f"Sub-Vectors (M):          {faiss_pq.M} (Sub-Dimension: {DIMENSION // faiss_pq.M} dims/sub-vector)")
print(f"Raw FP32 Memory:          {raw_mem_kb:.1f} KB")
print(f"Quantized Memory:         {pq_mem_kb:.1f} KB")
print(f"Memory Compression Ratio: {comp_ratio:.1f}x reduction")
print(f"ADC Query Latency:        {pq_t_us:.2f} us")
print(f"ADC Recall@10:            {pq_recall * 100:.1f}%")

# %% [markdown]
# ## Section 5: Systematic ANN Benchmark Suite & The Pareto Frontier
#
# We execute a comparative benchmark measuring the **Pareto Tradeoffs** across all FAISS indexing strategies.

# %%
class FAISSBenchmarkHarness:
    """Unified benchmarking harness evaluating FAISS Flat GPU vs IVF GPU vs HNSW vs PQ."""

    def __init__(self, dataset: np.ndarray, test_queries: List[np.ndarray]):
        self.dataset = dataset
        self.test_queries = test_queries
        self.N, self.D = dataset.shape

    def run_benchmark(self) -> List[Dict[str, Any]]:
        """Run all indexing engines and compute the Pareto frontier."""
        results = []

        # 1. Ground Truth Baseline (FAISS Flat GPU)
        flat_gpu = FAISSFlatEngine(dimension=self.D, use_gpu=True).add(self.dataset)
        
        flat_latencies = []
        gt_map = {}
        for i, q in enumerate(self.test_queries):
            t0 = time.perf_counter_ns()
            top_ids, _ = flat_gpu.search(q, top_k=10)
            flat_latencies.append((time.perf_counter_ns() - t0) / 1_000_000.0)
            gt_map[i] = set(top_ids)

        flat_mean_ms = float(np.mean(flat_latencies))
        results.append({
            "index_type": "1. FAISS Flat (CUDA GPU)",
            "recall_10": 1.0,
            "mean_latency_ms": round(flat_mean_ms, 4),
            "speedup_vs_flat": 1.0,
            "memory_kb": round((self.N * self.D * 4) / 1024.0, 1),
            "compression": "1.0x (Baseline)"
        })

        # 2. FAISS IVF (CUDA GPU)
        nlist = min(64, max(4, self.N // 16))
        ivf_gpu = FAISSIVFEngine(dimension=self.D, nlist=nlist, use_gpu=True).train_and_add(self.dataset)
        ivf_latencies, ivf_recalls = [], []
        for i, q in enumerate(self.test_queries):
            t0 = time.perf_counter_ns()
            top_ids, _ = ivf_gpu.search(q, top_k=10, nprobe=8)
            ivf_latencies.append((time.perf_counter_ns() - t0) / 1_000_000.0)
            ivf_recalls.append(len(set(top_ids).intersection(gt_map[i])) / 10.0)

        ivf_mean_ms = float(np.mean(ivf_latencies))
        results.append({
            "index_type": "2. FAISS IVF-64 (CUDA GPU, probe=8)",
            "recall_10": round(float(np.mean(ivf_recalls)), 3),
            "mean_latency_ms": round(ivf_mean_ms, 4),
            "speedup_vs_flat": round(flat_mean_ms / ivf_mean_ms, 2) if ivf_mean_ms > 0 else 0.0,
            "memory_kb": round((self.N * self.D * 4) / 1024.0, 1),
            "compression": "1.0x"
        })

        # 3. FAISS HNSW (CPU, efSearch=64)
        hnsw_engine = FAISSHNSWEngine(dimension=self.D, M=32, ef_construction=40).add(self.dataset)
        hnsw_latencies, hnsw_recalls = [], []
        for i, q in enumerate(self.test_queries):
            t0 = time.perf_counter_ns()
            top_ids, _ = hnsw_engine.search(q, top_k=10, ef_search=64)
            hnsw_latencies.append((time.perf_counter_ns() - t0) / 1_000_000.0)
            hnsw_recalls.append(len(set(top_ids).intersection(gt_map[i])) / 10.0)

        hnsw_mean_ms = float(np.mean(hnsw_latencies))
        results.append({
            "index_type": "3. FAISS HNSW-32 (efSearch=64)",
            "recall_10": round(float(np.mean(hnsw_recalls)), 3),
            "mean_latency_ms": round(hnsw_mean_ms, 4),
            "speedup_vs_flat": round(flat_mean_ms / hnsw_mean_ms, 2) if hnsw_mean_ms > 0 else 0.0,
            "memory_kb": round((self.N * (self.D * 4 + 32 * 4)) / 1024.0, 1),
            "compression": "0.8x"
        })

        # 4. FAISS Product Quantization (M=8, nbits=8)
        pq_engine = FAISSPQEngine(dimension=self.D, M=8, nbits=8).train_and_add(self.dataset)
        pq_latencies, pq_recalls = [], []
        for i, q in enumerate(self.test_queries):
            t0 = time.perf_counter_ns()
            top_ids, _ = pq_engine.search(q, top_k=10)
            pq_latencies.append((time.perf_counter_ns() - t0) / 1_000_000.0)
            pq_recalls.append(len(set(top_ids).intersection(gt_map[i])) / 10.0)

        pq_mean_ms = float(np.mean(pq_latencies))
        results.append({
            "index_type": "4. FAISS PQ-8 (64x Compression)",
            "recall_10": round(float(np.mean(pq_recalls)), 3),
            "mean_latency_ms": round(pq_mean_ms, 4),
            "speedup_vs_flat": round(flat_mean_ms / pq_mean_ms, 2) if pq_mean_ms > 0 else 0.0,
            "memory_kb": round((self.N * 8 * 1) / 1024.0, 1),
            "compression": f"{pq_engine.compression_ratio():.1f}x"
        })

        return results

# %% [markdown]
# ### Demo 5: Comprehensive Pareto Benchmark Execution
#
# Below, we execute the benchmark harness over 20 test queries on the $N = 10,000$ dataset.

# %%
np.random.seed(123)
eval_queries = [np.random.randn(DIMENSION).astype(np.float32) for _ in range(20)]
benchmark_suite = FAISSBenchmarkHarness(synthetic_dataset, eval_queries)
pareto_results = benchmark_suite.run_benchmark()

print("=== [FAISS ANN Vector Indexing Pareto Frontier Benchmark] ===")
print(f"{'Indexing Algorithm':<36}{'Recall@10':<12}{'Latency (ms)':<15}{'Speedup':<12}{'Memory (KB)':<14}{'Compression':<12}")
print("-" * 101)
for row in pareto_results:
    print(f"{row['index_type']:<36}{row['recall_10'] * 100:>5.1f}%     {row['mean_latency_ms']:<15.4f}{row['speedup_vs_flat']:<12.1f}{row['memory_kb']:<14.1f}{row['compression']:<12}")

# %% [markdown]
# ## Section 6: Presenter Dashboard & ASCII Pareto Visualizer
#
# Below is the consolidated presenter dashboard rendering an ASCII Pareto efficiency trade-off chart.

# %%
# collapse_input
def display_indexing_dashboard(benchmark_rows: List[Dict[str, Any]], N: int, D: int, device_str: str):
    """Render a clean ASCII visualizer of vector indexing tradeoffs."""
    print("=" * 80)
    print("           KNOWLEDGE RETRIEVAL A-Z: MODULE 03 VECTOR INDEXING DASHBOARD")
    print("=" * 80)
    
    print(f"\n[1] HARDWARE ACCELERATION & DATASET")
    print(f"  • Compute Device:    {device_str}")
    print(f"  • Total Vectors (N): {N:,}")
    print(f"  • Vector Dimension:  {D}")
    print(f"  • Raw Vector Space:  {(N * D * 4) / 1024:.1f} KB ({(N * D * 4) / (1024**2):.2f} MB)")

    print(f"\n[2] PARETO EFFICIENCY TRADEOFF MATRIX")
    print(f"  {'Index Strategy':<34}{'Recall@10':<12}{'Latency':<14}{'Speedup':<10}{'Memory Footprint':<18}")
    print("  " + "-" * 88)
    for r in benchmark_rows:
        print(f"  {r['index_type']:<34}{r['recall_10']*100:>5.1f}%      {r['mean_latency_ms']:<6.3f} ms   {r['speedup_vs_flat']:<5.1f}x     {r['memory_kb']:>6.1f} KB ({r['compression']})")

    print("\n[3] ARCHITECTURAL SELECTION GUIDELINES")
    print("  • Use FAISS Flat GPU:  When dataset N < 1,000,000 and 100% recall is mandatory on GPU.")
    print("  • Use FAISS IVF GPU:   When sub-millisecond query speed and dynamic updates are needed.")
    print("  • Use FAISS HNSW:      When >98% recall and ultra-fast CPU routing are required.")
    print("  • Use FAISS PQ:        When scaling to 10M+ vectors where 64x RAM compression is critical.")

    print("\n" + "=" * 80)
    print("  [OK] Track 1 Complete! Ready for Track 2: RAG Ingestion & Chunking.")
    print("=" * 80)

# Render dashboard
display_indexing_dashboard(pareto_results, N_VECTORS, DIMENSION, COMPUTE_DEVICE_STATUS)

# %% [markdown]
# ## Section 7: Summary & Transition to Track 2
#
# In Track 1 (Foundations & Classical Retrieval), we have built the foundational algorithmic engines of modern knowledge retrieval:
# 1. **Module 01**: Verified the workspace, built BPE tokenizers from scratch, calculated KV-cache memory requirements, and modeled high-dimensional vector spaces.
# 2. **Module 02**: Implemented production sparse search with `rank_bm25`, GPU-accelerated dense tensor search with PyTorch CUDA, and Reciprocal Rank Fusion (RRF).
# 3. **Module 03**: Engineered production vector indexing with **FAISS GPU** (`IndexFlatIP`, `IndexIVFFlat`, `IndexHNSWFlat`, `IndexPQ`) and charted the Pareto efficiency frontier.
#
# In **Track 2 (Module 04: Advanced Chunking & Hierarchical Ingestion)**, we advance to production **Retrieval-Augmented Generation (RAG)**, implementing semantic chunking, parent-child document architectures, and multi-format document parsing.
