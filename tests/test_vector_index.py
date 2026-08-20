import importlib.util
import math
import pathlib
import numpy as np
import pytest

def _load_vector_index_guide():
    file_path = pathlib.Path(__file__).parent.parent / "src/01_foundations_sparse_dense/module_03_vector_indexing/vector_index_guide.py"
    spec = importlib.util.spec_from_file_location("vector_index_guide", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

vector_index_guide = _load_vector_index_guide()
FlatIndex = vector_index_guide.FlatIndex
IVFIndex = vector_index_guide.IVFIndex
HNSWIndex = vector_index_guide.HNSWIndex
ProductQuantizer = vector_index_guide.ProductQuantizer
ANNBenchmarkHarness = vector_index_guide.ANNBenchmarkHarness


def test_flat_index_exact_search():
    np.random.seed(42)
    vectors = np.random.randn(50, 16).astype(np.float32)
    
    flat = FlatIndex(dimension=16)
    flat.add(vectors)

    assert flat.num_vectors == 50
    assert flat.memory_bytes() == 50 * 16 * 4

    # Query with exact first vector -> top match must be vector 0 with score ~1.0
    query = vectors[0]
    indices, scores = flat.search(query, top_k=5)
    
    assert len(indices) == 5
    assert indices[0] == 0
    assert math.isclose(scores[0], 1.0, rel_tol=1e-5)


def test_ivf_index_clustering_and_search():
    np.random.seed(42)
    vectors = np.random.randn(100, 16).astype(np.float32)
    
    ivf = IVFIndex(dimension=16, num_centroids=8, max_kmeans_iters=5)
    ivf.train_and_add(vectors)

    assert ivf.centroids is not None
    assert ivf.centroids.shape == (8, 16)
    assert ivf.num_vectors == 100

    # Total vectors across all inverted lists must equal N
    total_assigned = sum(len(lst) for lst in ivf.inverted_lists.values())
    assert total_assigned == 100

    # Search with n_probe=8 (full scan of all centroids)
    query = vectors[0]
    indices, scores = ivf.search(query, top_k=5, n_probe=8)
    assert len(indices) == 5
    assert indices[0] == 0


def test_hnsw_index_graph_construction_and_search():
    np.random.seed(42)
    vectors = np.random.randn(80, 16).astype(np.float32)

    hnsw = HNSWIndex(dimension=16, M=4, ef_construction=8)
    hnsw.add(vectors)

    assert hnsw.num_nodes == 80
    assert hnsw.max_layer >= 0
    assert len(hnsw.graphs[0]) == 80  # Layer 0 must contain all nodes

    # Search
    query = vectors[0]
    indices, scores = hnsw.search(query, top_k=5, ef_search=16)
    assert len(indices) == 5
    assert scores[0] > 0.5


def test_product_quantizer_compression_and_adc():
    np.random.seed(42)
    # D = 32, M = 4, sub_dim = 8, K_sub = 8
    vectors = np.random.randn(100, 32).astype(np.float32)

    pq = ProductQuantizer(dimension=32, num_subvectors=4, num_centroids=8)
    pq.train_and_encode(vectors)

    assert pq.codebooks is not None
    assert pq.codebooks.shape == (4, 8, 8)
    assert pq.codes is not None
    assert pq.codes.shape == (100, 4)

    # Theoretical compression: (32 * 4) / (4 * 1) = 128 / 4 = 32.0x
    assert math.isclose(pq.compression_ratio(), 32.0)

    # Search ADC
    query = vectors[0]
    indices, dists = pq.search(query, top_k=5)
    assert len(indices) == 5
    assert len(dists) == 5
    assert indices[0] == 0  # Nearest quantized code should match query vector
    assert dists[0] >= 0.0  # Euclidean distance must be non-negative


def test_ann_benchmark_harness():
    np.random.seed(42)
    dataset = np.random.randn(50, 16).astype(np.float32)
    queries = [np.random.randn(16).astype(np.float32) for _ in range(3)]

    harness = ANNBenchmarkHarness(dataset, queries)
    results = harness.run_benchmark()

    assert len(results) == 3
    assert results[0]["index_type"] == "1. Exact Flat KNN"
    assert results[0]["recall_10"] == 1.0
    assert results[1]["recall_10"] >= 0.0
    assert results[2]["recall_10"] >= 0.0
