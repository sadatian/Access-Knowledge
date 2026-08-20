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
FAISSFlatEngine = vector_index_guide.FAISSFlatEngine
FAISSIVFEngine = vector_index_guide.FAISSIVFEngine
FAISSHNSWEngine = vector_index_guide.FAISSHNSWEngine
FAISSPQEngine = vector_index_guide.FAISSPQEngine
FAISSBenchmarkHarness = vector_index_guide.FAISSBenchmarkHarness


def test_faiss_flat_engine():
    np.random.seed(42)
    vectors = np.random.randn(50, 16).astype(np.float32)
    
    flat = FAISSFlatEngine(dimension=16, use_gpu=True)
    flat.add(vectors)

    assert flat.ntotal == 50

    # Query with exact first vector -> top match must be vector 0 with score ~1.0
    query = vectors[0]
    indices, scores = flat.search(query, top_k=5)
    
    assert len(indices) == 5
    assert indices[0] == 0
    assert math.isclose(scores[0], 1.0, rel_tol=1e-4)


def test_faiss_ivf_engine():
    np.random.seed(42)
    vectors = np.random.randn(100, 16).astype(np.float32)
    
    ivf = FAISSIVFEngine(dimension=16, nlist=8, use_gpu=True)
    ivf.train_and_add(vectors)

    assert ivf.index.ntotal == 100

    query = vectors[0]
    indices, scores = ivf.search(query, top_k=5, nprobe=8)
    assert len(indices) == 5
    assert indices[0] == 0


def test_faiss_hnsw_engine():
    np.random.seed(42)
    vectors = np.random.randn(80, 16).astype(np.float32)

    hnsw = FAISSHNSWEngine(dimension=16, M=8, ef_construction=16)
    hnsw.add(vectors)

    assert hnsw.index.ntotal == 80

    query = vectors[0]
    indices, scores = hnsw.search(query, top_k=5, ef_search=32)
    assert len(indices) == 5
    assert scores[0] > 0.8


def test_faiss_pq_engine():
    np.random.seed(42)
    # N = 300 (>= 256 for nbits=8), D = 32, M = 4, nbits = 8
    vectors = np.random.randn(300, 32).astype(np.float32)

    pq = FAISSPQEngine(dimension=32, M=4, nbits=8)
    pq.train_and_add(vectors)

    assert pq.index.ntotal == 300
    assert math.isclose(pq.compression_ratio(), 32.0)

    query = vectors[0]
    indices, scores = pq.search(query, top_k=5)
    assert len(indices) == 5
    assert len(scores) == 5


def test_faiss_benchmark_harness():
    np.random.seed(42)
    dataset = np.random.randn(300, 16).astype(np.float32)
    queries = [np.random.randn(16).astype(np.float32) for _ in range(3)]

    harness = FAISSBenchmarkHarness(dataset, queries)
    results = harness.run_benchmark()

    assert len(results) == 4
    assert "FAISS Flat" in results[0]["index_type"]
    assert results[0]["recall_10"] == 1.0
    for r in results:
        assert r["recall_10"] >= 0.0
        assert r["mean_latency_ms"] >= 0.0
