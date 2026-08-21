import importlib.util
import math
import pathlib
import numpy as np
import pytest

# Helper to load modules from directory paths containing digits
def _load_env_guide():
    file_path = pathlib.Path(__file__).parent.parent / "src/01_foundations_sparse_dense/module_01_environment_setup/env_guide.py"
    spec = importlib.util.spec_from_file_location("env_guide", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

env_guide = _load_env_guide()
BPETokenizer = env_guide.BPETokenizer
ContextBudgetCalculator = env_guide.ContextBudgetCalculator
RetrievalBenchmarkHarness = env_guide.RetrievalBenchmarkHarness
cosine_similarity = env_guide.cosine_similarity
dot_product = env_guide.dot_product
euclidean_distance = env_guide.euclidean_distance
manhattan_distance = env_guide.manhattan_distance
normalize_vector = env_guide.normalize_vector
simulate_high_dimensional_orthogonality = env_guide.simulate_high_dimensional_orthogonality
verify_workspace_environment = env_guide.verify_workspace_environment


def test_verify_workspace_environment():
    status = verify_workspace_environment()
    assert isinstance(status, dict)
    assert "python_version" in status
    assert "python_supported" in status
    assert "dependencies" in status
    assert "endpoint_url" in status
    assert "endpoint_status" in status
    assert status["python_supported"] is True
    assert "numpy" in status["dependencies"]
    assert "faiss" in status["dependencies"]


def test_bpe_tokenizer_lossless_roundtrip():
    corpus = [
        "Retrieval Augmented Generation",
        "Dense vector embeddings",
        "Cache Augmented Generation preloading",
        "Hybrid search combines BM25 and dense vectors"
    ]
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, num_merges=30)

    assert len(tokenizer.vocab) > 0
    assert len(tokenizer.merges) > 0

    # Test standard sentence
    text1 = "Dense embeddings"
    tokens1 = tokenizer.encode(text1)
    assert len(tokens1) > 0
    assert isinstance(tokens1, list)
    assert tokenizer.decode(tokens1) == text1

    # Test complex spacing: leading, trailing, multiple spaces, tabs, newlines
    text2 = "  Cache-Augmented \t embeddings\n\noptimize   dense retrieval!  "
    tokens2 = tokenizer.encode(text2)
    assert tokenizer.decode(tokens2) == text2, "BPE Tokenizer must be strictly 1:1 lossless on arbitrary whitespace!"

    # Test ID encoding and decoding
    token_ids = tokenizer.encode_to_ids(text2)
    assert len(token_ids) == len(tokens2)
    assert tokenizer.decode_from_ids(token_ids) == text2

    comp_ratio = tokenizer.compression_ratio(text1)
    assert comp_ratio > 0.0


def test_context_budget_calculator_and_document_capacity():
    calc = ContextBudgetCalculator(
        total_context=8192,
        max_generation_tokens=1024,
        system_prompt_tokens=250,
        query_tokens=50,
        history_tokens=100,
    )
    budget = calc.calculate_chunk_budget(chunk_size=512, reserve_safety_tokens=100)

    fixed_overhead = 1024 + 250 + 50 + 100 + 100  # 1524
    expected_available = 8192 - fixed_overhead  # 6668
    expected_k = 6668 // 512  # 13
    
    assert budget["total_context"] == 8192
    assert budget["fixed_overhead"] == fixed_overhead
    assert budget["available_for_retrieval"] == expected_available
    assert budget["max_chunks_k"] == expected_k
    assert budget["allocated_retrieval_tokens"] == expected_k * 512
    assert budget["slack_tokens"] == 8192 - (fixed_overhead + expected_k * 512)
    assert 0 < budget["utilization_percent"] <= 100

    # Test physical document capacity calculation
    capacity = ContextBudgetCalculator.compute_document_capacity(
        token_count=budget["allocated_retrieval_tokens"],
        compression_ratio=4.0,
        avg_word_length=5.0
    )
    assert capacity["retrieval_tokens"] == expected_k * 512
    assert capacity["estimated_characters"] == (expected_k * 512) * 4
    assert capacity["estimated_words"] == ((expected_k * 512) * 4) // 5
    assert capacity["estimated_pages"] > 0
    assert capacity["raw_payload_kb"] > 0


def test_kv_cache_memory_calculation():
    # 32 layers, 8 kv heads, dim 128, 8192 tokens, 2 bytes/elem (FP16)
    # Total bytes: 2 * 32 * 8 * 128 * 8192 * 2 = 1,073,741,824 bytes = 1024 MB = 1.0 GB
    kv_stats = ContextBudgetCalculator.calculate_kv_cache_memory(
        context_tokens=8192,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        bytes_per_elem=2,
    )
    assert kv_stats["bytes"] == 1073741824.0
    assert kv_stats["megabytes"] == 1024.0
    assert kv_stats["gigabytes"] == 1.0
    assert kv_stats["bytes_per_token"] == (1073741824.0 / 8192)


def test_vector_distance_metrics_and_unit_norm_equivalence():
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])

    # Orthogonal vectors
    assert math.isclose(dot_product(u, v), 0.0)
    assert math.isclose(cosine_similarity(u, v), 0.0)
    assert math.isclose(euclidean_distance(u, v), math.sqrt(2.0))
    assert math.isclose(manhattan_distance(u, v), 2.0)

    # Identical vectors
    assert math.isclose(cosine_similarity(u, u), 1.0)
    assert math.isclose(euclidean_distance(u, u), 0.0)

    # Arbitrary unit vectors for unit-norm equivalence: ||u - v||_2^2 == 2(1 - cos(theta))
    vec1 = normalize_vector(np.array([3.0, 4.0, 1.0]))
    vec2 = normalize_vector(np.array([1.0, 2.0, 5.0]))
    cos_sim = cosine_similarity(vec1, vec2)
    l2_dist = euclidean_distance(vec1, vec2)

    assert math.isclose(l2_dist**2, 2.0 * (1.0 - cos_sim), rel_tol=1e-5)


def test_simulate_high_dimensional_orthogonality():
    dims = [16, 64, 256]
    stats = simulate_high_dimensional_orthogonality(dims, num_samples=100, seed=42)
    assert len(stats) == 3
    for d in dims:
        assert d in stats
        assert abs(stats[d]["mean_cosine"]) < 0.1
        assert stats[d]["std_cosine"] > 0
        assert math.isclose(stats[d]["expected_std"], 1.0 / math.sqrt(d), rel_tol=1e-4)
        assert "raw_samples" in stats[d]
        assert len(stats[d]["raw_samples"]) == 100


def test_retrieval_benchmark_harness():
    harness = RetrievalBenchmarkHarness()
    
    # Profile simple function
    res = harness.profile_callable("Test_Sum", lambda: sum(range(100)), iterations=10)
    assert res["benchmark_name"] == "Test_Sum"
    assert res["iterations"] == 10
    assert res["mean_latency_ms"] >= 0
    assert res["throughput_ops_sec"] > 0

    # Test exact vector similarity benchmark
    vec_bench = harness.benchmark_vector_similarity(num_vectors=100, dimension=64, iterations=5)
    assert vec_bench["num_vectors"] == 100
    assert vec_bench["dimension"] == 64
    assert vec_bench["vector_comparisons_per_sec"] > 0

    # Test indexed vector search benchmark with FAISS
    indexed_hnsw = harness.benchmark_indexed_vector_search(num_vectors=200, dimension=64, index_type="hnsw", iterations=5)
    assert indexed_hnsw["index_type"] == "HNSW"
    assert indexed_hnsw["queries_per_sec"] > 0

    indexed_ivf = harness.benchmark_indexed_vector_search(num_vectors=200, dimension=64, index_type="ivf", iterations=5)
    assert indexed_ivf["index_type"] == "IVF"
    assert indexed_ivf["queries_per_sec"] > 0
