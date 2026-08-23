import importlib.util
import math
import pathlib
import numpy as np
import scipy.spatial.distance as dist
import pytest

# Helper to load modules from directory paths containing digits
def _load_env_guide():
    file_path = pathlib.Path(__file__).parent.parent / "src/01_foundations_sparse_dense/module_01_environment_setup/env_guide.py"
    spec = importlib.util.spec_from_file_location("env_guide", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

env_guide = _load_env_guide()
verify_workspace_environment = env_guide.verify_workspace_environment
inspect_subword_tokens = env_guide.inspect_subword_tokens
profile_tokenizer_throughput = env_guide.profile_tokenizer_throughput
compute_document_capacity = env_guide.compute_document_capacity
calculate_chunk_budget = env_guide.calculate_chunk_budget
calculate_kv_cache_memory = env_guide.calculate_kv_cache_memory
simulate_high_dimensional_orthogonality = env_guide.simulate_high_dimensional_orthogonality
benchmark_faiss_scaling = env_guide.benchmark_faiss_scaling
benchmark_large_scale_memmap = env_guide.benchmark_large_scale_memmap


def test_verify_workspace_environment():
    status = verify_workspace_environment()
    assert isinstance(status, dict)
    assert "python_version" in status
    assert "python_supported" in status
    assert "dependencies" in status
    assert "gpu_available" in status
    assert "endpoint_url" in status
    assert "endpoint_status" in status
    assert status["python_supported"] is True
    assert "numpy" in status["dependencies"]
    assert "faiss" in status["dependencies"]
    assert "tiktoken" in status["dependencies"]
    assert "torch" in status["dependencies"]


def test_tiktoken_subword_inspection_and_lossless():
    text = "  Cache-Augmented \t embeddings\n\noptimize   dense retrieval!  "
    res = inspect_subword_tokens(text, encoding_name="cl100k_base")
    assert res["num_tokens"] > 0
    assert len(res["token_ids"]) == res["num_tokens"]
    assert len(res["subword_tokens"]) == res["num_tokens"]
    assert res["is_lossless"] is True
    assert res["decoded_text"] == text
    assert res["compression_ratio"] > 0.0


def test_tiktoken_throughput_profiling():
    corpus = ["Retrieval augmented generation with high throughput compiled tokenizers."] * 10
    res = profile_tokenizer_throughput(corpus, encoding_name="cl100k_base", iterations=5)
    assert res["encoding"] == "cl100k_base"
    assert res["total_tokens_processed"] > 0
    assert res["tokens_per_second"] > 0


def test_context_chunk_budgeting_and_document_capacity():
    budget = calculate_chunk_budget(
        total_context=8192,
        max_generation_tokens=1024,
        system_prompt_tokens=250,
        query_tokens=50,
        history_tokens=100,
        chunk_size=512,
        overlap=0,
        reserve_safety_tokens=100,
    )
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
    capacity = compute_document_capacity(
        token_count=budget["allocated_retrieval_tokens"],
        compression_ratio=4.0,
        avg_word_length=5.0,
    )
    assert capacity["retrieval_tokens"] == expected_k * 512
    assert capacity["estimated_characters"] == (expected_k * 512) * 4
    assert capacity["estimated_words"] == ((expected_k * 512) * 4) // 5
    assert capacity["estimated_pages"] > 0
    assert capacity["raw_payload_kib"] > 0
    assert capacity["raw_payload_kb"] > 0


def test_kv_cache_memory_calculation():
    # 32 layers, 8 kv heads, dim 128, 8192 tokens, 2 bytes/elem (FP16)
    # Total bytes: 2 * 32 * 8 * 128 * 8192 * 2 = 1,073,741,824 bytes = 1024 MiB = 1.0 GiB
    kv_stats = calculate_kv_cache_memory(
        context_tokens=8192,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        bytes_per_elem=2,
        batch_size=1,
    )
    assert kv_stats["bytes"] == 1073741824.0
    assert kv_stats["mebibytes"] == 1024.0
    assert kv_stats["gibibytes"] == 1.0
    assert kv_stats["megabytes"] == 1024.0
    assert kv_stats["gigabytes"] == 1.0
    assert kv_stats["gibibytes_ceiling"] == 1.0
    assert kv_stats["bytes_per_token"] == (1073741824.0 / 8192)


def test_vector_distance_metrics_and_unit_norm_equivalence():
    u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Orthogonal vectors via numpy and scipy
    assert np.isclose(float(np.dot(u, v)), 0.0)
    assert np.isclose(float(dist.cosine(u, v)), 1.0)
    assert np.isclose(float(dist.euclidean(u, v)), math.sqrt(2.0))
    assert np.isclose(float(dist.cityblock(u, v)), 2.0)

    # Arbitrary unit vectors for unit-norm equivalence: ||u - v||_2^2 == 2(1 - cos(theta))
    vec1 = np.array([3.0, 4.0, 1.0], dtype=np.float32)
    vec2 = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)

    cos_sim = float(np.dot(vec1, vec2))
    l2_dist = float(dist.euclidean(vec1, vec2))

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


def test_faiss_scaling_benchmark():
    perf = benchmark_faiss_scaling(num_vectors=1000, dimension=64, top_k=5)
    assert perf["num_vectors"] == 1000
    assert perf["dimension"] == 64
    assert "flat_ip" in perf
    assert "hnsw" in perf
    assert "ivf" in perf
    assert perf["flat_ip"]["queries_per_sec"] > 0
    assert perf["hnsw"]["queries_per_sec"] > 0
    assert perf["ivf"]["queries_per_sec"] > 0


def test_large_scale_vector_benchmark_memmap():
    # Test memory-safe memmap benchmark with moderate vector count for fast test execution
    perf = benchmark_large_scale_memmap(num_vectors=2000, dimension=64, top_k=5)
    assert perf["num_vectors"] == 2000
    assert perf["dimension"] == 64
    assert "hnsw" in perf
    assert "flat_ip" in perf
    assert "hnsw_speedup_vs_flat" in perf
    assert perf["hnsw"]["queries_per_sec"] > 0
    assert perf["flat_ip"]["queries_per_sec"] > 0
