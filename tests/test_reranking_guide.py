import importlib.util
import pathlib
import pytest

def _load_reranking_guide():
    file_path = pathlib.Path(__file__).parent.parent / "src/02_rag_architectures/module_06_reranking_compression/reranking_guide.py"
    spec = importlib.util.spec_from_file_location("reranking_guide", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

reranking_guide = _load_reranking_guide()
CrossEncoderReranker = reranking_guide.CrossEncoderReranker
ContextualTokenCompressor = reranking_guide.ContextualTokenCompressor
LostInTheMiddleReorderer = reranking_guide.LostInTheMiddleReorderer
RerankingBenchmarkHarness = reranking_guide.RerankingBenchmarkHarness


def test_cross_encoder_reranker():
    reranker = CrossEncoderReranker()
    candidates = [
        {"id": "doc_generic", "text": "Deep learning and computer vision architectures.", "bi_score": 0.88},
        {"id": "doc_exact", "text": "Cache-Augmented Generation preloads KV cache to cut TTFT latency.", "bi_score": 0.75}
    ]

    reranked = reranker.rerank("How does KV cache preloading reduce TTFT latency in CAG?", candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["id"] == "doc_exact"
    assert reranked[0]["cross_encoder_score"] > reranked[1]["cross_encoder_score"]


def test_contextual_token_compressor():
    compressor = ContextualTokenCompressor(target_compression_ratio=0.5)
    text = (
        "It is well known that Cache-Augmented Generation preloads context into KV cache. "
        "Furthermore, traditional Vector RAG searches HNSW graphs. "
        "Preloading ensures sub-20ms TTFT performance."
    )
    res = compressor.compress_document(text, query="CAG KV cache TTFT")
    assert res["compressed_tokens"] < res["raw_tokens"]
    assert res["tokens_saved"] > 0
    assert "Cache-Augmented Generation" in res["compressed_text"] or "TTFT" in res["compressed_text"]


def test_lost_in_the_middle_reorderer():
    reorderer = LostInTheMiddleReorderer()
    docs = [
        {"id": "d1", "rank": 1},
        {"id": "d2", "rank": 2},
        {"id": "d3", "rank": 3},
        {"id": "d4", "rank": 4},
        {"id": "d5", "rank": 5}
    ]
    reordered = reorderer.reorder_for_optimal_attention(docs)
    assert len(reordered) == 5
    assert reordered[0]["id"] == "d1"   # Rank 1 at start
    assert reordered[-1]["id"] == "d2"  # Rank 2 at end


def test_reranking_benchmark_harness():
    reranker = CrossEncoderReranker()
    compressor = ContextualTokenCompressor()
    harness = RerankingBenchmarkHarness(reranker, compressor)

    cases = [
        {
            "query": "ERR_KV_CACHE_OVERFLOW_503 on CUDA GPU",
            "target_id": "d_err",
            "candidates": [
                {"id": "d_gen", "text": "General PyTorch tensor operations. SIMD acceleration on CPU.", "bi_score": 0.85},
                {"id": "d_err", "text": "Error code ERR_KV_CACHE_OVERFLOW_503 indicates GPU memory exhaustion on CUDA device. It occurs when preloading large context tokens.", "bi_score": 0.78}
            ]
        }
    ]
    report = harness.run_benchmark(cases)
    assert report["cross_mrr3"] == 1.0
    assert report["overall_token_savings"] > 0.0
