import importlib.util
import math
import pathlib
import pytest

def _load_hybrid_guide():
    file_path = pathlib.Path(__file__).parent.parent / "src/01_foundations_sparse_dense/module_02_sparse_dense_hybrid/hybrid_search_guide.py"
    spec = importlib.util.spec_from_file_location("hybrid_search_guide", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

hybrid_guide = _load_hybrid_guide()
IndustryStandardBM25 = hybrid_guide.IndustryStandardBM25
GPUDenseEmbeddingEngine = hybrid_guide.GPUDenseEmbeddingEngine
reciprocal_rank_fusion = hybrid_guide.reciprocal_rank_fusion
convex_score_fusion = hybrid_guide.convex_score_fusion
DynamicHybridRouter = hybrid_guide.DynamicHybridRouter
HybridEvaluationHarness = hybrid_guide.HybridEvaluationHarness


def test_industry_standard_bm25_search():
    docs = [
        {"id": "d1", "text": "Python programming and data structures."},
        {"id": "d2", "text": "Dense vector retrieval for semantic search."},
        {"id": "d3", "text": "BM25 inverted index for keyword search and Python scripts."}
    ]
    bm25 = IndustryStandardBM25(k1=1.5, b=0.75)
    bm25.index_documents(docs)

    assert len(bm25.doc_ids) == 3
    assert bm25.bm25_model.avgdl > 0

    # Search exact keyword
    results = bm25.search("Python", top_k=2)
    assert len(results) >= 1
    matched_ids = [doc_id for doc_id, _ in results]
    assert "d1" in matched_ids or "d3" in matched_ids


def test_gpu_dense_embedding_engine_search():
    docs = [
        {"id": "d1", "text": "Machine learning algorithms optimize predictive accuracy."},
        {"id": "d2", "text": "Cache-Augmented Generation eliminates retrieval latency."},
        {"id": "d3", "text": "Knowledge graphs represent relational facts as triplets."}
    ]
    dense = GPUDenseEmbeddingEngine(dimension=128)
    dense.index_documents(docs)

    assert dense.gpu_embedding_matrix is not None
    assert list(dense.gpu_embedding_matrix.shape) == [3, 128]

    # Search
    results = dense.search("latency optimization and caching", top_k=3)
    assert len(results) == 3
    assert results[0][0] in ["d2", "d1"]


def test_reciprocal_rank_fusion():
    sparse_ranks = [("doc_a", 12.5), ("doc_b", 8.0), ("doc_c", 3.0)]
    dense_ranks = [("doc_b", 0.95), ("doc_a", 0.85), ("doc_d", 0.70)]

    rrf = reciprocal_rank_fusion(sparse_ranks, dense_ranks, k=60)
    assert len(rrf) == 4
    
    top_doc_ids = [doc_id for doc_id, _ in rrf[:2]]
    assert "doc_a" in top_doc_ids
    assert "doc_b" in top_doc_ids


def test_convex_score_fusion():
    sparse_scores = [("doc_1", 10.0), ("doc_2", 0.0)]
    dense_scores = [("doc_1", 0.0), ("doc_2", 1.0)]

    # Alpha 0.5 (equal balance)
    fused_half = convex_score_fusion(sparse_scores, dense_scores, alpha=0.5)
    fused_dict = dict(fused_half)
    assert math.isclose(fused_dict["doc_1"], 0.5)
    assert math.isclose(fused_dict["doc_2"], 0.5)


def test_dynamic_hybrid_router():
    router = DynamicHybridRouter()
    
    alpha_code, _ = router.compute_query_alpha("ERR_503_TIMEOUT in database")
    assert alpha_code < 0.5

    alpha_nl, _ = router.compute_query_alpha("How does reciprocal rank fusion combine multiple disparate retrieval lists?")
    assert alpha_nl > 0.5


def test_hybrid_evaluation_harness():
    docs = [
        {"id": "doc_1", "text": "Error code ERR_404_NOT_FOUND in web service."},
        {"id": "doc_2", "text": "Storing precomputed key value states to accelerate inference generation."},
        {"id": "doc_3", "text": "BM25 inverted indexes rank documents according to term frequency."},
        {"id": "doc_4", "text": "Graph retrieval traverses relational triplets in knowledge graphs."}
    ]
    bm25 = IndustryStandardBM25().index_documents(docs)
    dense = GPUDenseEmbeddingEngine(dimension=128).index_documents(docs)

    harness = HybridEvaluationHarness(bm25, dense)
    test_cases = [
        {"type": "Error Code", "query": "ERR_404_NOT_FOUND", "target_id": "doc_1"}
    ]
    report = harness.evaluate_test_cases(test_cases, top_k=2)
    assert report["sparse_mrr"] == 1.0
    assert report["hybrid_mrr"] == 1.0
