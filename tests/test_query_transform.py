import importlib.util
import pathlib
import pytest

def _load_query_transform_guide():
    file_path = pathlib.Path(__file__).parent.parent / "src/02_rag_architectures/module_05_query_transformation/query_transform_guide.py"
    spec = importlib.util.spec_from_file_location("query_transform_guide", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

query_transform_guide = _load_query_transform_guide()
DenseFeatureProjector = query_transform_guide.DenseFeatureProjector
QueryTransformationSuite = query_transform_guide.QueryTransformationSuite
HyDEGenerator = query_transform_guide.HyDEGenerator
SemanticCollectionRouter = query_transform_guide.SemanticCollectionRouter


def test_dense_feature_projector():
    projector = DenseFeatureProjector(dimension=64)
    v1 = projector.embed_text("Cache-Augmented Generation")
    v2 = projector.embed_text("Cache-Augmented Generation")
    v3 = projector.embed_text("Baking artisan sourdough bread")

    assert len(v1) == 64
    sim_same = projector.compute_similarity(v1, v2)
    sim_diff = projector.compute_similarity(v1, v3)
    assert pytest.approx(sim_same, 0.001) == 1.0
    assert sim_same > sim_diff


def test_multi_query_expansion():
    projector = DenseFeatureProjector(dimension=64)
    suite = QueryTransformationSuite(projector)
    
    variations = suite.expand_multi_query("HNSW Vector Indexing")
    assert len(variations) == 5
    assert "HNSW Vector Indexing" in variations[0]


def test_step_back_query_generation():
    projector = DenseFeatureProjector(dimension=64)
    suite = QueryTransformationSuite(projector)

    step_back_q, concept = suite.generate_step_back_query("ERR_KV_CACHE_OVERFLOW_503 memory crash")
    assert "GPU Memory" in concept
    assert "fundamental principles" in step_back_q.lower()


def test_sub_query_decomposition():
    projector = DenseFeatureProjector(dimension=64)
    suite = QueryTransformationSuite(projector)

    decomposed = suite.decompose_complex_query("Compare HNSW indexing latency versus Product Quantization memory compression")
    assert len(decomposed) >= 2


def test_hyde_retrieval():
    projector = DenseFeatureProjector(dimension=64)
    hyde = HyDEGenerator(projector)

    corpus = [
        {"id": "d1", "text": "Cache-Augmented Generation preloads static prompt tokens directly into the LLM KV-cache."},
        {"id": "d2", "text": "Recipe for chocolate chip cookies with organic butter."}
    ]

    res = hyde.retrieve_with_hyde("How does KV prefill eliminate runtime vector search?", corpus, top_k=2)
    assert len(res["hyde_top_k"]) == 2
    assert res["hyde_top_k"][0][0] == "d1"


def test_semantic_collection_router():
    projector = DenseFeatureProjector(dimension=64)
    router = SemanticCollectionRouter(projector)

    router.register_collection(
        collection_name="architecture",
        description="System design, CAG, and GraphRAG",
        exemplars=["Cache-Augmented Generation architectural blueprint", "GraphRAG community detection"]
    )
    router.register_collection(
        collection_name="troubleshooting",
        description="Error codes and CUDA crashes",
        exemplars=["ERR_KV_CACHE_OVERFLOW_503 memory crash", "CUDA out of memory in PyTorch"]
    )

    best_col, conf, _ = router.route_query("How do I fix ERR_KV_CACHE_OVERFLOW_503 memory crash on GPU?")
    assert best_col == "troubleshooting"
    assert conf > 0.0
