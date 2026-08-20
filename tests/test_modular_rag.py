import importlib.util
import pathlib
import pytest

def _load_modular_rag_guide():
    file_path = pathlib.Path(__file__).parent.parent / "src/02_rag_architectures/module_07_advanced_modular_rag/modular_rag_guide.py"
    spec = importlib.util.spec_from_file_location("modular_rag_guide", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

modular_rag_guide = _load_modular_rag_guide()
CRAGRetrievalEvaluator = modular_rag_guide.CRAGRetrievalEvaluator
SelfRAGSimulator = modular_rag_guide.SelfRAGSimulator
AdaptiveRAGRouter = modular_rag_guide.AdaptiveRAGRouter
ModularRAGSystem = modular_rag_guide.ModularRAGSystem


def test_crag_retrieval_evaluator():
    evaluator = CRAGRetrievalEvaluator(upper_threshold=0.70, lower_threshold=0.30)
    
    # 1. Correct
    valid_docs = [{"id": "d1", "text": "Cache-Augmented Generation preloads KV cache to eliminate retrieval latency."}]
    conf_c, action_c = evaluator.evaluate_retrieval_confidence("How does CAG eliminate retrieval latency using the KV-cache?", valid_docs)
    assert action_c == "CORRECT"
    assert conf_c >= 0.70

    # 2. Incorrect
    noise_docs = [{"id": "d2", "text": "Baking sourdough bread."}]
    conf_i, action_i = evaluator.evaluate_retrieval_confidence("What is quantum entanglement in physics?", noise_docs)
    assert action_i == "INCORRECT"
    assert conf_i < 0.30

    # 3. Knowledge Striping
    stripes = evaluator.stripe_knowledge(valid_docs, "CAG KV-cache latency")
    assert len(stripes) >= 1
    assert "Cache-Augmented Generation" in stripes[0]


def test_self_rag_simulator():
    self_rag = SelfRAGSimulator()

    # Retrieve need
    assert self_rag.evaluate_retrieval_need("Hello there!") == "[Retrieve=False]"
    assert self_rag.evaluate_retrieval_need("Explain KV cache preloading in CAG") == "[Retrieve=True]"

    # Grounding check
    context = "Cache-Augmented Generation preloads tokens directly into KV cache."
    good_resp = "Cache-Augmented Generation preloads tokens into KV cache."
    bad_resp = "Quantum computers teleport information across galaxies."
    assert self_rag.evaluate_groundedness(context, good_resp) == "[IsSup=FullySupported]"
    assert self_rag.evaluate_groundedness(context, bad_resp) == "[IsSup=NoSupport]"


def test_adaptive_rag_router():
    router = AdaptiveRAGRouter()

    tier_dir, _ = router.classify_complexity("Hello")
    assert tier_dir == "DIRECT"

    tier_single, _ = router.classify_complexity("What is BM25 term saturation?")
    assert tier_single == "SINGLE_HOP"

    tier_multi, _ = router.classify_complexity("Compare the latency and memory trade-offs of HNSW versus Product Quantization")
    assert tier_multi == "MULTI_HOP"


def test_modular_rag_system():
    router = AdaptiveRAGRouter()
    crag = CRAGRetrievalEvaluator()
    self_rag = SelfRAGSimulator()
    kb = [{"id": "d1", "text": "Cache-Augmented Generation preloads prompt tokens into KV cache."}]

    system = ModularRAGSystem(router, crag, self_rag, kb)

    # 1. Direct query
    res_direct = system.execute_pipeline("Hi!")
    assert res_direct["tier"] == "DIRECT"

    # 2. RAG query
    res_rag = system.execute_pipeline("What does CAG preload into KV cache?")
    assert res_rag["tier"] == "SINGLE_HOP"
    assert res_rag["crag_action"] in ["CORRECT", "AMBIGUOUS"]
    assert "KV cache" in res_rag["final_answer"]
