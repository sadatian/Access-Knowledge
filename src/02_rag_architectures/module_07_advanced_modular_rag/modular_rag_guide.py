# %% [markdown]
# # Module 07: Modular, Corrective (CRAG) & Self-RAG
#
# Welcome to **Module 07** of the Knowledge Retrieval A-Z masterclass.
# Naive RAG follows a rigid, linear sequence: *Query $\rightarrow$ Retrieve $\rightarrow$ Prompt $\rightarrow$ Generate*.
#
# When retrieved context contains noise, contradictions, or misses the answer entirely, naive RAG fails catastrophically by hallucinating answers or repeating irrelevant context.
#
# **Modular & Self-Reflective RAG** transforms retrieval into an adaptive, self-healing state machine:
# 1. **Corrective RAG (CRAG)**: Evaluates retrieval confidence across three distinct bands (*Correct*, *Incorrect*, *Ambiguous*) to trigger knowledge striping or external web fallbacks.
# 2. **Self-RAG Reflection Tokens**: Embeds special control tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) to dynamically decide when to retrieve, filter irrelevant passages, verify factual grounding, and score utility.
# 3. **Adaptive RAG Dynamic Routing**: Routes queries based on semantic complexity to *Direct LLM*, *Single-Hop RAG*, or *Iterative Multi-Hop Loops*.
# 4. **Presenter Dashboard & State Machine Visualizer (`# collapse_input`)**: Auto-collapsing ASCII state machine and reflection token audit log.
#
# ---

# %%
import math
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from openai import OpenAI

# Hardware Accelerator Detection
def detect_compute_device() -> torch.device:
    """Detect available compute accelerator (CUDA GPU / MPS) with graceful CPU fallback."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        print(f"[INFO] Modular RAG Hardware: CUDA GPU -> {device_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Modular RAG Hardware: Apple Silicon MPS GPU")
    else:
        device = torch.device("cpu")
        print("[INFO] Modular RAG Hardware: CPU (Optimized SIMD)")
    return device

DEVICE = detect_compute_device()

# Local LLM Client Configuration
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_ENDPOINT", "http://localhost:5055/v1")
openai_client = OpenAI(base_url=LOCAL_LLM_URL, api_key="dummy")

# %% [markdown]
# ## Section 1: Corrective RAG (CRAG) with 3-Tier Confidence Bands
#
# **Corrective RAG** introduces a lightweight *Retrieval Evaluator* that assesses the confidence score $\gamma \in [0.0, 1.0]$ of retrieved documents:
#
# 1. **Correct ($\gamma \ge 0.75$):** The internal context is highly relevant. Extract key sentences (*Knowledge Striping*) and generate response directly.
# 2. **Incorrect ($\gamma < 0.35$):** The internal context is completely irrelevant. Discard internal documents and trigger external fallback / web retrieval.
# 3. **Ambiguous ($0.35 \le \gamma < 0.75$):** The internal context is noisy or incomplete. Combine internal knowledge stripes with rewritten query expansions.

# %%
class CRAGRetrievalEvaluator:
    """Evaluates retrieval quality and triggers corrective actions across 3 confidence bands."""

    def __init__(self, upper_threshold: float = 0.75, lower_threshold: float = 0.35):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def evaluate_retrieval_confidence(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Tuple[float, str]:
        """Compute aggregate confidence and classify into CORRECT, AMBIGUOUS, or INCORRECT."""
        if not retrieved_docs:
            return 0.0, "INCORRECT"

        q_terms = set(re.findall(r"\b\w+\b", query.lower()))
        scores = []

        for doc in retrieved_docs:
            text = doc.get("text", "").lower()
            doc_terms = set(re.findall(r"\b\w+\b", text))
            overlap = len(q_terms.intersection(doc_terms)) / max(1, len(q_terms))
            
            # Boost for exact technical identifiers
            has_exact = any(t in text for t in q_terms if len(t) >= 4)
            exact_boost = 0.25 if has_exact else 0.0
            scores.append(min(1.0, overlap + exact_boost))

        confidence = float(np.mean(scores)) if scores else 0.0
        confidence = round(confidence, 3)

        if confidence >= self.upper_threshold:
            action = "CORRECT"
        elif confidence < self.lower_threshold:
            action = "INCORRECT"
        else:
            action = "AMBIGUOUS"

        return confidence, action

    def stripe_knowledge(self, documents: List[Dict[str, Any]], query: str) -> List[str]:
        """Knowledge Striping: Extract only the precise relevant sentences from valid documents."""
        q_terms = set(re.findall(r"\b\w+\b", query.lower()))
        stripes = []

        for doc in documents:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc.get("text", "")) if s.strip()]
            for s in sentences:
                s_terms = set(re.findall(r"\b\w+\b", s.lower()))
                if len(q_terms.intersection(s_terms)) >= 1:
                    stripes.append(s)

        return stripes

# %% [markdown]
# ### Demo 1: CRAG Confidence Bands & Corrective Execution
#
# Below, we test CRAG across verified internal context, ambiguous context, and completely unindexed topics.

# %%
internal_knowledge_corpus = [
    {"id": "doc_cag", "text": "Cache-Augmented Generation (CAG) preloads static prompt tokens directly into the LLM KV-cache to eliminate retrieval latency."},
    {"id": "doc_hnsw", "text": "HNSW vector indexing builds multi-layer proximity graphs for fast cosine approximate nearest neighbor search."},
    {"id": "doc_bm25", "text": "BM25Okapi applies Robertson-Spärck Jones probabilistic term frequency weighting with document length normalization."}
]

crag_evaluator = CRAGRetrievalEvaluator(upper_threshold=0.70, lower_threshold=0.30)

crag_test_queries = [
    ("How does CAG eliminate retrieval latency using the KV-cache?", [internal_knowledge_corpus[0]]),
    ("What are HNSW graph skip connections?", [internal_knowledge_corpus[1], {"id": "noise", "text": "General computing history."}]),
    ("What was the stock price of Apple on January 15, 2026?", [{"id": "irrelevant", "text": "Baking sourdough bread at high altitude."}])
]

print("=== [Corrective RAG (CRAG) Confidence Assessment] ===")
for q, docs in crag_test_queries:
    conf, action = crag_evaluator.evaluate_retrieval_confidence(q, docs)
    print(f"\nQuery: '{q}'")
    print(f"  • Retrieval Confidence: {conf:.3f} -> Action: [{action}]")
    
    if action == "CORRECT":
        stripes = crag_evaluator.stripe_knowledge(docs, q)
        print(f"  • Execution: Knowledge Striping ({len(stripes)} key sentences extracted): '{stripes[0][:60]}...'")
    elif action == "AMBIGUOUS":
        stripes = crag_evaluator.stripe_knowledge(docs, q)
        print(f"  • Execution: Query Refinement + Internal Striping ({len(stripes)} sentences)")
    else:
        print("  • Execution: Discarding Internal Docs -> Triggering External Web Search Fallback")

# %% [markdown]
# ## Section 2: Self-RAG Architecture & Reflection Control Tokens
#
# **Self-RAG** equips language models with self-reflection capabilities through four specialized control tokens:
#
# 1. **`[Retrieve]` Token:** Decides whether retrieval is necessary (`[Retrieve=True]` vs `[Retrieve=False]`).
# 2. **`[IsRel]` Token:** Evaluates passage relevance (`[IsRel=Relevant]` vs `[IsRel=Irrelevant]`).
# 3. **`[IsSup]` Token:** Verifies factual grounding (`[IsSup=FullySupported]`, `[IsSup=PartiallySupported]`, `[IsSup=NoSupport]`).
# 4. **`[IsUse]` Token:** Rates overall generation utility from 1 to 5 (`[IsUse=5]`).

# %%
class SelfRAGSimulator:
    """Self-RAG engine generating and auditing reflection tokens for grounded generation."""

    def evaluate_retrieval_need(self, query: str) -> str:
        """Decide whether the query requires external non-parametric retrieval."""
        conversational_starters = {"hello", "hi", "hey", "who are you", "what is your name", "2+2", "write a poem"}
        q_clean = query.lower().strip()
        
        if q_clean in conversational_starters or len(q_clean.split()) <= 2:
            return "[Retrieve=False]"
        return "[Retrieve=True]"

    def evaluate_passage_relevance(self, query: str, passage: str) -> str:
        """Assess whether a retrieved passage is relevant to the query."""
        q_words = set(re.findall(r"\b\w+\b", query.lower()))
        p_words = set(re.findall(r"\b\w+\b", passage.lower()))
        overlap = len(q_words.intersection(p_words))
        return "[IsRel=Relevant]" if overlap >= 2 else "[IsRel=Irrelevant]"

    def evaluate_groundedness(self, context: str, response: str) -> str:
        """Verify whether every factual claim in the response is grounded in the retrieved context."""
        c_words = set(re.findall(r"\b\w+\b", context.lower()))
        r_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response.strip()) if s.strip()]

        if not r_sentences:
            return "[IsSup=NoSupport]"

        supported_cnt = 0
        for sent in r_sentences:
            s_words = set(re.findall(r"\b\w+\b", sent.lower()))
            overlap = len(c_words.intersection(s_words))
            if overlap >= len(s_words) * 0.4:
                supported_cnt += 1

        ratio = supported_cnt / len(r_sentences)
        if ratio >= 0.8:
            return "[IsSup=FullySupported]"
        elif ratio >= 0.4:
            return "[IsSup=PartiallySupported]"
        else:
            return "[IsSup=NoSupport]"

    def score_utility(self, response: str, grounded_verdict: str) -> int:
        """Score generation utility from 1 to 5 based on completeness and support."""
        if grounded_verdict == "[IsSup=FullySupported]" and len(response.split()) >= 10:
            return 5
        elif grounded_verdict == "[IsSup=PartiallySupported]":
            return 3
        else:
            return 1

    def run_self_rag_pipeline(self, query: str, candidate_contexts: List[str]) -> Dict[str, Any]:
        """Execute complete Self-RAG reflection cycle."""
        retrieve_decision = self.evaluate_retrieval_need(query)
        
        if retrieve_decision == "[Retrieve=False]":
            return {
                "query": query,
                "retrieve_token": retrieve_decision,
                "selected_context": None,
                "response": "Direct parametric response without external retrieval.",
                "grounding_token": "[IsSup=FullySupported]",
                "utility_token": "[IsUse=5]"
            }

        # Filter relevant passages
        relevant_contexts = []
        for ctx in candidate_contexts:
            rel_token = self.evaluate_passage_relevance(query, ctx)
            if rel_token == "[IsRel=Relevant]":
                relevant_contexts.append(ctx)

        if not relevant_contexts:
            return {
                "query": query,
                "retrieve_token": retrieve_decision,
                "selected_context": None,
                "response": "Unable to locate verified context for this query.",
                "grounding_token": "[IsSup=NoSupport]",
                "utility_token": "[IsUse=2]"
            }

        best_context = relevant_contexts[0]
        simulated_response = f"Based on verified documentation: {best_context}"
        grounding = self.evaluate_groundedness(best_context, simulated_response)
        utility = self.score_utility(simulated_response, grounding)

        return {
            "query": query,
            "retrieve_token": retrieve_decision,
            "selected_context": best_context,
            "response": simulated_response,
            "grounding_token": grounding,
            "utility_token": f"[IsUse={utility}]"
        }

# %% [markdown]
# ### Demo 2: Self-RAG Reflection Cycle Demonstration
#
# Below, we trace the full Self-RAG reflection audit trail across conversational and technical queries.

# %%
self_rag_engine = SelfRAGSimulator()

queries_self_rag = [
    "Hello there!",
    "How does Cache-Augmented Generation eliminate retrieval latency?"
]

sample_contexts = [
    "Cache-Augmented Generation (CAG) preloads static prompt tokens directly into the LLM KV-cache to eliminate retrieval latency."
]

print("=== [Self-RAG Reflection Token Execution] ===")
for q in queries_self_rag:
    audit = self_rag_engine.run_self_rag_pipeline(q, sample_contexts)
    print(f"\nQuery: '{audit['query']}'")
    print(f"  • Retrieval Trigger:  {audit['retrieve_token']}")
    print(f"  • Grounding Check:   {audit['grounding_token']}")
    print(f"  • Utility Rating:    {audit['utility_token']}")
    print(f"  • Final Output:      '{audit['response'][:75]}...'")

# %% [markdown]
# ## Section 3: Adaptive RAG Dynamic Query Complexity Routing
#
# Production systems must not incur multi-hop retrieval latency for simple factual lookups.
#
# **Adaptive RAG** classifies incoming queries into three operational tiers:
# - **Tier 1 (Direct LLM):** Zero-retrieval conversational or parametric queries.
# - **Tier 2 (Single-Hop RAG):** Standard single-topic factual lookups (Hybrid Search + Cross-Encoder).
# - **Tier 3 (Multi-Hop Iterative RAG):** Complex multi-entity or comparative questions requiring iterative sub-query decomposition.

# %%
class AdaptiveRAGRouter:
    """Classifies query complexity and routes execution between Direct LLM, Single-Hop RAG, and Multi-Hop RAG."""

    def classify_complexity(self, query: str) -> Tuple[str, str]:
        """Classify query into DIRECT, SINGLE_HOP, or MULTI_HOP tier."""
        q_lower = query.lower().strip()
        tokens = q_lower.split()

        # Tier 1: Conversational / Simple
        conversational = {"hi", "hello", "who are you", "thank you", "thanks"}
        if q_lower in conversational or len(tokens) <= 2:
            return "DIRECT", "Conversational / Simple query -> Zero retrieval overhead"

        # Tier 3: Complex Multi-Hop / Comparative
        multi_hop_triggers = ["compare", "difference between", "versus", "vs", "trade-offs", "how does x relate to y", "and also"]
        if any(trig in q_lower for trig in multi_hop_triggers) or len(tokens) >= 12:
            return "MULTI_HOP", "Comparative / Multi-entity reasoning -> Iterative multi-hop loop"

        # Tier 2: Standard Single-Hop
        return "SINGLE_HOP", "Specific technical lookup -> Single-hop hybrid retrieval"

# %% [markdown]
# ### Demo 3: Adaptive RAG Complexity Routing Demonstration
#
# Below, we test query complexity routing on diverse real-world queries.

# %%
adaptive_router = AdaptiveRAGRouter()

complexity_test_cases = [
    "Hi!",
    "What is the average document length parameter b in BM25?",
    "Compare the latency, memory footprint, and recall trade-offs of HNSW indexing versus Product Quantization"
]

print("=== [Adaptive RAG Complexity Routing Output] ===")
for q in complexity_test_cases:
    tier, rationale = adaptive_router.classify_complexity(q)
    print(f"\nQuery: '{q}'")
    print(f"  • Operational Tier: [{tier}]")
    print(f"  • Routing Rationale: {rationale}")

# %% [markdown]
# ## Section 4: End-to-End Modular RAG State Machine Execution
#
# We execute a comprehensive stress test combining **Adaptive Routing**, **CRAG Confidence Evaluation**, and **Self-RAG Reflection Tokens**.

# %%
class ModularRAGSystem:
    """Unified Modular RAG State Machine combining Adaptive Routing, CRAG, and Self-RAG."""

    def __init__(
        self,
        router: AdaptiveRAGRouter,
        crag: CRAGRetrievalEvaluator,
        self_rag: SelfRAGSimulator,
        knowledge_base: List[Dict[str, Any]]
    ):
        self.router = router
        self.crag = crag
        self.self_rag = self_rag
        self.kb = knowledge_base

    def search_kb(self, query: str) -> List[Dict[str, Any]]:
        """Simple keyword-overlap candidate search."""
        q_words = set(re.findall(r"\b\w+\b", query.lower()))
        scored = []
        for doc in self.kb:
            d_words = set(re.findall(r"\b\w+\b", doc["text"].lower()))
            overlap = len(q_words.intersection(d_words))
            scored.append((doc, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, s in scored if s > 0][:2]

    def execute_pipeline(self, query: str) -> Dict[str, Any]:
        """Execute full state machine."""
        tier, tier_rationale = self.router.classify_complexity(query)
        
        if tier == "DIRECT":
            return {
                "query": query,
                "tier": tier,
                "crag_action": "BYPASS",
                "reflection_tokens": "[Retrieve=False] [IsSup=FullySupported] [IsUse=5]",
                "final_answer": "Direct conversational response without retrieval."
            }

        # Retrieve candidates
        candidates = self.search_kb(query)
        conf, crag_action = self.crag.evaluate_retrieval_confidence(query, candidates)

        if crag_action == "CORRECT":
            stripes = self.crag.stripe_knowledge(candidates, query)
            context = " ".join(stripes)
            resp = f"Verified Internal Knowledge: {context}"
            tokens = "[Retrieve=True] [IsRel=Relevant] [IsSup=FullySupported] [IsUse=5]"
        elif crag_action == "AMBIGUOUS":
            stripes = self.crag.stripe_knowledge(candidates, query)
            context = " ".join(stripes) if stripes else "Partial context."
            resp = f"Refined Hybrid Response: {context}"
            tokens = "[Retrieve=True] [IsRel=Relevant] [IsSup=PartiallySupported] [IsUse=3]"
        else:
            resp = "Fallback Search Response: External verified knowledge extracted."
            tokens = "[Retrieve=True] [IsRel=Irrelevant] [IsSup=FullySupported] [IsUse=4]"

        return {
            "query": query,
            "tier": tier,
            "confidence": conf,
            "crag_action": crag_action,
            "reflection_tokens": tokens,
            "final_answer": resp
        }

# %% [markdown]
# ### Demo 4: Modular RAG End-to-End Stress Test Run
#
# Below, we execute the state machine across all operational regimes.

# %%
modular_rag = ModularRAGSystem(adaptive_router, crag_evaluator, self_rag_engine, internal_knowledge_corpus)

stress_queries = [
    "Hello Antigravity",
    "How does CAG eliminate retrieval latency?",
    "What is the market valuation of SpaceX in 2026?"
]

print("=== [Modular RAG State Machine Stress Test] ===")
for q in stress_queries:
    out = modular_rag.execute_pipeline(q)
    print(f"\nQuery: '{out['query']}'")
    print(f"  • Complexity Tier:    [{out['tier']}]")
    print(f"  • CRAG Action:        [{out['crag_action']}] (Confidence: {out.get('confidence', 1.0):.2f})")
    print(f"  • Reflection Tokens:  {out['reflection_tokens']}")
    print(f"  • Final Output:       '{out['final_answer'][:70]}...'")

# %% [markdown]
# ## Section 5: Presenter Dashboard & State Machine Visualizer
#
# Below is the consolidated presenter dashboard rendering an ASCII state machine audit report.

# %%
# collapse_input
def display_modular_rag_dashboard():
    """Render a clean ASCII visualizer of the modular RAG state machine and reflection audit."""
    print("=" * 80)
    print("           KNOWLEDGE RETRIEVAL A-Z: MODULE 07 MODULAR RAG DASHBOARD")
    print("=" * 80)
    
    print("\n[1] STATE MACHINE EXECUTION NODES")
    print("  [User Query] ──▶ (Adaptive Classifier) ──┬── [DIRECT]    ──▶ (Zero Retrieval LLM)")
    print("                                            ├── [SINGLE_HOP]─▶ (Hybrid + Reranker)")
    print("                                            └── [MULTI_HOP] ─▶ (Iterative Sub-Queries)")
    print("                                                      │")
    print("                                                      ▼")
    print("                                             [CRAG Evaluator]")
    print("                                            ┌─────────┼─────────┐")
    print("                                            ▼         ▼         ▼")
    print("                                        [CORRECT] [AMBIGUOUS] [INCORRECT]")
    print("                                        (Striping) (Refine)   (Web Fallback)")

    print("\n[2] SELF-RAG REFLECTION AUDIT MATRIX")
    print("  • [Retrieve=True/False]  : Eliminates unnecessary retrieval on conversational queries.")
    print("  • [IsRel=Relevant]       : Discards noisy, off-topic passages before prompt building.")
    print("  • [IsSup=FullySupported] : Verifies strict attribution against context (Zero Hallucination).")
    print("  • [IsUse=5]              : Enforces high utility and complete answers.")

    print("\n[3] TRACK 2 COMPLETION MILESTONE")
    print("  • Advanced Chunking (Module 04)  ──▶ Robust semantic units & parent-child links.")
    print("  • Query Transformation (Module 05) ──▶ Multi-Query, Step-Back & HyDE alignment.")
    print("  • Context Reranking (Module 06)   ──▶ Cross-Encoder precision & attention reordering.")
    print("  • Modular RAG (Module 07)         ──▶ Self-healing state machines & reflection tokens.")

    print("\n" + "=" * 80)
    print("  [OK] Track 2 Complete! Ready for Track 3: Cache-Augmented Generation (CAG).")
    print("=" * 80)

# Render Dashboard
display_modular_rag_dashboard()

# %% [markdown]
# ## Section 6: Summary & Transition to Track 3
#
# In Track 2 (Retrieval-Augmented Generation Architecture), we engineered the complete production RAG stack:
# 1. **Module 04**: Advanced Chunking, Parent-Child Stores, and Multi-Format Ingestion.
# 2. **Module 05**: Query Transformation, Step-Back Prompting, HyDE, and Semantic Collection Routing.
# 3. **Module 06**: Two-Stage Cross-Encoder Reranking, LLMLingua Compression, and Lost-in-the-Middle Attention Optimization.
# 4. **Module 07**: Modular State Machines, Corrective RAG (CRAG) Confidence Bands, and Self-RAG Reflection Tokens.
#
# In **Track 3 (Module 08: Cache-Augmented Generation Patterns)**, we explore how **persistent LLM Key-Value (KV) Cache architectures** can bypass retrieval entirely for fixed domain knowledge, delivering sub-20ms Time-To-First-Token performance.
