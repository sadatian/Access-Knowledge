# %% [markdown]
# # Module 05: Query Transformation & Multi-Query Routing
#
# Welcome to **Module 05** of the Knowledge Retrieval A-Z masterclass.
# In production RAG systems, user queries are notoriously problematic:
# - **Brevity & Ambiguity:** Users search with 2 to 4 keywords, omitting essential domain context.
# - **The Modality Gap (Query Space vs Document Space):** Questions and explanatory documents occupy structurally distinct distributions in high-dimensional embedding space.
# - **Multi-Part Conflation:** Complex user questions bundle multiple sub-questions that cannot be satisfied by any single retrieved passage.
#
# In this module, we construct and master production-grade query transformation architectures:
# 1. **Multi-Query Expansion & Fusion**: Generating diverse lexical and semantic query perspectives and fusing multi-path retrieval candidates using Reciprocal Rank Fusion (RRF).
# 2. **Step-Back Prompting**: Abstracting narrow, specific queries into fundamental conceptual principles to retrieve foundational context.
# 3. **Sub-Query Decomposition**: Deconstructing complex multi-hop queries into atomic sub-questions for parallel retrieval.
# 4. **Hypothetical Document Embeddings (HyDE)**: Generating synthetic document passages to project search queries directly into *Document Vector Space*.
# 5. **Semantic Collection Routing**: Dynamically routing queries to specialized vector collections based on centroid cosine proximity.
# 6. **Presenter Dashboard & Transformation Inspector (`# collapse_input`)**: Auto-collapsing ASCII query transformation and routing inspector.
#
# ---
#
# ```mermaid
# graph TD
#     RawQuery["Raw User Query"] --> Router{"Semantic Collection Router"}
#     
#     RawQuery --> MultiQuery["1. Multi-Query Expander (K Variations)"]
#     RawQuery --> StepBack["2. Step-Back Abstractor (High-Level Concepts)"]
#     RawQuery --> Decompose["3. Sub-Query Decomposer (Atomic Queries)"]
#     RawQuery --> HyDE["4. HyDE Generator (Document Space Projection)"]
#     
#     MultiQuery --> ParallelSearch["Parallel Multi-Path Search"]
#     StepBack --> ParallelSearch
#     Decompose --> ParallelSearch
#     HyDE --> ParallelSearch
#     
#     ParallelSearch --> RRFFusion["Reciprocal Rank Fusion (RRF)"]
#     Router -->|Collection Target| TargetDB[("Target Vector Collection")]
#     RRFFusion --> TargetDB
#     TargetDB --> UnifiedTopK["Unified Re-Ranked Top-K Context"]
# ```
#
# ---

# %%
import math
import os
import re
from collections import defaultdict
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
        print(f"[INFO] Query Transformation Hardware: CUDA GPU -> {device_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Query Transformation Hardware: Apple Silicon MPS GPU")
    else:
        device = torch.device("cpu")
        print("[INFO] Query Transformation Hardware: CPU (Optimized SIMD)")
    return device

DEVICE = detect_compute_device()

# Local LLM Client Configuration
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_ENDPOINT", "http://localhost:5055/v1")
openai_client = OpenAI(base_url=LOCAL_LLM_URL, api_key="dummy")

# %% [markdown]
# ## Section 1: The Query-Document Alignment Problem
#
# In standard vector search, the inner product measures directional alignment:
#
# $$\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$
#
# When $\mathbf{q}$ is an interrogative question (*"Why does GPU memory overflow?"*) and $\mathbf{d}$ is an affirmative specification (*"Error 503 denotes KV-cache saturation on CUDA device"*), their representations in embedding space can be distant.
#
# **Query Transformation** bridges this gap by rewriting, abstracting, or expanding $\mathbf{q}$ prior to vector search.

# %%
class DenseFeatureProjector:
    """Fast GPU-accelerated subword and lexical feature projector into R^D."""

    def __init__(self, dimension: int = 128, device: Optional[torch.device] = None):
        self.dimension = dimension
        self.device = device or DEVICE

    def embed_text(self, text: str) -> np.ndarray:
        """Deterministic subword feature projection for query and document embedding."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return vec
        for i, word in enumerate(words):
            weight = 1.0 / math.sqrt(i + 1)
            h = abs(hash(word)) % self.dimension
            vec[h] += weight * 1.5
            if len(word) >= 3:
                for n in range(3, min(6, len(word) + 1)):
                    for start in range(len(word) - n + 1):
                        ngram = word[start : start + n]
                        sub_h = abs(hash(ngram)) % self.dimension
                        vec[sub_h] += 0.4
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity over unit vectors."""
        return float(np.dot(vec_a, vec_b))

# %% [markdown]
# ## Section 2: Multi-Query Expansion, Step-Back Prompting & Sub-Query Decomposition
#
# 1. **Multi-Query Expansion:** Formulates multiple stylistic and perspective variations of the query to capture different lexical facets.
# 2. **Step-Back Prompting:** Extracts the underlying high-level concept (e.g. *Specific CUDA crash* $\rightarrow$ *GPU memory management architecture*).
# 3. **Sub-Query Decomposition:** Breaks multi-entity queries into discrete single-concept searches.

# %%
class QueryTransformationSuite:
    """Production Query Transformation Suite implementing Multi-Query, Step-Back, and Sub-Query splitting."""

    def __init__(self, projector: DenseFeatureProjector):
        self.projector = projector

    # 1. Multi-Query Expansion
    def expand_multi_query(self, raw_query: str) -> List[str]:
        """Generate multiple diverse search variations covering lexical, technical, and conceptual facets."""
        clean_q = raw_query.strip()
        variations = [
            clean_q,
            f"Technical architecture and implementation of {clean_q}",
            f"How does {clean_q} work under the hood?",
            f"Trade-offs, performance benchmarks, and limitations of {clean_q}",
            f"Troubleshooting and common failure modes in {clean_q}"
        ]
        return variations

    # 2. Step-Back Conceptual Abstraction
    def generate_step_back_query(self, specific_query: str) -> Tuple[str, str]:
        """Abstract a specific low-level query into its overarching foundational principle."""
        q_lower = specific_query.lower()
        
        if "503" in q_lower or "overflow" in q_lower or "memory" in q_lower or "crash" in q_lower:
            step_back = "What are the fundamental principles of GPU memory allocation and KV-cache management in LLMs?"
            concept = "GPU Memory & KV-Cache Management"
        elif "hnsw" in q_lower or "ivf" in q_lower or "ann" in q_lower or "index" in q_lower:
            step_back = "What are the core mathematical trade-offs between graph-based and quantization-based Approximate Nearest Neighbor indexes?"
            concept = "Vector Indexing & ANN Algorithms"
        elif "bm25" in q_lower or "sparse" in q_lower or "rrf" in q_lower:
            step_back = "How do lexical probabilistic ranking models differ from continuous semantic dense representations?"
            concept = "Sparse vs Dense Retrieval Fundamentals"
        else:
            step_back = f"What is the high-level system architecture and operational mechanics governing {specific_query}?"
            concept = "General System Architecture"

        return step_back, concept

    # 3. Sub-Query Decomposition
    def decompose_complex_query(self, complex_query: str) -> List[str]:
        """Deconstruct complex multi-intent questions into atomic sub-queries."""
        delimiters = [r"\band\b", r"\bvs\b", r"\bversus\b", r"\bcompared to\b", r";", r","]
        pattern = "|".join(delimiters)
        parts = [p.strip() for p in re.split(pattern, complex_query, flags=re.IGNORECASE) if len(p.strip()) > 3]
        
        if len(parts) <= 1:
            return [complex_query]
        
        sub_queries = []
        for part in parts:
            if not any(part.lower().startswith(prefix) for prefix in ["how", "what", "why"]):
                sub_queries.append(f"What is {part}?")
            else:
                sub_queries.append(part)
        return sub_queries

# %% [markdown]
# ### Demo 1: Multi-Query, Step-Back & Sub-Query Decomposition
#
# Below, we execute the transformation suite on representative technical queries.

# %%
projector = DenseFeatureProjector(dimension=128, device=DEVICE)
query_suite = QueryTransformationSuite(projector)

test_query = "ERR_KV_CACHE_OVERFLOW_503 on CUDA GPU"
multi_queries = query_suite.expand_multi_query("Cache-Augmented Generation")
step_back_q, concept = query_suite.generate_step_back_query(test_query)
decomposed = query_suite.decompose_complex_query("Compare HNSW indexing latency versus Product Quantization memory compression")

print("=== [Query Transformation Suite Output] ===")
print("\n[1] Multi-Query Expansions:")
for i, mq in enumerate(multi_queries, 1):
    print(f"  [{i}] {mq}")

print(f"\n[2] Step-Back Prompting:")
print(f"  • Specific Query: '{test_query}'")
print(f"  • Abstracted Concept: '{concept}'")
print(f"  • Step-Back Query: '{step_back_q}'")

print(f"\n[3] Sub-Query Decomposition:")
for i, sq in enumerate(decomposed, 1):
    print(f"  [{i}] Sub-Query: '{sq}'")

# %% [markdown]
# ## Section 3: Hypothetical Document Embeddings (HyDE)
#
# **HyDE** inverts the traditional retrieval sequence:
# 1. Given an ambiguous query $Q$, an LLM generates a synthetic *hypothetical answer* $D_{\text{hypo}}$.
# 2. Even if $D_{\text{hypo}}$ contains minor hallucinations, its phrasing resides firmly in **Document Space** (declarative paragraphs, technical vocabulary).
# 3. Embedding $D_{\text{hypo}}$ produces a vector $\mathbf{e}(D_{\text{hypo}})$ that locates true relevant documents with significantly higher cosine proximity than embedding the raw query $\mathbf{e}(Q)$.

# %%
class HyDEGenerator:
    """Hypothetical Document Embeddings (HyDE) generator with local LLM routing and robust fallback."""

    def __init__(self, projector: DenseFeatureProjector, client: OpenAI = openai_client):
        self.projector = projector
        self.client = client

    def generate_hypothetical_document(self, query: str) -> str:
        """Generate a synthetic answer passage to shift the query into document vector space."""
        # Attempt local LLM call
        try:
            response = self.client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": "You are a technical knowledge engine. Write a concise, declarative passage answering the user question."},
                    {"role": "user", "content": query}
                ],
                max_tokens=120,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception:
            # Deterministic local surrogate when local LLM server is not reachable
            q_lower = query.lower()
            if "cag" in q_lower or "cache" in q_lower or "kv" in q_lower:
                return (
                    "Cache-Augmented Generation (CAG) preloads static context documents directly into the GPU "
                    "Key-Value (KV) cache of Large Language Models. By eliminating runtime vector database lookups, "
                    "CAG achieves sub-millisecond prefill latency and ensures 100% context recall across large documents."
                )
            elif "hnsw" in q_lower or "vector" in q_lower or "index" in q_lower:
                return (
                    "Hierarchical Navigable Small World (HNSW) is a multi-layer graph index for Approximate Nearest "
                    "Neighbor (ANN) search. It provides logarithmic search complexity and high recall by maintaining "
                    "sparse skip connections on upper layers and dense local neighborhoods at the base layer."
                )
            else:
                return (
                    f"Technical documentation detailing {query}. The system architecture provides high throughput, "
                    f"robust score fusion, and optimized memory allocation across GPU and CPU execution environments."
                )

    def retrieve_with_hyde(
        self,
        query: str,
        corpus: List[Dict[str, str]],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """Execute HyDE retrieval: Query -> Synthetic Document -> Document Space Search."""
        hypo_doc = self.generate_hypothetical_document(query)
        hypo_vec = self.projector.embed_text(hypo_doc)
        raw_vec = self.projector.embed_text(query)

        raw_scores = []
        hyde_scores = []

        for doc in corpus:
            doc_vec = self.projector.embed_text(doc["text"])
            raw_sim = self.projector.compute_similarity(raw_vec, doc_vec)
            hyde_sim = self.projector.compute_similarity(hypo_vec, doc_vec)
            raw_scores.append((doc["id"], raw_sim, doc["text"]))
            hyde_scores.append((doc["id"], hyde_sim, doc["text"]))

        raw_scores.sort(key=lambda x: x[1], reverse=True)
        hyde_scores.sort(key=lambda x: x[1], reverse=True)

        return {
            "query": query,
            "hypothetical_document": hypo_doc,
            "raw_top_k": raw_scores[:top_k],
            "hyde_top_k": hyde_scores[:top_k]
        }

# %% [markdown]
# ### Demo 2: HyDE Document Space Projection Demonstration
#
# Below, we compare standard raw query search vs. HyDE search on an enterprise knowledge corpus.

# %%
enterprise_knowledge_base = [
    {
        "id": "doc_cag_spec",
        "text": "Cache-Augmented Generation preloads static prompt tokens directly into the LLM KV-cache to eliminate retrieval latency."
    },
    {
        "id": "doc_hnsw_spec",
        "text": "HNSW graphs maintain logarithmic query routing across hierarchical layers for sub-millisecond vector retrieval."
    },
    {
        "id": "doc_bm25_spec",
        "text": "BM25Okapi implements Robertson-Spärck Jones probabilistic term weighting with document length normalization."
    },
    {
        "id": "doc_peft_spec",
        "text": "Parameter-Efficient Fine-Tuning adapts low-rank LoRA matrices without altering frozen base weights."
    }
]

hyde_engine = HyDEGenerator(projector)
hyde_res = hyde_engine.retrieve_with_hyde("How does KV prefill eliminate vector search?", enterprise_knowledge_base, top_k=2)

print("=== [HyDE Retrieval Execution] ===")
print(f"User Query: '{hyde_res['query']}'")
print(f"\nGenerated Hypothetical Document (Document Space):\n'{hyde_res['hypothetical_document']}'")

print(f"\nStandard Raw Query Top Match: {hyde_res['raw_top_k'][0][0]} (Cosine: {hyde_res['raw_top_k'][0][1]:.3f})")
print(f"HyDE Document Space Top Match: {hyde_res['hyde_top_k'][0][0]} (Cosine: {hyde_res['hyde_top_k'][0][1]:.3f})")

# %% [markdown]
# ## Section 4: Semantic Query Routing Across Specialized Collections
#
# In enterprise architectures, knowledge is partitioned across specialized vector databases:
# - **Collection A (Architecture & Deep Dives):** High-level design, whitepapers, benchmarks.
# - **Collection B (API & Code Reference):** Function signatures, parameters, return types.
# - **Collection C (Troubleshooting & Error Codes):** Stack traces, HTTP error codes, GPU memory issues.
# - **Collection D (General FAQs):** Billing, onboarding, general overviews.
#
# The **Semantic Collection Router** computes cosine proximity between the query embedding and collection exemplars to select the optimal database without hardcoded if-else trees.

# %%
class SemanticCollectionRouter:
    """Semantic Query Router classifying queries to target vector collections using exemplar centroids."""

    def __init__(self, projector: DenseFeatureProjector):
        self.projector = projector
        self.collections: Dict[str, Dict[str, Any]] = {}

    def register_collection(self, collection_name: str, description: str, exemplars: List[str]):
        """Register a collection and compute its semantic centroid vector."""
        vectors = np.array([self.projector.embed_text(ex) for ex in exemplars], dtype=np.float32)
        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm

        self.collections[collection_name] = {
            "description": description,
            "exemplars": exemplars,
            "centroid": centroid
        }

    def route_query(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        """Route query to the collection with highest semantic centroid similarity."""
        q_vec = self.projector.embed_text(query)
        scores = {}

        for name, data in self.collections.items():
            sim = self.projector.compute_similarity(q_vec, data["centroid"])
            scores[name] = round(sim, 4)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_collection, best_score = ranked[0]
        return best_collection, best_score, scores

# %% [markdown]
# ### Demo 3: Semantic Collection Routing Demonstration
#
# Below, we register four specialized collections and route diverse user queries.

# %%
collection_router = SemanticCollectionRouter(projector)

collection_router.register_collection(
    collection_name="col_architecture",
    description="High-level architecture, CAG, GraphRAG, and vector scaling blueprints",
    exemplars=[
        "Cache-Augmented Generation architectural blueprint and KV cache design",
        "GraphRAG hierarchical community detection knowledge graphs",
        "Vector database indexing strategies and Pareto efficiency"
    ]
)

collection_router.register_collection(
    collection_name="col_api_code",
    description="Python API signatures, class constructors, parameters, and code snippets",
    exemplars=[
        "def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float",
        "class FAISSIVFEngine constructor parameters and nprobe options",
        "AST code parser and multi-format ingestion engine functions"
    ]
)

collection_router.register_collection(
    collection_name="col_troubleshooting",
    description="Error codes, stack traces, CUDA memory leaks, and GPU crash resolution",
    exemplars=[
        "ERR_KV_CACHE_OVERFLOW_503 GPU VRAM exhaustion on CUDA device",
        "CUDA out of memory exception during batch tensor multiplication",
        "Fixing index corrupted error in FAISS GPU memory allocation"
    ]
)

routing_test_queries = [
    "How do I fix ERR_KV_CACHE_OVERFLOW_503 in my PyTorch script?",
    "Show me the function signature for compute_cosine_similarity",
    "What are the system design differences between CAG and GraphRAG?"
]

print("=== [Semantic Collection Router Output] ===")
for q in routing_test_queries:
    target, conf, all_conf = collection_router.route_query(q)
    print(f"\nQuery: '{q}'")
    print(f"  • Routed Target:   [{target}] (Confidence: {conf:.3f})")
    print(f"  • Collection Dist: {all_conf}")

# %% [markdown]
# ## Section 5: End-to-End Query Transformation Benchmark
#
# We evaluate the retrieval accuracy of **Raw Query** vs. **Multi-Query Expansion (RRF)** vs. **Step-Back** vs. **HyDE** on a complex benchmark test set.

# %%
def execute_rrf_multi_search(queries: List[str], corpus: List[Dict[str, str]], projector: DenseFeatureProjector, top_k: int = 3) -> List[str]:
    """Execute multiple query vectors and fuse rankings with RRF."""
    rrf_scores = defaultdict(float)
    k_const = 60
    
    for q in queries:
        q_vec = projector.embed_text(q)
        scored = []
        for doc in corpus:
            sim = projector.compute_similarity(q_vec, projector.embed_text(doc["text"]))
            scored.append((doc["id"], sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (doc_id, _) in enumerate(scored, 1):
            rrf_scores[doc_id] += 1.0 / (k_const + rank)

    sorted_res = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_res[:top_k]]

class QueryBenchmarkHarness:
    """Benchmark comparing retrieval accuracy across transformation techniques."""

    def __init__(self, suite: QueryTransformationSuite, hyde: HyDEGenerator, corpus: List[Dict[str, str]]):
        self.suite = suite
        self.hyde = hyde
        self.corpus = corpus

    def run_benchmark(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        raw_hits, multi_hits, step_hits, hyde_hits = 0, 0, 0, 0

        for case in test_cases:
            q = case["query"]
            target = case["target_id"]

            # 1. Raw Query Search
            raw_vec = self.suite.projector.embed_text(q)
            raw_scored = sorted([(d["id"], self.suite.projector.compute_similarity(raw_vec, self.suite.projector.embed_text(d["text"]))) for d in self.corpus], key=lambda x: x[1], reverse=True)
            raw_top1 = raw_scored[0][0]

            # 2. Multi-Query RRF
            mq_list = self.suite.expand_multi_query(q)
            mq_top1 = execute_rrf_multi_search(mq_list, self.corpus, self.suite.projector, top_k=1)[0]

            # 3. Step-Back Search
            sb_q, _ = self.suite.generate_step_back_query(q)
            sb_vec = self.suite.projector.embed_text(sb_q)
            sb_scored = sorted([(d["id"], self.suite.projector.compute_similarity(sb_vec, self.suite.projector.embed_text(d["text"]))) for d in self.corpus], key=lambda x: x[1], reverse=True)
            sb_top1 = sb_scored[0][0]

            # 4. HyDE Search
            hyde_res = self.hyde.retrieve_with_hyde(q, self.corpus, top_k=1)
            hyde_top1 = hyde_res["hyde_top_k"][0][0]

            raw_hits += int(raw_top1 == target)
            multi_hits += int(mq_top1 == target)
            step_hits += int(sb_top1 == target)
            hyde_hits += int(hyde_top1 == target)

            results.append({
                "query": q,
                "target": target,
                "raw_top1": raw_top1,
                "multi_top1": mq_top1,
                "step_top1": sb_top1,
                "hyde_top1": hyde_top1
            })

        n = len(test_cases)
        return {
            "detailed": results,
            "raw_acc": round(raw_hits / n, 2),
            "multi_acc": round(multi_hits / n, 2),
            "step_acc": round(step_hits / n, 2),
            "hyde_acc": round(hyde_hits / n, 2)
        }

# %% [markdown]
# ### Demo 4: Comprehensive Transformation Accuracy Benchmark Run
#
# Below, we execute the transformation benchmark suite across challenging ambiguous test queries.

# %%
benchmark_test_cases = [
    {
        "query": "prefill speedup and zero retrieval latency",
        "target_id": "doc_cag_spec"
    },
    {
        "query": "hierarchical skip connections logarithmic search",
        "target_id": "doc_hnsw_spec"
    },
    {
        "query": "term saturation parameter k1 and length normalization b",
        "target_id": "doc_bm25_spec"
    }
]

bench_harness = QueryBenchmarkHarness(query_suite, hyde_engine, enterprise_knowledge_base)
bench_report = bench_harness.run_benchmark(benchmark_test_cases)

print("=== [Query Transformation Retrieval Accuracy Benchmark] ===")
print(f"{'Query Snippet':<32}{'Target':<16}{'Raw Match':<14}{'Multi-RRF':<14}{'HyDE Match':<14}")
print("-" * 90)
for r in bench_report["detailed"]:
    print(f"{r['query'][:30]:<32}{r['target']:<16}{r['raw_top1']:<14}{r['multi_top1']:<14}{r['hyde_top1']:<14}")

print("\nTop-1 Accuracy Summary:")
print(f"  • Raw Direct Query:       {bench_report['raw_acc']*100:.1f}%")
print(f"  • Multi-Query Expansion:  {bench_report['multi_acc']*100:.1f}%")
print(f"  • Step-Back Prompting:    {bench_report['step_acc']*100:.1f}%")
print(f"  • HyDE Document Space:    {bench_report['hyde_acc']*100:.1f}%")

# %% [markdown]
# ## Section 6: Presenter Dashboard & Transformation Visualizer
#
# Below is the consolidated presenter dashboard rendering an ASCII transformation pipeline audit log.

# %%
# collapse_input
def display_query_transform_dashboard(router: SemanticCollectionRouter, bench_results: Dict[str, Any]):
    """Render a clean ASCII visualizer of query transformation and routing performance."""
    print("=" * 80)
    print("        KNOWLEDGE RETRIEVAL A-Z: MODULE 05 QUERY TRANSFORMATION DASHBOARD")
    print("=" * 80)
    
    print("\n[1] REGISTERED SPECIALIZED COLLECTIONS")
    for name, data in router.collections.items():
        print(f"  • [{name:<18}] Exemplars: {len(data['exemplars'])} | Centroid Norm: 1.000")
        print(f"    Description: {data['description']}")

    print("\n[2] TRANSFORMATION ACCURACY COMPARISON MATRIX")
    print(f"  {'Method':<28}{'Top-1 Accuracy':<18}{'Primary Strengths':<32}")
    print("  " + "-" * 78)
    print(f"  {'Raw Direct Query':<28}{bench_results['raw_acc']*100:>5.1f}%             {'Fastest, but fails on vocabulary mismatch':<32}")
    print(f"  {'Multi-Query Expansion (RRF)':<28}{bench_results['multi_acc']*100:>5.1f}%             {'High recall across diverse phrasings':<32}")
    print(f"  {'Step-Back Prompting':<28}{bench_results['step_acc']*100:>5.1f}%             {'Captures fundamental domain principles':<32}")
    print(f"  {'HyDE (Document Space)':<28}{bench_results['hyde_acc']*100:>5.1f}%             {'Bridges the query-document modality gap':<32}")

    print("\n[3] PRODUCTION RECOMMENDATION")
    print("  • Use Semantic Routing FIRST to isolate the correct domain collection.")
    print("  • Use HyDE for ambiguous single-phrase conceptual questions.")
    print("  • Use Multi-Query RRF for high-stakes enterprise search where recall is critical.")

    print("\n" + "=" * 80)
    print("  [OK] Module 05 Complete! Proceeding to Module 06: Context Reranking & Compression.")
    print("=" * 80)

# Render Dashboard
display_query_transform_dashboard(collection_router, bench_report)

# %% [markdown]
# ## Section 7: Summary & Transition to Module 06
#
# In this module, we engineered production query transformation and routing pipelines:
# - Solved the **Query-Document Modality Gap** using **Multi-Query Expansion** and **Reciprocal Rank Fusion (RRF)**.
# - Implemented **Step-Back Prompting** to retrieve foundational background knowledge.
# - Engineered **Hypothetical Document Embeddings (HyDE)** to project queries into declarative document space.
# - Built the **Semantic Collection Router** using centroid cosine proximity to direct queries to specialized vector collections.
#
# In **Module 06 (Context Reranking & Compression)**, we take the top-50 candidates retrieved by our transformed queries and apply **Cross-Encoder Rerankers**, **LLMLingua Token Compression**, and **Lost-in-the-Middle Attention Reordering**.
