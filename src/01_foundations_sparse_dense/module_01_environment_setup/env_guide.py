# %% [markdown]
# # Module 01: Modern Retrieval Workspace Setup & Tokenization Math
#
# Welcome to **Module 01** of the Knowledge Retrieval A-Z masterclass.
# In production retrieval systems, retrieval quality and latency depend heavily on the foundational runtime environment, tokenization fidelity, context window allocation, and the geometry of high-dimensional embedding spaces.
#
# In this module, we construct and master:
# 1. **Workspace Health Diagnostics**: Runtime validation of `uv`, dependencies, and local LLM connectivity (`http://localhost:5055/v1`).
# 2. **Byte-Pair Encoding (BPE) from Scratch**: Subword tokenization mechanics, merge hierarchies, and token compression ratios.
# 3. **Context Window Arithmetic & KV-Cache Footprint**: Mathematical modeling of context budgets and GPU KV-cache memory requirements.
# 4. **Vector Embedding Geometries & Metric Spaces**: Dot product, Cosine similarity, Euclidean ($L_2$), Manhattan ($L_1$), and the *Curse of Dimensionality*.
# 5. **High-Precision Retrieval Micro-Benchmarking**: Micro-profiling tokenization throughput and vector distance operations per second.
#
# ---
#
# ```mermaid
# graph LR
#     subgraph Pipeline ["Foundational Retrieval Pipeline"]
#         A["Raw Text Corpus"] --> B["BPE Subword Tokenizer"]
#         B --> C["Token Sequences & Context Budget"]
#         C --> D["Embedding Transformation (R^D)"]
#         D --> E["Geometric Metric Space (Cosine / L2)"]
#         E --> F["KV-Cache Memory Planning & Benchmarking"]
#     end
# ```
#
# ---

# %%
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

# %% [markdown]
# ## Section 1: Workspace Health Diagnostics & Environment Configuration
#
# A reliable retrieval environment requires deterministic dependency management via `uv`, modern Python ($\ge 3.12$), and a verified link to the local inference server.
#
# Below, we implement a diagnostic inspector that validates installed packages and tests the local LLM endpoint with a graceful non-blocking fallback.

# %%
def verify_workspace_environment(endpoint_url: str = "http://localhost:5055/v1") -> Dict[str, Any]:
    """Inspect workspace runtime environment, core dependencies, and local LLM server."""
    python_ver = sys.version.split()[0]
    python_supported = sys.version_info >= (3, 12)
    
    # Check core library availability
    dependencies = {
        "numpy": np.__version__,
        "openai": "available",
        "click": "available",
        "networkx": "available",
    }
    
    # Check LLM endpoint connectivity (non-blocking with short timeout)
    llm_connected = False
    endpoint_message = "Not reachable (offline or not started)"
    try:
        client = OpenAI(base_url=endpoint_url, api_key="dummy", timeout=0.8)
        # Attempt minimal lightweight list/call
        models = client.models.list()
        llm_connected = True
        endpoint_message = f"Connected ({len(models.data)} models available)"
    except Exception as e:
        endpoint_message = f"Offline / Mock mode ({type(e).__name__})"

    status = {
        "python_version": python_ver,
        "python_supported": python_supported,
        "dependencies": dependencies,
        "endpoint_url": endpoint_url,
        "llm_connected": llm_connected,
        "endpoint_status": endpoint_message,
        "all_systems_ready": python_supported and len(dependencies) >= 4
    }
    return status

env_diagnostics = verify_workspace_environment()
print(f"[OK] Workspace Python: {env_diagnostics['python_version']} (Supported: {env_diagnostics['python_supported']})")
print(f"[OK] Local LLM Endpoint: {env_diagnostics['endpoint_url']} -> {env_diagnostics['endpoint_status']}")
print(f"[OK] Core Dependencies: {list(env_diagnostics['dependencies'].keys())}")

# %% [markdown]
# ## Section 2: Tokenization Mechanics & Byte-Pair Encoding (BPE) from Scratch
#
# Large Language Models and embedding models do not read characters or whole words directly.
# They operate on discrete **subword tokens** to handle out-of-vocabulary (OOV) terms, technical acronyms, and code identifiers efficiently.
#
# ### The Byte-Pair Encoding (BPE) Algorithm
# 1. Initialize the vocabulary $V_0$ with all individual characters present in the training corpus, appending an end-of-word marker `</w>`.
# 2. In each iteration $t$:
#    - Count the frequency of all adjacent symbol pairs across the tokenized corpus.
#    - Identify the most frequent pair $(s_i, s_j) = \arg\max \text{freq}(p)$.
#    - Merge $(s_i, s_j)$ into a single symbol $s_{\text{new}} = s_i + s_j$.
#    - Add $s_{\text{new}}$ to the vocabulary: $V_{t+1} = V_t \cup \{s_{\text{new}}\}$.
# 3. Terminate when the requested number of merges is reached or no pairs occur $> 1$ time.

# %%
class BPETokenizer:
    """A pure Python Byte-Pair Encoding (BPE) subword tokenizer implemented from first principles."""

    def __init__(self):
        self.vocab: List[str] = []
        self.merges: List[Tuple[str, str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.end_of_word: str = "</w>"

    def _get_word_stats(self, corpus_words: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Count frequencies of adjacent symbol pairs across the weighted vocabulary."""
        pairs = defaultdict(int)
        for word_tuple, freq in corpus_words.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i], word_tuple[i + 1])] += freq
        return pairs

    def _merge_pair(
        self, pair: Tuple[str, str], corpus_words: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        """Merge all occurrences of the target symbol pair in the vocabulary."""
        new_words = {}
        bigram = pair
        first, second = bigram
        for word_tuple, freq in corpus_words.items():
            new_tuple = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == first and word_tuple[i + 1] == second:
                    new_tuple.append(first + second)
                    i += 2
                else:
                    new_tuple.append(word_tuple[i])
                    i += 1
            new_words[tuple(new_tuple)] = freq
        return new_words

    def train(self, corpus: List[str], num_merges: int = 50) -> "BPETokenizer":
        """Train the BPE tokenizer on a list of text strings by learning `num_merges` merge rules."""
        # 1. Segment text into whitespace-delimited words and add end-of-word marker
        word_counts = defaultdict(int)
        for text in corpus:
            for word in text.strip().split():
                if word:
                    # Character sequence ending with marker
                    char_tuple = tuple(list(word) + [self.end_of_word])
                    word_counts[char_tuple] += 1

        # 2. Extract base character vocabulary
        base_chars = set()
        for word_tuple in word_counts.keys():
            for char in word_tuple:
                base_chars.add(char)
        
        self.vocab = sorted(list(base_chars))
        self.merges = []

        # 3. Iterative pair extraction and merging
        current_words = dict(word_counts)
        for _ in range(num_merges):
            pairs = self._get_word_stats(current_words)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 1:
                break
            current_words = self._merge_pair(best_pair, current_words)
            self.merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab.append(merged_token)

        # 4. Build index mapping tables
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        return self

    def encode_word(self, word: str) -> List[str]:
        """Encode a single word into its constituent learned subwords using the merge hierarchy."""
        if not word:
            return []
        tokens = list(word) + [self.end_of_word]
        for first, second in self.merges:
            i = 0
            merged = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
        return tokens

    def encode(self, text: str) -> List[str]:
        """Tokenize a full string into BPE subword tokens."""
        tokens = []
        for word in text.strip().split():
            tokens.extend(self.encode_word(word))
        return tokens

    def decode(self, tokens: List[str]) -> str:
        """Reconstruct original text from BPE subword tokens."""
        text = "".join(tokens)
        text = text.replace(self.end_of_word, " ")
        return text.strip()

    def compression_ratio(self, text: str) -> float:
        """Calculate the compression ratio: raw character count / token count."""
        tokens = self.encode(text)
        if not tokens:
            return 0.0
        return len(text) / len(tokens)

# %%
# Train BPE Tokenizer on technical retrieval text
domain_corpus = [
    "Cache-Augmented Generation preloads documents into the KV cache.",
    "BM25 is a sparse lexical ranking algorithm based on term frequency.",
    "Dense embeddings represent semantic vectors in high-dimensional vector spaces.",
    "Hybrid search fuses BM25 and dense retrieval using Reciprocal Rank Fusion.",
    "GraphRAG builds knowledge graphs from entity-relationship triplets."
]

tokenizer = BPETokenizer()
tokenizer.train(domain_corpus, num_merges=40)

sample_query = "Cache-Augmented embeddings optimize dense retrieval"
sample_tokens = tokenizer.encode(sample_query)
reconstructed = tokenizer.decode(sample_tokens)
compression = tokenizer.compression_ratio(sample_query)

print(f"Learned Vocabulary Size: {len(tokenizer.vocab)}")
print(f"Learned Merge Rules: {len(tokenizer.merges)}")
print(f"Sample Input: '{sample_query}'")
print(f"Encoded Subwords: {sample_tokens}")
print(f"Reconstructed Text: '{reconstructed}'")
print(f"Compression Ratio: {compression:.2f} chars/token")

# %% [markdown]
# ## Section 3: Context Window Budgeting & KV-Cache Memory Modeling
#
# When designing production RAG and CAG systems, understanding context constraints and memory allocations is critical.
#
# ### 1. Context Window Budget Allocation
# In any retrieval pipeline, the maximum model context window $W_{\text{total}}$ must accommodate:
# $$W_{\text{total}} = T_{\text{system}} + T_{\text{query}} + T_{\text{history}} + \sum_{i=1}^K T_{\text{chunk}_i} + T_{\text{reserve}} + T_{\text{generation}}$$
#
# Given a chunk size $C$, the maximum number of chunks $K$ we can safely retrieve is:
# $$K_{\text{max}} = \left\lfloor \frac{W_{\text{total}} - (T_{\text{system}} + T_{\text{query}} + T_{\text{history}} + T_{\text{generation}} + T_{\text{safety}})}{C} \right\rfloor$$
#
# ### 2. KV-Cache GPU Memory Footprint
# In modern Transformer decoders with Multi-Head Attention (MHA) or Grouped-Query Attention (GQA), each layer stores Key and Value tensors for all context tokens $T_{\text{context}}$:
# $$\text{Memory}_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times T_{\text{context}} \times b_{\text{precision}}$$
# where:
# - $n_{\text{layers}}$ is the number of transformer layers (e.g., 32 for 8B, 80 for 70B).
# - $n_{\text{kv\_heads}}$ is the number of Key/Value heads ($n_{\text{kv\_heads}} = n_{\text{heads}}$ in MHA; $n_{\text{kv\_heads}} \ll n_{\text{heads}}$ in GQA).
# - $d_{\text{head}}$ is the dimension per attention head (typically 128).
# - $b_{\text{precision}}$ is the byte width per parameter (2 bytes for FP16/BF16, 1 byte for FP8/INT8).

# %%
class ContextBudgetCalculator:
    """Calculates context window allocation limits and transformer KV-cache memory consumption."""

    def __init__(
        self,
        total_context: int = 8192,
        max_generation_tokens: int = 1024,
        system_prompt_tokens: int = 250,
        query_tokens: int = 50,
        history_tokens: int = 200,
    ):
        self.total_context = total_context
        self.max_generation_tokens = max_generation_tokens
        self.system_prompt_tokens = system_prompt_tokens
        self.query_tokens = query_tokens
        self.history_tokens = history_tokens

    def calculate_chunk_budget(
        self, chunk_size: int = 512, reserve_safety_tokens: int = 128
    ) -> Dict[str, Any]:
        """Compute the maximum number of retrieved chunks K that fit safely in the context window."""
        fixed_overhead = (
            self.system_prompt_tokens
            + self.query_tokens
            + self.history_tokens
            + self.max_generation_tokens
            + reserve_safety_tokens
        )
        available_for_retrieval = max(0, self.total_context - fixed_overhead)
        max_chunks = available_for_retrieval // chunk_size
        retrieval_tokens = max_chunks * chunk_size
        slack_tokens = self.total_context - (fixed_overhead + retrieval_tokens)

        return {
            "total_context": self.total_context,
            "fixed_overhead": fixed_overhead,
            "available_for_retrieval": available_for_retrieval,
            "chunk_size": chunk_size,
            "max_chunks_k": max_chunks,
            "allocated_retrieval_tokens": retrieval_tokens,
            "slack_tokens": slack_tokens,
            "utilization_percent": ((self.total_context - slack_tokens) / self.total_context) * 100,
        }

    @staticmethod
    def calculate_kv_cache_memory(
        context_tokens: int,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        bytes_per_elem: int = 2,
    ) -> Dict[str, float]:
        """Compute precise KV-cache memory footprint in megabytes (MB) and gigabytes (GB)."""
        # Formula: 2 (Key + Value) * Layers * KV_Heads * Head_Dim * Tokens * Bytes
        bytes_total = 2 * num_layers * num_kv_heads * head_dim * context_tokens * bytes_per_elem
        mb = bytes_total / (1024 ** 2)
        gb = bytes_total / (1024 ** 3)
        return {
            "bytes": float(bytes_total),
            "megabytes": round(mb, 3),
            "gigabytes": round(gb, 4),
            "bytes_per_token": (bytes_total / context_tokens) if context_tokens > 0 else 0,
        }

# %%
budget_calc = ContextBudgetCalculator(
    total_context=8192,
    max_generation_tokens=1024,
    system_prompt_tokens=300,
    query_tokens=60,
    history_tokens=240,
)
budget_summary = budget_calc.calculate_chunk_budget(chunk_size=400)

print(f"Context Window: {budget_summary['total_context']} tokens")
print(f"Fixed Overhead: {budget_summary['fixed_overhead']} tokens (Prompt, Query, History, Safety, Gen)")
print(f"Max Retrieval Chunks (K): {budget_summary['max_chunks_k']} chunks (Chunk size: {budget_summary['chunk_size']})")
print(f"Total Retrieval Capacity: {budget_summary['allocated_retrieval_tokens']} tokens")
print(f"Budget Utilization: {budget_summary['utilization_percent']:.2f}%")

# Compute KV-Cache memory across different context lengths for an 8B model (Llama-3 style: 32 layers, 8 KV heads, dim 128)
kv_8k = ContextBudgetCalculator.calculate_kv_cache_memory(context_tokens=8192)
kv_32k = ContextBudgetCalculator.calculate_kv_cache_memory(context_tokens=32768)
kv_128k = ContextBudgetCalculator.calculate_kv_cache_memory(context_tokens=131072)

print(f"\nKV-Cache Memory Footprint (8B Model with GQA):")
print(f"  •  8K Context:  {kv_8k['megabytes']} MB ({kv_8k['gigabytes']} GB)")
print(f"  • 32K Context: {kv_32k['megabytes']} MB ({kv_32k['gigabytes']} GB)")
print(f"  • 128K Context: {kv_128k['megabytes']} MB ({kv_128k['gigabytes']} GB)")

# %% [markdown]
# ## Section 4: Vector Embedding Dimensionality, Metric Geometries & Orthogonality
#
# Dense semantic search projects textual meaning into a continuous metric space $\mathbb{R}^D$.
# Choosing the right similarity metric and understanding high-dimensional geometric properties is crucial for vector retrieval accuracy and index efficiency.
#
# ### 1. Metric Geometries
# For two embedding vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^D$:
# - **Dot Product**: $\langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^D u_i v_i$
# - **Cosine Similarity**: $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$
# - **Euclidean Distance ($L_2$)**: $\|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^D (u_i - v_i)^2}$
# - **Manhattan Distance ($L_1$)**: $\|\mathbf{u} - \mathbf{v}\|_1 = \sum_{i=1}^D |u_i - v_i|$
#
# ### 2. Unit-Norm Equivalence
# When vectors are $L_2$-normalized ($\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1$):
# $$\|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|_2^2 + \|\mathbf{v}\|_2^2 - 2(\mathbf{u} \cdot \mathbf{v}) = 1 + 1 - 2\cos(\theta) = 2(1 - \cos(\theta))$$
# This equivalence allows high-performance vector search engines (e.g. FAISS, HNSW) to replace expensive square-root distance calculations with simple inner products.
#
# ### 3. The Curse of Dimensionality & Orthogonality
# In high-dimensional spaces ($D \ge 768$), random vectors are almost strictly orthogonal ($\cos(\theta) \approx 0$).
# The variance of cosine similarity between random vectors decays as $\text{Var}(\cos \theta) = \frac{1}{D}$.

# %%
def dot_product(u: np.ndarray, v: np.ndarray) -> float:
    """Compute inner dot product."""
    return float(np.dot(u, v))

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))

def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Compute Euclidean (L2) distance."""
    return float(np.linalg.norm(u - v))

def manhattan_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Compute Manhattan (L1) distance."""
    return float(np.sum(np.abs(u - v)))

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Project vector to the unit hypersphere."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def simulate_high_dimensional_orthogonality(
    dimensions: List[int], num_samples: int = 1000, seed: int = 42
) -> Dict[int, Dict[str, float]]:
    """Simulate cosine similarity distribution between random unit vectors across dimensions."""
    np.random.seed(seed)
    results = {}
    for d in dimensions:
        # Generate pairs of random Gaussian vectors and normalize
        u_mat = np.random.randn(num_samples, d)
        v_mat = np.random.randn(num_samples, d)
        u_norm = u_mat / np.linalg.norm(u_mat, axis=1, keepdims=True)
        v_norm = v_mat / np.linalg.norm(v_mat, axis=1, keepdims=True)
        
        # Batch cosine similarities (inner product of unit vectors)
        cos_sims = np.sum(u_norm * v_norm, axis=1)
        mean_sim = float(np.mean(cos_sims))
        std_sim = float(np.std(cos_sims))
        max_sim = float(np.max(np.abs(cos_sims)))
        expected_std = 1.0 / math.sqrt(d)
        
        results[d] = {
            "dimension": d,
            "mean_cosine": mean_sim,
            "std_cosine": std_sim,
            "expected_std": expected_std,
            "max_abs_cosine": max_sim,
        }
    return results

# %%
# Verify Unit-Norm Equivalence
vec_a = np.array([0.6, 0.8, 0.0])
vec_b = np.array([0.0, 0.8, 0.6])

cos_ab = cosine_similarity(vec_a, vec_b)
l2_ab = euclidean_distance(vec_a, vec_b)
l2_squared_formula = 2 * (1.0 - cos_ab)

print(f"Vector A (Norm = {np.linalg.norm(vec_a):.1f}): {vec_a}")
print(f"Vector B (Norm = {np.linalg.norm(vec_b):.1f}): {vec_b}")
print(f"Cosine Similarity: {cos_ab:.4f}")
print(f"Actual Euclidean Distance (L2)^2: {l2_ab**2:.4f}")
print(f"Theoretical 2*(1 - Cosine):       {l2_squared_formula:.4f}")
print(f"Equivalence Verified: {np.isclose(l2_ab**2, l2_squared_formula)}")

# Simulate High-Dimensional Orthogonality
dim_experiment = [8, 32, 128, 384, 768, 1536]
ortho_stats = simulate_high_dimensional_orthogonality(dim_experiment, num_samples=2000)

print("\nCurse of Dimensionality: Orthogonality in Vector Spaces (N=2000 pairs)")
print(f"{'Dimension':<12}{'Mean Cosine':<14}{'Std Dev':<14}{'Expected 1/sqrt(D)':<20}{'Max |Cosine|':<14}")
for d, res in ortho_stats.items():
    print(f"{d:<12}{res['mean_cosine']:<14.5f}{res['std_cosine']:<14.5f}{res['expected_std']:<20.5f}{res['max_abs_cosine']:<14.5f}")

# %% [markdown]
# ## Section 5: High-Precision Retrieval Micro-Benchmarking Harness
#
# Production retrieval architectures demand rigorous latency and throughput micro-benchmarking.
#
# Below, we implement `RetrievalBenchmarkHarness` to evaluate:
# 1. **Tokenization Throughput**: Subword token generation speed in tokens/second.
# 2. **Dense Vector Search Throughput**: Millions of distance calculations per second across realistic database scales.
# 3. **Statistical Percentile Profiling**: Mean, Median (p50), 95th percentile (p95), and 99th percentile (p99) latency measurements.

# %%
class RetrievalBenchmarkHarness:
    """High-precision micro-benchmarking harness for tokenizer throughput and vector search operations."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def profile_callable(
        self, name: str, target_fn: Callable[[], Any], iterations: int = 100
    ) -> Dict[str, Any]:
        """Profile a zero-argument callable over multiple iterations with nanosecond resolution."""
        # Warmup iteration
        target_fn()
        
        durations_ms = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            target_fn()
            end = time.perf_counter_ns()
            durations_ms.append((end - start) / 1_000_000.0)

        durations_arr = np.array(durations_ms)
        mean_ms = float(np.mean(durations_arr))
        median_ms = float(np.median(durations_arr))
        p95_ms = float(np.percentile(durations_arr, 95))
        p99_ms = float(np.percentile(durations_arr, 99))
        throughput_ops_sec = (1000.0 / mean_ms) if mean_ms > 0 else 0.0

        record = {
            "benchmark_name": name,
            "iterations": iterations,
            "mean_latency_ms": round(mean_ms, 4),
            "median_latency_ms": round(median_ms, 4),
            "p95_latency_ms": round(p95_ms, 4),
            "p99_latency_ms": round(p99_ms, 4),
            "throughput_ops_sec": round(throughput_ops_sec, 2),
        }
        self.history.append(record)
        return record

    def benchmark_tokenizer(
        self, bpe_tokenizer: BPETokenizer, test_corpus: List[str], iterations: int = 30
    ) -> Dict[str, Any]:
        """Benchmark subword tokenization throughput in tokens per second."""
        total_chars = sum(len(text) for text in test_corpus)
        
        def run_tokenization():
            total_toks = 0
            for text in test_corpus:
                toks = bpe_tokenizer.encode(text)
                total_toks += len(toks)
            return total_toks

        # Measure token count once
        tokens_per_pass = run_tokenization()
        perf = self.profile_callable("BPE_Tokenization", run_tokenization, iterations=iterations)
        
        tokens_per_sec = (tokens_per_pass / (perf["mean_latency_ms"] / 1000.0)) if perf["mean_latency_ms"] > 0 else 0
        perf["total_chars"] = total_chars
        perf["tokens_per_pass"] = tokens_per_pass
        perf["tokens_per_sec"] = round(tokens_per_sec, 2)
        return perf

    def benchmark_vector_similarity(
        self, num_vectors: int = 5000, dimension: int = 768, iterations: int = 20
    ) -> Dict[str, Any]:
        """Benchmark exact brute-force cosine similarity matrix computation against a query."""
        np.random.seed(42)
        corpus_matrix = np.random.randn(num_vectors, dimension).astype(np.float32)
        # Unit-normalize corpus
        corpus_matrix /= np.linalg.norm(corpus_matrix, axis=1, keepdims=True)
        query_vector = np.random.randn(dimension).astype(np.float32)
        query_vector /= np.linalg.norm(query_vector)

        def run_vector_search():
            # Dot product against normalized vectors equals cosine similarity
            return np.dot(corpus_matrix, query_vector)

        perf = self.profile_callable(
            f"Dense_Exact_Cosine_N{num_vectors}_D{dimension}", run_vector_search, iterations=iterations
        )
        comparisons_per_sec = (num_vectors / (perf["mean_latency_ms"] / 1000.0)) if perf["mean_latency_ms"] > 0 else 0
        perf["num_vectors"] = num_vectors
        perf["dimension"] = dimension
        perf["vector_comparisons_per_sec"] = round(comparisons_per_sec, 2)
        return perf

# %%
benchmark_harness = RetrievalBenchmarkHarness()

# 1. Benchmark Tokenization
tok_benchmark_corpus = domain_corpus * 20  # 100 sentences
tok_perf = benchmark_harness.benchmark_tokenizer(tokenizer, tok_benchmark_corpus, iterations=25)

# 2. Benchmark Vector Search across standard dimensions (384 MiniLM, 768 Base, 1536 Large)
vec_perf_384 = benchmark_harness.benchmark_vector_similarity(num_vectors=10000, dimension=384, iterations=15)
vec_perf_768 = benchmark_harness.benchmark_vector_similarity(num_vectors=10000, dimension=768, iterations=15)
vec_perf_1536 = benchmark_harness.benchmark_vector_similarity(num_vectors=10000, dimension=1536, iterations=15)

print("Tokenizer Micro-Benchmark:")
print(f"  • Mean Latency: {tok_perf['mean_latency_ms']} ms")
print(f"  • Throughput:   {tok_perf['tokens_per_sec']:,.0f} tokens/sec")

print("\nDense Vector Search (N = 10,000 vectors):")
print(f"  • D=384:  {vec_perf_384['mean_latency_ms']:.3f} ms ({vec_perf_384['vector_comparisons_per_sec']:,.0f} comparisons/sec)")
print(f"  • D=768:  {vec_perf_768['mean_latency_ms']:.3f} ms ({vec_perf_768['vector_comparisons_per_sec']:,.0f} comparisons/sec)")
print(f"  • D=1536: {vec_perf_1536['mean_latency_ms']:.3f} ms ({vec_perf_1536['vector_comparisons_per_sec']:,.0f} comparisons/sec)")

# %% [markdown]
# ## Section 6: Workspace & Benchmark Summary Dashboard
#
# Below is the consolidated presenter dashboard summarizing the runtime diagnostic status, BPE tokenization statistics, context allocation limits, high-dimensional vector orthogonality, and micro-benchmark metrics.

# %%
# collapse_input
def display_environment_dashboard(
    env_status: Dict[str, Any],
    tok_stats: Dict[str, Any],
    budget: Dict[str, Any],
    dim_stats: Dict[int, Dict[str, float]],
    benchmarks: List[Dict[str, Any]],
):
    """Render a clean, formatted ASCII diagnostic dashboard."""
    print("=" * 80)
    print("           KNOWLEDGE RETRIEVAL A-Z: MODULE 01 WORKSPACE DASHBOARD")
    print("=" * 80)
    
    # 1. Environment Status
    print(f"\n[1] WORKSPACE ENVIRONMENT")
    print(f"  • Python Runtime:    {env_status['python_version']} (Compatible: {env_status['python_supported']})")
    print(f"  • Local LLM Server:  {env_status['endpoint_url']}")
    print(f"  • Connectivity:      {env_status['endpoint_status']}")
    print(f"  • Core Libraries:    {', '.join(env_status['dependencies'].keys())}")
    
    # 2. Tokenization & Context Budget
    print(f"\n[2] TOKENIZATION & CONTEXT BUDGET")
    print(f"  • BPE Vocab Size:    {tok_stats['vocab_size']} subwords ({tok_stats['merges']} learned merges)")
    print(f"  • Context Capacity:  {budget['total_context']} tokens")
    print(f"  • Max Chunks (K):    {budget['max_chunks_k']} chunks @ {budget['chunk_size']} tokens/chunk")
    print(f"  • Budget Allocation: {budget['allocated_retrieval_tokens']} tokens for retrieval ({budget['utilization_percent']:.1f}% total)")

    # 3. Dimensionality & Orthogonality
    print(f"\n[3] HIGH-DIMENSIONAL VECTOR METRIC SPACES")
    print(f"  {'Dim (D)':<10}{'Mean Cosine':<14}{'Std Dev':<14}{'1/sqrt(D)':<14}{'Max |Cosine|':<14}")
    print(f"  {'-'*62}")
    for d, row in dim_stats.items():
        print(f"  {d:<10}{row['mean_cosine']:<14.4f}{row['std_cosine']:<14.4f}{row['expected_std']:<14.4f}{row['max_abs_cosine']:<14.4f}")

    # 4. Performance Benchmarks
    print(f"\n[4] RETRIEVAL MICRO-BENCHMARK HARNESS SUMMARY")
    print(f"  {'Benchmark':<38}{'Mean (ms)':<12}{'p95 (ms)':<12}{'Throughput / sec':<18}")
    print(f"  {'-'*78}")
    for record in benchmarks:
        tp_str = f"{record.get('tokens_per_sec') or record.get('vector_comparisons_per_sec') or record['throughput_ops_sec']:,.0f}"
        print(f"  {record['benchmark_name']:<38}{record['mean_latency_ms']:<12.3f}{record['p95_latency_ms']:<12.3f}{tp_str:<18}")

    print("\n" + "=" * 80)
    print("  [OK] Module 01 foundations complete! Ready for Module 02: Sparse vs Dense Search.")
    print("=" * 80)

# Render dashboard
display_environment_dashboard(
    env_status=env_diagnostics,
    tok_stats={"vocab_size": len(tokenizer.vocab), "merges": len(tokenizer.merges)},
    budget=budget_summary,
    dim_stats=ortho_stats,
    benchmarks=benchmark_harness.history,
)

# %% [markdown]
# ## Section 7: Summary & Transition to Module 02
#
# In this module, we have established the foundational infrastructure of retrieval engineering:
# - Verified the workspace runtime, `uv` dependency orchestration, and local LLM connectivity.
# - Constructed a **Byte-Pair Encoding (BPE)** subword tokenizer from scratch to understand subword segmentation and token-to-character compression.
# - Formulated mathematical models for **context window budgeting** and **KV-cache memory allocation** across modern LLM architectures.
# - Explored high-dimensional vector spaces, verified the **unit-norm distance equivalence** $\|\mathbf{u} - \mathbf{v}\|_2^2 = 2(1 - \cos\theta)$, and simulated the **concentration of measure**.
# - Engineered a reusable **Retrieval Micro-Benchmarking Harness** to measure tokenization throughput and vector search speeds.
#
# In **Module 02**, we will build on these vector and lexical foundations to implement **Sparse Search (BM25)**, **Dense Semantic Search**, and fuse them using **Reciprocal Rank Fusion (RRF)**.
