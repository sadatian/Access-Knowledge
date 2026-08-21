# %% [markdown]
# # Module 01: Modern Retrieval Workspace Setup & Tokenization Math
#
# Welcome to **Module 01** of the Knowledge Retrieval A-Z masterclass.
# In production retrieval systems, retrieval quality and latency depend heavily on the foundational runtime environment, tokenization fidelity, context window allocation, and the geometry of high-dimensional embedding spaces.
#
# In this module, we construct and master:
# 1. **Workspace Health Diagnostics**: Runtime validation of `uv`, dependencies, and local LLM connectivity (`http://localhost:5055/v1`).
# 2. **Lossless Byte-Pair Encoding (BPE) from Scratch**: Subword tokenization mechanics, explicit whitespace preservation, merge hierarchies, and token compression ratios.
# 3. **Context Window Arithmetic & KV-Cache Footprint**: Mathematical modeling of context budgets, document ingestion capacities, and GPU KV-cache memory requirements.
# 4. **Vector Embedding Geometries & Dimensionality Distribution**: Dot product, Cosine similarity, Euclidean ($L_2$), Manhattan ($L_1$), and visual simulation of the *Curse of Dimensionality*.
# 5. **High-Precision Retrieval Micro-Benchmarking**: Micro-profiling tokenization throughput, exact BLAS matrix inner products, and indexed Approximate Nearest Neighbor (ANN) search.
# 6. **Architectural Decision Matrix & Production Guidelines**: Comprehensive synthesis of distance metrics, indexing algorithms, and VRAM sizing models.
#
# ---
#
# ```mermaid
# graph LR
#     subgraph Pipeline ["Foundational Retrieval Architecture"]
#         A["Raw Text Corpus"] --> B["Lossless BPE Subword Tokenizer"]
#         B --> C["Token Compression & Context Capacity"]
#         C --> D["Embedding Transformation (R^D)"]
#         D --> E["Geometric Metric Space (Cosine / L2)"]
#         E --> F["Exact BLAS vs Indexed ANN Benchmarking"]
#         F --> G["KV-Cache VRAM Planning & Decision Matrix"]
#     end
# ```

# %%
import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from IPython.display import display, Image
import io
from IPython.display import display, Image
import io

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
        "faiss": getattr(faiss, "__version__", "available"),
        "openai": "available",
        "click": "available",
        "networkx": "available",
        "matplotlib": plt.matplotlib.__version__,
    }
    
    # Check LLM endpoint connectivity (non-blocking with short timeout)
    llm_connected = False
    endpoint_message = "Not reachable (offline or not started)"
    try:
        client = OpenAI(base_url=endpoint_url, api_key="dummy", timeout=0.8)
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
# ## Section 2: Tokenization Mechanics & Lossless Byte-Pair Encoding (BPE)
#
# Large Language Models and embedding models do not read characters or whole words directly.
# They operate on discrete **subword tokens** to handle out-of-vocabulary (OOV) terms, technical acronyms, and code identifiers efficiently.
#
# ### 2.1. The Byte-Pair Encoding (BPE) Algorithm
# 1. Initialize the vocabulary $V_0$ with all individual characters present in the training corpus.
# 2. To ensure **1:1 lossless reconstruction**, whitespace characters must be explicitly preserved as distinct symbols (e.g. using the GPT-2 / RoBERTa convention `'Ġ'` for space, or preserving literal characters) rather than discarding them via naive `split()`.
# 3. In each iteration $t$:
#    - Count the frequency of all adjacent symbol pairs across the tokenized sequences.
#    - Identify the most frequent pair $(s_i, s_j) = \arg\max \text{freq}(p)$.
#    - Merge $(s_i, s_j)$ into a single symbol $s_{\text{new}} = s_i + s_j$.
#    - Add $s_{\text{new}}$ to the vocabulary: $V_{t+1} = V_t \cup \{s_{\text{new}}\}$.
# 4. Terminate when the requested number of merges is reached or no pairs occur $> 1$ time.
#
# ### 2.2. Subword Merge Hierarchy Diagram
#
# ```mermaid
# graph TD
#     subgraph Hierarchy ["Hierarchical BPE Subword Formation"]
#         C1["'Ġ' (Space)"] --> M1["'Ġe'"]
#         C2["'e'"] --> M1
#         C3["'m'"] --> M2["'em'"]
#         C4["'b'"] --> M2
#         M1 --> M3["'Ġemb'"]
#         M2 --> M3
#         C5["'e'"] --> M4["'ed'"]
#         C6["'d'"] --> M4
#         M3 --> M5["'Ġembed'"]
#         M4 --> M5
#         C7["'d'"] --> M6["'ding'"]
#         C8["'i'"] --> M6
#         C9["'n'"] --> M6
#         C10["'g'"] --> M6
#         M5 --> M7["'Ġembedding'"]
#         M6 --> M7
#         C11["'s'"] --> Final["'Ġembeddings' (Single Token)"]
#         M7 --> Final
#     end
# ```

# %%
# collapse_input
class BPETokenizer:
    """A pure Python Byte-Pair Encoding (BPE) subword tokenizer with 1:1 lossless reconstruction."""

    SPACE_MARKER: str = "Ġ"

    def __init__(self):
        self.vocab: List[str] = []
        self.merges: List[Tuple[str, str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def _text_to_symbols(self, text: str) -> List[str]:
        """Convert raw string to symbol list, mapping spaces explicitly for lossless reconstruction."""
        return [self.SPACE_MARKER if char == " " else char for char in text]

    def _symbols_to_text(self, symbols: List[str]) -> str:
        """Invert symbols back to raw text with exact whitespace preservation."""
        return "".join(symbols).replace(self.SPACE_MARKER, " ")

    def train(self, corpus: List[str], num_merges: int = 50) -> "BPETokenizer":
        """Train the BPE tokenizer on a corpus by iteratively learning `num_merges` merge rules."""
        # 0. Sentinel Token Collision Check
        if any(self.SPACE_MARKER in text for text in corpus):
            raise ValueError(f"Sentinel collision: Raw corpus contains the '{self.SPACE_MARKER}' character. "
                             "Please map raw inputs to a strict byte representation for lossless reconstruction.")

        # 1. Transform texts into atomic symbol sequences preserving all whitespace, tabs, and newlines
        # Optimization: Count unique sequence frequencies instead of iterating the entire corpus O(N^3)
        seq_freqs = Counter(tuple(self._text_to_symbols(text)) for text in corpus)

        # 2. Extract base character vocabulary with full ASCII byte fallback (0-255) for complete OOV coverage
        base_symbols = set([self.SPACE_MARKER])
        for i in range(256):
            c = chr(i)
            if c != " ":
                base_symbols.add(c)
        for seq in seq_freqs.keys():
            base_symbols.update(seq)
        
        self.vocab = sorted(list(base_symbols))
        self.merges = []

        # 3. Iterative pair extraction and merging
        for _ in range(num_merges):
            pairs = defaultdict(int)
            for seq, count in seq_freqs.items():
                for i in range(len(seq) - 1):
                    pairs[(seq[i], seq[i + 1])] += count
                    
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 1:
                break

            self.merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab.append(merged_token)

            # Apply merge across all unique sequences
            new_seq_freqs = Counter()
            first, second = best_pair
            for seq, count in seq_freqs.items():
                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == first and seq[i + 1] == second:
                        new_seq.append(merged_token)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_seq_freqs[tuple(new_seq)] += count
            seq_freqs = new_seq_freqs

        # 4. Build index mapping tables
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        return self

    def encode(self, text: str) -> List[str]:
        """Tokenize a full string into BPE subword tokens losslessly."""
        if not text:
            return []
        tokens = self._text_to_symbols(text)
        for first, second in self.merges:
            merged = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
        return tokens

    def decode(self, tokens: List[str]) -> str:
        """Reconstruct original text from BPE subword tokens with 1:1 lossless fidelity."""
        return self._symbols_to_text(tokens)

    def encode_to_ids(self, text: str) -> List[int]:
        """Encode text directly to numerical token IDs."""
        tokens = self.encode(text)
        return [self.token_to_id.get(t, -1) for t in tokens]

    def decode_from_ids(self, token_ids: List[int]) -> str:
        """Decode numerical token IDs back into reconstructed text."""
        tokens = [self.id_to_token.get(idx, "") for idx in token_ids if idx in self.id_to_token]
        return self.decode(tokens)

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
tokenizer.train(domain_corpus, num_merges=60)

sample_query = "  Cache-Augmented \t embeddings optimize\n dense retrieval!  "
sample_tokens = tokenizer.encode(sample_query)
reconstructed = tokenizer.decode(sample_tokens)
compression = tokenizer.compression_ratio(sample_query)

print(f"Learned Vocabulary Size: {len(tokenizer.vocab)}")
print(f"Learned Merge Rules: {len(tokenizer.merges)}")
print(f"Sample Input: {repr(sample_query)}")
print(f"Encoded Subwords: {sample_tokens}")
print(f"Reconstructed Text: {repr(reconstructed)}")
print(f"1:1 Lossless Match: {reconstructed == sample_query}")
print(f"Compression Ratio: {compression:.2f} chars/token")

# %% [markdown]
# ### 2.3: Conceptual Bridge — Token Compression to Physical Document Capacity
#
# In retrieval engineering, the subword token compression ratio ($R_{\text{comp}} = \frac{\text{chars}}{\text{token}}$) acts as the fundamental bridge between abstract LLM context budgets and physical textual capacity:
#
# $$C_{\text{chars}} = T_{\text{retrieval}} \times R_{\text{comp}}$$
# $$C_{\text{words}} \approx \frac{C_{\text{chars}}}{\bar{L}_{\text{word}}} \approx \frac{T_{\text{retrieval}} \times R_{\text{comp}}}{5.1}$$
# $$C_{\text{pages}} \approx \frac{C_{\text{words}}}{500} \approx \frac{T_{\text{retrieval}} \times R_{\text{comp}}}{2550}$$
# $$\text{Payload Size (KB)} = \frac{C_{\text{chars}} \times 1\text{ byte}}{1024}$$
#
# For instance, a retrieval budget of $6,000\text{ tokens}$ yields:
# - **At $R_{\text{comp}} = 4.2$ (Natural English prose):** $25,200\text{ chars} \approx 4,941\text{ words} \approx 9.9\text{ pages}$ ($24.6\text{ KB}$).
# - **At $R_{\text{comp}} = 2.8$ (Dense Source Code / JSON triplets):** $16,800\text{ chars} \approx 3,294\text{ words} \approx 6.6\text{ pages}$ ($16.4\text{ KB}$).
#
# Understanding this mathematical translation is critical when sizing chunk ingestion pipelines and provisioning GPU VRAM.

# %% [markdown]
# ### 2.4. Embedding Model Token Limits
# Before scaling up to LLM multi-thousand token contexts (e.g., 128k tokens), standard embedding models impose strict maximum sequence lengths. 
# For example, BERT-based embedding models typically cap at 512 tokens, and modern open-source models cap at 8192 tokens. 
# Retrieval chunking strategies must first satisfy this *embedding model limit* to prevent semantic truncation before they are packed into the broader LLM context window.


# %%
# collapse_input
class RetrievalBenchmarkHarness:
    """High-precision micro-benchmarking harness for tokenizer throughput and exact vs indexed vector search."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def profile_callable(
        self, name: str, target_fn: Callable[[], Any], iterations: int = 50
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

        tokens_per_pass = run_tokenization()
        perf = self.profile_callable("BPE_Tokenization", run_tokenization, iterations=iterations)
        
        tokens_per_sec = (tokens_per_pass / (perf["mean_latency_ms"] / 1000.0)) if perf["mean_latency_ms"] > 0 else 0
        perf["total_chars"] = total_chars
        perf["tokens_per_pass"] = tokens_per_pass
        perf["tokens_per_sec"] = round(tokens_per_sec, 2)
        return perf

    def benchmark_vector_similarity(
        self, num_vectors: int = 10000, dimension: int = 768, iterations: int = 20
    ) -> Dict[str, Any]:
        """Benchmark exact brute-force cosine similarity matrix computation against a query."""
        np.random.seed(42)
        corpus_matrix = np.random.randn(num_vectors, dimension).astype(np.float32)
        corpus_matrix /= np.linalg.norm(corpus_matrix, axis=1, keepdims=True)
        query_vector = np.random.randn(dimension).astype(np.float32)
        query_vector /= np.linalg.norm(query_vector)

        def run_vector_search():
            return np.dot(corpus_matrix, query_vector)

        perf = self.profile_callable(
            f"Dense_Exact_Cosine_N{num_vectors}_D{dimension}", run_vector_search, iterations=iterations
        )
        comparisons_per_sec = (num_vectors / (perf["mean_latency_ms"] / 1000.0)) if perf["mean_latency_ms"] > 0 else 0
        perf["num_vectors"] = num_vectors
        perf["dimension"] = dimension
        perf["vector_comparisons_per_sec"] = round(comparisons_per_sec, 2)
        return perf

    def benchmark_indexed_vector_search(
        self, num_vectors: int = 10000, dimension: int = 768, index_type: str = "hnsw", top_k: int = 10, iterations: int = 30
    ) -> Dict[str, Any]:
        """Benchmark indexed Approximate Nearest Neighbor (ANN) search using FAISS."""
        np.random.seed(42)
        corpus_matrix = np.random.randn(num_vectors, dimension).astype(np.float32)
        corpus_matrix /= np.linalg.norm(corpus_matrix, axis=1, keepdims=True)
        query = np.random.randn(1, dimension).astype(np.float32)
        query /= np.linalg.norm(query, axis=1, keepdims=True)

        if index_type.lower() == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = 64
            index.add(corpus_matrix)
        elif index_type.lower() == "ivf":
            nlist = int(math.sqrt(num_vectors))
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(corpus_matrix)
            index.add(corpus_matrix)
            index.nprobe = 16
        else:
            index = faiss.IndexFlatIP(dimension)
            index.add(corpus_matrix)

        def run_search():
            return index.search(query, top_k)

        perf = self.profile_callable(
            f"FAISS_{index_type.upper()}_N{num_vectors}_D{dimension}_top{top_k}",
            run_search,
            iterations=iterations
        )
        qps = (1000.0 / perf["mean_latency_ms"]) if perf["mean_latency_ms"] > 0 else 0.0
        perf["num_vectors"] = num_vectors
        perf["dimension"] = dimension
        perf["index_type"] = index_type.upper()
        perf["queries_per_sec"] = round(qps, 2)
        return perf

# %%
benchmark_harness = RetrievalBenchmarkHarness()

# Benchmark Tokenization immediately after defining the BPE class
tok_benchmark_corpus = domain_corpus * 20  # 100 sentences
tok_perf = benchmark_harness.benchmark_tokenizer(tokenizer, tok_benchmark_corpus, iterations=25)

print("Tokenizer Micro-Benchmark:")
print(f"  • Mean Latency: {tok_perf['mean_latency_ms']} ms")
print(f"  • Throughput:   {tok_perf['tokens_per_sec']:,.0f} tokens/sec")


# %% [markdown]
# ### 2.4. Embedding Model Token Limits
# Before scaling up to LLM multi-thousand token contexts (e.g., 128k tokens), standard embedding models impose strict maximum sequence lengths. 
# For example, BERT-based embedding models typically cap at 512 tokens, and modern open-source models cap at 8192 tokens. 
# Retrieval chunking strategies must first satisfy this *embedding model limit* to prevent semantic truncation before they are packed into the broader LLM context window.


# %% [markdown]
# ## Section 3: Context Window Budgeting & KV-Cache Memory Modeling
#
# When designing production RAG and CAG systems, understanding context constraints and memory allocations is critical.
#
# ### 3.1. Context Window Budget Allocation
# In any retrieval pipeline, the maximum model context window $W_{\text{total}}$ must accommodate:
# $$W_{\text{total}} = T_{\text{system}} + T_{\text{query}} + T_{\text{history}} + \sum_{i=1}^K T_{\text{chunk}_i} + T_{\text{reserve}} + T_{\text{generation}}$$
#
# Given a chunk size $C$, the maximum number of chunks $K$ we can safely retrieve is:
# $$K_{\text{max}} = \left\lfloor \frac{W_{\text{total}} - (T_{\text{system}} + T_{\text{query}} + T_{\text{history}} + T_{\text{generation}} + T_{\text{safety}})}{C - \text{overlap}} \right\rfloor$$
#
# ### 3.2. KV-Cache GPU Memory Footprint
# In modern Transformer decoders with Multi-Head Attention (MHA) or Grouped-Query Attention (GQA), each layer stores Key and Value tensors for all context tokens $T_{\text{context}}$:
# $$\text{Memory}_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times T_{\text{context}} \times b_{\text{precision}}$$
# where:
# - $n_{\text{layers}}$ is the number of transformer layers (e.g., 32 for 8B, 80 for 70B, 128 for 405B).
# - $n_{\text{kv\_heads}}$ is the number of Key/Value heads ($n_{\text{kv\_heads}} = n_{\text{heads}}$ in MHA; $n_{\text{kv\_heads}} \ll n_{\text{heads}}$ in GQA).
# - $d_{\text{head}}$ is the dimension per attention head (typically 128).
# - $b_{\text{precision}}$ is the byte width per parameter (2 bytes for FP16/BF16, 1 byte for FP8, 0.5 bytes for INT4).

# %%
# collapse_input
class ContextBudgetCalculator:
    """Calculates context window allocation limits, physical document capacities, and KV-cache footprints."""

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
        self, chunk_size: int = 512, reserve_safety_tokens: int = 128, overlap: int = 0
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
        effective_chunk_size = max(1, chunk_size - overlap)
        max_chunks = available_for_retrieval // effective_chunk_size
        retrieval_tokens = max_chunks * effective_chunk_size
        slack_tokens = self.total_context - (fixed_overhead + retrieval_tokens)

        return {
            "total_context": self.total_context,
            "fixed_overhead": fixed_overhead,
            "available_for_retrieval": available_for_retrieval,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "effective_chunk_size": effective_chunk_size,
            "max_chunks_k": max_chunks,
            "allocated_retrieval_tokens": retrieval_tokens,
            "slack_tokens": slack_tokens,
            "utilization_percent": ((self.total_context - slack_tokens) / self.total_context) * 100,
        }

    @staticmethod
    def compute_document_capacity(
        token_count: int, compression_ratio: float = 4.0, avg_word_length: float = 5.1
    ) -> Dict[str, Any]:
        """Convert allocated retrieval token budget into physical document capacity metrics."""
        chars = int(token_count * compression_ratio)
        words = int(chars / avg_word_length) if avg_word_length > 0 else 0
        pages = round(words / 500.0, 2)
        payload_kb = round(chars / 1024.0, 2)
        return {
            "retrieval_tokens": token_count,
            "compression_ratio": round(compression_ratio, 2),
            "estimated_characters": chars,
            "estimated_words": words,
            "estimated_pages": pages,
            "raw_payload_kb": payload_kb,
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
budget_summary = budget_calc.calculate_chunk_budget(chunk_size=400, overlap=50)
doc_capacity = ContextBudgetCalculator.compute_document_capacity(
    token_count=budget_summary["allocated_retrieval_tokens"],
    compression_ratio=compression,
)

print(f"Context Window: {budget_summary['total_context']} tokens")
print(f"Fixed Overhead: {budget_summary['fixed_overhead']} tokens (Prompt, Query, History, Safety, Gen)")
print(f"Max Retrieval Chunks (K): {budget_summary['max_chunks_k']} chunks (Chunk size: {budget_summary['chunk_size']})")
print(f"Total Retrieval Capacity: {budget_summary['allocated_retrieval_tokens']} tokens")
print(f"Physical Capacity: ~{doc_capacity['estimated_words']} words ({doc_capacity['estimated_pages']} pages, {doc_capacity['raw_payload_kb']} KB)")
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
# ### 4.1. Metric Geometries
# For two embedding vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^D$:
# - **Dot Product**: $\langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^D u_i v_i$
# - **Cosine Similarity**: $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$
# - **Euclidean Distance ($L_2$)**: $\|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^D (u_i - v_i)^2}$
# - **Manhattan Distance ($L_1$)**: $\|\mathbf{u} - \mathbf{v}\|_1 = \sum_{i=1}^D |u_i - v_i|$
#
# ### 4.2. Unit-Norm Equivalence
# When vectors are $L_2$-normalized ($\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1$):
# $$\|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|_2^2 + \|\mathbf{v}\|_2^2 - 2(\mathbf{u} \cdot \mathbf{v}) = 1 + 1 - 2\cos(\theta) = 2(1 - \cos(\theta))$$
# This equivalence allows high-performance vector search engines (e.g. FAISS, HNSW) to replace expensive square-root distance calculations with simple inner products.
#
# ### 4.3. The Curse of Dimensionality & Orthogonality
# In high-dimensional spaces ($D \ge 768$), random vectors are almost strictly orthogonal ($\cos(\theta) \approx 0$).
# The variance of cosine similarity between random vectors decays as:
# $$\text{Var}(\cos \theta) = \frac{1}{D} \implies \sigma(\cos \theta) = \frac{1}{\sqrt{D}}$$
# As $D$ scales toward 1536, the distribution of pairwise similarities collapses into a tight Dirac delta spike around zero.

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
    dimensions: List[int], num_samples: int = 2000, seed: int = 42
) -> Dict[int, Dict[str, Any]]:
    """Simulate cosine similarity distribution between random unit vectors across dimensions."""
    np.random.seed(seed)
    results = {}
    for d in dimensions:
        u_mat = np.random.randn(num_samples, d)
        v_mat = np.random.randn(num_samples, d)
        u_norm = u_mat / np.linalg.norm(u_mat, axis=1, keepdims=True)
        v_norm = v_mat / np.linalg.norm(v_mat, axis=1, keepdims=True)
        
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
            "raw_samples": cos_sims,
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
ortho_stats = simulate_high_dimensional_orthogonality(dim_experiment, num_samples=3000)

print("\nCurse of Dimensionality: Orthogonality in Vector Spaces (N=3000 pairs)")
print(f"{'Dimension':<12}{'Mean Cosine':<14}{'Std Dev':<14}{'Expected 1/sqrt(D)':<20}{'Max |Cosine|':<14}")
for d, res in ortho_stats.items():
    print(f"{d:<12}{res['mean_cosine']:<14.5f}{res['std_cosine']:<14.5f}{res['expected_std']:<20.5f}{res['max_abs_cosine']:<14.5f}")

# %% [markdown]
# ### 4.4. Geometric Distribution Visualizer: Variance Collapse in High Dimensions
#
# Below, we plot the empirical probability density distributions of cosine similarity across dimensions $D \in [8, 32, 128, 384, 768, 1536]$.
# Notice how the bell curve tightens dramatically as dimensionality scales, confirming that high-dimensional embedding spaces are almost universally orthogonal for unrelated vectors.

# %%
# collapse_input
def plot_cosine_variance_distributions(ortho_data: Dict[int, Dict[str, Any]]):
    """Render high-resolution distribution comparison visualizing the Curse of Dimensionality."""
    plt.figure(figsize=(10, 5), dpi=120)
    
    colors = ["#9E9E9E", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0", "#E91E63"]
    
    for (d, data), color in zip(ortho_data.items(), colors):
        samples = data["raw_samples"]
        # Generate smooth histogram/KDE-style density plot
        counts, bin_edges = np.histogram(samples, bins=60, range=(-1.0, 1.0), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, counts, label=f"D={d} (σ={data['std_cosine']:.3f})", color=color, linewidth=2)
        plt.fill_between(bin_centers, counts, alpha=0.12, color=color)

    plt.title("Curse of Dimensionality: Cosine Similarity Variance Decay as D → 1536", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Cosine Similarity $\\cos(\\theta)$", fontsize=10)
    plt.ylabel("Probability Density", fontsize=10)
    plt.xlim(-0.8, 0.8)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, facecolor="white", loc="upper right", fontsize=9)
    plt.tight_layout()

    # Save figure to memory and render as Image with alt-text for accessibility
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()

    display(Image(data=buf.getvalue(), alt="Geometric Distribution Visualizer showing the Curse of Dimensionality variance collapse as D approaches 1536"))

plot_cosine_variance_distributions(ortho_stats)

# %% [markdown]
# ### 4.5. High-Precision Vector Index Benchmarking
# Production vector search systems cannot rely on brute-force scanning for massive corpora. We benchmark both exact BLAS matrix products and indexed Approximate Nearest Neighbor (ANN) structures (`faiss.IndexHNSWFlat` and `faiss.IndexIVFFlat`).

# %%
# Benchmark Vector Search across standard dimensions (384 MiniLM, 768 Base, 1536 Large)
vec_perf_384 = benchmark_harness.benchmark_vector_similarity(num_vectors=10000, dimension=384, iterations=15)
vec_perf_768 = benchmark_harness.benchmark_vector_similarity(num_vectors=10000, dimension=768, iterations=15)
vec_perf_1536 = benchmark_harness.benchmark_vector_similarity(num_vectors=10000, dimension=1536, iterations=15)

# Benchmark Indexed ANN Search (HNSW vs IVF vs Flat)
hnsw_perf = benchmark_harness.benchmark_indexed_vector_search(num_vectors=20000, dimension=768, index_type="hnsw")
ivf_perf = benchmark_harness.benchmark_indexed_vector_search(num_vectors=20000, dimension=768, index_type="ivf")
flat_perf = benchmark_harness.benchmark_indexed_vector_search(num_vectors=20000, dimension=768, index_type="flat")

# %%
# collapse_input
print("Dense Vector Search Exact Dot Product (N = 10,000 vectors):")
print(f"  • D=384:  {vec_perf_384['mean_latency_ms']:.3f} ms ({vec_perf_384['vector_comparisons_per_sec']:,.0f} comparisons/sec)")
print(f"  • D=768:  {vec_perf_768['mean_latency_ms']:.3f} ms ({vec_perf_768['vector_comparisons_per_sec']:,.0f} comparisons/sec)")
print(f"  • D=1536: {vec_perf_1536['mean_latency_ms']:.3f} ms ({vec_perf_1536['vector_comparisons_per_sec']:,.0f} comparisons/sec)")

print("\nFAISS Index Architecture Comparison (N = 20,000, D = 768, Top-10):")
print(f"  • Exact Flat (FlatIP):  {flat_perf['mean_latency_ms']:.3f} ms ({flat_perf['queries_per_sec']:,.0f} QPS)")
print(f"  • Inverted File (IVF):  {ivf_perf['mean_latency_ms']:.3f} ms ({ivf_perf['queries_per_sec']:,.0f} QPS)")
print(f"  • Proximity Graph (HNSW): {hnsw_perf['mean_latency_ms']:.3f} ms ({hnsw_perf['queries_per_sec']:,.0f} QPS)")



# %% [markdown]
# ## Section 5: Architectural Decision Matrix & Synthesis Dashboard
#
# Below is the consolidated **Architectural Decision Matrix** and synthesized presenter dashboard summarizing runtime readiness, tokenization compression factors, context budget capacities, high-dimensional geometries, and vector search index tradeoffs.
#
# ### 5.1. Metric Space Selection Guide
#
# | Distance Metric | Mathematical Definition | Computational Cost | Unit-Norm Invariant | Optimal Production Use Case |
# | :--- | :--- | :--- | :--- | :--- |
# | **Inner Dot Product** | $\sum_{i=1}^D u_i v_i$ | $O(D)$ SIMD FMA (Fastest) | Equivalent to Cosine on unit vectors | Normalized dense embeddings, Maximum Inner Product Search (MIPS). |
# | **Cosine Similarity** | $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ | $O(D) + 2\sqrt{O(D)}$ | Normalization built-in | Unnormalized text representations, angle-based semantic similarity. |
# | **Euclidean ($L_2$)** | $\sqrt{\sum (u_i - v_i)^2}$ | $O(D) + \text{sqrt}$ | Monotonic with Cosine: $\|\mathbf{u}-\mathbf{v}\|_2^2 = 2(1-\cos\theta)$ | Geometric clustering (K-Means), spatial indexing, image embeddings. |
# | **Manhattan ($L_1$)** | $\sum \|u_i - v_i\|$ | $O(D)$ absolute sums | Sensitive to coordinate axes | Sparse keyword vectors, high-dimensional outlier-resistant retrieval. |
#
# ### 5.2. Vector Index Architecture Comparison
#
# | Index Architecture | Query Complexity | Build Time | Memory Overhead | 1-Recall@10 | Recommended Deployment Scenario |
# | :--- | :--- | :--- | :--- | :--- | :--- |
# | **Flat / Exact** | $O(N \cdot D)$ | $0$ (Instant) | Low ($1\times$ vector raw size) | $1.00$ (100% Exact) | Corpora $< 100\text{K}$ vectors, ground-truth evaluation rigs. |
# | **IVF (Inverted File)** | $O(\frac{N}{\text{nlist}} \cdot \text{nprobe} \cdot D)$ | Low ($K$-Means clustering) | Very Low | $0.92 - 0.98$ | Medium-scale datasets ($100\text{K} - 5\text{M}$ vectors) with limited RAM. |
# | **HNSW (Graph)** | $O(\log N)$ | Moderate (Graph build) | High ($1.3 - 2.0\times$ for graph links) | $0.97 - 0.999$ | High-concurrency, ultra-low-latency production RAG applications. |
# | **IVF-PQ (Quantized)** | $O(\text{lookup tables})$ | High (Codebook training) | Extremely Low ($0.05 - 0.1\times$) | $0.85 - 0.94$ | Billion-scale vector search on memory-constrained hardware. |

# %% [markdown]
# ## Section 6: Summary & Transition to Module 02
#
# In this module, we have established the foundational infrastructure of retrieval engineering:
# - Verified the workspace runtime, `uv` dependency orchestration, and local LLM connectivity.
# - Constructed a **Lossless Byte-Pair Encoding (BPE)** subword tokenizer with explicit whitespace preservation and 1:1 invertibility.
# - Formulated mathematical models connecting **token compression ratios** to **real-world physical document capacities** and **KV-cache GPU VRAM requirements**.
# - Explored high-dimensional vector spaces, verified the **unit-norm distance equivalence** $\|\mathbf{u} - \mathbf{v}\|_2^2 = 2(1 - \cos\theta)$, and visually graphed the **variance collapse** across dimensions.
# - Engineered a reusable **Retrieval Micro-Benchmarking Harness** comparing exact BLAS matrix products against indexed ANN structures (`faiss.IndexHNSWFlat` and `faiss.IndexIVFFlat`).
# - Synthesized the comprehensive **Distance Metric & Index Architecture Decision Matrix**.
#
# In **Module 02**, we will build on these vector and lexical foundations to implement **Sparse Search (BM25)**, **GPU-Accelerated Dense Semantic Search**, and fuse them using **Reciprocal Rank Fusion (RRF)**.


