import os

file_path = "/home/t/Access Knowledge/src/01_foundations_sparse_dense/module_01_environment_setup/env_guide.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Imports
content = content.replace("from openai import OpenAI\n", "from openai import OpenAI\nfrom IPython.display import display, Image\nimport io\n")

# 2. Add collapse input to BPETokenizer
content = content.replace("# %%\nclass BPETokenizer:", "# %%\n# collapse_input\nclass BPETokenizer:")

# 3. Optimize BPETokenizer.train and add collision check
old_train = '''    def train(self, corpus: List[str], num_merges: int = 50) -> "BPETokenizer":
        """Train the BPE tokenizer on a corpus by iteratively learning `num_merges` merge rules."""
        # 1. Transform texts into atomic symbol sequences preserving all whitespace, tabs, and newlines
        tokenized_corpus = [self._text_to_symbols(text) for text in corpus]

        # 2. Extract base character vocabulary with full ASCII byte fallback (0-255) for complete OOV coverage
        base_symbols = set([self.SPACE_MARKER])
        for i in range(256):
            c = chr(i)
            if c != " ":
                base_symbols.add(c)
        for seq in tokenized_corpus:
            base_symbols.update(seq)
        
        self.vocab = sorted(list(base_symbols))
        self.merges = []

        # 3. Iterative pair extraction and merging
        current_sequences = tokenized_corpus
        for _ in range(num_merges):
            pairs = defaultdict(int)
            for seq in current_sequences:
                for i in range(len(seq) - 1):
                    pairs[(seq[i], seq[i + 1])] += 1
                    
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 1:
                break

            self.merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab.append(merged_token)

            # Apply merge across all sequences
            new_sequences = []
            first, second = best_pair
            for seq in current_sequences:
                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == first and seq[i + 1] == second:
                        new_seq.append(merged_token)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_sequences.append(new_seq)
            current_sequences = new_sequences

        # 4. Build index mapping tables
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        return self'''

new_train = '''    def train(self, corpus: List[str], num_merges: int = 50) -> "BPETokenizer":
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
        return self'''

content = content.replace(old_train, new_train)


# 4. Context Budget Calculator Overlap logic
old_calc = '''# %%
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
        }'''

new_calc = '''# %%
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
        }'''

content = content.replace(old_calc, new_calc)

# Update chunk budget call
content = content.replace("budget_summary = budget_calc.calculate_chunk_budget(chunk_size=400)", "budget_summary = budget_calc.calculate_chunk_budget(chunk_size=400, overlap=50)")

# Update context math markdown
old_math = "$$K_{\\text{max}} = \\left\\lfloor \\frac{W_{\\text{total}} - (T_{\\text{system}} + T_{\\text{query}} + T_{\\text{history}} + T_{\\text{generation}} + T_{\\text{safety}})}{C} \\right\\rfloor$$"
new_math = "$$K_{\\text{max}} = \\left\\lfloor \\frac{W_{\\text{total}} - (T_{\\text{system}} + T_{\\text{query}} + T_{\\text{history}} + T_{\\text{generation}} + T_{\\text{safety}})}{C - \\text{overlap}} \\right\\rfloor$$"
content = content.replace(old_math, new_math)


# 5. Move RetrievalBenchmarkHarness
harness_start = content.find("# %% [markdown]\n# ## Section 5: High-Precision Retrieval Micro-Benchmarking Harness")
if harness_start == -1:
    print("Could not find section 5")
harness_end = content.find("# %% [markdown]\n# ## Section 6: Architectural Decision Matrix")

section_5_text = content[harness_start:harness_end]

# Extract harness class
class_start = section_5_text.find("class RetrievalBenchmarkHarness:")
class_end = section_5_text.find("# %%\nbenchmark_harness = RetrievalBenchmarkHarness()")

harness_class = "# %%\n# collapse_input\n" + section_5_text[class_start:class_end]

# Insert class and embedding limits after Section 2.3
bridge_text = "# Understanding this mathematical translation is critical when sizing chunk ingestion pipelines and provisioning GPU VRAM.\n"
insertion_point = content.find(bridge_text) + len(bridge_text)

embedding_limits_text = """
# %% [markdown]
# ### 2.4. Embedding Model Token Limits
# Before scaling up to LLM multi-thousand token contexts (e.g., 128k tokens), standard embedding models impose strict maximum sequence lengths. 
# For example, BERT-based embedding models typically cap at 512 tokens, and modern open-source models cap at 8192 tokens. 
# Retrieval chunking strategies must first satisfy this *embedding model limit* to prevent semantic truncation before they are packed into the broader LLM context window.

"""

benchmark_tokenizer_text = """# %%
benchmark_harness = RetrievalBenchmarkHarness()

# Benchmark Tokenization immediately after defining the BPE class
tok_benchmark_corpus = domain_corpus * 20  # 100 sentences
tok_perf = benchmark_harness.benchmark_tokenizer(tokenizer, tok_benchmark_corpus, iterations=25)

print("Tokenizer Micro-Benchmark:")
print(f"  • Mean Latency: {tok_perf['mean_latency_ms']} ms")
print(f"  • Throughput:   {tok_perf['tokens_per_sec']:,.0f} tokens/sec")

"""

content = content[:insertion_point] + "\n" + harness_class + benchmark_tokenizer_text + embedding_limits_text + content[insertion_point:]

# Remove old Section 5 and replace with Section 4.5 vector benchmark
vector_benchmark_text = """# %% [markdown]
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

print("\\nFAISS Index Architecture Comparison (N = 20,000, D = 768, Top-10):")
print(f"  • Exact Flat (FlatIP):  {flat_perf['mean_latency_ms']:.3f} ms ({flat_perf['queries_per_sec']:,.0f} QPS)")
print(f"  • Inverted File (IVF):  {ivf_perf['mean_latency_ms']:.3f} ms ({ivf_perf['queries_per_sec']:,.0f} QPS)")
print(f"  • Proximity Graph (HNSW): {hnsw_perf['mean_latency_ms']:.3f} ms ({hnsw_perf['queries_per_sec']:,.0f} QPS)")

"""

# Re-find the positions because content length changed
harness_start = content.find("# %% [markdown]\n# ## Section 5: High-Precision Retrieval Micro-Benchmarking Harness")
harness_end = content.find("# %% [markdown]\n# ## Section 6: Architectural Decision Matrix")

content = content[:harness_start] + vector_benchmark_text + content[harness_end:]


# 6. Matplotlib Alt Text fix
old_plt = '''    plt.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.show()'''

new_plt = '''    plt.tight_layout()
    
    # Save figure to memory and render as Image with alt-text for accessibility
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    
    display(Image(data=buf.getvalue(), alt="Geometric Distribution Visualizer showing the Curse of Dimensionality variance collapse as D approaches 1536"))'''

content = content.replace(old_plt, new_plt)

# Replace Section 6 and 7 headings
content = content.replace("## Section 6:", "## Section 5:")
content = content.replace("### 6.1.", "### 5.1.")
content = content.replace("### 6.2.", "### 5.2.")
content = content.replace("## Section 7:", "## Section 6:")


with open(file_path, "w") as f:
    f.write(content)

print("Patch applied successfully.")
