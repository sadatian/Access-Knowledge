# %% [markdown]
# # Module 04: Advanced Chunking & Hierarchical Ingestion
#
# Welcome to **Module 04** of the Knowledge Retrieval A-Z masterclass.
# Chunking is the fundamental gatekeeper of any Retrieval-Augmented Generation (RAG) system:
# - **Too Large Chunks:** Embeddings average out fine-grained facts into generic representations, introducing context pollution and wasting prompt budget.
# - **Too Small Chunks:** Critical contextual relationships and dependencies across sentences are severed, leading to retrieval mismatches and fragmented generations.
# - **Structure-Agnostic Slicing:** Naively splitting across character lengths breaks code functions, markdown tables, and multi-sentence assertions.
#
# In this module, we construct and master production-grade chunking architectures:
# 1. **The Chunking Taxonomy**: Fixed-Size with Overlap, Sentence-Boundary, Recursive Markdown Structural, and Semantic Similarity Gradient Chunking.
# 2. **Hierarchical & Parent-Child Document Ingestion**: Decoupling the indexing unit (fine-grained child chunks for vector precision) from the generation unit (rich parent documents for LLM context).
# 3. **Multi-Format Ingestion Engines**: Specialized structural parsers for Markdown, Python Source Code AST, and Tabular Data with persistent headers.
# 4. **Systematic Retrieval Quality Benchmark**: Comparative evaluation measuring retrieval precision, context relevance, and token efficiency.
# 5. **Presenter Dashboard & Visualizer (`# collapse_input`)**: Auto-collapsing ASCII chunk tree and boundary visualizer.
#
# ---

# %%
import ast
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

# Hardware Accelerator Detection
def detect_compute_device() -> torch.device:
    """Detect available compute accelerator (CUDA GPU / MPS) with graceful CPU fallback."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        print(f"[INFO] Ingestion Compute Hardware: CUDA GPU -> {device_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Ingestion Compute Hardware: Apple Silicon MPS GPU")
    else:
        device = torch.device("cpu")
        print("[INFO] Ingestion Compute Hardware: CPU (Optimized SIMD)")
    return device

DEVICE = detect_compute_device()

# %% [markdown]
# ## Section 1: The Taxonomy of Chunking Strategies
#
# Modern RAG systems employ four primary chunking strategies depending on document structure and latency constraints:
#
# 1. **Fixed-Size Chunking with Sliding Window Overlap:** Fast, naive character/word slicing with step overlap $(C_{\text{size}} - O_{\text{size}})$.
# 2. **Sentence-Level Chunking:** Preserves complete linguistic sentences using regex punctuation boundaries.
# 3. **Recursive Structural Chunking:** Hierarchically splits along structural delimiters (`\n## `, `\n### `, `\n\n`, `\n`, ` `).
# 4. **Semantic Similarity Gradient Chunking:** Calculates cosine distance between adjacent sentence embeddings and splits at statistical distance spikes:
#
# $$\Delta_{\text{sim}}(s_i, s_{i+1}) = 1.0 - \frac{\mathbf{e}_i \cdot \mathbf{e}_{i+1}}{\|\mathbf{e}_i\| \|\mathbf{e}_{i+1}\|}$$
#
# A boundary is declared whenever $\Delta_{\text{sim}} > \mu_{\Delta} + k \cdot \sigma_{\Delta}$.

# %%
class TextChunkingEngine:
    """Production Text Chunking Suite implementing Fixed, Sentence, Recursive, and Semantic strategies."""

    def __init__(self, dimension: int = 128, device: Optional[torch.device] = None):
        self.dimension = dimension
        self.device = device or DEVICE

    # 1. Fixed-Size Chunking
    @staticmethod
    def fixed_size_chunk(text: str, chunk_size: int = 150, overlap: int = 30) -> List[Dict[str, Any]]:
        """Slice text into fixed-length windows with sliding overlap."""
        if not text:
            return []
        chunks = []
        start = 0
        step = max(1, chunk_size - overlap)
        chunk_idx = 0

        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({
                    "chunk_id": f"fixed_{chunk_idx:03d}",
                    "strategy": "fixed_size",
                    "text": chunk_content,
                    "char_span": (start, end),
                    "token_count_approx": len(chunk_content.split())
                })
                chunk_idx += 1
            if end >= len(text):
                break
            start += step
        return chunks

    # 2. Sentence-Level Chunking
    @staticmethod
    def sentence_chunk(text: str, max_sentences_per_chunk: int = 3, sentence_overlap: int = 1) -> List[Dict[str, Any]]:
        """Split text along natural sentence punctuation boundaries with sentence overlap."""
        if not text:
            return []
        # Split on sentence terminals: period, exclamation, question followed by space
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        if not sentences:
            return []

        chunks = []
        start_idx = 0
        step = max(1, max_sentences_per_chunk - sentence_overlap)
        chunk_idx = 0

        while start_idx < len(sentences):
            end_idx = min(len(sentences), start_idx + max_sentences_per_chunk)
            group = sentences[start_idx:end_idx]
            chunk_content = " ".join(group)
            chunks.append({
                "chunk_id": f"sent_{chunk_idx:03d}",
                "strategy": "sentence_boundary",
                "text": chunk_content,
                "sentence_count": len(group),
                "token_count_approx": len(chunk_content.split())
            })
            chunk_idx += 1
            if end_idx >= len(sentences):
                break
            start_idx += step
        return chunks

    # 3. Recursive Character & Markdown Structural Chunking
    @classmethod
    def recursive_markdown_chunk(
        cls,
        text: str,
        max_chunk_size: int = 250,
        delimiters: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Recursively split text by structural Markdown headers, paragraphs, and newlines."""
        if delimiters is None:
            delimiters = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]

        def _split_recursive(content: str, delim_idx: int) -> List[str]:
            if len(content) <= max_chunk_size or delim_idx >= len(delimiters):
                return [content.strip()] if content.strip() else []

            delimiter = delimiters[delim_idx]
            sub_parts = content.split(delimiter)
            result = []
            accumulator = ""

            for part in sub_parts:
                part_with_delim = part if delimiter == " " else (part + ("" if part.endswith("\n") else "\n"))
                if len(accumulator) + len(part_with_delim) <= max_chunk_size:
                    accumulator += part_with_delim
                else:
                    if accumulator.strip():
                        result.append(accumulator.strip())
                    if len(part) > max_chunk_size:
                        result.extend(_split_recursive(part, delim_idx + 1))
                        accumulator = ""
                    else:
                        accumulator = part_with_delim

            if accumulator.strip():
                result.append(accumulator.strip())
            return result

        raw_chunks = _split_recursive(text, 0)
        return [
            {
                "chunk_id": f"rec_md_{i:03d}",
                "strategy": "recursive_markdown",
                "text": c,
                "char_length": len(c),
                "token_count_approx": len(c.split())
            }
            for i, c in enumerate(raw_chunks)
        ]

    # 4. Semantic Similarity Gradient Chunking
    def _project_sentence(self, sentence: str) -> np.ndarray:
        """Deterministic subword feature projection for fast semantic distance tracking."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = re.findall(r"\b\w+\b", sentence.lower())
        for i, w in enumerate(words):
            h = abs(hash(f"{w}_{i}")) % self.dimension
            vec[h] += 1.0 / math.sqrt(i + 1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def semantic_chunk(
        self,
        text: str,
        similarity_threshold_percentile: float = 70.0,
        min_sentences_per_chunk: int = 1,
        max_sentences_per_chunk: int = 5
    ) -> List[Dict[str, Any]]:
        """Split text dynamically where semantic distance between adjacent sentences spikes."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        if len(sentences) <= 1:
            return [{"chunk_id": "sem_000", "strategy": "semantic_gradient", "text": text, "token_count_approx": len(text.split())}]

        # Compute sentence embeddings
        embeddings = [self._project_sentence(s) for s in sentences]
        
        # Calculate cosine distances between adjacent sentences: 1.0 - (u . v)
        distances = []
        for i in range(len(embeddings) - 1):
            cos_sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            cos_dist = 1.0 - max(0.0, cos_sim)
            distances.append(cos_dist)

        # Threshold based on distance distribution
        threshold = float(np.percentile(distances, similarity_threshold_percentile)) if distances else 0.5
        
        chunks = []
        current_sentences = [sentences[0]]
        chunk_idx = 0

        for i, dist in enumerate(distances):
            should_split = (dist >= threshold and len(current_sentences) >= min_sentences_per_chunk) or (len(current_sentences) >= max_sentences_per_chunk)
            if should_split:
                chunk_content = " ".join(current_sentences)
                chunks.append({
                    "chunk_id": f"sem_{chunk_idx:03d}",
                    "strategy": "semantic_gradient",
                    "text": chunk_content,
                    "sentence_count": len(current_sentences),
                    "token_count_approx": len(chunk_content.split())
                })
                chunk_idx += 1
                current_sentences = [sentences[i + 1]]
            else:
                current_sentences.append(sentences[i + 1])

        if current_sentences:
            chunk_content = " ".join(current_sentences)
            chunks.append({
                "chunk_id": f"sem_{chunk_idx:03d}",
                "strategy": "semantic_gradient",
                "text": chunk_content,
                "sentence_count": len(current_sentences),
                "token_count_approx": len(chunk_content.split())
            })

        return chunks

# %% [markdown]
# ### Demo 1: Comprehensive Comparison of the 4 Chunking Strategies
#
# Below, we ingest a multi-paragraph technical passage explaining LLM inference and compare the output boundaries of all 4 chunkers.

# %%
sample_technical_text = (
    "Cache-Augmented Generation (CAG) bypasses runtime vector searches by preloading static knowledge directly into "
    "the LLM KV-cache. This ensures full document attention while dramatically cutting Time-To-First-Token (TTFT). "
    "In contrast, standard Vector RAG relies on external Approximate Nearest Neighbor (ANN) search over indexed chunks. "
    "When documents scale into millions of pages, vector databases use HNSW graphs and Product Quantization to compress memory. "
    "However, chunking boundaries frequently sever relational context between adjacent paragraphs. "
    "Hierarchical parent-child architectures solve this trade-off by indexing small child chunks for vector precision "
    "while returning the complete parent document to the LLM for generation."
)

chunk_engine = TextChunkingEngine(dimension=128, device=DEVICE)

fixed_res = chunk_engine.fixed_size_chunk(sample_technical_text, chunk_size=160, overlap=30)
sent_res = chunk_engine.sentence_chunk(sample_technical_text, max_sentences_per_chunk=2, sentence_overlap=1)
rec_res = chunk_engine.recursive_markdown_chunk(sample_technical_text, max_chunk_size=180)
sem_res = chunk_engine.semantic_chunk(sample_technical_text, similarity_threshold_percentile=60.0)

print("=== [Chunking Strategy Taxonomy Comparison] ===")
print(f"1. Fixed-Size Chunking:       {len(fixed_res)} chunks generated")
print(f"2. Sentence-Level Chunking:   {len(sent_res)} chunks generated")
print(f"3. Recursive Markdown:        {len(rec_res)} chunks generated")
print(f"4. Semantic Gradient:         {len(sem_res)} chunks generated")

print("\n[Sample Chunk Inspection]")
print(f"• Fixed-Size [0]:     '{fixed_res[0]['text'][:80]}...'")
print(f"• Sentence-Level [0]: '{sent_res[0]['text'][:80]}...'")
print(f"• Recursive MD [0]:   '{rec_res[0]['text'][:80]}...'")
print(f"• Semantic [0]:       '{sem_res[0]['text'][:80]}...'")

# %% [markdown]
# ## Section 2: Hierarchical & Parent-Child Document Ingestion
#
# In production RAG, a critical mismatch exists between the **Retriever** and the **Generator**:
# - **Retriever Need:** Small, dense, highly focused chunks (e.g., 64 tokens) that match user queries with minimal semantic dilution.
# - **Generator (LLM) Need:** Broad, comprehensive context (e.g., 512 tokens or full section) to reason across multiple sentences without hallucinating missing context.
#
# The **Hierarchical Parent-Child Store** resolves this:
# 1. Parent documents are stored in an in-memory or key-value store.
# 2. Each parent is decomposed into child chunks, which are embedded into the vector index with a foreign key pointer `parent_id`.
# 3. At query time, searching retrieves child chunks, but the pipeline resolves back to the parent document for generation.

# %%
class HierarchicalDocumentStore:
    """Production Parent-Child Hierarchical Ingestion and Resolution Store."""

    def __init__(self, chunk_engine: TextChunkingEngine):
        self.chunk_engine = chunk_engine
        self.parents: Dict[str, Dict[str, Any]] = {}
        self.children: List[Dict[str, Any]] = []
        self.parent_lookup: Dict[str, str] = {}  # child_id -> parent_id

    def ingest_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        child_chunk_size: int = 100,
        child_overlap: int = 20
    ) -> "HierarchicalDocumentStore":
        """Ingest a complete parent document, partition into child chunks, and establish relational links."""
        meta = metadata or {}
        self.parents[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "full_content": content,
            "metadata": meta,
            "child_ids": []
        }

        # Generate child chunks
        child_chunks = self.chunk_engine.fixed_size_chunk(content, chunk_size=child_chunk_size, overlap=child_overlap)
        for idx, c in enumerate(child_chunks):
            child_id = f"{doc_id}_c{idx:03d}"
            child_record = {
                "child_id": child_id,
                "parent_id": doc_id,
                "parent_title": title,
                "text": c["text"],
                "token_count": c["token_count_approx"],
                "metadata": meta
            }
            self.children.append(child_record)
            self.parent_lookup[child_id] = doc_id
            self.parents[doc_id]["child_ids"].append(child_id)

        return self

    def resolve_parent(self, child_id: str) -> Optional[Dict[str, Any]]:
        """Resolve a retrieved child chunk back to its full parent context."""
        parent_id = self.parent_lookup.get(child_id)
        if parent_id and parent_id in self.parents:
            return self.parents[parent_id]
        return None

    def search_children_and_resolve_parents(
        self,
        query: str,
        top_k_children: int = 3
    ) -> List[Dict[str, Any]]:
        """Simulate vector search over children and resolve unique parent contexts for LLM generation."""
        # Score children via lexical-semantic match
        q_words = set(re.findall(r"\b\w+\b", query.lower()))
        scored_children = []

        for child in self.children:
            c_words = set(re.findall(r"\b\w+\b", child["text"].lower()))
            overlap_score = len(q_words.intersection(c_words)) / max(1, len(q_words))
            scored_children.append((child, overlap_score))

        scored_children.sort(key=lambda x: x[1], reverse=True)
        top_children = scored_children[:top_k_children]

        # Resolve unique parents preserving highest child relevance rank
        seen_parents = set()
        resolved_results = []

        for child, score in top_children:
            parent = self.resolve_parent(child["child_id"])
            if parent and parent["doc_id"] not in seen_parents:
                seen_parents.add(parent["doc_id"])
                resolved_results.append({
                    "matched_child_id": child["child_id"],
                    "matched_child_snippet": child["text"],
                    "match_score": score,
                    "resolved_parent_id": parent["doc_id"],
                    "parent_title": parent["title"],
                    "parent_full_content": parent["full_content"]
                })

        return resolved_results

# %% [markdown]
# ### Demo 2: Parent-Child Ingestion & Context Resolution Demonstration
#
# Below, we ingest multiple full architecture documents and show how child chunk matching resolves directly into complete parent context.

# %%
hierarchical_store = HierarchicalDocumentStore(chunk_engine)

hierarchical_store.ingest_document(
    doc_id="arch_cag_01",
    title="Cache-Augmented Generation Architecture Specification",
    content=(
        "Cache-Augmented Generation (CAG) stores complete precomputed prompt contexts directly into the Key-Value (KV) cache of LLMs. "
        "Unlike Vector RAG, which retrieves 3 to 5 chunk snippets per query, CAG preloads up to 128k tokens of static domain knowledge. "
        "When an inference request arrives, the decoder skips the prompt prefill phase entirely, yielding 10x lower TTFT latency."
    ),
    metadata={"track": "cag", "author": "Architecture Team"}
)

hierarchical_store.ingest_document(
    doc_id="arch_vector_02",
    title="Vector Database HNSW Indexing Blueprint",
    content=(
        "Hierarchical Navigable Small World (HNSW) graphs organize high-dimensional embeddings into multi-layer proximity skip-graphs. "
        "Top layers contain long-range sparse highway edges, while the base layer contains dense local nearest neighbors. "
        "During search, greedy routing at upper layers quickly zooms in on candidate clusters before executing beam search on layer 0."
    ),
    metadata={"track": "vector_indexing", "author": "Database Team"}
)

print("=== [Hierarchical Document Store Status] ===")
print(f"Total Parent Documents: {len(hierarchical_store.parents)}")
print(f"Total Child Chunks:     {len(hierarchical_store.children)}")

query_cag = "How does KV cache preloading lower TTFT latency in CAG?"
resolved_cag = hierarchical_store.search_children_and_resolve_parents(query_cag, top_k_children=2)

print(f"\nQuery: '{query_cag}'")
for res in resolved_cag:
    print(f"  • Matched Child Chunk:  [{res['matched_child_id']}] (Score: {res['match_score']:.2f})")
    print(f"    Child Snippet:        '{res['matched_child_snippet'][:70]}...'")
    print(f"  • Resolved Parent ID:   [{res['resolved_parent_id']}] -> '{res['parent_title']}'")
    print(f"    Full Context for LLM: '{res['parent_full_content'][:110]}...'")

# %% [markdown]
# ## Section 3: Multi-Format Ingestion Engines (Markdown, Code AST, Tables)
#
# Production RAG pipelines must ingest heterogeneous document formats without destroying their inherent semantics:
#
# 1. **Markdown Header AST Parser:** Maintains the full ancestral header breadcrumb trail (e.g. `Architecture > Storage > KV-Cache`) on every chunk.
# 2. **Python Code AST Parser:** Uses Python's native `ast` module to split source code strictly at class and function boundaries, preserving full docstrings and signatures.
# 3. **Tabular Header-Preserving Chunker:** Slices large tables row-by-row while automatically repeating the schema header row on every segmented chunk.

# %%
class MultiFormatIngestionEngine:
    """Specialized ingestion parsers for Markdown, Python Code AST, and Tabular Data."""

    # 1. Header-Breadcrumb Markdown Parser
    @staticmethod
    def parse_markdown_with_breadcrumbs(markdown_text: str) -> List[Dict[str, Any]]:
        """Parse markdown and attach complete hierarchical heading breadcrumbs to every section."""
        lines = markdown_text.strip().split("\n")
        sections = []
        breadcrumbs = []
        current_body = []
        current_level = 0
        section_idx = 0

        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.*)$", line)
            if header_match:
                # Flush previous section
                if current_body:
                    body_text = "\n".join(current_body).strip()
                    if body_text:
                        sections.append({
                            "section_id": f"md_sec_{section_idx:03d}",
                            "format": "markdown",
                            "breadcrumb": " > ".join(breadcrumbs) if breadcrumbs else "Root",
                            "header": breadcrumbs[-1] if breadcrumbs else "Root",
                            "content": body_text
                        })
                        section_idx += 1
                    current_body = []

                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Adjust breadcrumb depth
                if level > current_level:
                    breadcrumbs.append(title)
                else:
                    # Pop back to appropriate depth
                    breadcrumbs = breadcrumbs[: level - 1]
                    breadcrumbs.append(title)
                current_level = level
            else:
                current_body.append(line)

        if current_body:
            body_text = "\n".join(current_body).strip()
            if body_text:
                sections.append({
                    "section_id": f"md_sec_{section_idx:03d}",
                    "format": "markdown",
                    "breadcrumb": " > ".join(breadcrumbs) if breadcrumbs else "Root",
                    "header": breadcrumbs[-1] if breadcrumbs else "Root",
                    "content": body_text
                })

        return sections

    # 2. Python Code AST Parser
    @staticmethod
    def parse_python_code_ast(source_code: str) -> List[Dict[str, Any]]:
        """Parse Python source code into function and class chunks using the AST parser."""
        chunks = []
        try:
            tree = ast.parse(source_code)
            lines = source_code.splitlines()
            
            for idx, node in enumerate(tree.body):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno or len(lines)
                    code_block = "\n".join(lines[start_line:end_line])
                    node_type = "Class" if isinstance(node, ast.ClassDef) else "Function"
                    
                    chunks.append({
                        "chunk_id": f"code_{node.name}_{idx:02d}",
                        "format": "python_ast",
                        "node_type": node_type,
                        "symbol_name": node.name,
                        "line_range": (start_line + 1, end_line),
                        "code": code_block
                    })
        except SyntaxError:
            # Fallback for code snippets that are not complete modules
            chunks.append({
                "chunk_id": "code_snippet_fallback",
                "format": "python_raw",
                "node_type": "Snippet",
                "symbol_name": "unknown",
                "line_range": (1, len(source_code.splitlines())),
                "code": source_code
            })

        return chunks

    # 3. Tabular Data Header-Preserving Chunker
    @staticmethod
    def parse_table_with_persistent_header(table_csv_or_md: str, rows_per_chunk: int = 2) -> List[Dict[str, Any]]:
        """Slice table rows while preserving the schema header on every segment."""
        lines = [line.strip() for line in table_csv_or_md.strip().split("\n") if line.strip()]
        if len(lines) <= 1:
            return []

        header = lines[0]
        separator = lines[1] if lines[1].startswith("|---") or lines[1].startswith("---") else None
        data_rows = lines[2:] if separator else lines[1:]

        chunks = []
        chunk_idx = 0
        for i in range(0, len(data_rows), rows_per_chunk):
            segment = data_rows[i : i + rows_per_chunk]
            table_piece = [header]
            if separator:
                table_piece.append(separator)
            table_piece.extend(segment)
            
            chunks.append({
                "chunk_id": f"table_chunk_{chunk_idx:03d}",
                "format": "tabular_header_preserved",
                "header_schema": header,
                "row_range": (i + 1, i + len(segment)),
                "content": "\n".join(table_piece)
            })
            chunk_idx += 1

        return chunks

# %% [markdown]
# ### Demo 3: Multi-Format Parser Demonstration
#
# Below, we parse Markdown with breadcrumb lineages, Python source code ASTs, and partitioned tabular datasets.

# %%
markdown_doc = """
# Knowledge Retrieval Architecture
Overview of retrieval systems.

## Vector Indexing
High dimensional embeddings in vector space.

### HNSW Skip Graphs
HNSW uses multi-layer proximity graphs.

## Cache-Augmented Generation
Preloads prompt context into LLM KV cache.
"""

python_code_sample = """
import numpy as np

def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    \"\"\"Compute inner product over normalized vectors.\"\"\"
    return float(np.dot(vec_a, vec_b))

class VectorIndexManager:
    \"\"\"Manages FAISS GPU indexes and inverted posting lists.\"\"\"
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.ntotal = 0
"""

table_sample = """
| Metric Name | BM25 Sparse | Dense HNSW | Hybrid RRF |
|---|---|---|---|
| Latency (ms) | 1.2 ms | 0.4 ms | 1.5 ms |
| Recall@10 | 72.0% | 88.5% | 94.0% |
| Memory Footprint | 12 MB | 64 MB | 76 MB |
| Code SKU Search | 99.0% | 45.0% | 99.0% |
"""

multi_parser = MultiFormatIngestionEngine()

md_sections = multi_parser.parse_markdown_with_breadcrumbs(markdown_doc)
code_sections = multi_parser.parse_python_code_ast(python_code_sample)
table_sections = multi_parser.parse_table_with_persistent_header(table_sample, rows_per_chunk=2)

print("=== [Multi-Format Ingestion Results] ===")
print(f"\n[1] Markdown Header Breadcrumbs ({len(md_sections)} sections):")
for s in md_sections:
    print(f"  • [{s['section_id']}] Breadcrumb: '{s['breadcrumb']}' -> Content: '{s['content'][:40]}...'")

print(f"\n[2] Python Code AST Blocks ({len(code_sections)} blocks):")
for c in code_sections:
    print(f"  • [{c['chunk_id']}] {c['node_type']} '{c['symbol_name']}' (Lines {c['line_range'][0]}-{c['line_range'][1]}): {c['code'].splitlines()[0]}")

print(f"\n[3] Tabular Header-Preserved Slices ({len(table_sections)} slices):")
for t in table_sections:
    print(f"  • [{t['chunk_id']}] Rows {t['row_range'][0]}-{t['row_range'][1]}:\n{t['content']}\n")

# %% [markdown]
# ## Section 4: Systematic Chunking Evaluation & Retrieval Quality Benchmark
#
# We evaluate the 4 chunking strategies across three challenging enterprise query scenarios:
# 1. **Scenario A (Specific Entity Needle):** Pinpoints a precise technical keyword.
# 2. **Scenario B (Multi-Sentence Complex Fact):** Requires reasoning across adjacent sentences without broken context.
# 3. **Scenario C (Tabular Schema Query):** Requires column-header context aligned with data rows.

# %%
class ChunkingEvaluationHarness:
    """Evaluates chunking strategies across retrieval precision and token overhead."""

    def __init__(self, strategies_chunks: Dict[str, List[Dict[str, Any]]]):
        self.strategies = strategies_chunks

    def evaluate_query(self, query: str, target_keywords: List[str]) -> Dict[str, Any]:
        """Measure top-1 chunk match precision and token efficiency across strategies."""
        results = {}
        for name, chunks in self.strategies.items():
            scored = []
            for c in chunks:
                text = c.get("text", c.get("content", c.get("code", "")))
                overlap = sum(1 for kw in target_keywords if kw.lower() in text.lower())
                score = overlap / len(target_keywords) if target_keywords else 0.0
                scored.append((c, score, len(text.split())))

            scored.sort(key=lambda x: x[1], reverse=True)
            top_chunk, top_score, tokens = scored[0] if scored else ({}, 0.0, 0)
            results[name] = {
                "top_score": round(top_score, 2),
                "matched_chunk_id": top_chunk.get("chunk_id", top_chunk.get("section_id", "none")),
                "chunk_tokens": tokens,
                "hit": top_score >= 0.75
            }
        return results

# %% [markdown]
# ### Demo 4: Comprehensive Evaluation Benchmark Run
#
# Below, we benchmark the chunking strategies against precision, recall, and token efficiency.

# %%
benchmark_corpus = (
    "Cache-Augmented Generation (CAG) preloads context directly into the GPU KV-cache. "
    "This avoids retrieval latency entirely and speeds up Time-To-First-Token. "
    "In contrast, Vector RAG searches HNSW vector indexes to locate top-k candidate chunks. "
    "When chunk size is too small, parent-child store links 50-token child chunks back to 500-token parent documents. "
    "For tabular data, persistent header slices maintain column metadata across table segments."
)

strategies_pool = {
    "Fixed (Size=80)": chunk_engine.fixed_size_chunk(benchmark_corpus, chunk_size=80, overlap=10),
    "Sentence-Level": chunk_engine.sentence_chunk(benchmark_corpus, max_sentences_per_chunk=2, sentence_overlap=1),
    "Recursive MD": chunk_engine.recursive_markdown_chunk(benchmark_corpus, max_chunk_size=150),
    "Semantic Gradient": chunk_engine.semantic_chunk(benchmark_corpus, similarity_threshold_percentile=60.0)
}

eval_harness = ChunkingEvaluationHarness(strategies_pool)

test_scenarios = [
    {"name": "Scenario A: Specific Entity Needle", "query": "GPU KV-cache preloading TTFT", "keywords": ["KV-cache", "TTFT", "CAG"]},
    {"name": "Scenario B: Multi-Sentence Fact", "query": "Parent-child 50-token child links to 500-token parent", "keywords": ["Parent-child", "50-token", "500-token"]},
    {"name": "Scenario C: Tabular Header Schema", "query": "Persistent header slices maintain column metadata", "keywords": ["Persistent", "header", "metadata"]}
]

print("=== [Chunking Strategies Retrieval Precision Benchmark] ===")
print(f"{'Scenario Name':<34}{'Strategy':<20}{'Precision':<12}{'Chunk Tokens':<15}{'Hit Status':<12}")
print("-" * 93)

for scen in test_scenarios:
    res = eval_harness.evaluate_query(scen["query"], scen["keywords"])
    for strat, metrics in res.items():
        hit_str = "[MATCH]" if metrics["hit"] else "[MISS]"
        print(f"{scen['name']:<34}{strat:<20}{metrics['top_score']*100:>5.1f}%      {metrics['chunk_tokens']:<15}{hit_str:<12}")

# %% [markdown]
# ## Section 5: Presenter Dashboard & Chunk Visualizer
#
# Below is the consolidated presenter dashboard rendering an ASCII chunk boundary tree and strategy decision matrix.

# %%
# collapse_input
def display_chunking_dashboard(strategies: Dict[str, List[Dict[str, Any]]]):
    """Render a clean ASCII visualizer of chunking distributions and structural boundaries."""
    print("=" * 80)
    print("           KNOWLEDGE RETRIEVAL A-Z: MODULE 04 CHUNKING DASHBOARD")
    print("=" * 80)
    
    print("\n[1] CHUNKING STRATEGY PARTITION METRICS")
    print(f"  {'Strategy Name':<24}{'Chunk Count':<14}{'Avg Tokens/Chunk':<20}{'Boundary Coherence':<20}")
    print("  " + "-" * 76)
    
    for name, chunks in strategies.items():
        cnt = len(chunks)
        token_lens = [c.get("token_count_approx", len(c.get("text", "").split())) for c in chunks]
        avg_tok = np.mean(token_lens) if token_lens else 0.0
        
        if "Fixed" in name:
            coherence = "Low (Arbitrary cut)"
        elif "Sentence" in name:
            coherence = "High (Sentence boundary)"
        elif "Recursive" in name:
            coherence = "Very High (Header AST)"
        else:
            coherence = "Adaptive (Embedding cosine)"
            
        print(f"  {name:<24}{cnt:<14}{avg_tok:<20.1f}{coherence:<20}")

    print("\n[2] ARCHITECTURAL CHUNKING SELECTION MATRIX")
    print("  • Fixed-Size Chunker:     Best for homogeneous flat text where ingestion speed is paramount.")
    print("  • Sentence-Level Chunker: Best for narrative prose and linguistic summarization pipelines.")
    print("  • Recursive MD Chunker:   Best for technical documentation, wikis, and structured markdown.")
    print("  • Semantic Chunker:       Best for conversational transcripts and multi-topic articles.")
    print("  • Parent-Child Store:     MANDATORY for enterprise RAG to resolve retrieval vs context trade-off.")

    print("\n" + "=" * 80)
    print("  [OK] Module 04 Complete! Proceeding to Module 05: Query Transformation & HyDE.")
    print("=" * 80)

# Render Dashboard
display_chunking_dashboard(strategies_pool)

# %% [markdown]
# ## Section 6: Summary & Transition to Module 05
#
# In this module, we engineered production-grade chunking and ingestion pipelines:
# - Implemented the **4 core chunking strategies**: Fixed-Size with Overlap, Sentence-Boundary, Recursive Markdown Structural, and Semantic Similarity Gradient drops.
# - Built the **Hierarchical Parent-Child Store** resolving fine-grained vector retrieval chunks directly into rich parent context for LLM generation.
# - Created specialized multi-format parsers for **Markdown Breadcrumbs**, **Python Code AST**, and **Tabular Data with Persistent Headers**.
# - Benchmarked retrieval precision, context coherence, and token efficiency.
#
# In **Module 05 (Query Transformation & Multi-Query Routing)**, we address the other side of the retrieval equation: transforming ambiguous user queries into multi-perspective expansions, Step-Back conceptual prompts, and Hypothetical Document Embeddings (HyDE).
