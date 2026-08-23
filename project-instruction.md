# Knowledge Retrieval A-Z: Project Instructions & Roadmap

Welcome to the **Knowledge Retrieval A-Z** repository blueprint. This document outlines the roadmap, coding standards, local LLM protocols, and architectural guidelines for the interactive knowledge retrieval curriculum.

---

## 1. Project Goal & Design Philosophy

The purpose of this project is to provide a complete, hands-on, step-by-step masterclass on modern **Knowledge Retrieval & Augmentation Systems** for Large Language Models.

Rather than reading passive text, tutorials are written as **interactive, celled Python files (`.py`)** utilizing the `# %%` Jupytext percent format.

These scripts serve two key purposes:
1. **Interactive Execution:** Users can execute them cell-by-cell in an IDE interactive window (VS Code, PyCharm, Antigravity IDE).
2. **Auto-compiled Documentation:** `mkdocs` compiles these files into rich HTML notebooks at build time via `mkdocs-jupyter`.

---

## 2. Core Technical Guidelines

### 2.1 Dependency Management
- All Python packages must be managed using `uv`.
- Virtual environments and lockfiles are synchronized using `uv sync`.
- Run commands with `uv run python <script_path>`.

### 2.2 Python Script Cell Formatting (`# %%`)
- Always use standard Jupytext percent format for scripts.
- Text blocks reside in markdown cells:
  ```python
  # %% [markdown]
  # # Title of the Section
  # Explanatory markdown content here...
  ```
- Executable blocks reside in standard code cells:
  ```python
  # %%
  import numpy as np
  print("Running Knowledge Retrieval module...")
  ```

### 2.3 Presenter Code & Auto-Collapsing (`# collapse_input`)
- When a code cell contains purely "presenter" or visualization code (e.g., plot configurations, ASCII tables, or verbose display routines lacking substantive algorithmic calculations), insert `# collapse_input` at the top of the code block:
  ```python
  # %%
  # collapse_input
  import matplotlib.pyplot as plt
  # Figure setup and display code...
  ```
- The `docs/js/code_toggle.js` script will automatically collapse the input cell on initial load in `mkdocs-jupyter` while keeping the resulting output visible. Users can toggle visibility by clicking the prompt `In [ ]`.
- Core algorithmic implementations, mathematical operations, and data structures must remain in standard uncollapsed cells (`# %%`).

### 2.4 Professional Typography & Emoji Restraint
- Restrict emoji use to top-level hierarchies only (e.g., Main Title / Hero H1 and top-level Track headers).
- Do not use emojis in lower-level hierarchies (H2, H3, H4, H5 subheadings, leaf navigation titles, bullet points, function docstrings, or terminal outputs).
- Maintain an authoritative, technical, and distraction-free presentation style.

### 2.5 Local LLM API Routing
- Any LLM calls are routed directly to `http://localhost:5055/v1` using the standard OpenAI client:
  ```python
  from openai import OpenAI
  client = OpenAI(base_url="http://localhost:5055/v1", api_key="dummy")
  ```
- Never make requests to external cloud endpoints.

### 2.6 Use Pre-Built Packages & Industry-Standard Approaches (Do Not Reinvent the Wheel)
- **Leverage Standard Libraries:** Use pre-built, production-grade packages and industry-standard approaches whenever available. Do not reinvent the wheel or write verbose low-level boilerplate if a commonly used library already exists with pre-defined functions and optimized functionalities.
- **Idiomatic APIs:** Prefer idiomatic, battle-tested APIs from standard ecosystems (e.g., standard vector stores, tokenizers, chunkers, graph libraries, evaluation frameworks).
- **Common Corpora & Realistic Data Sources:** Always utilize realistic, domain-representative corpora and standard benchmark data sources rather than trivial toy snippets, demonstrating real-world retrieval engineering workflows.
- **GPU Acceleration First:** ALWAYS use GPU acceleration whenever possible (e.g., CUDA / ROCm / MPS / GPU tensor operations / GPU-accelerated vector indexes and embeddings). Ensure code auto-detects GPU availability and defaults to GPU execution with graceful CPU fallback.

### 2.7 Comprehensive Complete System Demos for Every Section
- Every tutorial section must provide a fully working, complete end-to-end system utilizing industry-standard packages and robust architectures rather than partial stubs or placeholders.
- Every section must include a dedicated, rich, and exhaustive code execution demonstration that exercises virtually every feature, method, parameter, pipeline stage, and realistic edge case of the system.
- Output from demonstrations must be structured, clear, and informative, showing real inputs, intermediate pipeline states, and final evaluation results.

### 2.8 Hierarchical Subsection Numbering
- All subsection headings must explicitly include their parent section number as a prefix.
  - In Section 2: `### 2.1. The Byte-Pair Encoding (BPE) Algorithm`, `### 2.2. Subword Merge Hierarchy Diagram`, `### 2.3. Conceptual Bridge...`.
  - In Section 4: `#### 4.1. Metric Geometries`, `#### 4.2. Unit-Norm Equivalence`, `#### 4.3. The Curse of Dimensionality...`.
  - In Section 6: `### 6.1. Metric Space Selection Guide`, `### 6.2. Vector Index Architecture Comparison`.
- Never use un-prefixed numbering (e.g., `### 1.`, `### 2.`, `#### 1.`) within a numbered major section; always prefix with the section number (`X.1`, `X.2`, `X.3`, etc.).

### 2.9 Cell Execution Timing & Runtime Profiling
- In compiled documentation, `mkdocs-jupyter` is configured with `record_timing: true`, capturing execution timestamps and rendering execution badges (`⏱️ <time>`) in the cell toolbar of all executable code cells.
- In interactive IDE execution (`# %%`), `knowledge.timing` connects to IPython event channels (`pre_run_cell` and `post_run_cell`) to print real-time wall and CPU execution timings.

### 2.10 Avoid Mermaid Diagrams
- Do NOT use Mermaid diagrams anywhere in this project. Avoid all ` ```mermaid ` code fences, Mermaid scripts, and custom Mermaid styling.
- Use clean Markdown lists/tables, ASCII art/Unicode box diagrams, LaTeX mathematical equations, or static Python image/SVG plots generated via `matplotlib`/`IPython.display.SVG` within `# collapse_input` presenter cells instead.

---

## 3. Module Roadmap & Curriculum Checklist

### Track 1: Foundations & Classical Retrieval
#### Module 01: Modern Retrieval Workspace Setup (`uv`, environment, test rigs, tokenizers)
- [x] Initialize Python environment and dependencies with `uv`.
- [x] Explore tokenizer mechanics, context windows, and embeddings math.
- [x] Establish foundational benchmarking harness.

#### Module 02: Dense vs Sparse Retrieval & Hybrid Search
- [x] Implement Sparse Search (BM25 / TF-IDF) with inverted indexes.
- [x] Implement Dense Semantic Search with embedding models.
- [x] Fuse sparse and dense scores using Reciprocal Rank Fusion (RRF) and convex combination.

#### Module 03: Vector Databases & Indexing Strategies
- [x] Analyze vector indexing algorithms: Exact KNN vs Flat, IVF, HNSW, and PQ.
- [x] Implement and evaluate vector stores (Chroma, FAISS, Qdrant).
- [x] Benchmark recall vs search latency tradeoffs.

---

### Track 2: Retrieval-Augmented Generation (RAG) Architecture
#### Module 04: Advanced Chunking & Ingestion Strategies
- [x] Compare fixed-size, sentence-level, recursive character, and semantic chunking.
- [x] Implement hierarchical and parent-child document chunking.
- [x] Handle multi-format ingestion (Markdown, PDF, Code, Tables).

#### Module 05: Query Transformation & Multi-Query Routing
- [x] Implement Query Rewriting, Expansion, and Step-Back Prompting.
- [x] Build Hypothetical Document Embeddings (HyDE) generation and retrieval.
- [x] Construct semantic routing between specialized vector collections.

#### Module 06: Context Reranking & Compression
- [x] Apply Cross-Encoder rerankers (e.g., BGE-Reranker, Cohere-style) to top-$K$ candidates.
- [x] Implement Contextual Compression & LLMLingua prompt token pruning.
- [x] Resolve the "Lost in the Middle" attention degradation phenomenon.

#### Module 07: Modular, Corrective (CRAG) & Self-RAG
- [x] Build Corrective RAG (CRAG) with retrieval evaluator & web-search fallback.
- [x] Construct Self-RAG reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`).
- [x] Design Adaptive RAG routing based on query complexity.

---

### Track 3: Cache-Augmented Generation (CAG) & Long-Context Architectures
#### Module 08: Cache-Augmented Generation (CAG) Patterns
- [ ] Understand Key-Value (KV) Cache architectures in LLM decoders.
- [ ] Implement persistent prefix caching and preloaded context sessions.
- [ ] Benchmark latency and throughput improvements of CAG vs standard RAG.

#### Module 09: Long-Context LLMs vs RAG Tradeoff Analysis
- [ ] Compare 1M+ token context ingestion vs dynamic chunk retrieval.
- [ ] Run "Needle in a Haystack" (NIAH) retrieval depth tests.
- [ ] Formulate cost-latency-accuracy decision matrices for RAG vs Long-Context.

---

### Track 4: Knowledge-Augmented Generation (KAG) & Graph Retrieval
#### Module 10: Knowledge Graphs & Triplet Extraction
- [ ] Extract Entities and Relations from unstructured text using local LLMs.
- [ ] Build and query Knowledge Graphs with NetworkX / Neo4j interfaces.
- [ ] Perform multi-hop graph traversals and relational reasoning.

#### Module 11: GraphRAG & Hybrid Knowledge Fusion
- [ ] Implement GraphRAG community detection and hierarchical summaries.
- [ ] Perform Global Search (dataset-wide themes) vs Local Search (entity neighborhoods).
- [ ] Fuse vector embedding similarity with graph topology paths.

---

### Track 5: Fine-Tuning for Retrieval & Knowledge Infusion
#### Module 12: Embedding Model Fine-Tuning
- [ ] Construct contrastive training pairs and triplets (Query, Positive, Hard Negative).
- [ ] Fine-tune embeddings using Multiple Negatives Ranking Loss (MNRL).
- [ ] Evaluate domain adaptation metrics (MTEB / custom validation sets).

#### Module 13: Parameter-Efficient Fine-Tuning (PEFT / LoRA) for Knowledge Infusion
- [ ] Understand LoRA (Low-Rank Adaptation) and QLoRA quantization mechanics.
- [ ] Prepare domain instruction datasets for internal knowledge infusion.
- [ ] Compare parametric knowledge retention vs non-parametric retrieval.

#### Module 14: Direct Preference Optimization (DPO) for Grounding
- [ ] Construct chosen vs rejected response pairs focusing on factual grounding.
- [ ] Train DPO policy models to penalize hallucinations and reward source citation.
- [ ] Verify attribution compliance against retrieved context.

---

### Track 6: Agentic Retrieval, MCP & Tool-Augmented Systems
#### Module 15: Model Context Protocol (MCP) & Tool-Augmented LLMs
- [ ] Understand the Model Context Protocol (MCP) architecture (Clients, Servers, Resources, Tools).
- [ ] Build a custom MCP server exposing search, file retrieval, and database tools.
- [ ] Integrate MCP tools into LLM function calling workflows.

#### Module 16: Agentic RAG & Autonomous Search (ReAct)
- [ ] Implement ReAct (Reason + Act) dynamic retrieval loops.
- [ ] Enable self-directed query refinement, pagination, and multi-step investigation.
- [ ] Add execution bounds and loop termination guardrails.

#### Module 17: Multi-Agent Collaborative Retrieval
- [ ] Build Planner-Retriever-Critic-Synthesizer multi-agent team.
- [ ] Implement debate and consensus protocols for ambiguous retrieval tasks.
- [ ] Parallelize distributed knowledge retrieval across specialized agents.

---

### Track 7: Evaluation, Observability & Production Guardrails
#### Module 18: Retrieval Evaluation & Benchmarking (Ragas / TruLens)
- [ ] Measure RAG Triad: Context Relevance, Groundedness (Faithfulness), and Answer Relevance.
- [ ] Compute classical ranking metrics: MRR, NDCG@K, Precision@K, Recall@K.
- [ ] Automate continuous evaluation pipelines with synthetic test set generation.

#### Module 19: Security, Guardrails & Hallucination Mitigation
- [ ] Defend against Indirect Prompt Injection through retrieved documents.
- [ ] Implement PII detection and contextual data masking.
- [ ] Add strict schema validation, confidence scoring, and fallback handoffs.
