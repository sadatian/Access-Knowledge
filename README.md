# 🧠 Knowledge Retrieval A-Z

> A comprehensive, hands-on, and interactive curriculum mastering modern **Retrieval-Augmented Generation (RAG)**, **Cache-Augmented Generation (CAG)**, **Knowledge-Augmented Generation (KAG / GraphRAG)**, **Fine-Tuning (PEFT/LoRA)**, **Model Context Protocol (MCP)**, and **Agentic Retrieval Systems**.

---

## Overview & Philosophy

Modern Large Language Model applications rely heavily on external context to eliminate hallucinations, inject domain-specific proprietary information, and reason over vast corpuses. However, building production-grade knowledge systems requires mastering a spectrum of architectures:

```
[ Traditional Search ] ──▶ [ Hybrid RAG ] ──▶ [ GraphRAG / KAG ] ──▶ [ Cache-Augmented (CAG) ] ──▶ [ Agentic MCP ]
```

This repository is designed with a **Python-first, interactive notebook approach**:
- Every module is a standalone, executable `.py` script using the **`# %%` Jupytext percent format**.
- You can run code cell-by-cell in your IDE or browse auto-compiled documentation powered by `mkdocs-jupyter`.
- Local LLM execution is seamlessly routed to `http://localhost:5055/v1` via standard OpenAI client integration.

---

## Curriculum Tracks

| Track | Focus | Key Topics |
| :--- | :--- | :--- |
| **Track 1: Foundations & Hybrid Retrieval** | Sparse vs Dense | BM25, TF-IDF, Vector Embeddings, HNSW, IVFFlat, Reciprocal Rank Fusion (RRF) |
| **Track 2: RAG Architectures** | Advanced Ingestion & Search | Semantic Chunking, Parent-Child Docs, HyDE, Reranking, CRAG, Self-RAG |
| **Track 3: Cache-Augmented Generation (CAG)** | Ultra-Low Latency Context | KV-Cache Preloading, Static Prompt Prefix Caching, RAG vs CAG Tradeoffs |
| **Track 4: Knowledge-Augmented Generation (KAG)** | Structured & Relational Graphs | Entity-Relation Extraction, NetworkX, GraphRAG Community Summarization |
| **Track 5: Fine-Tuning for Knowledge Infusion** | Model Parametric Adaptation | Embedding FT (MNRL), LoRA / QLoRA QA Infusion, DPO Grounding |
| **Track 6: Agentic Retrieval & MCP** | Autonomous & Tool-Augmented | Model Context Protocol (MCP) Servers, ReAct Loops, Multi-Agent Retrieval Teams |
| **Track 7: Evaluation & Production Guardrails** | Quality & Security | RAG Triad (Context Relevance, Groundedness, Answer Relevance), Injection Defenses |

---

## Quick Start

```bash
# 1. Synchronize environment and dependencies
uv sync

# 2. Run the interactive CLI to inspect system status
uv run knowledge status

# 3. Execute any interactive module directly
uv run python src/intro.py
```

---

## Project Structure

```
.scratch/Knowledge/
├── .agents/                    # Agent guidance & rules
├── docs/                       # MkDocs documentation site & theme
│   ├── assets/                 # SVGs, diagrams, and logos
│   ├── stylesheets/            # Custom CSS & glassmorphic styling
│   └── index.md                # Interactive curriculum roadmap
├── src/
│   ├── intro.py                # Visual introduction & conceptual overview
│   ├── knowledge/              # Unified Knowledge CLI module
│   ├── 01_foundations_sparse_dense/
│   ├── 02_rag_architectures/
│   ├── 03_cag_long_context/
│   ├── 04_kag_graph_rag/
│   ├── 05_fine_tuning_infusion/
│   ├── 06_agentic_retrieval_mcp/
│   └── 07_eval_observability_guardrails/
├── tests/                      # Pytest verification test suite
├── mkdocs.yml                  # MkDocs Material & Jupyter configuration
├── pyproject.toml              # Dependencies & CLI entrypoints
└── project-instruction.md      # Repository blueprint & track checklist
```
