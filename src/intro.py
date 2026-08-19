# %% [markdown]
# # 📖 Introduction to Knowledge Retrieval A-Z
#
# Welcome to the **Knowledge Retrieval A-Z Masterclass**!
#
# As Large Language Models evolve, static parametric memory is insufficient for enterprise applications requiring high precision, up-to-date facts, and private data access.
# This curriculum systematically covers the architectural landscape of modern Knowledge Retrieval and Augmentation.
#
# ---
#
# ## 🗺️ The Modern Knowledge Retrieval Spectrum
#
# ```mermaid
# graph LR
#     subgraph Spectrum ["The Knowledge Augmentation Spectrum"]
#         A["1. Lexical / BM25"] --> B["2. Dense Vector RAG"]
#         B --> C["3. Hybrid & Reranked RAG"]
#         C --> D["4. GraphRAG / KAG"]
#         D --> E["5. Cache-Augmented Gen (CAG)"]
#         E --> F["6. Fine-Tuning & LoRA"]
#         F --> G["7. Agentic RAG & MCP"]
#     end
# ```
#
# ---

# %%
import os
import sys
from openai import OpenAI

# %% [markdown]
# ## ⚡ Connecting to the Local LLM Endpoint
#
# In this environment, all LLM operations route to a local server running at `http://localhost:5055/v1`.
# We instantiate the standard OpenAI client pointing to this local address:

# %%
client = OpenAI(
    base_url="http://localhost:5055/v1",
    api_key="dummy"
)

print(f"✅ Local LLM Client initialized targeting: {client.base_url}")

# %% [markdown]
# ## 📊 Comparing Knowledge Augmentation Paradigms
#
# | Architecture | Best For | Latency Profile | Accuracy & Grounding | Compute Overhead |
# | :--- | :--- | :--- | :--- | :--- |
# | **Sparse (BM25)** | Exact keyword matches, IDs, part numbers | Extremely Low (<5ms) | Low semantic awareness | Minimal CPU |
# | **Dense Vector RAG** | Semantic conceptual queries | Low (20-50ms) | High for matched chunks | Embedding GPU/CPU |
# | **Hybrid RAG + Rerank** | Complex enterprise search | Medium (100-250ms) | State-of-the-Art top-k | Cross-encoder GPU |
# | **CAG (Cache-Augmented)** | Repetitive queries over fixed docs | Ultra-Low (<15ms TTFT) | 100% full-document attention | High KV RAM |
# | **KAG (GraphRAG)** | Multi-hop reasoning & community themes | High (500ms - 2s) | Superior relational reasoning | Graph DB + LLM extraction |
# | **Fine-Tuning (LoRA)** | Internalizing style & domain vocabulary | Zero retrieval latency | Moderate risk of hallucination | High training time |
# | **Agentic RAG / MCP** | Autonomous research & multi-source tasks | Variable (Multi-turn) | Maximum flexibility | High token usage |

# %%
def summarize_curriculum():
    tracks = [
        "Track 1: Foundations & Classical Retrieval (BM25, Dense, HNSW)",
        "Track 2: RAG Architectures (Chunking, HyDE, Reranking, CRAG/Self-RAG)",
        "Track 3: Cache-Augmented Generation (CAG & KV-Cache)",
        "Track 4: Knowledge-Augmented Generation (KAG & GraphRAG)",
        "Track 5: Fine-Tuning for Knowledge Infusion (LoRA & DPO)",
        "Track 6: Agentic Retrieval & MCP (Model Context Protocol)",
        "Track 7: Evaluation & Production Guardrails (Ragas & Injection Defenses)"
    ]
    print("🚀 Knowledge Retrieval A-Z Curriculum Ready:")
    for idx, track in enumerate(tracks, 1):
        print(f"  {idx}. {track}")

summarize_curriculum()
