# %% [markdown]
# # 🎯 Module 06: Context Reranking & Compression
#
# Bi-encoder vector search is extremely fast but computes embeddings independently for query and document.
# Cross-Encoders evaluate full cross-attention between query and document tokens, achieving vastly superior relevance scores at the cost of higher latency.
#
# This tutorial covers:
# 1. **Two-Stage Retrieval: Bi-Encoder Top-50 -> Cross-Encoder Top-5**
# 2. **Context Compression & Pruning**
# 3. **Mitigating "Lost in the Middle" Attention Traps**
#
# ---

# %%
from typing import List, Dict, Tuple

# %% [markdown]
# ## 🔄 Section 1: Two-Stage Reranking Pipeline

# %%
candidates = [
    {"id": "doc_a", "text": "Python is a high-level programming language known for readable syntax.", "bi_score": 0.82},
    {"id": "doc_b", "text": "Cache-Augmented Generation preloads tokens directly into KV-cache.", "bi_score": 0.79},
    {"id": "doc_c", "text": "LLM KV-Cache optimization techniques speed up attention computation.", "bi_score": 0.75},
]

def simulate_cross_encoder_rerank(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simulate cross-attention scoring between query and document text."""
    reranked = []
    for d in docs:
        # Cross-encoder evaluates contextual interaction
        if "KV-cache" in d["text"]:
            cross_score = 0.96
        elif "Cache-Augmented" in d["text"]:
            cross_score = 0.92
        else:
            cross_score = 0.30
        reranked.append({**d, "cross_score": cross_score})
    return sorted(reranked, key=lambda x: x["cross_score"], reverse=True)

query = "How does KV cache relate to Cache-Augmented Generation?"
reranked_docs = simulate_cross_encoder_rerank(query, candidates)

print(f"Query: '{query}'")
print("\nResults after Cross-Encoder Reranking:")
for rank, d in enumerate(reranked_docs, 1):
    print(f"  [{rank}] {d['id']} (Bi-Score: {d['bi_score']:.2f} -> Cross-Score: {d['cross_score']:.2f}): {d['text']}")

# %% [markdown]
# ## 🧩 Section 2: Mitigating "Lost in the Middle"
#
# LLM attention heads tend to focus strongly on the start and end of context prompts, degrading attention in the middle.
# Reordering top documents to place the most relevant items at the extreme edges improves generation fidelity:
# `[Rank 1, Rank 3, Rank 4, Rank 2]`

# %%
def edge_reorder_context(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Place most relevant docs at the start and end of prompt context."""
    if len(docs) <= 2:
        return docs
    reordered = []
    left, right = 0, len(docs) - 1
    toggle = True
    while left <= right:
        if toggle:
            reordered.append(docs[left])
            left += 1
        else:
            reordered.append(docs[right])
            right -= 1
        toggle = not toggle
    return reordered

reordered = edge_reorder_context(reranked_docs)
print(f"\nOrder for prompt injection to prevent attention degradation: {[d['id'] for d in reordered]}")
