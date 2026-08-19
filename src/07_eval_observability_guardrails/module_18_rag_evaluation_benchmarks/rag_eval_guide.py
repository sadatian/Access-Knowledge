# %% [markdown]
# # 📊 Module 18: Retrieval Evaluation & Benchmarking (Ragas)
#
# How do you measure whether your retrieval pipeline is actually improving or degrading?
#
# Production RAG evaluation requires evaluating two distinct subsystems:
# 1. **Retrieval Component Quality:**
#    - **Context Relevance:** Does the retrieved context contain *only* relevant information?
#    - **Context Recall:** Did we retrieve all the information needed to answer the question?
#    - **Ranking Metrics:** MRR (Mean Reciprocal Rank), NDCG@K.
# 2. **Generation Component Quality:**
#    - **Faithfulness / Groundedness:** Is the answer strictly derived from the retrieved context (no hallucinations)?
#    - **Answer Relevance:** Does the answer directly address the user query?
#
# ---

# %%
import numpy as np
from typing import List, Dict, Any

# %% [markdown]
# ## 📐 Section 1: Computing Classical Ranking Metrics (MRR & NDCG)

# %%
def compute_mrr(rank_positions: List[int]) -> float:
    """Compute Mean Reciprocal Rank given 1-based ranks of first relevant item."""
    reciprocals = [1.0 / r if r > 0 else 0.0 for r in rank_positions]
    return float(np.mean(reciprocals))

def compute_ndcg_at_k(relevance_scores: List[int], k: int = 3) -> float:
    """Compute Normalized Discounted Cumulative Gain at K."""
    scores = relevance_scores[:k]
    dcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(scores))
    
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(ideal_scores))
    
    return float(dcg / idcg) if idcg > 0 else 0.0

rank_positions = [1, 2, 1, 3, 1]  # 1st item found at rank 1, 2, 1, 3, 1
mrr_val = compute_mrr(rank_positions)
ndcg_val = compute_ndcg_at_k([3, 2, 0, 1], k=3)

print(f"Ranking Evaluation Results:")
print(f"  • Mean Reciprocal Rank (MRR): {mrr_val:.4f}")
print(f"  • NDCG@3 Score:              {ndcg_val:.4f}")

# %% [markdown]
# ## 🔺 Section 2: Simulating RAG Triad Evaluation

# %%
rag_triad_scores = {
    "Context Relevance": 0.93,
    "Faithfulness / Groundedness": 0.97,
    "Answer Relevance": 0.91
}

print("\nRAG Triad Scores:")
for metric, score in rag_triad_scores.items():
    status = "✅ PASS" if score >= 0.85 else "❌ FAIL"
    print(f"  • {metric:<30}: {score:.2f} [{status}]")
