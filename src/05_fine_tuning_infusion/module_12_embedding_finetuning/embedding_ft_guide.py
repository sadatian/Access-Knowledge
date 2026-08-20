# %% [markdown]
# # Module 12: Embedding Model Fine-Tuning (MNRL)
#
# Off-the-shelf embedding models (like BGE or MiniLM) perform well on general web text, but underperform on domain-specific corpora with specialized acronyms, medical terms, or internal codebases.
#
# In this module, we learn:
# 1. **Preparing Contrastive Training Triplets:** `(Query, Positive_Doc, Hard_Negative_Doc)`
# 2. **Multiple Negatives Ranking Loss (MNRL)**
# 3. **Evaluating Retrieval Lift Post-Adaptation**
#
# ---

# %%
import numpy as np
from typing import List, Dict, Tuple

# %% [markdown]
# ## Section 1: Contrastive Triplet Generation

# %%
training_triplets = [
    {
        "query": "How to eliminate TTFT latency in fixed document retrieval?",
        "positive": "Cache-Augmented Generation (CAG) preloads the entire document into the LLM KV-cache to achieve sub-20ms TTFT.",
        "hard_negative": "Vector database indexing with HNSW reduces average search latency for million-scale collections."
    },
    {
        "query": "What ranking algorithm fuses sparse and dense candidate lists?",
        "positive": "Reciprocal Rank Fusion (RRF) merges candidate ranks across sparse BM25 and dense embedding systems.",
        "hard_negative": "Cross-encoders use full self-attention to score document query pairs directly."
    }
]

print("Fine-Tuning Triplet Samples:")
for i, t in enumerate(training_triplets, 1):
    print(f"\nTriplet {i}:")
    print(f"  • Query:         {t['query']}")
    print(f"  • Positive (+):  {t['positive']}")
    print(f"  • Negative (-):  {t['hard_negative']}")

# %% [markdown]
# ## Section 2: Multiple Negatives Ranking Loss (MNRL)
#
# MNRL treats all other positive documents in the mini-batch as in-batch negative examples, enabling efficient training without explicitly mining billions of hard negatives.

# %%
def simulate_contrastive_loss(pos_sim: float, neg_sim: float, temperature: float = 0.05) -> float:
    # InfoNCE loss: -log( exp(pos / T) / (exp(pos / T) + exp(neg / T)) )
    import math
    pos_exp = math.exp(pos_sim / temperature)
    neg_exp = math.exp(neg_sim / temperature)
    loss = -math.log(pos_exp / (pos_exp + neg_exp))
    return loss

loss_before = simulate_contrastive_loss(pos_sim=0.70, neg_sim=0.65)
loss_after = simulate_contrastive_loss(pos_sim=0.95, neg_sim=0.30)

print(f"\nTraining Loss Progress:")
print(f"  • Initial Contrastive Loss: {loss_before:.4f}")
print(f"  • Post-Tuning Loss:         {loss_after:.4f} (Clean separation achieved)")
