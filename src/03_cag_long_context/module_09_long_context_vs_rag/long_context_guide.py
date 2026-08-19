# %% [markdown]
# # 📏 Module 09: Long-Context LLMs vs RAG Tradeoff Analysis
#
# With frontier models supporting 1M+ token context windows, engineers must decide:
# *Should we stuff everything into prompt context, or build an external RAG / CAG pipeline?*
#
# This tutorial models:
# 1. **"Needle in a Haystack" (NIAH) Retrieval Accuracy at Depth**
# 2. **Token Ingestion Cost & Latency Tradeoffs**
# 3. **The Hybrid Architecture Decision Matrix**
#
# ---

# %%
from typing import Dict, Any

# %% [markdown]
# ## 🪡 Section 1: Simulating Needle in a Haystack (NIAH)

# %%
def simulate_niah_benchmark(context_length_tokens: int, needle_depth_pct: float) -> float:
    """
    Simulate retrieval recall degradation as context length increases and needle depth varies.
    Attention heads often suffer degradation around 50-70% context depth ('Lost in the Middle').
    """
    base_accuracy = 1.0 - (context_length_tokens / 1000000) * 0.15
    if 40.0 <= needle_depth_pct <= 70.0:
        base_accuracy -= 0.10  # Middle degradation penalty
    return max(0.0, min(1.0, base_accuracy))

for tokens in [32000, 128000, 500000, 1000000]:
    top_acc = simulate_niah_benchmark(tokens, needle_depth_pct=10.0)
    mid_acc = simulate_niah_benchmark(tokens, needle_depth_pct=55.0)
    print(f"Context {tokens:>8} tokens -> Recall @ Top: {top_acc*100:.1f}% | Recall @ Middle: {mid_acc*100:.1f}%")

# %% [markdown]
# ## 📊 Section 2: Tradeoff Analysis Decision Rules

# %%
def recommend_architecture(corpus_tokens: int, query_volume_per_sec: int, update_rate: str) -> str:
    if corpus_tokens <= 100000 and query_volume_per_sec >= 10:
        return "⚡ Recommendation: Cache-Augmented Generation (CAG) with preloaded KV-cache."
    elif corpus_tokens > 2000000 or update_rate == "realtime":
        return "🔍 Recommendation: Modular Hybrid RAG (BM25 + Dense + Cross-Encoder Reranking)."
    else:
        return "📏 Recommendation: Long-Context Direct Ingestion or Hierarchical GraphRAG."

print("\nArchitecture Recommendations:")
print(recommend_architecture(corpus_tokens=50000, query_volume_per_sec=50, update_rate="daily"))
print(recommend_architecture(corpus_tokens=10000000, query_volume_per_sec=5, update_rate="realtime"))
