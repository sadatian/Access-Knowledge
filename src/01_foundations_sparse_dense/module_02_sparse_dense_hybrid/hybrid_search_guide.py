# %% [markdown]
# # 🔍 Module 02: Sparse vs Dense Search & Hybrid Fusion
#
# In modern information retrieval, relying purely on keyword search (Sparse) fails on semantic nuance, while relying purely on vector search (Dense) fails on exact keywords, product SKUs, and acronyms.
#
# This tutorial demonstrates how to implement:
# 1. **Sparse Lexical Search (BM25 / Inverted Index)**
# 2. **Dense Vector Search (Cosine Similarity)**
# 3. **Hybrid Search Fusion via Reciprocal Rank Fusion (RRF)**
#
# ---

# %%
import math
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any

# %% [markdown]
# ## 📚 Section 1: Corpus & BM25 Sparse Search Engine

# %%
corpus = [
    {"id": "doc_1", "text": "Cache-Augmented Generation preloads tokens directly into the LLM KV cache for near-instant inference."},
    {"id": "doc_2", "text": "BM25 is a ranking function used in information retrieval to estimate document relevance to a search query based on term frequencies."},
    {"id": "doc_3", "text": "Hybrid search combines BM25 lexical keyword matching with dense vector embeddings to achieve superior retrieval accuracy."},
    {"id": "doc_4", "text": "Knowledge Graphs structure facts as entity-relationship triplets for GraphRAG multi-hop reasoning."}
]

class SimpleBM25:
    def __init__(self, docs: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_len = [len(d["text"].lower().split()) for d in docs]
        self.avg_doc_len = sum(self.doc_len) / len(docs)
        self.doc_freqs = defaultdict(int)
        self.inverted_index = []

        for d in docs:
            words = d["text"].lower().split()
            counts = Counter(words)
            self.inverted_index.append(counts)
            for word in counts.keys():
                self.doc_freqs[word] += 1

    def search(self, query: str) -> List[Tuple[str, float]]:
        query_terms = query.lower().split()
        scores = []
        N = len(self.docs)

        for idx, d in enumerate(self.docs):
            score = 0.0
            doc_counts = self.inverted_index[idx]
            dl = self.doc_len[idx]
            for term in query_terms:
                if term in doc_counts:
                    df = self.doc_freqs[term]
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                    tf = doc_counts[term]
                    denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_doc_len))
                    score += idf * (tf * (self.k1 + 1)) / denom
            scores.append((d["id"], score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)

bm25 = SimpleBM25(corpus)
sparse_results = bm25.search("BM25 lexical keyword")
print("BM25 Sparse Search Results:")
for doc_id, score in sparse_results:
    print(f"  {doc_id}: Score = {score:.4f}")

# %% [markdown]
# ## 🧠 Section 2: Dense Vector Search Simulation

# %%
def simulate_dense_search(query: str, docs: List[Dict[str, str]]) -> List[Tuple[str, float]]:
    # Mock dense embedding cosine similarities
    np.random.seed(42)
    scores = []
    for d in docs:
        sim = float(np.random.uniform(0.4, 0.95))
        scores.append((d["id"], sim))
    return sorted(scores, key=lambda x: x[1], reverse=True)

dense_results = simulate_dense_search("BM25 lexical keyword", corpus)
print("Dense Vector Search Results:")
for doc_id, score in dense_results:
    print(f"  {doc_id}: Similarity = {score:.4f}")

# %% [markdown]
# ## 🔀 Section 3: Reciprocal Rank Fusion (RRF)
#
# Reciprocal Rank Fusion combines rankings from different retrieval strategies without requiring score normalization:
#
# $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
#
# where $k$ is typically set to 60.

# %%
def reciprocal_rank_fusion(sparse_rankings: List[Tuple[str, float]], 
                           dense_rankings: List[Tuple[str, float]], 
                           k: int = 60) -> List[Tuple[str, float]]:
    rrf_scores = defaultdict(float)
    
    for rank, (doc_id, _) in enumerate(sparse_rankings, 1):
        rrf_scores[doc_id] += 1.0 / (k + rank)
        
    for rank, (doc_id, _) in enumerate(dense_rankings, 1):
        rrf_scores[doc_id] += 1.0 / (k + rank)
        
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

hybrid_results = reciprocal_rank_fusion(sparse_results, dense_results)
print("\n🏆 Hybrid RRF Fusion Results:")
for rank, (doc_id, score) in enumerate(hybrid_results, 1):
    doc_text = next(d["text"] for d in corpus if d["id"] == doc_id)
    print(f"  [{rank}] {doc_id} (RRF Score: {score:.5f}) -> {doc_text[:60]}...")
