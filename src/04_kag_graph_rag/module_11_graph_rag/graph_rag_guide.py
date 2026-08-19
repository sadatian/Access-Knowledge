# %% [markdown]
# # 🌐 Module 11: GraphRAG & Community Knowledge Fusion
#
# Standard vector RAG struggles to answer holistic, corpus-wide questions like *"What are the overarching themes in this entire dataset?"* because vector search only retrieves specific localized text chunks.
#
# **GraphRAG** solves this through hierarchical community clustering:
# 1. **Entity & Relationship Extraction:** Build an entity graph from documents.
# 2. **Community Detection:** Group connected entities into thematic clusters (e.g., Leiden algorithm).
# 3. **Hierarchical Summarization:** Generate structured summaries for each community cluster.
# 4. **Global vs Local Search:** Route global queries across community summaries and local queries across entity neighborhoods.
#
# ---

# %%
from typing import Dict, List, Any

# %% [markdown]
# ## 🏙️ Section 1: Hierarchical Community Clusters

# %%
communities = {
    "community_0": {
        "title": "Low-Latency Inference Architectures",
        "entities": ["CAG", "KV_Cache", "GPU_VRAM", "Prefix_Caching"],
        "summary": "This cluster covers memory-level caching techniques including CAG and prefix caches designed to eliminate inference prefill latency."
    },
    "community_1": {
        "title": "Hybrid & Relational Retrieval",
        "entities": ["BM25", "Hybrid_Search", "GraphRAG", "Knowledge_Graphs"],
        "summary": "This cluster explores multi-modal retrieval fusing lexical inverted indexes, dense vectors, and structured relational graphs."
    }
}

print("Hierarchical Knowledge Communities:")
for cid, info in communities.items():
    print(f"\n🏷️ {cid.upper()}: {info['title']}")
    print(f"   Entities: {', '.join(info['entities'])}")
    print(f"   Summary: {info['summary']}")

# %% [markdown]
# ## 🌐 Section 2: Global Search vs Local Search Routing

# %%
def route_graphrag_query(query: str) -> str:
    global_triggers = ["overview", "themes", "summarize dataset", "main concepts", "trends"]
    if any(trigger in query.lower() for trigger in global_triggers):
        return "GLOBAL_SEARCH (Querying community cluster summaries)"
    else:
        return "LOCAL_SEARCH (Querying specific entity neighborhoods & 1-hop triplets)"

q1 = "What are the main architectural themes in this repository?"
q2 = "How does KV cache specifically function in CAG?"

print(f"\nQuery: '{q1}' -> Routed to: {route_graphrag_query(q1)}")
print(f"Query: '{q2}' -> Routed to: {route_graphrag_query(q2)}")
