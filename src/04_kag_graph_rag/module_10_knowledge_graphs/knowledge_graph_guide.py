# %% [markdown]
# # 🕸️ Module 10: Knowledge Graphs & Triplet Extraction
#
# Vector databases search by geometric cosine similarity, but struggle with multi-hop relational questions (e.g., *"Which colleagues of Person A worked on Project X before Year Y?"*).
#
# **Knowledge-Augmented Generation (KAG)** structures unstructured text into explicit **Entity-Relation-Entity Triplets** $(Subject, Predicate, Object)$ to enable deterministic graph traversal.
#
# In this module, we implement:
# 1. **LLM Triplet Extraction Schema**
# 2. **Building a Knowledge Graph with NetworkX**
# 3. **Multi-Hop Graph Queries & Path Traversal**
#
# ---

# %%
import networkx as nx
from typing import List, Tuple, Dict, Any

# %% [markdown]
# ## 🔍 Section 1: Entity-Relation Triplet Construction

# %%
triplets = [
    ("CAG", "relies_on", "KV_Cache"),
    ("CAG", "improves", "Time_To_First_Token"),
    ("KV_Cache", "stored_in", "GPU_VRAM"),
    ("Hybrid_Search", "combines", "BM25"),
    ("Hybrid_Search", "combines", "Dense_Embeddings"),
    ("GraphRAG", "uses", "Knowledge_Graphs"),
    ("GraphRAG", "solves", "Multi_Hop_Reasoning")
]

G = nx.DiGraph()
for src, rel, dst in triplets:
    G.add_edge(src, dst, relation=rel)

print(f"Constructed Knowledge Graph:")
print(f"  • Total Nodes (Entities): {G.number_of_nodes()}")
print(f"  • Total Edges (Relations): {G.number_of_edges()}")

# %% [markdown]
# ## 🚀 Section 2: Multi-Hop Graph Traversal
#
# Finding indirect multi-hop reasoning paths (e.g., How is `CAG` linked to `GPU_VRAM`?).

# %%
def find_reasoning_paths(graph: nx.DiGraph, start_node: str, target_node: str) -> List[List[str]]:
    """Find all simple paths linking two knowledge concepts."""
    try:
        return list(nx.all_simple_paths(graph, source=start_node, target=target_node))
    except nx.NetworkXNoPath:
        return []

paths = find_reasoning_paths(G, "CAG", "GPU_VRAM")
print(f"\nMulti-hop reasoning path from CAG -> GPU_VRAM:")
for p in paths:
    print(f"  Path: {' -> '.join(p)}")
    for i in range(len(p) - 1):
        rel = G[p[i]][p[i+1]]["relation"]
        print(f"    ({p[i]}) --[{rel}]--> ({p[i+1]})")
