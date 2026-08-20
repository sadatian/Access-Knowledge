# %% [markdown]
# # Module 05: Query Transformation & HyDE
#
# Raw user queries are frequently ambiguous, brief, or poorly aligned with the phrasing found inside indexed documents.
#
# In this module, we implement:
# 1. **Multi-Query Expansion:** generating diverse query variations to maximize recall.
# 2. **Hypothetical Document Embeddings (HyDE):** generating a synthetic answer first, then retrieving documents similar to the answer.
# 3. **Step-Back Prompting:** abstracting specific questions into high-level conceptual inquiries.
#
# ---

# %%
from typing import List

# %% [markdown]
# ## Section 1: Multi-Query Rewriting & Expansion

# %%
def expand_query(raw_query: str) -> List[str]:
    """Generate multiple search query variations targeting different phrasing styles."""
    variations = [
        raw_query,
        f"Technical documentation on {raw_query}",
        f"How does {raw_query} work under the hood?",
        f"Comparison and trade-offs of {raw_query}"
    ]
    return variations

query = "Cache-Augmented Generation"
expanded = expand_query(query)
print(f"Original Query: '{query}'")
print("Expanded Multi-Queries:")
for q in expanded:
    print(f"  • {q}")

# %% [markdown]
# ## Section 2: Hypothetical Document Embeddings (HyDE)
#
# HyDE generates a synthetic hypothetical answer using an LLM. 
# Even if the hypothetical answer contains minor hallucinations, its embedding vector resides in "document space" rather than "query space", significantly improving semantic similarity matches.

# %%
def generate_hypothetical_document(query: str) -> str:
    """Simulate HyDE hypothetical document generation."""
    return (
        f"Cache-Augmented Generation (CAG) is an LLM inference architecture that stores static "
        f"knowledge bases directly in the GPU Key-Value (KV) cache. By removing vector database lookups, "
        f"CAG achieves deterministic retrieval and sub-20ms time-to-first-token responses."
    )

hyde_doc = generate_hypothetical_document(query)
print(f"\nHyDE Generated Document for retrieval query:\n'{hyde_doc}'")
