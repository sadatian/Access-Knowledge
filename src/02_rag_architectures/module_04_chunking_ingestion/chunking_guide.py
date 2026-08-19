# %% [markdown]
# # 📄 Module 04: Advanced Chunking & Hierarchical Ingestion
#
# Chunking is the foundational step of any RAG pipeline. How you slice source documents dictates the boundary of semantic units, context preservation, and retriever accuracy.
#
# In this module, we implement and compare:
# 1. **Fixed-Size Chunking with Overlap**
# 2. **Recursive Character & Markdown Chunking**
# 3. **Hierarchical / Parent-Child Document Chunking**
#
# ---

# %%
from typing import List, Dict

# %% [markdown]
# ## ✂️ Section 1: Fixed-Size Chunking with Sliding Window Overlap

# %%
def fixed_chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Slice text into fixed-character chunks with sliding overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        if start >= len(text):
            break
    return chunks

raw_text = (
    "Cache-Augmented Generation (CAG) bypasses runtime vector searches by preloading static knowledge directly into "
    "the LLM KV-cache. This ensures full document attention while dramatically cutting Time-To-First-Token (TTFT)."
)

chunks = fixed_chunk_text(raw_text, chunk_size=80, overlap=15)
print(f"Generated {len(chunks)} fixed chunks:")
for i, c in enumerate(chunks, 1):
    print(f"  Chunk {i}: '{c}'")

# %% [markdown]
# ## 🌳 Section 2: Parent-Child / Hierarchical Chunking
#
# **Parent-Child Strategy:**
# - Small child chunks (e.g., 100 tokens) are indexed in the vector database for high-precision retrieval matching.
# - When a child matches, the larger parent document (e.g., 500 tokens) is returned to the LLM for rich context.

# %%
class HierarchicalDocumentStore:
    def __init__(self):
        self.parents: Dict[str, str] = {}
        self.children: List[Dict[str, str]] = []

    def ingest(self, parent_id: str, full_doc: str, child_size: int = 50):
        self.parents[parent_id] = full_doc
        raw_children = fixed_chunk_text(full_doc, chunk_size=child_size, overlap=10)
        for idx, c_text in enumerate(raw_children):
            self.children.append({
                "child_id": f"{parent_id}_c{idx}",
                "parent_id": parent_id,
                "text": c_text
            })

store = HierarchicalDocumentStore()
store.ingest("parent_cag_doc", raw_text)
print(f"\nIngested into Hierarchical Store:")
print(f"  • Parent docs: {len(store.parents)}")
print(f"  • Child chunks indexed: {len(store.children)}")
print(f"  • Sample Child -> Parent link: {store.children[0]['child_id']} -> {store.children[0]['parent_id']}")
