# %% [markdown]
# # 🔄 Module 07: Modular, Corrective (CRAG) & Self-RAG
#
# Standard naive RAG follows a rigid single-shot path: *Retrieve -> Prompt -> Generate*.
#
# If the retrieved context is irrelevant or noisy, the LLM hallucinates. Advanced RAG architectures introduce dynamic evaluation loops:
# 1. **Corrective RAG (CRAG):** Evaluates retrieved document confidence. If low, triggers web search fallback or query refinement.
# 2. **Self-RAG:** Embeds special reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) to dynamically decide when to retrieve and self-correct.
# 3. **Adaptive RAG:** Classifies query complexity to route between Direct LLM, Single-Hop RAG, and Multi-Hop Iterative RAG.
#
# ---

# %%
from typing import Dict, Any, List

# %% [markdown]
# ## 🛡️ Section 1: Corrective RAG (CRAG) Decision Flow

# %%
def crag_evaluator(query: str, retrieved_docs: List[str]) -> str:
    """Evaluate retrieval quality and return action: 'CORRECT', 'INCORRECT', or 'AMBIGUOUS'."""
    has_keywords = any("CAG" in d or "Cache" in d for d in retrieved_docs)
    if has_keywords:
        return "CORRECT"
    elif len(retrieved_docs) == 0:
        return "INCORRECT"
    else:
        return "AMBIGUOUS"

def run_crag_pipeline(query: str, mock_docs: List[str]):
    verdict = crag_evaluator(query, mock_docs)
    print(f"CRAG Evaluation for '{query}': Verdict = {verdict}")
    if verdict == "CORRECT":
        print("  -> Context verified. Proceeding directly to generation.")
    elif verdict == "INCORRECT":
        print("  -> Context rejected. Initiating web search / fallback retrieval.")
    else:
        print("  -> Ambiguous context. Combining internal retrieval with query refinement.")

run_crag_pipeline("Explain CAG architectures", ["Cache-Augmented Generation preloads KV cache."])
run_crag_pipeline("Explain CAG architectures", ["Completely unrelated recipe for baking bread."])

# %% [markdown]
# ## 🪞 Section 2: Self-RAG Reflection Tokens
#
# Self-RAG models output special control tokens:
# - `[Retrieve=True]` / `[Retrieve=False]`
# - `[IsRel=Relevant]` / `[IsRel=Irrelevant]`
# - `[IsSup=FullySupported]` / `[IsSup=PartiallySupported]` / `[IsSup=NoSupport]`
# - `[IsUse=5]` (Utility score from 1 to 5)

# %%
class SelfRAGSimulator:
    def evaluate_generation(self, context: str, response: str) -> Dict[str, str]:
        return {
            "retrieval_decision": "[Retrieve=True]",
            "relevance_verdict": "[IsRel=Relevant]",
            "grounding_check": "[IsSup=FullySupported]",
            "utility_rating": "[IsUse=5]"
        }

self_rag = SelfRAGSimulator()
assessment = self_rag.evaluate_generation(
    context="CAG preloads KV cache.",
    response="CAG improves TTFT by preloading KV cache."
)
print("\nSelf-RAG Reflection Assessment:")
for token, val in assessment.items():
    print(f"  • {token}: {val}")
