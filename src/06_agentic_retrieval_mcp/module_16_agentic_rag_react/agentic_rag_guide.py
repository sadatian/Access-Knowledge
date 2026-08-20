# %% [markdown]
# # Module 16: Agentic RAG & Autonomous Search (ReAct)
#
# Traditional RAG is passive and brittle: it performs one vector search and immediately forces the LLM to generate an answer.
#
# **Agentic RAG** turns the retriever into an active tool within a reasoning loop. The agent:
# 1. Inspects initial search results.
# 2. Determines if the retrieved context is sufficient.
# 3. If insufficient, generates a refined sub-query and searches again.
# 4. Synthesizes a verified final answer once confident.
#
# ---

# %%
from typing import List, Dict, Any

# %% [markdown]
# ## Section 1: The ReAct (Thought -> Action -> Observation) Loop

# %%
# collapse_input
class AgenticRAGRunner:
    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns

    def run(self, user_question: str):
        print(f"User Question: '{user_question}'\n")
        
        # Turn 1
        print("  [Thought 1]: I need to find the definition of CAG.")
        print("  [Action 1]: search_tool(query='Cache-Augmented Generation definition')")
        print("  [Observation 1]: CAG preloads static context into GPU KV-cache.")
        
        # Turn 2
        print("  [Thought 2]: Now I need to identify why that reduces TTFT latency compared to RAG.")
        print("  [Action 2]: search_tool(query='CAG TTFT latency comparison')")
        print("  [Observation 2]: CAG removes prefill token computation and vector DB network hops.")
        
        # Final Synthesis
        print("  [Thought 3]: I have sufficient verified information to answer the question.")
        print("  [Final Answer]: Cache-Augmented Generation preloads context into KV-cache, bypassing runtime vector search and reducing TTFT to sub-20ms.\n")

agent = AgenticRAGRunner()
agent.run("What makes CAG faster than standard RAG?")
