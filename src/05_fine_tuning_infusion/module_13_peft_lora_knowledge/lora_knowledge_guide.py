# %% [markdown]
# # 🎛️ Module 13: PEFT / LoRA / QLoRA for Knowledge Infusion
#
# Can we teach an LLM new facts purely through weights rather than runtime context injection?
#
# **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA (Low-Rank Adaptation)** freezes base model weights $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank-decomposition matrices:
#
# $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \times A)$$
#
# where $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ with rank $r \ll \min(d, k)$.
#
# ---

# %%
import numpy as np
from typing import Dict, Any

# %% [markdown]
# ## 🧮 Section 1: LoRA Parameter Math & Memory Savings

# %%
def compute_lora_parameter_footprint(d: int = 4096, k: int = 4096, r: int = 16) -> Dict[str, Any]:
    full_weights = d * k
    lora_weights = (d * r) + (r * k)
    savings_pct = (1.0 - (lora_weights / full_weights)) * 100.0
    return {
        "full_weights": full_weights,
        "lora_weights": lora_weights,
        "reduction": f"{savings_pct:.2f}%"
    }

stats = compute_lora_parameter_footprint(d=4096, k=4096, r=16)
print(f"Weight Parameter Comparison for Single Linear Layer:")
print(f"  • Full Matrix (4096 x 4096): {stats['full_weights']:,} parameters")
print(f"  • LoRA Rank-16 Matrices:     {stats['lora_weights']:,} parameters")
print(f"  • Parameter Reduction:       {stats['reduction']} memory saved!")

# %% [markdown]
# ## 📝 Section 2: Knowledge Infusion Instruction Datasets

# %%
instruction_sample = {
    "instruction": "Explain Cache-Augmented Generation and its advantage over standard RAG.",
    "input": "",
    "output": "Cache-Augmented Generation (CAG) preloads context documents directly into the LLM's KV-cache. Unlike standard RAG which performs vector lookups per query, CAG provides deterministic zero-lookup latency and attends over the full document."
}

print(f"\nSample LoRA Knowledge Fine-Tuning Format:")
print(f"  [PROMPT] {instruction_sample['instruction']}")
print(f"  [RESPONSE] {instruction_sample['output']}")
