# %% [markdown]
# # Module 14: DPO & Preference Alignment for Grounding
#
# Even with retrieved documents placed in prompt context, standard LLMs sometimes hallucinate facts or ignore retrieved passages.
#
# **Direct Preference Optimization (DPO)** trains models directly on paired comparisons:
# - **Chosen ($y_w$):** Factual, completely grounded in the retrieved context, with precise citations.
# - **Rejected ($y_l$):** Plausible-sounding hallucination, ungrounded speculation, or ignoring retrieved facts.
#
# ---

# %%
from typing import Dict, Any, List

# %% [markdown]
# ## Section 1: Grounding Preference Pairs

# %%
grounding_pairs = [
    {
        "context": "CAG preloads 32k tokens into the KV-cache, reducing TTFT to under 20ms.",
        "prompt": "What is the TTFT latency achieved by CAG?",
        "chosen": "According to the provided documentation, CAG reduces Time-To-First-Token (TTFT) to under 20ms by preloading 32k tokens into the KV-cache.",
        "rejected": "CAG operates in real-time by querying external vector databases and takes approximately 250ms to generate the first token."
    }
]

print("DPO Grounding Training Example:")
for idx, pair in enumerate(grounding_pairs, 1):
    print(f"\nExample {idx}:")
    print(f"  [Context]  {pair['context']}")
    print(f"  [Prompt]   {pair['prompt']}")
    print(f"  [Chosen]   {pair['chosen']}")
    print(f"  [Rejected] {pair['rejected']}")

# %% [markdown]
# ## Section 2: DPO Mathematical Objective
#
# DPO optimizes the log-likelihood ratio between policy model $\pi_\theta$ and frozen reference model $\pi_{\text{ref}}$:
#
# $$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

# %%
def simulate_dpo_reward_margin(beta: float = 0.1, log_ratio_w: float = 2.5, log_ratio_l: float = -1.2) -> float:
    import math
    margin = beta * (log_ratio_w - log_ratio_l)
    prob_chosen_preferred = 1.0 / (1.0 + math.exp(-margin))
    return prob_chosen_preferred

prob = simulate_dpo_reward_margin()
print(f"\nImplicit Reward Probability for Grounded Response: {prob*100:.2f}% (High alignment preference)")
