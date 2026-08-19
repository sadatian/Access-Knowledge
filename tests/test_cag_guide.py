import pytest

def test_cag_kv_cache_calculation():
    # KV Cache Size = 2 * num_layers * hidden_dim * context_tokens * precision_bytes
    num_layers = 32
    hidden_dim = 4096
    context_tokens = 32768
    precision_bytes = 2 # FP16
    total_bytes = 2 * num_layers * hidden_dim * context_tokens * precision_bytes
    mb = total_bytes / (1024 * 1024)
    assert mb > 0
    assert 16000 < mb < 17000 # ~16.38 GB for full 32-layer 4096-dim 32k context
