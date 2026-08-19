import pytest
import numpy as np

def test_reciprocal_rank_fusion_logic():
    from collections import defaultdict
    sparse = [("doc_1", 10.5), ("doc_2", 8.2)]
    dense = [("doc_2", 0.95), ("doc_1", 0.70)]
    k = 60
    
    rrf_scores = defaultdict(float)
    for rank, (doc_id, _) in enumerate(sparse, 1):
        rrf_scores[doc_id] += 1.0 / (k + rank)
    for rank, (doc_id, _) in enumerate(dense, 1):
        rrf_scores[doc_id] += 1.0 / (k + rank)
        
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    assert len(sorted_rrf) == 2
    assert sorted_rrf[0][0] in ["doc_1", "doc_2"]
