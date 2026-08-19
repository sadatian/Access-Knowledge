import pytest
import networkx as nx

def test_knowledge_graph_traversal():
    G = nx.DiGraph()
    G.add_edge("CAG", "KV_Cache", relation="relies_on")
    G.add_edge("KV_Cache", "GPU_VRAM", relation="stored_in")
    
    paths = list(nx.all_simple_paths(G, source="CAG", target="GPU_VRAM"))
    assert len(paths) == 1
    assert paths[0] == ["CAG", "KV_Cache", "GPU_VRAM"]
