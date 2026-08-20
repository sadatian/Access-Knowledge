import importlib.util
import pathlib
import pytest

def _load_chunking_guide():
    file_path = pathlib.Path(__file__).parent.parent / "src/02_rag_architectures/module_04_chunking_ingestion/chunking_guide.py"
    spec = importlib.util.spec_from_file_location("chunking_guide", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

chunking_guide = _load_chunking_guide()
TextChunkingEngine = chunking_guide.TextChunkingEngine
HierarchicalDocumentStore = chunking_guide.HierarchicalDocumentStore
MultiFormatIngestionEngine = chunking_guide.MultiFormatIngestionEngine
ChunkingEvaluationHarness = chunking_guide.ChunkingEvaluationHarness


def test_fixed_size_chunking():
    engine = TextChunkingEngine()
    text = "Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda Mu."
    chunks = engine.fixed_size_chunk(text, chunk_size=30, overlap=10)
    assert len(chunks) >= 2
    assert all("text" in c and "char_span" in c for c in chunks)
    assert chunks[0]["strategy"] == "fixed_size"


def test_sentence_chunking():
    engine = TextChunkingEngine()
    text = "First sentence here. Second sentence follows! Is this the third sentence? Yes it is."
    chunks = engine.sentence_chunk(text, max_sentences_per_chunk=2, sentence_overlap=1)
    assert len(chunks) >= 2
    assert "First sentence here" in chunks[0]["text"]
    assert chunks[0]["sentence_count"] == 2


def test_recursive_markdown_chunking():
    engine = TextChunkingEngine()
    md_text = "# Header 1\nIntroduction paragraph.\n\n## Header 2\nDetailed explanation of retrieval algorithms."
    chunks = engine.recursive_markdown_chunk(md_text, max_chunk_size=50)
    assert len(chunks) >= 2
    assert all(c["strategy"] == "recursive_markdown" for c in chunks)


def test_semantic_chunking():
    engine = TextChunkingEngine()
    text = (
        "Cache-Augmented Generation preloads KV cache. This eliminates retrieval latency. "
        "Baking sourdough bread requires flour and yeast. Fermentation creates gas bubbles."
    )
    chunks = engine.semantic_chunk(text, similarity_threshold_percentile=50.0)
    assert len(chunks) >= 1
    assert all("semantic_gradient" == c["strategy"] for c in chunks)


def test_hierarchical_document_store():
    engine = TextChunkingEngine()
    store = HierarchicalDocumentStore(engine)
    
    store.ingest_document(
        doc_id="parent_01",
        title="CAG Architecture Overview",
        content="Cache-Augmented Generation preloads static tokens into the GPU Key-Value cache directly.",
        child_chunk_size=40,
        child_overlap=10
    )
    
    assert len(store.parents) == 1
    assert len(store.children) >= 1
    
    child_id = store.children[0]["child_id"]
    resolved = store.resolve_parent(child_id)
    assert resolved is not None
    assert resolved["doc_id"] == "parent_01"

    # Search children and resolve parent
    results = store.search_children_and_resolve_parents("GPU Key-Value cache", top_k_children=1)
    assert len(results) == 1
    assert results[0]["resolved_parent_id"] == "parent_01"


def test_multi_format_ingestion():
    parser = MultiFormatIngestionEngine()

    # 1. Markdown with breadcrumbs
    md_doc = "# Architecture\nOverview.\n## Storage\nKV-Cache details."
    md_sections = parser.parse_markdown_with_breadcrumbs(md_doc)
    assert len(md_sections) >= 2
    assert "Architecture" in md_sections[0]["breadcrumb"]
    assert "Storage" in md_sections[1]["breadcrumb"]

    # 2. Python code AST
    code_str = "def search(query: str):\n    return []\n\nclass VectorStore:\n    pass"
    code_blocks = parser.parse_python_code_ast(code_str)
    assert len(code_blocks) == 2
    assert code_blocks[0]["symbol_name"] == "search"
    assert code_blocks[1]["symbol_name"] == "VectorStore"

    # 3. Tabular parser
    table_str = "| Col1 | Col2 |\n|---|---|\n| Val1 | Val2 |\n| Val3 | Val4 |"
    table_slices = parser.parse_table_with_persistent_header(table_str, rows_per_chunk=1)
    assert len(table_slices) == 2
    assert "| Col1 | Col2 |" in table_slices[0]["content"]
    assert "| Col1 | Col2 |" in table_slices[1]["content"]
