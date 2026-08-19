<!-- Knowledge Retrieval A-Z Overhauled Home Page -->
<div class="homepage-container" markdown="1">

<div class="custom-hero">
    <span class="hero-badge">Interactive Labs &amp; Code</span>
    <h1>Knowledge Retrieval A-Z</h1>
    <p class="hero-subtitle">Master modern Knowledge Augmentation for LLMs: from Sparse/Dense Hybrid Search to RAG, Cache-Augmented Generation (CAG), GraphRAG (KAG), Fine-Tuning (LoRA), and Agentic MCP systems.</p>
    <div class="hero-buttons">
        <a href="src/intro/" class="hero-btn btn-primary">📖 Read Introduction</a>
        <a href="src/01_foundations_sparse_dense/module_01_environment_setup/env_guide/" class="hero-btn btn-outline">🚀 Start First Lab</a>
        <a href="#curriculum-roadmap" class="hero-btn btn-outline">🧭 Explore Roadmap</a>
    </div>
</div>

<p class="homepage-intro">
    Welcome to <b>Knowledge Retrieval A-Z</b>. This interactive curriculum provides a comprehensive, hands-on engineering environment designed to take you from foundational retrieval algorithms to cutting-edge cognitive architectures.
    <br><br>
    As Large Language Models scale, static parameters alone cannot capture real-time, proprietary, or evolving enterprise knowledge. Building production-grade AI systems requires a deep understanding of the full retrieval continuum: combining lexical and semantic search, optimizing KV-cache architectures in <b>CAG</b>, traversing relational graphs in <b>KAG</b>, infusing facts through parameter-efficient fine-tuning (<b>PEFT/LoRA</b>), and enabling autonomous reasoning through the <b>Model Context Protocol (MCP)</b>.
</p>

---

## 🛠️ Local Environment Quick Start

To execute any interactive tutorial script locally:

<div class="terminal-window">
    <div class="terminal-header">
        <div class="terminal-buttons">
            <span class="terminal-btn close"></span>
            <span class="terminal-btn minimize"></span>
            <span class="terminal-btn maximize"></span>
        </div>
        <div class="terminal-title">bash</div>
        <div></div>
    </div>
    <div class="terminal-body">
        <span class="terminal-comment"># 1. Enter the project workspace</span><br>
        <span class="terminal-prompt">$</span> <span class="terminal-command">cd .scratch/Knowledge</span><br><br>
        <span class="terminal-comment"># 2. Synchronize all dependencies inside isolated virtual environment</span><br>
        <span class="terminal-prompt">$</span> <span class="terminal-command">uv sync</span><br><br>
        <span class="terminal-comment"># 3. Execute any interactive percent-celled python tutorial</span><br>
        <span class="terminal-prompt">$</span> <span class="terminal-command">uv run python src/intro.py</span>
    </div>
</div>

---

## 🧭 Curriculum Roadmap

<div class="roadmap-section" id="curriculum-roadmap">
    
    <div class="roadmap-grid">
        <!-- Track 1 -->
        <a href="src/01_foundations_sparse_dense/module_01_environment_setup/env_guide/" class="roadmap-item">
            <div class="roadmap-num">⚙️</div>
            <div class="roadmap-content">
                <h4>Track 1: Foundations &amp; Hybrid Search</h4>
                <p>BM25, TF-IDF, dense vector embeddings, HNSW/IVF indexing, and Reciprocal Rank Fusion.</p>
            </div>
        </a>

        <!-- Track 2 -->
        <a href="src/02_rag_architectures/module_04_chunking_ingestion/chunking_guide/" class="roadmap-item">
            <div class="roadmap-num">📄</div>
            <div class="roadmap-content">
                <h4>Track 2: RAG Architectures</h4>
                <p>Semantic chunking, parent-child documents, HyDE query transformation, reranking, and CRAG / Self-RAG.</p>
            </div>
        </a>

        <!-- Track 3 -->
        <a href="src/03_cag_long_context/module_08_cache_augmented_generation/cag_guide/" class="roadmap-item">
            <div class="roadmap-num">⚡</div>
            <div class="roadmap-content">
                <h4>Track 3: Cache-Augmented Generation (CAG)</h4>
                <p>KV-cache preloading, static prompt prefix caching, and RAG vs Long-Context decision boundaries.</p>
            </div>
        </a>

        <!-- Track 4 -->
        <a href="src/04_kag_graph_rag/module_10_knowledge_graphs/knowledge_graph_guide/" class="roadmap-item">
            <div class="roadmap-num">🕸️</div>
            <div class="roadmap-content">
                <h4>Track 4: Knowledge-Augmented Generation (KAG)</h4>
                <p>Entity-relation extraction, knowledge graphs, and GraphRAG hierarchical community summarization.</p>
            </div>
        </a>

        <!-- Track 5 -->
        <a href="src/05_fine_tuning_infusion/module_12_embedding_finetuning/embedding_ft_guide/" class="roadmap-item">
            <div class="roadmap-num">🧪</div>
            <div class="roadmap-content">
                <h4>Track 5: Fine-Tuning for Knowledge Infusion</h4>
                <p>Embedding adaptation with MNRL, PEFT / LoRA / QLoRA for factual QA, and DPO alignment for grounding.</p>
            </div>
        </a>

        <!-- Track 6 -->
        <a href="src/06_agentic_retrieval_mcp/module_15_mcp_tool_retrieval/mcp_tools_guide/" class="roadmap-item">
            <div class="roadmap-num">🔌</div>
            <div class="roadmap-content">
                <h4>Track 6: Agentic Retrieval &amp; MCP</h4>
                <p>Model Context Protocol (MCP) servers, ReAct search loops, and multi-agent collaborative retrieval teams.</p>
            </div>
        </a>

        <!-- Track 7 -->
        <a href="src/07_eval_observability_guardrails/module_18_rag_evaluation_benchmarks/rag_eval_guide/" class="roadmap-item">
            <div class="roadmap-num">📊</div>
            <div class="roadmap-content">
                <h4>Track 7: Evaluation &amp; Production Guardrails</h4>
                <p>Ragas evaluation triad (Groundedness, Relevance, Recall), prompt injection defense, and security guardrails.</p>
            </div>
        </a>
    </div>

</div>

</div>
