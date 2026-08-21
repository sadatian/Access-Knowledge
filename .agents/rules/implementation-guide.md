# Agent Rules for Knowledge Retrieval A-Z Project

Whenever you are asked to make changes, debug, or write new tutorials in this workspace, you MUST adhere to the following rules:

## 1. Review Project Instructions First
- Before performing any code edits or file creation, you MUST read and review the [project-instruction.md](file:///home/t/Access%20Knowledge/project-instruction.md) file.
- This file acts as the repository blueprint, containing structural roadmaps, todo lists, and styling standards. Do not deviate from the guidelines defined there.

## 2. Interactive Python Scripts (`# %%` Jupytext Percent Format)
- Every tutorial script must be written as an interactive Jupytext percent format file (`.py`).
- Markdown sections use `# %% [markdown]`.
- Code sections use `# %%`.
- This enables cell-by-cell interactive IDE execution and clean rendering via `mkdocs-jupyter`.

## 3. Use Pre-Built Packages & Industry-Standard Approaches (Do Not Reinvent the Wheel)
- **Leverage Standard Libraries:** Use pre-built, production-grade packages and industry-standard approaches whenever available. Do not reinvent the wheel or write verbose low-level boilerplate if a commonly used library already exists with pre-defined functions and optimized functionalities.
- **Idiomatic APIs:** Prefer idiomatic, battle-tested APIs from standard ecosystems (e.g., standard vector stores, tokenization engines, chunkers, graph libraries, evaluation frameworks).
- **Common Corpora & Realistic Data Sources:** Always utilize realistic, domain-representative corpora and standard data sources rather than trivial toy snippets, demonstrating real-world retrieval engineering workflows.
- **GPU Acceleration First:** ALWAYS use GPU acceleration whenever possible (e.g., CUDA / ROCm / MPS / GPU tensor operations / GPU-accelerated vector indexes and embeddings). Ensure code auto-detects GPU availability and defaults to GPU execution with graceful CPU fallback.

## 4. Presenter Code & Auto-Collapsing (`# collapse_input`)
- When a code cell serves purely as "presenter" code (e.g., figure/plot setup, visualization rendering, ASCII tables, or verbose print formatting routines that lack substantial algorithmic calculations), add `# collapse_input` at the top of the code cell:
  ```python
  # %%
  # collapse_input
  import matplotlib.pyplot as plt
  # plotting / display logic...
  ```
- This triggers `docs/js/code_toggle.js` to auto-collapse the input block on page load in `mkdocs-jupyter`, displaying the resulting visualization/output cleanly while allowing readers to click `In [ ]` to expand the source code on demand.
- Keep core pipeline logic, configuration parameters, and substantive system orchestrations in standard uncollapsed cells.

## 5. Local LLM Routing
- All LLM interactions must utilize the standard OpenAI library pointed to `http://localhost:5055/v1` with a dummy API key:
  ```python
  from openai import OpenAI
  client = OpenAI(base_url="http://localhost:5055/v1", api_key="dummy")
  ```
- Do not make calls to external cloud API endpoints.

## 6. Professional Typography & Emoji Restraint
- Restrict emoji usage to top-level hierarchies only (e.g., Main Title / Hero H1 and top-level Track headers).
- Eliminate emojis from lower hierarchies: subheadings (H2, H3, H4, H5), leaf module titles in `mkdocs.yml`, inline text, function/class docstrings, and terminal logs.
- Use clean text tags (e.g., `[INFO]`, `[OK]`, `[ERROR]`, or simple bullets `•`) for logging instead of scattered emojis.

## 7. Token-Efficiency & Resource Controls
- **NO Automatic Docs Build:** Do NOT run `mkdocs build` or `mkdocs serve` automatically unless explicitly requested by the user.
- **Judicious Test Execution:** Exercise common sense on when to run tests. Only run `pytest` or script verification when computational code, modules, algorithms, or execution logic have been modified. Never run `pytest` for documentation, markdown rules, prompt updates, or text-only edits.
- **Python Execution:** When code is actually modified, verify correctness directly using Python via `uv run python path/to/script.py` and targeted unit tests.
- **Targeted Reading:** When inspecting existing files, use specific ranges (`StartLine` / `EndLine`) in file-viewing tools.
- **Concise Reporting:** Keep responses concise and focused on high-level outcomes and diffs.

## 8. Comprehensive Complete System Demos for Every Section
- **Complete, Working Systems:** Every tutorial section must provide a fully functional, complete end-to-end system utilizing industry-standard packages and robust architectures rather than partial stubs or placeholders.
- **Exhaustive Demos:** Every section must include a dedicated, rich, and exhaustive code execution demonstration exercising almost every feature, method, parameter, pipeline stage, and realistic edge case of the system.
- **Structured Explanatory Output:** Output from demonstrations must be clear, structured, and informative, showing real inputs, intermediate pipeline states, and final evaluation results.

## 9. Hierarchical Subsection Numbering
- **Include Parent Section Prefix:** All subsection headings must explicitly include their parent section number.
  - In Section 2: `### 2.1. The Byte-Pair Encoding (BPE) Algorithm`, `### 2.2. Subword Merge Hierarchy Diagram`, `### 2.3. Conceptual Bridge...`.
  - In Section 4: `#### 4.1. Metric Geometries`, `#### 4.2. Unit-Norm Equivalence`, `#### 4.3. The Curse of Dimensionality...`.
  - In Section 6: `### 6.1. Metric Space Selection Guide`, `### 6.2. Vector Index Architecture Comparison`.
- **Never Use Isolated Numbers:** Do not use un-prefixed numbering (e.g. `### 1.`, `### 2.`, `#### 1.`) within a numbered major section; always prefix with the section number (`X.1`, `X.2`, `X.3`, etc.).

