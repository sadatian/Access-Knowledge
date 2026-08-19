# Agent Rules for Knowledge Retrieval A-Z Project

Whenever you are asked to make changes, debug, or write new tutorials in this workspace, you MUST adhere to the following rules:

## 1. Review Project Instructions First
- Before performing any code edits or file creation, you MUST read and review the [project-instruction.md](file:///home/t/MLOps/.scratch/Knowledge/project-instruction.md) file.
- This file acts as the repository blueprint, containing structural roadmaps, todo lists, and styling standards. Do not deviate from the guidelines defined there.

## 2. Interactive Python Scripts (`# %%` Jupytext Percent Format)
- Every tutorial script must be written as an interactive Jupytext percent format file (`.py`).
- Markdown sections use `# %% [markdown]`.
- Code sections use `# %%`.
- This enables cell-by-cell interactive IDE execution and clean rendering via `mkdocs-jupyter`.

## 3. Local LLM Routing
- All LLM interactions must utilize the standard OpenAI library pointed to `http://localhost:5055/v1` with a dummy API key:
  ```python
  from openai import OpenAI
  client = OpenAI(base_url="http://localhost:5055/v1", api_key="dummy")
  ```
- Do not make calls to external cloud API endpoints.

## 4. Token-Efficiency & Resource Controls
- **NO Automatic Docs Build:** Do NOT run `mkdocs build` or `mkdocs serve` automatically unless explicitly requested by the user.
- **Python Execution:** Verify code correctness directly using Python via `uv run python path/to/script.py`.
- **Targeted Reading:** When inspecting existing files, use specific ranges (`StartLine` / `EndLine`) in file-viewing tools.
- **Concise Reporting:** Keep responses concise and focused on high-level outcomes and diffs.
