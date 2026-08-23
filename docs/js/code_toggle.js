/**
 * Subtle Native Code Collapse Handler for Interactive Notebook Pages
 * Ensures presenter/plotting code marked with # collapse_input is collapsible (not permanently hidden),
 * with smooth interactive expand/collapse toggling via the prompt indicator.
 */
function initCodeToggle() {
  const codeCells = document.querySelectorAll(".jp-Cell.jp-CodeCell");

  codeCells.forEach(function (cell) {
    if (cell.dataset.toggleInitialized === "true") return;
    cell.dataset.toggleInitialized = "true";

    const prompt = cell.querySelector(".jp-InputPrompt");
    const editor = cell.querySelector(".jp-InputArea-editor");
    if (!prompt || !editor) return;

    // Check if cell contains presenter markers or plotting routines
    const codeText = editor.innerText || editor.textContent || "";
    const isCollapsedByDefault = (
      codeText.includes("# collapse_input") ||
      codeText.includes("# auto_collapse") ||
      codeText.includes("sp.make_subplots") ||
      codeText.includes("go.Scatter") ||
      codeText.includes("go.Figure") ||
      codeText.includes("fig.show()") ||
      codeText.includes("fig1.") ||
      codeText.includes("fig2.") ||
      codeText.includes("fig3.") ||
      codeText.includes("plt.subplots") ||
      codeText.includes("plt.show()")
    );

    // Auto-collapse cells marked for collapse on initial load
    if (isCollapsedByDefault) {
      cell.classList.add("is-collapsed");
      prompt.setAttribute("aria-expanded", "false");
    } else {
      prompt.setAttribute("aria-expanded", "true");
    }

    // Configure accessibility and tooltip with execution timing
    const execTimeBadge = editor.querySelector(".jp-ExecutionTime");
    const execTimeText = execTimeBadge ? execTimeBadge.textContent.trim() : "";
    if (execTimeText) {
      prompt.dataset.execTime = execTimeText;
      prompt.title = `Execution time: ${execTimeText} | Click to collapse/expand code`;
    } else {
      prompt.title = "Click to collapse/expand code";
    }
    prompt.setAttribute("role", "button");
    prompt.setAttribute("tabindex", "0");

    function toggleCell(e) {
      if (e) e.stopPropagation();
      const isCurrentlyCollapsed = cell.classList.toggle("is-collapsed");
      prompt.setAttribute("aria-expanded", isCurrentlyCollapsed ? "false" : "true");
    }

    prompt.addEventListener("click", toggleCell);

    prompt.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggleCell(e);
      }
    });
  });
}

// Initialize on DOM load and Material for MkDocs instant navigation
document.addEventListener("DOMContentLoaded", initCodeToggle);
if (typeof document$ !== "undefined") {
  document$.subscribe(initCodeToggle);
}
