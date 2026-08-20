/**
 * Subtle Native Code Collapse Handler
 */
document.addEventListener("DOMContentLoaded", function () {
  const codeCells = document.querySelectorAll(".jp-Cell.jp-CodeCell");

  codeCells.forEach(function (cell) {
    const prompt = cell.querySelector(".jp-InputPrompt");
    const editor = cell.querySelector(".jp-InputArea-editor");
    if (!prompt || !editor) return;

    // Check if cell contains figure setup, plotting code, or '# collapse_input' marker
    const codeText = editor.innerText || editor.textContent || "";
    const isCollapsedByDefault = (
      codeText.includes("sp.make_subplots") ||
      codeText.includes("go.Scatter") ||
      codeText.includes("go.Figure") ||
      codeText.includes("fig.show()") ||
      codeText.includes("fig1.") ||
      codeText.includes("fig2.") ||
      codeText.includes("fig3.") ||
      codeText.includes("plt.subplots") ||
      codeText.includes("plt.show()") ||
      codeText.includes("display(") ||
      codeText.includes("# collapse_input") ||
      codeText.includes("# auto_collapse")
    );

    // Auto-collapse cells marked for collapse on load
    if (isCollapsedByDefault) {
      cell.classList.add("is-collapsed");
    }

    // Toggle collapse state when clicking the prompt
    prompt.title = "Click to collapse/expand code";
    prompt.addEventListener("click", function () {
      cell.classList.toggle("is-collapsed");
    });
  });
});
