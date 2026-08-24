/**
 * Interactive Guide Enhancements:
 * 1. Subtle Native Code Collapse Handler (# collapse_input / plotting routines)
 * 2. Dynamic GitHub-Style Alert / Callout Transformer (> [!NOTE], > [!IMPORTANT], > [NOTE!], etc.)
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

function initAlerts() {
  // GitHub Octicon SVGs for distinct, standard callouts
  const alertIcons = {
    note: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM6.5 7.75A.75.75 0 0 1 7.25 7h1.5a.75.75 0 0 1 .75.75v2.75h.25a.75.75 0 0 1 0 1.5h-3.5a.75.75 0 0 1 0-1.5h.25v-2h-.25a.75.75 0 0 1-.75-.75ZM8 6a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"></path></svg>',
    info: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM6.5 7.75A.75.75 0 0 1 7.25 7h1.5a.75.75 0 0 1 .75.75v2.75h.25a.75.75 0 0 1 0 1.5h-3.5a.75.75 0 0 1 0-1.5h.25v-2h-.25a.75.75 0 0 1-.75-.75ZM8 6a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"></path></svg>',
    tip: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M8 1.5c-2.363 0-4 1.69-4 3.75 0 .984.424 1.625.984 2.304l.214.253c.223.264.47.556.673.848.284.411.537.896.621 1.49a.75.75 0 0 1-1.484.211c-.04-.282-.163-.547-.37-.847a8.456 8.456 0 0 0-.542-.68c-.09-.1-.18-.2-.26-.301C3.049 7.599 2.5 6.649 2.5 5.25 2.5 2.31 4.863 0 8 0s5.5 2.31 5.5 5.25c0 1.399-.549 2.349-1.326 3.253-.08.1-.17.2-.26.301-.164.183-.353.396-.542.68-.207.3-.33.565-.37.847a.751.751 0 0 1-1.485-.212c.084-.593.337-1.078.621-1.489.203-.292.45-.584.673-.848.075-.088.147-.173.213-.253.561-.679.985-1.32.985-2.304 0-2.06-1.637-3.75-4-3.75ZM5.75 12h4.5a.75.75 0 0 1 0 1.5h-4.5a.75.75 0 0 1 0-1.5ZM6 15.25a.75.75 0 0 1 .75-.75h2.5a.75.75 0 0 1 0 1.5h-2.5a.75.75 0 0 1-.75-.75Z"></path></svg>',
    success: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path></svg>',
    important: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v9.5A1.75 1.75 0 0 1 14.25 13H8.06l-2.573 2.573A1.458 1.458 0 0 1 3 14.543V13H1.75A1.75 1.75 0 0 1 0 11.25Zm1.75-.25a.25.25 0 0 0-.25.25v9.5c0 .138.112.25.25.25h2a.75.75 0 0 1 .75.75v2.19l2.72-2.72a.749.749 0 0 1 .53-.22h6.5a.25.25 0 0 0 .25-.25v-9.5a.25.25 0 0 0-.25-.25Zm7 2.25v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 9a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path></svg>',
    warning: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M6.457 1.047c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0 1 14.082 15H1.918a1.75 1.75 0 0 1-1.543-2.575Zm1.763.707a.25.25 0 0 0-.44 0L1.698 13.132a.25.25 0 0 0 .22.368h12.164a.25.25 0 0 0 .22-.368Zm.53 3.996v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path></svg>',
    caution: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M4.47.04A1.75 1.75 0 0 1 5.71 0h4.58c.464 0 .909.184 1.237.513l3.96 3.96c.329.328.513.773.513 1.237v4.58c0 .464-.184.909-.513 1.237l-3.96 3.96a1.748 1.748 0 0 1-1.237.513H5.71a1.748 1.748 0 0 1-1.237-.513L.513 11.53A1.75 1.75 0 0 1 0 10.293V5.71c0-.464.184-.909.513-1.237L4.47.04Zm.883 1.453a.25.25 0 0 0-.177.073L1.216 5.526a.25.25 0 0 0-.073.177v4.587c0 .066.026.13.073.177l3.96 3.96c.047.047.111.073.177.073h4.587c.066 0 .13-.026.177-.073l3.96-3.96c.047-.047.073-.111.073-.177V5.703a.25.25 0 0 0-.073-.177l-3.96-3.96a.25.25 0 0 0-.177-.073Zm3.397 2.257v4.5a.75.75 0 0 1-1.5 0v-4.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path></svg>',
    danger: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M4.47.04A1.75 1.75 0 0 1 5.71 0h4.58c.464 0 .909.184 1.237.513l3.96 3.96c.329.328.513.773.513 1.237v4.58c0 .464-.184.909-.513 1.237l-3.96 3.96a1.748 1.748 0 0 1-1.237.513H5.71a1.748 1.748 0 0 1-1.237-.513L.513 11.53A1.75 1.75 0 0 1 0 10.293V5.71c0-.464.184-.909.513-1.237L4.47.04Zm.883 1.453a.25.25 0 0 0-.177.073L1.216 5.526a.25.25 0 0 0-.073.177v4.587c0 .066.026.13.073.177l3.96 3.96c.047.047.111.073.177.073h4.587c.066 0 .13-.026.177-.073l3.96-3.96c.047-.047.073-.111.073-.177V5.703a.25.25 0 0 0-.073-.177l-3.96-3.96a.25.25 0 0 0-.177-.073Zm3.397 2.257v4.5a.75.75 0 0 1-1.5 0v-4.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path></svg>'
  };

  const alertTitles = {
    note: "Note",
    info: "Info",
    tip: "Tip",
    success: "Success",
    important: "Important",
    warning: "Warning",
    caution: "Caution",
    danger: "Danger"
  };

  const blockquotes = document.querySelectorAll("blockquote");
  const alertRegex = /^\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION|DANGER|INFO|SUCCESS)\]|^\s*\[(NOTE|TIP|IMPORTANT|WARNING|CAUTION|DANGER|INFO|SUCCESS)!\]/i;

  blockquotes.forEach(function (bq) {
    if (bq.dataset.alertProcessed === "true") return;

    const firstP = bq.querySelector("p") || bq;
    const textContent = firstP.textContent || "";
    const match = textContent.match(alertRegex);
    if (!match) return;

    bq.dataset.alertProcessed = "true";
    const typeRaw = (match[1] || match[2]).toLowerCase();
    const alertType = (typeRaw === "info") ? "note" : ((typeRaw === "caution") ? "caution" : ((typeRaw === "success") ? "tip" : typeRaw));
    const titleText = alertTitles[typeRaw] || alertTitles[alertType] || "Note";
    const iconSvg = alertIcons[typeRaw] || alertIcons[alertType] || alertIcons.note;

    // Remove the tag from the matching text node
    for (let node of firstP.childNodes) {
      if (node.nodeType === Node.TEXT_NODE && alertRegex.test(node.nodeValue)) {
        node.nodeValue = node.nodeValue.replace(alertRegex, "").trimStart();
        if (node.nodeValue.length === 0) {
          node.remove();
        }
        break;
      }
    }

    // Clean up any leading <br> in the first paragraph
    if (firstP.firstChild && firstP.firstChild.nodeName === "BR") {
      firstP.firstChild.remove();
    }

    // Create styled admonition container
    const admonition = document.createElement("div");
    admonition.className = `admonition ${alertType} github-alert github-alert-${alertType}`;

    const titleEl = document.createElement("p");
    titleEl.className = "admonition-title";
    titleEl.innerHTML = `<span class="github-alert-icon">${iconSvg}</span><span class="github-alert-title-text">${titleText}</span>`;
    admonition.appendChild(titleEl);

    // Transfer contents
    while (bq.firstChild) {
      admonition.appendChild(bq.firstChild);
    }

    // Replace blockquote with admonition
    bq.parentNode.replaceChild(admonition, bq);
  });
}

function runEnhancements() {
  initCodeToggle();
  initAlerts();
}

// Initialize on DOM load and Material for MkDocs instant navigation
document.addEventListener("DOMContentLoaded", runEnhancements);
if (typeof document$ !== "undefined") {
  document$.subscribe(runEnhancements);
}
