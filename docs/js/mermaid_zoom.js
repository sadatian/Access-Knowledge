(function() {
  function initMermaidZoom() {
    document.addEventListener("click", function(event) {
      // Don't expand if user clicked a link or button
      if (event.target.closest("a") || event.target.closest("button")) {
        return;
      }

      // Find the closest mermaid chart container
      var container = event.target.closest(".mermaid, .jp-RenderedMermaid");
      if (!container) return;

      // Check if it's already expanded
      var isExpanded = container.classList.contains("mermaid-expanded");
      if (isExpanded) {
        container.classList.remove("mermaid-expanded");
        document.body.classList.remove("mermaid-expanded-active");
      } else {
        // Retract any other expanded charts first
        var currentlyExpanded = document.querySelectorAll(".mermaid-expanded");
        currentlyExpanded.forEach(function(el) {
          el.classList.remove("mermaid-expanded");
        });

        // Expand the clicked chart
        container.classList.add("mermaid-expanded");
        document.body.classList.add("mermaid-expanded-active");
      }
    });

    // Support retracting on Escape key press
    document.addEventListener("keydown", function(event) {
      if (event.key === "Escape") {
        var currentlyExpanded = document.querySelectorAll(".mermaid-expanded");
        if (currentlyExpanded.length > 0) {
          currentlyExpanded.forEach(function(el) {
            el.classList.remove("mermaid-expanded");
          });
          document.body.classList.remove("mermaid-expanded-active");
        }
      }
    });
  }

  // Initialize
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initMermaidZoom);
  } else {
    initMermaidZoom();
  }
})();
