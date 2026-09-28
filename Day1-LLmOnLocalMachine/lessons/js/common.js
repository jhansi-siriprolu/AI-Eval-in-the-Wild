// Shared helpers for the AI Evals lesson pages.
(function () {
  // Highlight current page in the lesson nav strip.
  document.addEventListener("DOMContentLoaded", () => {
    const here = location.pathname.split("/").pop();
    document.querySelectorAll(".lesson-nav a").forEach((a) => {
      if (a.getAttribute("href") === here) a.classList.add("current");
    });

    // Copy-to-clipboard buttons on every <pre><code> block.
    document.querySelectorAll("pre").forEach((pre) => {
      const btn = document.createElement("button");
      btn.className = "copy-btn";
      btn.textContent = "Copy";
      btn.addEventListener("click", () => {
        navigator.clipboard.writeText(pre.innerText.replace(/Copy$/, "").trim());
        btn.textContent = "Copied!";
        setTimeout(() => (btn.textContent = "Copy"), 1200);
      });
      pre.appendChild(btn);
    });

    // Auto-persist any textarea/input with data-persist="key" to localStorage.
    document.querySelectorAll("[data-persist]").forEach((el) => {
      const key = "ai-evals:" + el.getAttribute("data-persist");
      try {
        const saved = localStorage.getItem(key);
        if (saved !== null) el.value = saved;
      } catch (e) {}
      el.addEventListener("input", () => {
        try { localStorage.setItem(key, el.value); } catch (e) {}
      });
    });
  });

  window.AIEvals = {
    fmtBytes(bytes) {
      const units = ["B", "KB", "MB", "GB", "TB"];
      let i = 0;
      while (bytes >= 1024 && i < units.length - 1) { bytes /= 1024; i++; }
      return bytes.toFixed(bytes < 10 && i > 0 ? 2 : 1) + " " + units[i];
    },
    fmtNum(n) {
      return new Intl.NumberFormat().format(n);
    },
  };
})();
