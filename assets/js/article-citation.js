(() => {
    "use strict";

    function initialize() {
        document.querySelectorAll("[data-copy-citation]").forEach((button) => {
            const block = button.closest(".article-citation");
            const code = block && block.querySelector(".article-citation-code");
            const label = button.querySelector("[data-copy-label]");
            const status = block && block.querySelector(".article-citation-status");
            if (!code || !label || !status) return;
            let resetTimer;

            function showStatus(message) {
                status.textContent = message;
                status.hidden = false;
            }

            function selectCitation() {
                const selection = window.getSelection();
                if (!selection) return false;
                code.focus({ preventScroll: true });
                const range = document.createRange();
                range.selectNodeContents(code);
                selection.removeAllRanges();
                selection.addRange(range);
                return true;
            }

            button.addEventListener("click", async () => {
                window.clearTimeout(resetTimer);
                button.disabled = true;
                try {
                    if (!navigator.clipboard || !navigator.clipboard.writeText) {
                        throw new Error("Clipboard API unavailable");
                    }
                    await navigator.clipboard.writeText(code.textContent.trim());
                    label.textContent = "已复制";
                    showStatus("BibTeX 引用已复制。");
                    resetTimer = window.setTimeout(() => {
                        label.textContent = "复制";
                        status.hidden = true;
                    }, 2200);
                } catch {
                    label.textContent = "复制";
                    showStatus(selectCitation()
                        ? "已选中引用，请按 ⌘C 或 Ctrl+C 手动复制。"
                        : "请选中上方引用并手动复制。");
                } finally {
                    button.disabled = false;
                }
            });
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initialize, { once: true });
    } else {
        initialize();
    }
})();
