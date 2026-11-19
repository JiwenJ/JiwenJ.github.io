(() => {
    "use strict";

    function initialize() {
        document.querySelectorAll(".article-share-dialog").forEach((dialog) => {
            if (dialog.dataset.shareReady || typeof dialog.showModal !== "function") return;
            dialog.dataset.shareReady = "true";
            const text = dialog.querySelector(".article-share-text");
            const copyButton = dialog.querySelector("[data-share-copy]");
            const nativeButton = dialog.querySelector("[data-share-native]");
            const status = dialog.querySelector(".article-share-status");
            const triggers = Array.from(document.querySelectorAll("[data-share-dialog]"))
                .filter((button) => button.dataset.shareDialog === dialog.id);
            let opener = null;
            let resetTimer = null;
            let session = 0;
            // Labels come from the page (data/article_ui.yaml) so they match the post's language.
            const labels = dialog.dataset;
            const copyLabel = copyButton.textContent;

            triggers.forEach((button) => {
                button.hidden = false;
                button.addEventListener("click", () => {
                    session += 1;
                    window.clearTimeout(resetTimer);
                    opener = button;
                    text.value = [dialog.dataset.shareTitle, `${labels.shareSourcePrefix ?? "原文："}${dialog.dataset.shareUrl}`].join("\n\n");
                    copyButton.disabled = false;
                    copyButton.textContent = copyLabel;
                    nativeButton.hidden = typeof navigator.share !== "function";
                    status.textContent = "";
                    dialog.showModal();
                });
            });

            nativeButton.addEventListener("click", async () => {
                const shareSession = session;
                nativeButton.disabled = true;
                try {
                    await navigator.share({
                        title: dialog.dataset.shareTitle,
                        text: dialog.dataset.shareTitle,
                        url: dialog.dataset.shareUrl
                    });
                } catch (error) {
                    if (error.name !== "AbortError" && dialog.open && shareSession === session) {
                        status.textContent = labels.shareFailed || "无法打开系统分享，请复制链接。";
                    }
                } finally {
                    nativeButton.disabled = false;
                }
            });

            copyButton.addEventListener("click", async () => {
                window.clearTimeout(resetTimer);
                const copySession = session;
                const payload = text.value;
                copyButton.disabled = true;
                try {
                    if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
                    await navigator.clipboard.writeText(payload);
                    if (!dialog.open || copySession !== session) return;
                    copyButton.textContent = labels.shareCopied || "已复制";
                    status.textContent = labels.shareCopiedStatus || "标题和链接已复制，可以粘贴分享。";
                    resetTimer = window.setTimeout(() => { copyButton.textContent = copyLabel; }, 2200);
                } catch {
                    if (!dialog.open || copySession !== session) return;
                    text.focus();
                    text.select();
                    status.textContent = labels.shareManual || "请手动复制已选中的内容：电脑按 ⌘C 或 Ctrl+C，手机可长按文字复制。";
                } finally {
                    if (dialog.open && copySession === session) copyButton.disabled = false;
                }
            });
            dialog.querySelector(".article-share-close").addEventListener("click", () => dialog.close());
            dialog.addEventListener("click", (event) => {
                if (event.target === dialog) dialog.close();
            });
            dialog.addEventListener("close", () => {
                session += 1;
                window.clearTimeout(resetTimer);
                opener?.focus({ preventScroll: true });
            });
        });
    }

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, { once: true });
    else initialize();
})();
