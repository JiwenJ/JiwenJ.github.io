(() => {
    "use strict";

    const platforms = {
        wechat: {
            title: "分享到微信",
            description: "用微信扫一扫打开文章，再通过右上角菜单发送给朋友或分享到朋友圈。",
            copyLabel: "复制链接"
        },
        zhihu: {
            title: "分享到知乎",
            description: "复制标题、摘要和原文链接，在知乎的想法或文章中粘贴。",
            copyLabel: "复制分享内容",
            openLabel: "打开知乎",
            url: "https://www.zhihu.com/"
        },
        xiaohongshu: {
            title: "分享到小红书",
            description: "复制分享内容，在小红书笔记中粘贴，再按需编辑标题和配图。",
            copyLabel: "复制分享内容",
            openLabel: "打开小红书",
            url: "https://www.xiaohongshu.com/"
        }
    };

    function initialize() {
        document.querySelectorAll(".article-share-dialog").forEach((dialog) => {
            if (dialog.dataset.shareReady || typeof dialog.showModal !== "function") return;
            dialog.dataset.shareReady = "true";
            const heading = dialog.querySelector("h2");
            const description = dialog.querySelector(".article-share-description");
            const qr = dialog.querySelector(".article-share-qr");
            const text = dialog.querySelector(".article-share-text");
            const copyButton = dialog.querySelector("[data-share-copy]");
            const openLink = dialog.querySelector("[data-share-open]");
            const status = dialog.querySelector(".article-share-status");
            const triggers = Array.from(document.querySelectorAll("[data-share-dialog]"))
                .filter((button) => button.dataset.shareDialog === dialog.id);
            let opener = null;
            let resetTimer = null;
            let copyLabel = "复制分享内容";

            triggers.forEach((button) => {
                const platform = platforms[button.dataset.sharePlatform];
                if (!platform) return;
                button.hidden = false;
                button.addEventListener("click", () => {
                    window.clearTimeout(resetTimer);
                    opener = button;
                    heading.textContent = platform.title;
                    description.textContent = platform.description;
                    const isWechat = button.dataset.sharePlatform === "wechat";
                    qr.hidden = !isWechat;
                    text.value = isWechat ? dialog.dataset.shareUrl : [
                        dialog.dataset.shareTitle,
                        dialog.dataset.shareSummary,
                        `原文：${dialog.dataset.shareUrl}`
                    ].filter(Boolean).join("\n\n");
                    text.rows = isWechat ? 2 : 6;
                    text.classList.toggle("share-link-only", isWechat);
                    copyLabel = platform.copyLabel;
                    copyButton.textContent = copyLabel;
                    status.textContent = "";
                    openLink.hidden = !platform.url;
                    if (platform.url) {
                        openLink.href = platform.url;
                        openLink.textContent = platform.openLabel;
                    }
                    dialog.showModal();
                });
            });

            copyButton.addEventListener("click", async () => {
                window.clearTimeout(resetTimer);
                try {
                    if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
                    await navigator.clipboard.writeText(text.value);
                    copyButton.textContent = "已复制";
                    status.textContent = qr.hidden ? "分享内容已复制，可前往平台粘贴。" : "文章链接已复制。";
                    resetTimer = window.setTimeout(() => { copyButton.textContent = copyLabel; }, 2200);
                } catch {
                    text.focus();
                    text.select();
                    status.textContent = "已选中内容，请按 ⌘C 或 Ctrl+C 复制。";
                }
            });
            dialog.querySelector(".article-share-close").addEventListener("click", () => dialog.close());
            dialog.addEventListener("click", (event) => {
                if (event.target === dialog) dialog.close();
            });
            dialog.addEventListener("close", () => opener?.focus({ preventScroll: true }));
        });
    }

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, { once: true });
    else initialize();
})();
