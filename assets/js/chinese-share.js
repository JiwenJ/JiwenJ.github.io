(() => {
    "use strict";

    const platforms = {
        native: {
            title: "分享这篇文章",
            description: "复制文章标题和链接，粘贴到聊天或动态中分享。",
            copyLabel: "复制标题和链接",
            copiedMessage: "标题和链接已复制，可以粘贴分享。"
        },
        wechat: {
            title: "微信扫码分享",
            description: "用微信扫一扫打开文章，再通过右上角菜单分享。正在用手机阅读？复制下方链接，切换到微信粘贴发送即可。",
            copyLabel: "复制链接",
            copiedMessage: "文章链接已复制，可在微信中粘贴发送。"
        },
        zhihu: {
            title: "分享到知乎",
            description: "复制文章标题和链接，打开知乎，在「发布想法」中粘贴分享。",
            copyLabel: "复制标题和链接",
            copiedMessage: "已复制。打开知乎，在「发布想法」中粘贴即可。",
            openLabel: "打开知乎",
            url: "https://www.zhihu.com/"
        },
        xiaohongshu: {
            title: "分享到小红书",
            description: "复制下方标题、摘要和原文地址，到小红书笔记中粘贴，也可以添加自己的阅读心得和配图。",
            copyLabel: "复制文案",
            copiedMessage: "文案已复制，请到小红书编辑笔记并发布。",
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
            let activePlatform = null;
            let session = 0;

            function openDialog(button, platform) {
                session += 1;
                window.clearTimeout(resetTimer);
                activePlatform = platform;
                opener = button;
                heading.textContent = platform.title;
                description.textContent = platform.description;
                const isWechat = button.dataset.sharePlatform === "wechat";
                qr.hidden = !isWechat;
                text.value = isWechat ? dialog.dataset.shareUrl : [
                    dialog.dataset.shareTitle,
                    button.dataset.sharePlatform === "xiaohongshu" ? dialog.dataset.shareSummary : "",
                    `原文：${dialog.dataset.shareUrl}`
                ].filter(Boolean).join("\n\n");
                text.rows = isWechat ? 2 : button.dataset.sharePlatform === "xiaohongshu" ? 6 : 4;
                text.classList.toggle("share-link-only", isWechat);
                copyLabel = platform.copyLabel;
                copyButton.disabled = false;
                copyButton.textContent = copyLabel;
                status.textContent = "";
                openLink.hidden = !platform.url;
                if (platform.url) {
                    openLink.href = platform.url;
                    openLink.textContent = platform.openLabel;
                }
                dialog.showModal();
            }

            triggers.forEach((button) => {
                const platform = platforms[button.dataset.sharePlatform];
                if (!platform) return;
                button.hidden = false;
                button.addEventListener("click", async () => {
                    if (button.dataset.sharePlatform === "native" && typeof navigator.share === "function") {
                        const shareSession = ++session;
                        button.disabled = true;
                        // Invoke directly from the click to preserve the browser's user activation.
                        try {
                            await navigator.share({
                                title: dialog.dataset.shareTitle,
                                text: dialog.dataset.shareTitle,
                                url: dialog.dataset.shareUrl
                            });
                            return;
                        } catch (error) {
                            if (error.name === "AbortError" || shareSession !== session) return;
                        } finally {
                            button.disabled = false;
                        }
                    }
                    openDialog(button, platform);
                });
            });

            copyButton.addEventListener("click", async () => {
                window.clearTimeout(resetTimer);
                const copySession = session;
                const payload = text.value;
                const copiedMessage = activePlatform.copiedMessage;
                copyButton.disabled = true;
                try {
                    if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
                    await navigator.clipboard.writeText(payload);
                    if (!dialog.open || copySession !== session) return;
                    copyButton.textContent = "已复制";
                    status.textContent = copiedMessage;
                    resetTimer = window.setTimeout(() => { copyButton.textContent = copyLabel; }, 2200);
                } catch {
                    if (!dialog.open || copySession !== session) return;
                    text.focus();
                    text.select();
                    status.textContent = "请手动复制已选中的内容：电脑按 ⌘C 或 Ctrl+C，手机可长按文字复制。";
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
