(() => {
    "use strict";

    function initialize() {
        const targets = new Map();
        document.querySelectorAll(".post-single.blog-post a.footnote-ref").forEach((reference) => {
            let target;
            try {
                const href = reference.getAttribute("href");
                if (!href || !href.startsWith("#")) return;
                target = document.getElementById(decodeURIComponent(href.slice(1)));
            } catch {
                return;
            }
            if (target && target.closest(".footnotes")) targets.set(reference, target);
        });
        if (!targets.size) return;

        const preview = document.createElement("div");
        let previewId = "footnote-preview";
        for (let suffix = 1; document.getElementById(previewId); suffix += 1) {
            previewId = `footnote-preview-${suffix}`;
        }
        preview.id = previewId;
        preview.className = "footnote-preview";
        preview.hidden = true;
        preview.setAttribute("role", "dialog");
        preview.setAttribute("aria-modal", "false");
        preview.tabIndex = -1;

        const header = document.createElement("div");
        header.className = "footnote-preview-header";
        const title = document.createElement("span");
        const closeButton = document.createElement("button");
        closeButton.className = "footnote-preview-close";
        closeButton.type = "button";
        closeButton.textContent = "×";
        closeButton.setAttribute("aria-label", "关闭脚注预览");
        header.append(title, closeButton);

        const content = document.createElement("div");
        content.className = "footnote-preview-content";
        const jumpLink = document.createElement("a");
        jumpLink.className = "footnote-preview-jump";
        jumpLink.textContent = "跳至脚注";
        preview.append(header, content, jumpLink);
        document.body.append(preview);

        let activeReference = null;
        let pinned = false;
        let closeTimeout = null;
        let positionFrame = null;
        let restoringFocus = false;

        function cancelClose() {
            window.clearTimeout(closeTimeout);
        }

        function hasVisibleSidenote(reference) {
            const note = document.getElementById(reference.dataset.sidenoteTarget || "");
            return note && note.getClientRects().length > 0;
        }

        function updatePreviewMode() {
            targets.forEach((_target, reference) => {
                if (hasVisibleSidenote(reference)) {
                    reference.removeAttribute("aria-controls");
                    reference.removeAttribute("aria-expanded");
                    reference.removeAttribute("aria-haspopup");
                    reference.setAttribute("aria-describedby", reference.dataset.sidenoteTarget);
                } else {
                    reference.removeAttribute("aria-describedby");
                    reference.setAttribute("aria-controls", previewId);
                    reference.setAttribute("aria-expanded", "false");
                    reference.setAttribute("aria-haspopup", "dialog");
                }
            });
            if (activeReference && hasVisibleSidenote(activeReference)) closePreview();
        }

        function positionPreview() {
            positionFrame = null;
            if (!activeReference || preview.hidden) return;
            const anchor = activeReference.getBoundingClientRect();
            const bounds = preview.getBoundingClientRect();
            const viewportWidth = document.documentElement.clientWidth;
            const viewportHeight = window.innerHeight;
            const gap = 10;
            const margin = 12;
            const centered = anchor.left + anchor.width / 2 - bounds.width / 2;
            const left = Math.max(margin, Math.min(centered, viewportWidth - bounds.width - margin));
            let top = anchor.bottom + gap;
            if (top + bounds.height > viewportHeight - margin) {
                top = anchor.top - bounds.height - gap;
            }
            top = Math.max(margin, Math.min(top, viewportHeight - bounds.height - margin));
            preview.style.left = `${left}px`;
            preview.style.top = `${top}px`;
        }

        function schedulePosition() {
            if (!preview.hidden && positionFrame === null) {
                positionFrame = window.requestAnimationFrame(positionPreview);
            }
        }

        function closePreview(restoreFocus = false) {
            cancelClose();
            const reference = activeReference;
            if (reference) reference.setAttribute("aria-expanded", "false");
            activeReference = null;
            pinned = false;
            preview.hidden = true;
            if (restoreFocus && reference) {
                restoringFocus = true;
                reference.focus({ preventScroll: true });
                restoringFocus = false;
            }
        }

        function scheduleClose() {
            cancelClose();
            closeTimeout = window.setTimeout(() => {
                if (!activeReference || pinned || activeReference === document.activeElement ||
                    preview.contains(document.activeElement) || preview.matches(":hover") ||
                    activeReference.matches(":hover")) return;
                closePreview();
            }, 220);
        }

        function openPreview(reference, pin = false) {
            if (hasVisibleSidenote(reference)) return;
            cancelClose();
            if (reference !== activeReference) {
                if (activeReference) activeReference.setAttribute("aria-expanded", "false");
                const clone = targets.get(reference).cloneNode(true);
                clone.querySelectorAll(".footnote-backref, [role='doc-backlink']").forEach((node) => node.remove());
                clone.removeAttribute("id");
                clone.querySelectorAll("[id]").forEach((node) => node.removeAttribute("id"));
                content.replaceChildren(...Array.from(clone.childNodes));
                const number = reference.textContent.trim();
                title.textContent = `脚注 ${number}`;
                preview.setAttribute("aria-label", `脚注 ${number}`);
                jumpLink.href = reference.getAttribute("href");
                jumpLink.setAttribute("aria-label", `跳至文末脚注 ${number}`);
                // Only offer a jump when the original endnote is visible in this layout.
                jumpLink.hidden = !targets.get(reference).getClientRects().length;
                activeReference = reference;
                pinned = false;
            }
            pinned = pinned || pin;
            reference.setAttribute("aria-expanded", "true");
            preview.hidden = false;
            positionPreview();
        }

        targets.forEach((_target, reference) => {
            if (!reference.hasAttribute("aria-label")) {
                reference.setAttribute("aria-label", `查看脚注 ${reference.textContent.trim()}`);
            }
            reference.addEventListener("pointerenter", (event) => {
                if (event.pointerType !== "touch") openPreview(reference);
            });
            reference.addEventListener("pointerleave", scheduleClose);
            reference.addEventListener("focus", () => {
                if (!restoringFocus) openPreview(reference);
            });
            reference.addEventListener("blur", scheduleClose);
        });

        // Capture only footnote references before the theme's generic hash-link handler.
        document.addEventListener("click", (event) => {
            if (!(event.target instanceof Element)) return;
            const reference = event.target.closest("a.footnote-ref");
            if (targets.has(reference) && !hasVisibleSidenote(reference) && event.button === 0 && !event.defaultPrevented &&
                !event.metaKey && !event.ctrlKey && !event.altKey && !event.shiftKey) {
                event.preventDefault();
                event.stopPropagation();
                if (reference === activeReference && pinned) {
                    closePreview();
                } else {
                    openPreview(reference, true);
                    if (event.detail === 0) preview.focus({ preventScroll: true });
                }
                return;
            }
            if (!preview.hidden && !preview.contains(event.target) && reference !== activeReference) {
                closePreview();
            }
        }, true);

        preview.addEventListener("pointerenter", cancelClose);
        preview.addEventListener("pointerleave", scheduleClose);
        preview.addEventListener("focusin", cancelClose);
        preview.addEventListener("focusout", scheduleClose);
        closeButton.addEventListener("click", () => closePreview(true));
        jumpLink.addEventListener("click", () => closePreview());
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape" && !preview.hidden) {
                event.preventDefault();
                closePreview(preview.contains(document.activeElement));
            }
        });
        updatePreviewMode();
        window.addEventListener("resize", () => {
            updatePreviewMode();
            schedulePosition();
        }, { passive: true });
        window.addEventListener("scroll", schedulePosition, { passive: true, capture: true });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initialize, { once: true });
    } else {
        initialize();
    }
})();
