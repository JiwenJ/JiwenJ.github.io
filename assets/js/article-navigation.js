(() => {
    "use strict";

    function initializeArticle(article) {
        const headings = Array.from(article.querySelectorAll(".post-content h2[id], .post-content h3[id]"));
        const tocLinks = Array.from(article.querySelectorAll("aside.article-toc a[href^='#'], .toc a[href^='#']"));
        const content = article.querySelector(".post-content");
        const rail = article.querySelector(".reference-sidebar");
        const notes = [];
        const anchorNotes = new Map();
        const noteAnchors = new Map();
        let currentHeading = null;
        let headingFrame = null;
        let noteFrame = null;
        let highlightedNote = null;
        let hoveredNote = null;
        let focusedNote = null;
        let clickedNote = null;
        let highlightTimeout = null;

        function fragmentId(link) {
            try {
                return decodeURIComponent(link.getAttribute("href").slice(1));
            } catch {
                return "";
            }
        }

        const linkedHeadings = tocLinks.map((link) => ({ link, id: fragmentId(link) }));

        function updateCurrentHeading() {
            headingFrame = null;
            if (!headings.length || !tocLinks.length) return;
            const readingLine = Math.min(160, Math.max(80, window.innerHeight * 0.22));
            let active = headings[0];
            for (const heading of headings) {
                if (heading.getBoundingClientRect().top > readingLine) break;
                active = heading;
            }
            if (window.scrollY > 0 && window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
                active = headings[headings.length - 1];
            }
            if (currentHeading === active.id) return;
            currentHeading = active.id;
            for (const { link, id } of linkedHeadings) {
                if (id === currentHeading) link.setAttribute("aria-current", "location");
                else link.removeAttribute("aria-current");
            }
        }

        function scheduleHeadingUpdate() {
            if (headingFrame === null) headingFrame = window.requestAnimationFrame(updateCurrentHeading);
        }

        function removeDuplicateIds(element) {
            element.removeAttribute("id");
            element.querySelectorAll("[id]").forEach((node) => node.removeAttribute("id"));
        }

        if (rail && content) {
            const list = rail.querySelector("ol");
            const references = new Map(Array.from(list.children, (entry) => [entry.dataset.reference, entry]));
            const noteElements = [];
            // References follow citation order; explanatory footnotes stay at the article's end.
            content.querySelectorAll("a.citation[data-reference]").forEach((anchor) => {
                const reference = references.get(anchor.dataset.reference);
                if (!reference) return;
                const note = reference.cloneNode(true);
                removeDuplicateIds(note);
                note.id = `sidenote-${notes.length + 1}`;
                note.classList.add("article-sidenote");
                note.tabIndex = 0;
                anchor.dataset.sidenoteTarget = note.id;
                notes.push({ anchor, note });
                anchorNotes.set(anchor, note);
                noteAnchors.set(note, anchor);
                noteElements.push(note);
            });
            if (notes.length) {
                list.replaceChildren(...noteElements);
                article.classList.add("has-sidenotes");
            }
        }

        function layoutSidenotes() {
            noteFrame = null;
            if (!rail || !notes.length) return;
            if (!rail.getClientRects().length) {
                rail.style.removeProperty("min-height");
                hoveredNote = null;
                focusedNote = null;
                clickedNote = null;
                window.clearTimeout(highlightTimeout);
                updateHighlight();
                return;
            }
            const railTop = rail.getBoundingClientRect().top;
            const measurements = notes.map(({ anchor, note }) => ({
                note,
                visible: anchor.getClientRects().length > 0,
                anchorTop: anchor.getBoundingClientRect().top - railTop - 6,
                height: note.getBoundingClientRect().height
            }));
            let bottom = 0;
            for (const { note, visible, anchorTop, height } of measurements) {
                note.hidden = !visible;
                if (!visible) continue;
                const top = Math.max(0, anchorTop, bottom);
                note.style.top = `${Math.round(top)}px`;
                bottom = top + height + 18;
            }
            // Keep any final margin note within the article, above the site footer.
            rail.style.minHeight = `${Math.ceil(bottom)}px`;
        }

        function scheduleSidenotes() {
            if (noteFrame === null) noteFrame = window.requestAnimationFrame(layoutSidenotes);
        }

        function noteFor(target) {
            if (!(target instanceof Element)) return null;
            const anchor = target.closest("a[data-sidenote-target]");
            const note = anchorNotes.get(anchor) || target.closest(".article-sidenote");
            if (!noteAnchors.has(note)) return null;
            return note && note.getClientRects().length ? note : null;
        }

        function updateHighlight() {
            const next = hoveredNote || focusedNote || clickedNote;
            if (highlightedNote === next) return;
            if (highlightedNote) {
                highlightedNote.classList.remove("is-highlighted");
                noteAnchors.get(highlightedNote).classList.remove("is-reference-highlighted");
            }
            highlightedNote = next;
            if (next) {
                next.classList.add("is-highlighted");
                noteAnchors.get(next).classList.add("is-reference-highlighted");
            }
        }

        // Highlight the nearby note without scrolling either the article or a separate pane.
        article.addEventListener("click", (event) => {
            if (event.defaultPrevented || event.button !== 0 || event.metaKey ||
                event.ctrlKey || event.altKey || event.shiftKey) return;
            if (!(event.target instanceof Element) || !event.target.closest("a[data-sidenote-target]")) return;
            const note = noteFor(event.target);
            if (!note) return;
            event.preventDefault();
            event.stopPropagation();
            window.clearTimeout(highlightTimeout);
            clickedNote = note;
            updateHighlight();
            highlightTimeout = window.setTimeout(() => {
                clickedNote = null;
                updateHighlight();
            }, 2200);
        }, true);
        article.addEventListener("pointerover", (event) => {
            hoveredNote = noteFor(event.target);
            updateHighlight();
        });
        article.addEventListener("pointerout", (event) => {
            // Descendant links and text belong to the same note; keep their pair lit.
            hoveredNote = noteFor(event.relatedTarget);
            if (!hoveredNote) clickedNote = null;
            updateHighlight();
        });
        article.addEventListener("focusin", (event) => {
            focusedNote = event.target instanceof Element && event.target.matches(":focus-visible")
                ? noteFor(event.target) : null;
            updateHighlight();
        });
        article.addEventListener("focusout", (event) => {
            focusedNote = event.relatedTarget instanceof Element && event.relatedTarget.matches(":focus-visible")
                ? noteFor(event.relatedTarget) : null;
            clickedNote = null;
            updateHighlight();
        });

        window.addEventListener("scroll", scheduleHeadingUpdate, { passive: true });
        window.addEventListener("hashchange", scheduleHeadingUpdate);
        window.addEventListener("load", () => {
            scheduleHeadingUpdate();
            scheduleSidenotes();
        }, { once: true });
        window.addEventListener("resize", () => {
            scheduleHeadingUpdate();
            scheduleSidenotes();
        }, { passive: true });
        content?.addEventListener("toggle", scheduleSidenotes, true);

        if ("ResizeObserver" in window && content) {
            const observer = new ResizeObserver(scheduleSidenotes);
            observer.observe(content);
            notes.forEach(({ note }) => observer.observe(note));
        }
        if (document.fonts) document.fonts.ready.then(() => {
            scheduleHeadingUpdate();
            scheduleSidenotes();
        });
        if (window.MathJax?.startup?.promise) window.MathJax.startup.promise.then(scheduleSidenotes);
        scheduleHeadingUpdate();
        scheduleSidenotes();
    }

    function initialize() {
        document.querySelectorAll(".article-with-references").forEach(initializeArticle);
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize, { once: true });
    else initialize();
})();
