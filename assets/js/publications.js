(() => {
  const page = document.querySelector('.publications-page');
  if (!page) return;

  const filters = page.querySelector('.pub-filters');
  const filterButtons = Array.from(page.querySelectorAll('.pub-filter'));
  const groups = Array.from(page.querySelectorAll('.pub-year-group'));
  const cards = Array.from(page.querySelectorAll('.pub-card'));
  const liveStatus = page.querySelector('.pub-live-status');
  const cardTopics = new Map(cards.map(card => {
    try {
      return [card, JSON.parse(card.dataset.topics || '[]')];
    } catch {
      return [card, []];
    }
  }));

  const announce = message => {
    liveStatus.textContent = message;
  };

  filterButtons.forEach(button => {
    button.addEventListener('click', () => {
      const topic = button.dataset.topic;
      let visibleCount = 0;
      filterButtons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      cards.forEach(card => {
        card.hidden = !!topic && !cardTopics.get(card).includes(topic);
        if (!card.hidden) visibleCount += 1;
      });
      groups.forEach(group => {
        const count = Array.from(group.querySelectorAll('.pub-card')).filter(card => !card.hidden).length;
        group.hidden = count === 0;
        group.querySelector('.pub-year-count').textContent = `${count} ${count === 1 ? 'paper' : 'papers'}`;
      });
      announce(`${visibleCount} ${visibleCount === 1 ? 'paper' : 'papers'} shown${topic ? ` for ${topic}` : ''}.`);
    });
  });
  if (filters) filters.hidden = false;

  const fallbackCopy = source => {
    const previousFocus = document.activeElement;
    source.hidden = false;
    source.focus({ preventScroll: true });
    source.select();
    let copied = false;
    try {
      copied = document.execCommand('copy');
    } catch {
      copied = false;
    }
    source.hidden = copied;
    if (copied && previousFocus instanceof HTMLElement) {
      previousFocus.focus({ preventScroll: true });
    }
    return copied;
  };

  page.querySelectorAll('.pub-copy-bib').forEach(button => {
    button.hidden = false;
    let resetTimer;
    button.addEventListener('click', async () => {
      const source = button.closest('.pub-card').querySelector('.pub-bibtex');
      const label = button.querySelector('.pub-copy-label');
      let copied = false;
      if (navigator.clipboard && window.isSecureContext) {
        try {
          await navigator.clipboard.writeText(source.value.trim());
          copied = true;
        } catch {
          copied = fallbackCopy(source);
        }
      } else {
        copied = fallbackCopy(source);
      }
      clearTimeout(resetTimer);
      label.textContent = copied ? 'Copied!' : 'Select BibTeX';
      announce(copied ? 'BibTeX citation copied to clipboard.' : 'Copy was unavailable. The BibTeX citation is selected below; press Control+C or Command+C to copy.');
      if (copied) {
        source.hidden = true;
        resetTimer = setTimeout(() => { label.textContent = 'Copy Bib'; }, 2400);
      }
    });
  });
})();
