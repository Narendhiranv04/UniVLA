(function () {
  const gallery = document.querySelector('[data-gallery]');
  const toggle = document.querySelector('[data-toggle]');
  const label = document.querySelector('[data-toggle-label]');

  if (!gallery || !toggle || !label) {
    return;
  }

  const expandedText = 'Collapse gallery';
  const collapsedText = 'Expand gallery';

  const updateState = (isExpanded) => {
    toggle.setAttribute('aria-expanded', String(isExpanded));
    label.textContent = isExpanded ? expandedText : collapsedText;
    gallery.classList.toggle('is-expanded', isExpanded);

    // Sync aria-hidden for secondary frames so screen readers announce them only when visible.
    gallery
      .querySelectorAll('.project__frame--secondary')
      .forEach((frame) => frame.setAttribute('aria-hidden', String(!isExpanded)));
  };

  // Ensure initial state keeps the primary image centered vertically.
  updateState(false);

  toggle.addEventListener('click', () => {
    const isExpanded = toggle.getAttribute('aria-expanded') === 'true';
    updateState(!isExpanded);
  });
})();
