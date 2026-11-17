(() => {
  const menuButton = document.querySelector('.nav-toggle');
  const navigation = document.querySelector('#site-nav');
  if (menuButton && navigation) {
    menuButton.addEventListener('click', () => {
      const open = navigation.classList.toggle('is-open');
      menuButton.setAttribute('aria-expanded', String(open));
    });
    navigation.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        navigation.classList.remove('is-open');
        menuButton.setAttribute('aria-expanded', 'false');
      });
    });
  }

  document.querySelectorAll('[data-storyboard]').forEach((storyboard) => {
    const slides = [...storyboard.querySelectorAll('[data-slide]')];
    const targets = [...storyboard.querySelectorAll('[data-slide-target]')];
    const status = storyboard.querySelector('[data-story-status]');
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let current = 0;
    let timer;

    const show = (index) => {
      current = (index + slides.length) % slides.length;
      slides.forEach((slide, slideIndex) => slide.classList.toggle('is-active', slideIndex === current));
      targets.forEach((target, targetIndex) => target.classList.toggle('is-active', targetIndex === current));
      if (status) status.textContent = `${current + 1} / ${slides.length}`;
    };
    const stop = () => window.clearInterval(timer);
    const start = () => {
      stop();
      if (!reducedMotion) timer = window.setInterval(() => show(current + 1), 7000);
    };

    targets.forEach((target, index) => target.addEventListener('click', () => {
      show(index);
      start();
    }));
    storyboard.querySelector('[data-story-prev]')?.addEventListener('click', () => {
      show(current - 1);
      start();
    });
    storyboard.querySelector('[data-story-next]')?.addEventListener('click', () => {
      show(current + 1);
      start();
    });
    storyboard.addEventListener('mouseenter', stop);
    storyboard.addEventListener('mouseleave', start);
    storyboard.addEventListener('focusin', stop);
    storyboard.addEventListener('focusout', start);
    show(0);
    start();
  });

  document.querySelectorAll('video[data-clip-start]').forEach((video) => {
    const start = Number(video.dataset.clipStart || 0);
    const end = Number(video.dataset.clipEnd || 0);
    video.addEventListener('loadedmetadata', () => {
      if (Number.isFinite(start) && video.currentTime < start) video.currentTime = start;
    });
    video.addEventListener('timeupdate', () => {
      if (end > start && video.currentTime >= end) {
        video.pause();
        video.currentTime = start;
      }
    });
  });
})();
