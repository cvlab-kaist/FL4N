(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const V = 'static/videos/';
  const MODELS = [
    ['ours', 'FL4N (Ours)', 'ours'], ['gt', 'Ground truth', 'gt'], ['c4g', 'C4G', ''],
    ['neoverse', 'NeoVerse', ''], ['movies', 'MoVieS', ''], ['4dgt', '4DGT', ''],
  ];

  const progress = $('.progress');
  const navLinks = $$('.nav-links a, .side-links a');
  const sections = $$('.side-links a').map(a => $(a.getAttribute('href'))).filter(Boolean);
  function onScroll() {
    const h = document.documentElement;
    const p = h.scrollTop / (h.scrollHeight - h.clientHeight);
    progress.style.width = (p * 100).toFixed(2) + '%';
    let cur = null;
    for (const s of sections) if (s.getBoundingClientRect().top - 120 <= 0) cur = s;
    navLinks.forEach(a => a.classList.toggle('active', cur && a.getAttribute('href') === '#' + cur.id));
  }
  document.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
  $('.nav-toggle').addEventListener('click', () => $('.nav-links').classList.toggle('open'));
  navLinks.forEach(a => a.addEventListener('click', () => $('.nav-links').classList.remove('open')));

  const io = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
  }), { threshold: 0, rootMargin: '0px 0px -4% 0px' });
  $$('.reveal').forEach(el => io.observe(el));
  function revealSeen() { $$('.reveal:not(.in)').forEach(el => { if (el.getBoundingClientRect().top < window.innerHeight) el.classList.add('in'); }); }
  document.addEventListener('scroll', revealSeen, { passive: true });
  window.addEventListener('load', () => setTimeout(revealSeen, 800));

  const vio = new IntersectionObserver(es => es.forEach(e => {
    const v = e.target;
    if (e.isIntersecting) {
      if (v.dataset.src && !v.src) { v.src = v.dataset.src; v.load(); }
      v.play().catch(() => {});
    } else if (!v.paused) v.pause();
  }), { rootMargin: '200px 0px' });
  function watch(v) { vio.observe(v); }
  $$('video.lazy').forEach(watch);

  const lb = $('.lightbox'), lbBody = $('.lb-body'), lbCap = $('.lb-cap');
  function openLb(src, cap) {
    const srcs = Array.isArray(src) ? src.filter(Boolean) : [src];
    lbBody.innerHTML = srcs.map(() => '<img alt="">').join('');
    $$('img', lbBody).forEach((im, i) => (im.src = srcs[i]));
    lbBody.className = 'lb-body' + (srcs.length === 2 ? ' pair' : srcs.length > 2 ? ' triple' : '');
    lbCap.textContent = cap || ''; lb.classList.add('open'); document.body.style.overflow = 'hidden';
  }
  function closeLb() { lb.classList.remove('open'); lbBody.innerHTML = ''; document.body.style.overflow = ''; }
  window.FL4N_lightbox = openLb;
  $$('.figure img, .car-slide img').forEach(img => img.addEventListener('click', () => {
    const cap = img.closest('figure')?.querySelector('figcaption')?.textContent || img.closest('.car-slide')?.querySelector('.cap')?.textContent || img.alt;
    openLb(img.dataset.full || img.src, cap);
  }));
  lb.addEventListener('click', e => { if (e.target === lb || e.target.classList.contains('lb-close')) closeLb(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLb(); });

  $$('.carousel').forEach(car => {
    const track = $('.car-track', car), slides = $$('.car-slide', car), dots = $('.car-dots', car);
    let i = 0;
    slides.forEach((_, k) => { const b = document.createElement('button'); b.setAttribute('aria-label', 'slide ' + (k + 1)); b.addEventListener('click', () => go(k)); dots.appendChild(b); });
    const vp = $('.car-viewport', car);
    function fit() { vp.style.height = slides[i].offsetHeight + 'px'; }
    function go(k) { i = (k + slides.length) % slides.length; track.style.transform = `translateX(-${i * 100}%)`; $$('button', dots).forEach((b, j) => b.classList.toggle('active', j === i)); fit(); }
    window.addEventListener('resize', fit);
    $$('img', car).forEach(im => im.addEventListener('load', fit));
    $('.car-btn.prev', car).addEventListener('click', () => go(i - 1));
    $('.car-btn.next', car).addEventListener('click', () => go(i + 1));
    go(0);
  });

  function tabs(container, items, onPick, active = 0) {
    container.innerHTML = '';
    items.forEach((it, k) => {
      const b = document.createElement('button');
      b.className = 'tab' + (k === active ? ' active' : ''); b.textContent = it.label;
      b.addEventListener('click', () => { $$('.tab', container).forEach(x => x.classList.remove('active')); b.classList.add('active'); onPick(it, k); });
      container.appendChild(b);
    });
  }
  function chips(container, items, onPick, active = 0) {
    container.innerHTML = '';
    items.forEach((it, k) => {
      const b = document.createElement('button');
      b.className = 'chip' + (k === active ? ' active' : '');
      b.textContent = it.label;
      b.addEventListener('click', () => { $$('.chip', container).forEach(x => x.classList.remove('active')); b.classList.add('active'); onPick(it, k); });
      container.appendChild(b);
    });
  }

  function gallery(root) {
    const data = {};
    ['ctvs', 'nvs', 'nts'].forEach(proto => Object.keys(FL4N_SCENES[proto]).forEach(ds => {
      const d = data[ds] = data[ds] || { name: FL4N_SCENES[proto][ds].name, scenes: [] };
      FL4N_SCENES[proto][ds].scenes.forEach(s => { if (!d.scenes.some(x => x.id === s.id)) d.scenes.push({ ...s, proto }); });
    }));
    const dsItems = Object.keys(data).map(ds => ({ label: data[ds].name, ds }));
    const inputV = $('.cmp-input video', root), cells = $$('.cell', root);
    const inputLbl = $('.cmp-input .lbl span', root), sceneName = $('.gal-scene', root);
    const vids = [inputV, ...cells.map(c => $('video', c))];
    function show(ds, s) {
      const base = `${V}compare/${s.proto}_${ds}_${s.id}__`;
      inputV.dataset.src = base + 'inp.mp4';
      cells.forEach(c => { $('video', c).dataset.src = base + c.dataset.key + '.mp4'; });
      vids.forEach(v => { v.pause(); v.src = v.dataset.src; v.load(); v.oncanplay = () => { v.oncanplay = null; v.currentTime = 0; v.play().catch(() => {}); }; });
      sceneName.textContent = `${data[ds].name} · ${s.name}`;
      inputLbl.textContent = s.proto === 'nts' ? 'moving camera' : 'one frame per camera';
    }
    function pickDs(it) { chips($('.chips', root), data[it.ds].scenes.map(s => ({ label: s.name, s })), c => show(it.ds, c.s)); show(it.ds, data[it.ds].scenes[0]); }
    tabs($('.tabs', root), dsItems, pickDs);
    pickDs(dsItems[0]);
    vids.forEach(watch);
  }
  setInterval(() => {
    $$('.cmp').forEach(root => {
      const vs = $$('.cell video', root).filter(v => v.src && !v.paused && isFinite(v.duration));
      if (vs.length < 2) return;
      const m = vs[0];
      vs.slice(1).forEach(v => { if (Math.abs(v.currentTime - m.currentTime) > 0.1) v.currentTime = m.currentTime; });
    });
  }, 500);
  $$('.gallery').forEach(root => gallery(root));

  $$('.restabs').forEach(rt => {
    const panes = $$('.table-scroll', rt.closest('section'));
    tabs(rt, panes.map(p => ({ label: p.dataset.label, p })), it => panes.forEach(p => (p.style.display = p === it.p ? '' : 'none')));
    panes.forEach((p, i) => (p.style.display = i ? 'none' : ''));
  });

  const copyBtn = $('.copy');
  if (copyBtn) copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText($('.bib pre').textContent).then(() => { copyBtn.textContent = 'Copied'; setTimeout(() => (copyBtn.textContent = 'Copy'), 1500); });
  });
})();
