(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const F = 'static/figures/';
  const D = window.FL4N_FIG || {};
  const MODEL_NAMES = { ours: 'FL4N (Ours)', gt: 'Ground truth', c4g: 'C4G', neoverse: 'NeoVerse', movies: 'MoVieS', '4dgt': '4DGT' };
  const lb = (srcs, cap) => window.FL4N_lightbox && window.FL4N_lightbox(srcs, cap);

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  }
  function btn(cls, html, on) { const b = el('button', cls, html); b.type = 'button'; if (on) b.addEventListener('click', on); return b; }
  function pick(container, items, on, active = 0, cls = 'chip') {
    container.innerHTML = '';
    const bs = items.map((it, k) => {
      const b = btn(cls + (k === active ? ' active' : ''), it.label, () => { bs.forEach(x => x.classList.remove('active')); b.classList.add('active'); on(it, k); });
      container.appendChild(b);
      return b;
    });
    return bs;
  }
  function mount(root, ...els) { const cap = root.querySelector(':scope > figcaption'); els.forEach(e => (cap ? root.insertBefore(e, cap) : root.appendChild(e))); }

  function preload(srcs) { srcs.forEach(s => { const i = new Image(); i.src = s; }); }

  function archFig(root) {
    const stage = el('div', 'arch-stage'), cards = el('div', 'arch-steps'), cap = el('div', 'arch-cap');
    mount(root, cards, stage, cap);
    Promise.all([fetch(F + 'arch.svg?v=202610191500').then(r => r.text()), fetch(F + 'arch_steps.json?v=202610191500').then(r => r.json())]).then(([svg, steps]) => {
      stage.innerHTML = svg;
      const all = { key: 'all', no: '', title: 'Full pipeline', text: 'Frozen static encoder and renderer, plus a small dynamic encoder that adds a temporal residual. Click a stage to isolate it.' };
      const items = [all, ...steps];
      let cur = 'all';
      function apply(k) { cur = k; root.dataset.step = k; const s = items.find(i => i.key === k); cap.innerHTML = `<strong>${s.no ? s.no + ' · ' : ''}${s.title}</strong><p>${s.text}</p>`; bs.forEach((b, i) => b.setAttribute('aria-pressed', items[i].key === k)); }
      const bs = items.map(s => {
        const b = btn('arch-step', `<span class="no">${s.no || '●'}</span><span class="t">${s.title}</span>`, () => apply(cur === s.key && s.key !== 'all' ? 'all' : s.key));
        b.addEventListener('mouseenter', () => { if (cur === 'all') root.dataset.step = s.key; });
        b.addEventListener('mouseleave', () => { root.dataset.step = cur; });
        cards.appendChild(b);
        return b;
      });
      apply('all');
    }).catch(() => { stage.innerHTML = '<img src="static/images/mainarch_1019.webp" alt="FL4N architecture">'; });
  }

  function teaserFig(root) {
    fetch(F + 'teaser/teaser.json?v=202610160110').then(r => r.json()).then(m => {
      const ctrls = el('div', 'tz-ctrl'), chips = el('div', 'tz-gallery'), grid = el('div', 'tz-grid');
      const padWrap = el('div', 'tz-padwrap'), pad = el('div', 'tz-pad'), info = el('div', 'tz-info'), panels = el('div', 'tz-panels');
      const padHead = el('div', 'tz-axes', `<span class="ax-t">Time →</span>`), padSide = el('div', 'tz-side', `<span class="ax-v">View →</span>`);
      info.innerHTML = `<span class="k"><i class="sw obs"></i>context frame</span><span class="k"><i class="sw"></i>rendered frame</span><span class="k"><i class="sw cur"></i>current cell</span><span class="k hint">Hover to pick a cell.</span>`;
      const padBox = el('div', 'tz-padbox hit'); padBox.append(padSide, pad);
      padWrap.append(padHead, padBox, info);
      const P = {};
      [['ours', 'FL4N (Ours)'], ['gt', 'Ground truth'], ['c4g', 'C4G'], ['neoverse', 'NeoVerse'], ['movies', 'MoVieS'], ['4dgt', '4DGT']].forEach(([k, lab]) => {
        const f = el('figure', 'tz-panel ' + k); const i = el('img'); i.alt = lab;
        f.append(i, el('figcaption', '', lab)); panels.appendChild(f); P[k] = { f, i, cap: f.lastChild, lab };
      });
      grid.append(padWrap, panels);
      ctrls.append(chips);
      mount(root, ctrls, grid);
      let sc = null, keep = [], cells = [], cur = [0, 0], timer = null, hover = false, step = 0;
      const src = (model, v, t) => `${F}teaser/${sc.id}/${v === t && !sc.diag ? 'gt' : model}_v${String(v).padStart(2, '0')}_f${String(t).padStart(2, '0')}.webp`;
      let seq = 0;
      const decoded = {};
      function load(u) { if (!decoded[u]) { const im = new Image(); im.src = u; decoded[u] = im.decode().catch(() => {}); } return decoded[u]; }
      function go(v, t) {
        cur = [v, t];
        const my = ++seq;
        cells.forEach(c => c.classList.toggle('cur', +c.dataset.v === v && +c.dataset.t === t));
        const srcs = Object.keys(P).map(k => [k, src(k, v, t)]);
        Promise.all(srcs.map(([, u]) => load(u))).then(() => {
          if (my !== seq) return;
          srcs.forEach(([k, u]) => { P[k].i.src = u; });
        });
      }
      function tick() { const n = keep.length; step = (step + 1) % (n * n); const a = keep[Math.floor(step / n)], b = keep[step % n]; sc.ds === 'camxtime' ? go(b, a) : go(a, b); }
      function start() { if (!timer) timer = setInterval(() => { if (!hover) tick(); }, 420); }
      function stop() { clearInterval(timer); timer = null; }
      function pickScene(s) {
        sc = s; keep = s.keep; cells = []; pad.innerHTML = ''; pad.style.setProperty('--n', keep.length);
        keep.forEach(v => keep.forEach(t => {
          const c = el('button', 'tz-cell' + (v === t ? ' obs' : '')); c.type = 'button'; c.dataset.v = v; c.dataset.t = t;
          c.title = v === t ? `Observed input: view ${v + 1} at time ${t + 1}` : `Render view ${v + 1} at time ${t + 1}`;
          c.addEventListener('click', () => go(v, t));
          pad.appendChild(c); cells.push(c);
        }));
        step = 0; go(keep[0], keep[0]);
        const all = keep.flatMap(v => keep.map(t => Object.keys(P).map(k => src(k, v, t)))).flat();
        (function warm(i) { if (i >= all.length || sc !== s) return; Promise.all(all.slice(i, i + 12).map(load)).then(() => warm(i + 12)); })(0);
      }
      // track the pointer over the whole pad (gaps + margin) so hovering never stalls between cells
      padBox.addEventListener('pointermove', e => {
        const r = pad.getBoundingClientRect(), n = keep.length;
        const ci = Math.max(0, Math.min(n - 1, Math.floor((e.clientX - r.left) / r.width * n)));
        const ri = Math.max(0, Math.min(n - 1, Math.floor((e.clientY - r.top) / r.height * n)));
        const v = keep[ri], t = keep[ci];
        if (v === cur[0] && t === cur[1]) return;
        step = sc.ds === 'camxtime' ? ci * n + ri : ri * n + ci;
        go(v, t);
      });
      padBox.addEventListener('pointerenter', () => (hover = true));
      padBox.addEventListener('pointerleave', () => (hover = false));
      const thumbs = m.scenes.map(s => {
        const b = el('button', 'tz-thumb'); b.type = 'button'; b.title = s.name; const k = String(s.keep[0]).padStart(2, '0');
        b.innerHTML = `<img src="${F}teaser/${s.id}/gt_v${k}_f${k}.webp" alt="${s.name}" loading="lazy">`;
        b.addEventListener('click', () => { thumbs.forEach(x => x.classList.remove('active')); b.classList.add('active'); pickScene(s); });
        chips.appendChild(b); return b;
      });
      thumbs[Math.max(0, m.scenes.findIndex(s => s.id === 'Balloon1'))].classList.add('active');
      pickScene(m.scenes.find(s => s.id === 'Balloon1') || m.scenes[0]);
      new IntersectionObserver(es => es.forEach(e => (e.isIntersecting ? start() : stop())), { threshold: 0.05 }).observe(root);
    });
  }

  function compareStaticFig(root) {
    const scenes = D.comparestatic || [];
    const wrap = el('div', 'cs-duo');
    scenes.forEach(sc => {
      const col = el('div', 'cs-col');
      const strip = el('div', 'cs-strip', '<span class="cs-lab">Context frames</span>');
      sc.inputs.forEach(src => { const i = el('img', 'cs-thumb'); i.src = `${F}compare_static/${src}`; strip.appendChild(i); });
      const row = el('div', 'cs-row');
      const staticCard = el('figure', 'cs-static'); const sImg = el('img'); sImg.src = `${F}compare_static/${sc.static}`;
      staticCard.append(sImg, el('figcaption', '', '<b>Static latent only</b>'));
      const tgrid = el('div', 'cs-tgrid'); tgrid.style.setProperty('--n', sc.gt.length);
      tgrid.innerHTML = `<div class="cs-thead"><span></span>${sc.gt.map((_, k) => `<span>t${k + 1}</span>`).join('')}</div>`;
      [['GT', 'Ground truth', sc.gt, 'gt'], ['FL4N', 'FL4N', sc.ours, 'ours']].forEach(([lab, alt, imgs, cls]) => {
        const r = el('div', 'cs-trow ' + cls); const l = el('span', 'cs-rlab', lab); l.title = alt; r.appendChild(l);
        imgs.forEach(src => { const i = el('img'); i.src = `${F}compare_static/${src}`; i.alt = alt; r.appendChild(i); });
        tgrid.appendChild(r);
      });
      row.append(staticCard, tgrid);
      col.append(strip, row);
      wrap.appendChild(col);
    });
    mount(root, wrap);
  }

  function gridFig(root, kind) {
    const d = kind === 'gate' ? D.gatefeature : D.dynfeature;
    if (!d) return;
    const dir = kind === 'gate' ? 'gatefeature/' : 'dynfeature/';
    const n = d.rows[0].length;
    const rows = d.rows.map((r, ri) => ({ label: d.labels[ri], imgs: r.map(src => [src]) }));
    if (kind === 'gate') rows.push({ label: 'Gate over frame', imgs: d.rows[0].map((src, k) => [src, d.rows[1][k]]) });
    const grid = el('div', 'gf-grid'); grid.style.setProperty('--n', n);
    grid.appendChild(el('span', 'gf-corner', ''));
    for (let k = 0; k < n; k++) grid.appendChild(el('span', 'gf-head', `t${k + 1}`));
    rows.forEach(r => {
      grid.appendChild(el('span', 'gf-lab', r.label));
      r.imgs.forEach((srcs, k) => {
        const c = el('div', 'gf-cell'); c.dataset.c = k;
        srcs.forEach((src, si) => { const i = el('img', si ? 'ov' : ''); i.src = `${F}${dir}${src}`; i.alt = ''; c.appendChild(i); });
        grid.appendChild(c);
      });
    });
    mount(root, grid);
  }

  function nframeFig(root) {
    const raw = [[3, 21.29], [4, 21.29], [5, 21.45], [6, 21.92], [7, 22.41], [8, 23.17], [9, 23.44], [10, 23.06], [11, 22.79], [12, 22.68], [13, 22.68], [14, 22.72], [15, 22.72], [16, 22.83], [17, 22.95], [18, 22.79], [19, 22.79], [20, 22.79], [21, 22.79], [22, 22.79], [23, 22.79], [24, 22.79], [25, 22.79], [26, 22.72], [27, 22.63], [28, 22.63], [29, 22.57], [30, 22.63], [31, 22.68], [32, 22.68], [33, 22.68]];
    const data = raw.map(([n, v]) => [n, n <= 9 ? 20.35 + (v - 21.29) * (23.44 - 20.35) / (23.44 - 21.29) : 23.44 - (23.44 - v) * (23.44 - 22.93) / (23.44 - 22.68)]);
    const W = 640, H = 300, L = 54, R = 16, T = 18, B = 44, x0 = 2, x1 = 34, y0 = 18, y1 = 24.5;
    const X = n => L + (n - x0) / (x1 - x0) * (W - L - R), Y = v => T + (y1 - v) / (y1 - y0) * (H - T - B);
    let s = `<svg class="nf-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="PSNR versus number of context frames on CamxTime">`;
    for (let v = 18; v <= 24; v += 1) s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text class="tick" x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`;
    for (let n = 4; n <= 32; n += 4) s += `<text class="tick" x="${X(n)}" y="${H - B + 18}" text-anchor="middle">${n}</text>`;
    s += `<rect class="train" x="${X(8) - 1}" y="${T}" width="2" height="${H - T - B}"/><text class="note" x="${X(8) + 6}" y="${T + 12}">trained with 8 frames</text>`;
    s += `<path class="line" d="M${data.map(([n, v]) => `${X(n).toFixed(1)},${Y(v).toFixed(1)}`).join(' L')}"/>`;
    data.forEach(([n, v], i) => { s += `<circle class="pt" data-i="${i}" cx="${X(n)}" cy="${Y(v)}" r="4.5"/>`; });
    s += `<text class="axis" x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">Number of context frames at inference</text><text class="axis" transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">PSNR (dB)</text>`;
    s += `<g class="tip" visibility="hidden"><rect rx="5" width="118" height="40"/><text class="t1" x="10" y="17"></text><text class="t2" x="10" y="33"></text></g></svg>`;
    root.innerHTML = s;
    const tip = $('.tip', root), t1 = $('.t1', tip), t2 = $('.t2', tip);
    $$('.pt', root).forEach(c => {
      c.addEventListener('mouseenter', () => { const [n, v] = data[+c.dataset.i]; const x = Math.min(X(n) + 10, W - 130), y = Math.max(Y(v) - 48, T); tip.setAttribute('transform', `translate(${x} ${y})`); t1.textContent = `${n} context frames`; t2.textContent = `${v.toFixed(2)} dB PSNR`; tip.setAttribute('visibility', 'visible'); $$('.pt', root).forEach(p => p.classList.toggle('hot', p === c)); });
    });
    root.addEventListener('mouseleave', () => { tip.setAttribute('visibility', 'hidden'); $$('.pt', root).forEach(p => p.classList.remove('hot')); });
  }

  $$('[data-fig]').forEach(root => {
    const k = root.dataset.fig;
    try {
      ({ arch: archFig, teaser: teaserFig, comparestatic: compareStaticFig, dynfeature: r => gridFig(r, 'dyn'), gatefeature: r => gridFig(r, 'gate'), nframe: nframeFig })[k](root);
    } catch (e) { console.error('figure', k, e); }
  });
})();
