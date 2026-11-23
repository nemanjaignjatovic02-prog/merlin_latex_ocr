(() => {
  const $ = (id) => document.getElementById(id);
  const drop = $('drop'), fileInput = $('file'), view = $('view'), dropEmpty = $('dropEmpty');
  const img = $('modelImg'), canvas = $('overlay'), busy = $('busy');
  const editor = $('editor'), preview = $('preview'), warn = $('warn');
  const tokensWrap = $('tokensWrap'), tokensEl = $('tokens'), tokenDetail = $('tokenDetail');
  const statusEl = $('status'), statsEl = $('stats'), rerun = $('rerun'), resetBtn = $('resetBtn');
  let lastFile = null, result = null, activeToken = null, modelReady = false;

  const fmtPct = (x, d = 1) => (x == null ? '–' : x.toFixed(d) + '%');
  const level = (p) => (p >= 0.9 ? 'sure' : p >= 0.6 ? 'mid' : 'unsure');

  // ---------- model status ----------
  fetch('/api/info').then((r) => r.json()).then((info) => {
    if (!info.ready) {
      statusEl.textContent = 'Model not loaded';
      statusEl.className = 'status err';
      statusEl.title = info.error || '';
      return;
    }
    modelReady = true;
    statusEl.textContent = 'Model ready';
    statusEl.className = 'status ok';
    const m = info.metrics || {};
    const items = [
      ['Exact match, full test set', fmtPct(m.test_em_full)],
      ['Image match', fmtPct(m.test_img_full)],
      ['Compiles', fmtPct(m.test_compile_full)],
      ['BLEU-4', m.test_bleu_full == null ? '–' : m.test_bleu_full.toFixed(1)],
    ];
    if (m.font_img_known != null) items.push(['Other fonts (image match)', fmtPct(m.font_img_known)]);
    if (m.font_img_unseen != null) items.push(['Unseen fonts (image match)', fmtPct(m.font_img_unseen)]);
    $('metrics').innerHTML = items.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
  }).catch(() => {
    statusEl.textContent = 'Server unavailable';
    statusEl.className = 'status err';
  });

  // ---------- image input ----------
  fileInput.addEventListener('change', () => fileInput.files[0] && run(fileInput.files[0]));
  drop.addEventListener('click', (e) => { if (!result && e.target.tagName !== 'LABEL') fileInput.click(); });
  drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('drag'); }));
  drop.addEventListener('drop', (e) => {
    const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith('image/'));
    if (f) run(f);
  });
  document.addEventListener('paste', (e) => {
    if (document.activeElement === editor) return;               // inside the editor Ctrl+V pastes text
    const item = [...(e.clipboardData?.items || [])].find((x) => x.type.startsWith('image/'));
    if (item) { e.preventDefault(); run(item.getAsFile()); }
  });
  document.querySelectorAll('[data-example]').forEach((b) => b.addEventListener('click', async () => {
    const blob = await (await fetch(b.dataset.example)).blob();
    run(new File([blob], 'example.png', { type: 'image/png' }));
  }));
  rerun.addEventListener('click', () => lastFile && run(lastFile));
  document.querySelectorAll('input[name=mode]').forEach((r) => r.addEventListener('change', syncOptions));
  function syncOptions() {
    const beam = document.querySelector('input[name=mode]:checked').value === 'beam';
    $('earlyExit').disabled = beam;
    $('earlyExit').parentElement.title = beam ? 'Early exit works only in fast mode' : '';
  }
  syncOptions();

  async function run(file) {
    if (!modelReady) { showWarn('The model is not loaded on the server yet. Refresh the page in a minute.'); return; }
    lastFile = file;
    busy.hidden = false;
    hideWarn();
    const fd = new FormData();
    fd.append('file', file);
    fd.append('mode', document.querySelector('input[name=mode]:checked').value);
    fd.append('early_exit', $('earlyExit').checked && !$('earlyExit').disabled ? 'true' : 'false');
    try {
      const r = await fetch('/api/predict', { method: 'POST', body: fd });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || 'Reading failed.');
      show(data);
    } catch (err) {
      showWarn(err.message || 'Reading failed. Please try again.');
    } finally {
      busy.hidden = true;
    }
  }

  // ---------- result display ----------
  function show(data) {
    result = data;
    activeToken = null;
    dropEmpty.hidden = true;
    view.hidden = false;
    drop.classList.add('has-image');
    rerun.disabled = false;
    resetBtn.disabled = false;
    img.onload = () => {
      img.style.width = Math.min($('stage').clientWidth, img.naturalWidth * 1.8) + 'px';
      drawOverlay();
    };
    img.src = data.model_input;
    editor.value = data.latex;
    renderPreview();
    renderTokens();
    renderStats();
    const low = data.tokens.filter((t) => t.p < 0.6).length;
    if (low) showWarn(`The model is unsure about ${low} ${low === 1 ? 'token' : 'tokens'} (red). Check them in the preview.`);
    if (!data.tokens.length) showWarn('No formula was recognised. Crop just the formula, without surrounding text.');
  }

  function renderTokens() {
    tokensWrap.hidden = false;
    tokensEl.innerHTML = '';
    result.tokens.forEach((t, i) => {
      const b = document.createElement('button');
      b.className = `tok ${level(t.p)}` + (i > 0 && !result.tokens[i - 1].eow ? ' join' : '');
      b.textContent = t.t;
      b.setAttribute('role', 'listitem');
      b.setAttribute('aria-label', `${t.t}, confidence ${Math.round(t.p * 100)} percent`);
      const pick = () => selectToken(i);
      b.addEventListener('mouseenter', pick);
      b.addEventListener('focus', pick);
      b.addEventListener('click', pick);
      b.addEventListener('mouseleave', () => { if (document.activeElement !== b) selectToken(null); });
      b.addEventListener('blur', () => selectToken(null));
      tokensEl.appendChild(b);
    });
    selectToken(null);
  }

  function selectToken(i) {
    activeToken = i;
    [...tokensEl.children].forEach((c, k) => c.classList.toggle('active', k === i));
    if (i == null) {
      tokenDetail.textContent = 'All tokens: total attention is shown (where the model looked across the whole formula).';
    } else {
      const t = result.tokens[i];
      tokenDetail.textContent = `Token ${i + 1} of ${result.tokens.length}: “${t.t}”, confidence ${Math.round(t.p * 100)}%, ` +
        (result.early_exit ? `exited at layer ${t.layer} of 4.` : `early exit would happen at layer ${t.layer} of 4.`);
    }
    drawOverlay();
  }

  function drawOverlay() {
    if (!result || !img.naturalWidth) return;
    const { rows, cols, stride, maps } = result.attention;
    const w = img.clientWidth, h = img.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const sx = w / (cols * stride), sy = h / (rows * stride);
    let map;
    if (activeToken != null) map = maps[activeToken];
    else if (maps.length) {                                         // total attention = sum of all maps
      map = maps[0].map((r) => r.map(() => 0));
      maps.forEach((m) => m.forEach((r, y) => r.forEach((v, x) => { map[y][x] += v; })));
      const mx = Math.max(...map.flat(), 1e-9);
      map = map.map((r) => r.map((v) => v / mx));
    }
    if (map) {
      for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
        const v = map[y][x];
        if (v < 0.04) continue;
        const a = activeToken == null ? 0.06 + 0.4 * v : 0.12 + 0.6 * v;
        ctx.fillStyle = `rgba(232, 161, 58, ${a.toFixed(3)})`;
        ctx.fillRect(x * stride * sx, y * stride * sy, stride * sx, stride * sy);
      }
    }
    ctx.strokeStyle = 'rgba(31, 78, 140, 0.18)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x <= cols; x++) { const px = Math.round(x * stride * sx) + 0.5; ctx.moveTo(px, 0); ctx.lineTo(px, h); }
    for (let y = 0; y <= rows; y++) { const py = Math.round(y * stride * sy) + 0.5; ctx.moveTo(0, py); ctx.lineTo(w, py); }
    ctx.stroke();
  }
  window.addEventListener('resize', drawOverlay);

  function renderStats() {
    const t = result.tokens;
    const avgP = t.length ? t.reduce((s, x) => s + x.p, 0) / t.length : 0;
    const items = [
      ['Reading time', `${result.ms} ms`],
      ['Tokens', `${t.length}`],
      ['Mean confidence', fmtPct(avgP * 100, 0)],
      ['Layers per token', result.avg_layers ? result.avg_layers.toFixed(2) + ' of 4' : '4 of 4'],
    ];
    statsEl.innerHTML = items.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
    statsEl.hidden = false;
  }

  // ---------- editor and preview ----------
  function renderPreview() {
    const src = editor.value.trim();
    if (!src) { preview.innerHTML = '<span class="muted">The rendered formula will appear here.</span>'; return; }
    if (!window.katex) { preview.textContent = src; return; }
    try {
      katex.render(src, preview, { displayMode: true, throwOnError: true, strict: 'ignore' });
    } catch (e) {
      preview.innerHTML = '';
      const msg = document.createElement('span');
      msg.className = 'katex-error';
      msg.textContent = 'Cannot render: ' + (e.message || '').replace(/^KaTeX parse error: /, '');
      preview.appendChild(msg);
    }
  }
  let t;
  editor.addEventListener('input', () => { clearTimeout(t); t = setTimeout(renderPreview, 120); });
  window.addEventListener('load', renderPreview);
  resetBtn.addEventListener('click', () => { if (result) { editor.value = result.latex; renderPreview(); } });

  document.querySelectorAll('[data-copy]').forEach((b) => b.addEventListener('click', async () => {
    const s = editor.value.trim();
    if (!s) return;
    const text = b.dataset.copy === 'inline' ? `$${s}$` : b.dataset.copy === 'display' ? `\\[\n${s}\n\\]` : s;
    try { await navigator.clipboard.writeText(text); flash('Copied.'); }
    catch { editor.select(); flash('Press Ctrl+C to copy.'); }
  }));
  function flash(msg) { const c = $('copied'); c.textContent = msg; setTimeout(() => { c.textContent = ''; }, 1800); }
  function showWarn(msg) { warn.textContent = msg; warn.hidden = false; }
  function hideWarn() { warn.hidden = true; }
})();
