// Smart Tile AI — editor frontend (vanilla JS, no build step).
// Served by FastAPI at "/"; falls back to the local API when opened from disk.
const API = location.protocol.startsWith('http') ? '' : 'http://localhost:8000';
const MAX_SIDE = 1600;   // must match MAX_IMAGE_SIDE in app/config.py

const DEMO_ROOMS = [
  { name: 'Living room', src: '/static/demo/living-room.jpg' },
  { name: 'Sunlit lounge', src: '/static/demo/sunlit-lounge.jpg' },
  { name: 'Open-plan living', src: '/static/demo/open-plan.jpg' },
  { name: 'Bathroom', src: '/static/demo/bathroom.jpg' },
];
const SIZE_PRESETS = [[300, 300], [300, 600], [600, 600], [600, 1200], [800, 800], [800, 1600], [1200, 1200]];
const RENDER_STEPS = [
  'Estimating room depth',
  'Detecting structural lines',
  'Solving vanishing geometry',
  'Warping the tile mesh',
  'Relighting with room shadows',
  'Compositing final image',
];

const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

const state = {
  roomBlob: null, roomImg: null, roomName: '',
  pos: [], neg: [], order: [],          // order: stack of 'pos' | 'neg' for undo
  maskBlob: null, overlayImg: null, maskSeq: 0, maskTimer: null,
  tile: null,                            // { name, blob, url }
  surface: 'auto',
  autoExclude: true, excludedSummary: {},
  resultImg: null, resultUrl: null, view: 'edit', split: 0.5, dragging: false,
  fit: { x: 0, y: 0, s: 1 },
};

// ---------------------------------------------------------------- utilities
function toast(msg, ms = 4200) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add('hidden'), ms);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not load image'));
    img.src = src;
  });
}

// Downscale once on the client; the same blob is reused for every request so
// click coordinates and the returned mask always line up with the server image.
async function normaliseRoom(src) {
  const img = await loadImage(src);
  const k = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const c = document.createElement('canvas');
  c.width = Math.round(img.naturalWidth * k);
  c.height = Math.round(img.naturalHeight * k);
  c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
  const blob = await new Promise((r) => c.toBlob(r, 'image/jpeg', 0.93));
  return { blob, img: await loadImage(URL.createObjectURL(blob)) };
}

async function apiError(res) {
  try { const j = await res.json(); return j.detail || j.message || res.statusText; } catch { return res.statusText; }
}

// ---------------------------------------------------------------- screens
function showEditor() {
  $('screen-landing').classList.add('hidden');
  $('screen-editor').classList.remove('hidden');
  requestAnimationFrame(draw);
}

function showLanding() {
  $('screen-editor').classList.add('hidden');
  $('screen-landing').classList.remove('hidden');
}

async function openRoom(src, name) {
  try {
    const { blob, img } = await normaliseRoom(src);
    state.roomBlob = blob;
    state.roomImg = img;
    state.roomName = name;
    $('room-name').textContent = `${name} · ${img.naturalWidth}×${img.naturalHeight}`;
    resetSelection();
    clearResult();
    $('search-results').innerHTML = '';
    showEditor();
  } catch (e) {
    toast('That file could not be opened as an image.');
  }
}

// ---------------------------------------------------------------- landing
function buildLanding() {
  const grid = $('demo-grid');
  DEMO_ROOMS.forEach((room) => {
    const card = document.createElement('button');
    card.className = 'group relative aspect-[4/3] overflow-hidden rounded-2xl bg-stone-200 shadow-sm ring-1 ring-black/5 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500';
    card.innerHTML = `
      <img src="${room.src}" alt="${room.name}" loading="lazy" class="absolute inset-0 h-full w-full object-cover transition-transform duration-700 group-hover:scale-[1.04]">
      <div class="absolute inset-0 bg-gradient-to-t from-black/60 via-black/0 to-black/0"></div>
      <div class="absolute bottom-0 inset-x-0 p-4 flex items-end justify-between">
        <span class="text-white font-medium drop-shadow">${room.name}</span>
        <span class="text-xs text-white/90 bg-white/15 backdrop-blur px-2.5 py-1 rounded-full opacity-0 translate-y-1 group-hover:opacity-100 group-hover:translate-y-0 transition">Try it →</span>
      </div>`;
    card.addEventListener('click', () => openRoom(room.src, room.name));
    grid.appendChild(card);
  });

  const dz = $('dropzone');
  const input = $('room-input');
  input.addEventListener('change', () => {
    const f = input.files[0];
    if (f) openRoom(URL.createObjectURL(f), f.name);
    input.value = '';
  });
  ['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', (e) => {
    const f = [...e.dataTransfer.files].find((x) => x.type.startsWith('image/'));
    if (f) openRoom(URL.createObjectURL(f), f.name);
    else toast('Drop an image file (JPG, PNG or WebP).');
  });
}

// ---------------------------------------------------------------- canvas
function fitCanvas() {
  const stage = $('stage');
  const img = state.roomImg;
  if (!img) return;
  const pad = window.innerWidth < 640 ? 32 : 64;
  const maxW = stage.clientWidth - pad;
  const maxH = stage.clientHeight - pad;
  const s = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight);
  const cssW = Math.max(1, Math.floor(img.naturalWidth * s));
  const cssH = Math.max(1, Math.floor(img.naturalHeight * s));
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  state.fit = { s: canvas.width / img.naturalWidth, dpr };
}

function draw() {
  if (!state.roomImg) return;
  fitCanvas();
  const { s, dpr } = state.fit;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  if (state.view === 'result' && state.resultImg) {
    const sx = Math.round(W * state.split);
    ctx.drawImage(state.roomImg, 0, 0, W, H);
    ctx.save();
    ctx.beginPath();
    ctx.rect(sx, 0, W - sx, H);
    ctx.clip();
    ctx.drawImage(state.resultImg, 0, 0, W, H);
    ctx.restore();
    // divider + handle
    ctx.fillStyle = '#fff';
    ctx.fillRect(sx - dpr, 0, 2 * dpr, H);
    ctx.beginPath();
    ctx.arc(sx, H / 2, 16 * dpr, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#16191c';
    ctx.font = `${12 * dpr}px Inter, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('◀ ▶', sx, H / 2 + dpr * 0.5);
    label('Before', 12 * dpr, 'left');
    label('After', W - 12 * dpr, 'right');
    return;
  }

  ctx.drawImage(state.overlayImg || state.roomImg, 0, 0, W, H);
  const r = 6.5 * dpr;
  const dot = (p, fill) => {
    ctx.beginPath();
    ctx.arc(p[0] * s, p[1] * s, r, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.lineWidth = 2.5 * dpr;
    ctx.strokeStyle = '#fff';
    ctx.stroke();
  };
  state.pos.forEach((p) => dot(p, '#10b981'));
  state.neg.forEach((p) => dot(p, '#f43f5e'));
}

function label(text, x, align) {
  const { dpr } = state.fit;
  ctx.font = `600 ${11 * dpr}px Inter, sans-serif`;
  ctx.textAlign = align;
  ctx.textBaseline = 'top';
  const w = ctx.measureText(text).width + 16 * dpr;
  const bx = align === 'left' ? x : x - w;
  ctx.fillStyle = 'rgba(15,17,19,.7)';
  ctx.beginPath();
  ctx.roundRect(bx, 12 * dpr, w, 22 * dpr, 11 * dpr);
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.fillText(text, align === 'left' ? x + 8 * dpr : x - 8 * dpr, 17.5 * dpr);
}

function eventToImage(e) {
  const rect = canvas.getBoundingClientRect();
  const img = state.roomImg;
  const x = ((e.clientX - rect.left) / rect.width) * img.naturalWidth;
  const y = ((e.clientY - rect.top) / rect.height) * img.naturalHeight;
  return [Math.max(0, Math.min(img.naturalWidth - 1, x)), Math.max(0, Math.min(img.naturalHeight - 1, y))];
}

function bindCanvas() {
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    if (!state.roomImg) return;
    if (state.view === 'result') {
      state.dragging = true;
      canvas.setPointerCapture(e.pointerId);
      moveSplit(e);
      return;
    }
    if (e.button !== 0 && e.button !== 2) return;
    const p = eventToImage(e);
    if (e.button === 2) { state.neg.push(p); state.order.push('neg'); }
    else { state.pos.push(p); state.order.push('pos'); }
    draw();
    scheduleMask();
  });
  canvas.addEventListener('pointermove', (e) => { if (state.dragging) moveSplit(e); });
  canvas.addEventListener('pointerup', () => { state.dragging = false; });
  window.addEventListener('resize', draw);
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && state.view === 'edit') { e.preventDefault(); undo(); }
  });
}

function moveSplit(e) {
  const rect = canvas.getBoundingClientRect();
  state.split = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  draw();
}

// ---------------------------------------------------------------- masking
function resetSelection() {
  state.pos = []; state.neg = []; state.order = [];
  state.maskBlob = null; state.overlayImg = null; state.maskSeq++;
  state.excludedSummary = {};
  updateMaskStatus();
}

function undo() {
  const last = state.order.pop();
  if (!last) return;
  state[last].pop();
  draw();
  scheduleMask();
}

function scheduleMask() {
  clearTimeout(state.maskTimer);
  updateMaskStatus();
  if (!state.pos.length) {
    state.maskBlob = null; state.overlayImg = null; state.maskSeq++;
    updateMaskStatus();
    draw();
    return;
  }
  state.maskTimer = setTimeout(requestMask, 140);
}

async function requestMask() {
  const seq = ++state.maskSeq;
  $('seg-busy').classList.replace('hidden', 'flex');
  const fd = new FormData();
  fd.append('room_image', state.roomBlob, 'room.jpg');
  fd.append('positive_clicks', JSON.stringify(state.pos));
  fd.append('negative_clicks', JSON.stringify(state.neg));
  fd.append('auto_exclude', state.autoExclude);
  try {
    const res = await fetch(`${API}/api/get-mask`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await apiError(res));
    const data = await res.json();
    if (seq !== state.maskSeq) return;               // a newer click superseded this one
    const [overlay, maskBlob] = await Promise.all([loadImage(data.overlay), fetch(data.mask).then((r) => r.blob())]);
    if (seq !== state.maskSeq) return;
    state.overlayImg = overlay;
    state.maskBlob = maskBlob;
    state.coverage = data.coverage;
    state.excludedSummary = data.auto_excluded || {};
    draw();
  } catch (e) {
    if (seq === state.maskSeq) toast(`Segmentation failed: ${e.message}`);
  } finally {
    if (seq === state.maskSeq) $('seg-busy').classList.replace('flex', 'hidden');
    updateMaskStatus();
  }
}

function describeExcluded() {
  const el = $('auto-excluded');
  if (!state.autoExclude) { el.textContent = 'Off: only your right-clicks are excluded'; return; }
  const parts = Object.entries(state.excludedSummary || {}).map(([label, n]) => `${n} ${label}${n > 1 ? 's' : ''}`);
  el.textContent = parts.length ? `Cut out: ${parts.join(' · ')}` : 'Detected objects are outlined in rose';
}

function updateMaskStatus() {
  describeExcluded();
  const n = state.pos.length + state.neg.length;
  $('btn-undo').disabled = !n;
  $('btn-clear').disabled = !n;
  const status = $('mask-status');
  if (state.maskBlob) {
    status.textContent = `${Math.round((state.coverage || 0) * 100)}% of photo · ${state.pos.length}+ / ${state.neg.length}−`;
    status.className = 'text-xs text-accent-700 font-medium';
  } else {
    status.textContent = n ? 'Segmenting…' : 'No selection';
    status.className = 'text-xs text-stone-400';
  }
  const ready = Boolean(state.maskBlob && state.tile);
  $('btn-render').disabled = !ready;
  $('render-hint').textContent = ready
    ? 'Depth, lighting and reflections are computed locally'
    : !state.maskBlob ? 'Select a surface to enable rendering' : 'Choose a tile to enable rendering';
}

// ---------------------------------------------------------------- tiles
async function selectTile(tile, card) {
  document.querySelectorAll('.tile-card').forEach((c) => c.setAttribute('aria-selected', 'false'));
  if (card) card.setAttribute('aria-selected', 'true');
  try {
    const blob = tile.blob || (await fetch(tile.url).then((r) => { if (!r.ok) throw new Error(); return r.blob(); }));
    state.tile = { ...tile, blob };
    $('tile-name').textContent = tile.name;
  } catch {
    toast(`Could not load tile "${tile.name}".`);
  }
  updateMaskStatus();
}

function addTileCard(tile, prepend = false) {
  const grid = $('tile-grid');
  const card = document.createElement('button');
  card.className = 'tile-card aspect-square rounded-lg overflow-hidden bg-stone-100 ring-1 ring-black/5 hover:ring-stone-400 transition';
  card.title = tile.name;
  card.setAttribute('aria-selected', 'false');
  card.innerHTML = `<img src="${tile.url}" alt="${tile.name}" loading="lazy" class="h-full w-full object-cover">`;
  card.addEventListener('click', () => selectTile(tile, card));
  prepend ? grid.prepend(card) : grid.appendChild(card);
  return card;
}

async function loadCatalog() {
  try {
    const res = await fetch(`${API}/api/catalog`);
    const tiles = await res.json();
    if (!tiles.length) {
      $('tile-grid').innerHTML = '<p class="col-span-4 text-sm text-stone-500">No tiles in <code>catalog_tiles/</code> yet. Upload one above.</p>';
      return;
    }
    tiles.forEach((t) => addTileCard({ ...t, url: `${API}${t.url}` }));
    const first = $('tile-grid').querySelector('.tile-card');
    selectTile({ ...tiles[0], url: `${API}${tiles[0].url}` }, first);
  } catch {
    toast('Could not reach the API. Is uvicorn running on port 8000?');
  }
}

function bindControls() {
  $('tile-input').addEventListener('change', (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const tile = { name: f.name.replace(/\.[^.]+$/, ''), blob: f, url: URL.createObjectURL(f) };
    selectTile(tile, addTileCard(tile, true));
    e.target.value = '';
  });

  const presets = $('size-presets');
  SIZE_PRESETS.forEach(([w, h]) => {
    const b = document.createElement('button');
    b.className = 'px-2.5 py-1 rounded-full border border-stone-200 text-stone-600 hover:border-accent-500 hover:text-accent-700';
    b.textContent = `${w}×${h}`;
    b.addEventListener('click', () => { $('tile-w').value = w; $('tile-h').value = h; });
    presets.appendChild(b);
  });
  $('btn-swap').addEventListener('click', () => {
    const w = $('tile-w').value;
    $('tile-w').value = $('tile-h').value;
    $('tile-h').value = w;
  });
  $('scale').addEventListener('input', (e) => { $('scale-val').textContent = `${Number(e.target.value).toFixed(2)}×`; });

  $('auto-exclude').addEventListener('change', (e) => {
    state.autoExclude = e.target.checked;
    if (state.pos.length) scheduleMask(); else updateMaskStatus();
  });

  document.querySelectorAll('#surface-seg button').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('#surface-seg button').forEach((x) => x.setAttribute('aria-pressed', 'false'));
    b.setAttribute('aria-pressed', 'true');
    state.surface = b.dataset.v;
  }));

  $('btn-undo').addEventListener('click', undo);
  $('btn-clear').addEventListener('click', () => { resetSelection(); draw(); });
  $('btn-view').addEventListener('click', () => setView(state.view === 'result' ? 'edit' : 'result'));
  $('btn-render').addEventListener('click', render);
  $('btn-home').addEventListener('click', showLanding);
  $('btn-new-room').addEventListener('click', showLanding);
  $('btn-keep').addEventListener('click', keepResult);
  $('btn-search').addEventListener('click', searchProducts);
}

// ---------------------------------------------------------------- render
function setView(view) {
  state.view = view;
  const result = view === 'result';
  $('hint-edit').classList.toggle('hidden', result);
  $('hint-result').classList.toggle('hidden', !result);
  $('btn-undo').classList.toggle('hidden', result);
  $('btn-clear').classList.toggle('hidden', result);
  $('btn-view').classList.toggle('hidden', !state.resultImg);
  $('btn-view').textContent = result ? 'Edit selection' : 'Show result';
  canvas.style.cursor = result ? 'ew-resize' : 'crosshair';
  draw();
}

function clearResult() {
  if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
  state.resultImg = null; state.resultUrl = null;
  ['btn-keep', 'btn-download', 'render-info'].forEach((id) => $(id).classList.add('hidden'));
  setView('edit');
}

function startLoading() {
  const steps = $('loading-steps');
  steps.innerHTML = RENDER_STEPS.map((s) => `<li class="flex items-center gap-2"><span class="h-1.5 w-1.5 rounded-full bg-stone-300"></span>${s}</li>`).join('');
  $('loading').classList.remove('hidden');
  let i = 0;
  const tick = () => {
    const items = steps.children;
    for (let k = 0; k < items.length; k++) {
      items[k].className = `flex items-center gap-2 ${k < i ? 'text-stone-600' : k === i ? 'text-ink-900 font-medium' : 'text-stone-400'}`;
      items[k].firstChild.className = `h-1.5 w-1.5 rounded-full ${k < i ? 'bg-accent-500' : k === i ? 'bg-accent-500 animate-pulse' : 'bg-stone-300'}`;
    }
    $('loading-stage').textContent = `${RENDER_STEPS[Math.min(i, RENDER_STEPS.length - 1)]}…`;
    $('loading-bar').style.width = `${Math.min(92, ((i + 1) / RENDER_STEPS.length) * 92)}%`;
    if (i < RENDER_STEPS.length - 1) i++;
  };
  tick();
  return setInterval(tick, 1100);
}

function stopLoading(timer) {
  clearInterval(timer);
  $('loading-bar').style.width = '100%';
  setTimeout(() => { $('loading').classList.add('hidden'); $('loading-bar').style.width = '0'; }, 250);
}

async function render() {
  if (!state.maskBlob || !state.tile) return;
  const w = Number($('tile-w').value);
  const h = Number($('tile-h').value);
  if (!(w >= 10 && h >= 10)) { toast('Enter the tile size in millimetres (at least 10).'); return; }

  const fd = new FormData();
  fd.append('room_image', state.roomBlob, 'room.jpg');
  fd.append('tile_image', state.tile.blob, 'tile');
  fd.append('mask_image', state.maskBlob, 'mask.png');
  fd.append('tile_width', w);
  fd.append('tile_height', h);
  fd.append('scale', $('scale').value);
  fd.append('is_glossy', $('glossy').checked);
  fd.append('surface_type', state.surface);

  $('btn-render').disabled = true;
  const timer = startLoading();
  try {
    const res = await fetch(`${API}/api/apply-tile`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await apiError(res));
    const blob = await res.blob();
    clearResult();
    state.resultUrl = URL.createObjectURL(blob);
    state.resultImg = await loadImage(state.resultUrl);
    state.split = 0.5;
    showRenderInfo(res.headers.get('X-Render-Info'));
    $('btn-download').href = state.resultUrl;
    ['btn-keep', 'btn-download'].forEach((id) => $(id).classList.remove('hidden'));
    setView('result');
  } catch (e) {
    toast(`Render failed: ${e.message}`);
  } finally {
    stopLoading(timer);
    updateMaskStatus();
  }
}

function showRenderInfo(raw) {
  const el = $('render-info');
  try {
    const info = JSON.parse(raw);
    el.innerHTML = `<span class="text-white font-medium capitalize">${info.surface}</span> · perspective from ${info.horizon_source}` +
      (info.grid_rotation_deg ? ` · grid ${info.grid_rotation_deg}°` : '');
    el.classList.remove('hidden');
  } catch { el.classList.add('hidden'); }
}

// Use the render as the new base photo, e.g. to tile the walls after the floor.
async function keepResult() {
  if (!state.resultImg) return;
  const blob = await fetch(state.resultUrl).then((r) => r.blob());
  state.roomBlob = blob;
  state.roomImg = await loadImage(URL.createObjectURL(blob));
  resetSelection();
  clearResult();
  toast('Render kept. Select the next surface.');
}

// ---------------------------------------------------------------- product search (/search-tiles/)
async function searchProducts() {
  if (!state.roomBlob) return;
  const out = $('search-results');
  out.innerHTML = '<div class="h-16 rounded-xl bg-stone-100 shimmer"></div>';
  const fd = new FormData();
  fd.append('query_image', state.roomBlob, 'room.jpg');
  try {
    const res = await fetch(`${API}/search-tiles/`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await apiError(res));
    const data = await res.json();
    const groups = [['Wall matches', data.wall_matches], ['Floor matches', data.floor_matches], ['Overall matches', data.overall_matches]]
      .filter(([, list]) => list && list.length);
    if (!groups.length) { out.innerHTML = '<p class="text-sm text-stone-500">No close matches in the catalog.</p>'; return; }
    out.innerHTML = '';
    groups.forEach(([title, list]) => {
      const sec = document.createElement('div');
      sec.innerHTML = `<p class="text-xs font-medium text-stone-500 mb-2">${title}</p><div class="space-y-2"></div>`;
      list.forEach((m) => {
        const price = typeof m.price_per_sqft === 'number' ? `₹${m.price_per_sqft}/sq.ft` : (m.price_per_sqft || '');
        const row = document.createElement('div');
        row.className = 'flex items-center gap-3 rounded-xl border border-stone-200 p-2';
        row.innerHTML = `
          <img src="${m.image_url}" class="h-12 w-12 rounded-lg object-cover bg-stone-100" alt="">
          <div class="min-w-0 flex-1">
            <p class="text-sm font-medium truncate">${m.name}</p>
            <p class="text-xs text-stone-500">${price ? price + ' · ' : ''}${m.match_score}% match</p>
          </div>
          <button class="text-xs font-medium text-accent-700 hover:underline shrink-0">Use</button>`;
        row.querySelector('button').addEventListener('click', () => {
          const tile = { name: m.name, url: m.image_url };
          selectTile(tile, addTileCard(tile, true));
        });
        sec.lastElementChild.appendChild(row);
      });
      out.appendChild(sec);
    });
  } catch (e) {
    out.innerHTML = `<p class="text-sm text-rose-600">Search failed: ${e.message}</p>`;
  }
}

// ---------------------------------------------------------------- boot
buildLanding();
bindCanvas();
bindControls();
loadCatalog();
updateMaskStatus();
