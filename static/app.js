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
const SIZE_PRESETS = [[300, 300], [300, 600], [600, 600], [600, 1200], [800, 800], [800, 1600], [1200, 1200], [1200, 200]];
const RENDER_STEPS = [
  'Estimating room depth',
  'Detecting structural lines',
  'Solving vanishing geometry',
  'Laying the tile pattern',
  'Relighting with room shadows',
  'Compositing final image',
];
const GROUT_COLORS = [
  { name: 'Auto (tile-matched)', hex: 'auto' },
  { name: 'Pure White', hex: '#F5F5F2' },
  { name: 'Silver Grey', hex: '#B8B8B4' },
  { name: 'Charcoal Black', hex: '#2B2B2B' },
  { name: 'Warm Beige', hex: '#D6C7B2' },
  { name: 'Terracotta', hex: '#9E5B40' },
];
const PATTERN_NAMES = { standard: 'Standard Grid', brick: 'Brick / Running Bond', herringbone: 'Herringbone', checkerboard: 'Checkerboard' };
const ORIENT_NAMES = { 0: '0° Standard', 90: '90° Rotated', 45: '45° Diamond' };
const MOOD_NAMES = { natural: 'Natural', warm: 'Warm Evening (3000K)', cool: 'Cool Daylight (6500K)' };
const ADVISOR_STYLE = {
  success: 'bg-emerald-50 text-emerald-900 ring-emerald-200',
  info: 'bg-sky-50 text-sky-900 ring-sky-200',
  warning: 'bg-amber-50 text-amber-900 ring-amber-200',
};

const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

const state = {
  roomBlob: null, roomImg: null, roomUrl: null, roomName: '', roomSeq: 0,
  originalImg: null,                     // the photo as opened (for the spec sheet's "before")
  pos: [], neg: [], order: [],           // order: stack of 'pos' | 'neg' for undo
  mode: 'pos',                           // what a plain tap / left-click adds
  maskBlob: null, overlayImg: null, maskSeq: 0, maskTimer: null, coverage: 0,
  tile: null,                            // { name, blob, url, path? }
  surface: 'auto', orientation: 0, pattern: 'standard', groutMm: 2, grout: GROUT_COLORS[0], mood: 'natural',
  autoExclude: true, straighten: true, excludedSummary: {}, straightened: 0,
  bestMatches: [], analysis: null,
  rendering: false, renderParams: null,
  resultImg: null, resultUrl: null, view: 'edit', split: 0.5, dragging: false,
  fit: { s: 1, dpr: 1 },
};

// ---------------------------------------------------------------- utilities
const DISTANCE_WARNING = 'Photo seems taken from far away. For best details and grid scale, try capturing closer to the surface.';

function toast(msg, ms = 4200) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add('hidden'), ms);
}

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ESC[c]);

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not load image'));
    img.src = src;
  });
}

const toBlob = (c, type = 'image/jpeg', q = 0.93) => new Promise((resolve, reject) =>
  c.toBlob((b) => (b ? resolve(b) : reject(new Error('Could not encode image'))), type, q));

// Downscale once on the client; the same blob is reused for every request so
// click coordinates and the returned mask always line up with the server image.
// The browser applies EXIF orientation when decoding, and the re-encoded JPEG has none.
async function normaliseRoom(src) {
  const img = await loadImage(src);
  if (!img.naturalWidth || !img.naturalHeight) throw new Error('Empty image');
  const k = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(img.naturalWidth * k));
  c.height = Math.max(1, Math.round(img.naturalHeight * k));
  c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
  const blob = await toBlob(c, 'image/jpeg', 0.93);
  const url = URL.createObjectURL(blob);
  return { blob, url, img: await loadImage(url) };
}

async function apiError(res) {
  try { const j = await res.json(); return j.detail || j.message || res.statusText; } catch { return res.statusText || `HTTP ${res.status}`; }
}

function setPressed(groupId, value) {
  document.querySelectorAll(`#${groupId} button`).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === String(value))));
}

function bindSeg(groupId, onPick) {
  document.querySelectorAll(`#${groupId} button`).forEach((b) => b.addEventListener('click', () => {
    setPressed(groupId, b.dataset.v);
    onPick(b.dataset.v);
  }));
}

// ---------------------------------------------------------------- screens
function showEditor() {
  $('screen-landing').classList.add('hidden');
  $('screen-editor').classList.remove('hidden');
  requestAnimationFrame(draw);
}

function showLanding() {
  closeDownloadMenu();
  $('screen-editor').classList.add('hidden');
  $('screen-landing').classList.remove('hidden');
}

async function openRoom(src, name) {
  try {
    const { blob, url, img } = await normaliseRoom(src);
    if (state.roomUrl) URL.revokeObjectURL(state.roomUrl);
    state.roomSeq++;
    state.roomBlob = blob;
    state.roomImg = img;
    state.roomUrl = url;
    state.originalImg = img;
    state.roomName = name;
    // A Floor/Wall choice belongs to the previous photo: carried over, it forced the next
    // room's walls through the floor projection.
    state.surface = 'auto';
    setPressed('surface-seg', 'auto');
    $('room-name').textContent = `${name} · ${img.naturalWidth}×${img.naturalHeight}`;
    resetSelection();
    clearResult();
    $('search-results').innerHTML = '';
    showEditor();
    analyzeRoom();
  } catch (e) {
    toast('That file could not be opened as an image.');
  }
}

function openFile(f) {
  if (!f) return;
  if (f.type && !f.type.startsWith('image/')) { toast('Choose an image file (JPG, PNG or WebP).'); return; }
  const url = URL.createObjectURL(f);
  openRoom(url, f.name || 'Photo').finally(() => URL.revokeObjectURL(url));
}

// ---------------------------------------------------------------- landing
function buildLanding() {
  const grid = $('demo-grid');
  DEMO_ROOMS.forEach((room) => {
    const card = document.createElement('button');
    card.className = 'group relative aspect-[4/3] overflow-hidden rounded-2xl bg-stone-200 shadow-sm ring-1 ring-black/5 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500';
    card.innerHTML = `
      <img src="${esc(room.src)}" alt="${esc(room.name)}" loading="lazy" class="absolute inset-0 h-full w-full object-cover transition-transform duration-700 group-hover:scale-[1.04]">
      <div class="absolute inset-0 bg-gradient-to-t from-black/60 via-black/0 to-black/0"></div>
      <div class="absolute bottom-0 inset-x-0 p-3 sm:p-4 flex items-end justify-between">
        <span class="text-white text-sm sm:text-base font-medium drop-shadow">${esc(room.name)}</span>
        <span class="hidden sm:inline text-xs text-white/90 bg-white/15 backdrop-blur px-2.5 py-1 rounded-full opacity-0 translate-y-1 group-hover:opacity-100 group-hover:translate-y-0 transition">Try it →</span>
      </div>`;
    card.addEventListener('click', () => openRoom(room.src, room.name));
    grid.appendChild(card);
  });

  const dz = $('dropzone');
  const input = $('room-input');
  input.addEventListener('change', () => { openFile(input.files[0]); input.value = ''; });
  const cam = $('camera-input');
  cam.addEventListener('change', () => { closeCamera(); openFile(cam.files[0]); cam.value = ''; });
  ['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
  dz.addEventListener('drop', (e) => {
    const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith('image/'));
    if (f) openFile(f);
    else toast('Drop an image file (JPG, PNG or WebP).');
  });
  // The camera button sits inside the dropzone <label>: keep it from also opening the file picker.
  $('btn-camera').addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); openCamera(); });
  $('btn-camera-top').addEventListener('click', openCamera);
}

// ---------------------------------------------------------------- pre-flight analysis
async function analyzeRoom() {
  const seq = state.roomSeq;
  state.bestMatches = [];
  state.analysis = null;
  $('advisor').classList.add('hidden');
  renderBestMatches();
  const fd = new FormData();
  fd.append('room_image', state.roomBlob, 'room.jpg');
  try {
    const res = await fetch(`${API}/api/analyze-image`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await apiError(res));
    const data = await res.json();
    if (seq !== state.roomSeq) return;               // another room was opened meanwhile
    state.analysis = data;
    state.bestMatches = data.best_matches || [];
    showAdvisor(data.quality);
    renderBestMatches(data.palette || []);
  } catch (e) {
    // Advice is optional; never block editing over it.
    console.warn('Photo analysis unavailable:', e.message);
  }
}

function showAdvisor(quality) {
  const msgs = quality?.messages || [];
  if (!msgs.length) return;
  $('advisor-list').innerHTML = msgs.map((m) =>
    `<p class="rounded-lg px-3 py-2 text-sm ring-1 ${ADVISOR_STYLE[m.level] || ADVISOR_STYLE.info}" data-code="${esc(m.code)}">${esc(m.text)}</p>`).join('');
  $('advisor').classList.remove('hidden');
  requestAnimationFrame(draw);                       // the stage height changed on desktop
}

function renderBestMatches(palette = []) {
  const box = $('best-matches');
  const list = $('best-list');
  const paths = new Set(state.bestMatches.map((m) => m.path));
  document.querySelectorAll('.tile-card').forEach((card) => {
    card.querySelector('.best-pill')?.remove();
    if (paths.has(card.dataset.path)) {
      const pill = document.createElement('span');
      pill.className = 'best-pill pointer-events-none absolute left-1 top-1 rounded-full px-1.5 py-0.5 text-[9px] font-semibold leading-none shadow';
      pill.textContent = '✨ Best Match';
      card.appendChild(pill);
    }
  });
  if (!state.bestMatches.length) { box.classList.add('hidden'); return; }
  $('room-palette').innerHTML = palette.slice(0, 6).map((p) =>
    `<span class="h-4 w-4 rounded-full ring-2 ring-white" style="background:${esc(p.hex)}" title="${esc(p.name)} · ${Math.round(p.share * 100)}%"></span>`).join('');
  list.innerHTML = '';
  state.bestMatches.forEach((m) => {
    const url = `${API}/catalog/${m.path.split('/').map(encodeURIComponent).join('/')}`;
    const name = prettyName(m.path);
    const row = document.createElement('button');
    row.className = 'w-full flex items-center gap-2.5 rounded-lg bg-white/80 p-1.5 text-left ring-1 ring-amber-200/60 hover:ring-amber-400';
    row.innerHTML = `<img src="${esc(url)}" alt="" class="h-9 w-9 rounded-md object-cover bg-stone-100">
      <span class="min-w-0 flex-1"><span class="block text-xs font-medium truncate">${esc(name)}</span>
      <span class="block text-[11px] text-stone-500 truncate">${esc(m.reason)}</span></span>
      <span class="text-[11px] font-semibold text-amber-800 tabular-nums">${Math.round(m.score)}%</span>`;
    row.addEventListener('click', () => {
      const card = document.querySelector(`.tile-card[data-path="${CSS.escape(m.path)}"]`);
      selectTile({ name, url, path: m.path }, card);
      card?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    });
    list.appendChild(row);
  });
  box.classList.remove('hidden');
}

function prettyName(path) {
  const file = path.split('/').pop().replace(/\.[^.]+$/, '');
  return file.replace(/[_-]+/g, ' ').trim().replace(/\b\w/g, (c) => c.toUpperCase());
}

// ---------------------------------------------------------------- canvas
function fitCanvas() {
  const stage = $('stage');
  const img = state.roomImg;
  if (!img || !stage.clientWidth) return;
  const pad = window.innerWidth < 640 ? 24 : 64;
  const maxW = Math.max(1, stage.clientWidth - pad);
  const maxH = Math.max(1, stage.clientHeight - pad);
  const s = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight);
  const cssW = Math.max(1, Math.floor(img.naturalWidth * s));
  const cssH = Math.max(1, Math.floor(img.naturalHeight * s));
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
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
  if (ctx.roundRect) ctx.roundRect(bx, 12 * dpr, w, 22 * dpr, 11 * dpr); else ctx.rect(bx, 12 * dpr, w, 22 * dpr);
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.fillText(text, align === 'left' ? x + 8 * dpr : x - 8 * dpr, 17.5 * dpr);
}

function eventToImage(e) {
  const rect = canvas.getBoundingClientRect();
  const img = state.roomImg;
  const x = ((e.clientX - rect.left) / Math.max(rect.width, 1)) * img.naturalWidth;
  const y = ((e.clientY - rect.top) / Math.max(rect.height, 1)) * img.naturalHeight;
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
    if (state.rendering) return;                      // the selection is frozen while it renders
    if (e.button !== 0 && e.button !== 2) return;
    if (!e.isPrimary) return;                         // ignore the second finger of a pinch
    const p = eventToImage(e);
    const kind = e.button === 2 ? 'neg' : state.mode;
    state[kind].push(p);
    state.order.push(kind);
    draw();
    scheduleMask();
  });
  canvas.addEventListener('pointermove', (e) => { if (state.dragging) moveSplit(e); });
  const stopDrag = () => { state.dragging = false; };
  canvas.addEventListener('pointerup', stopDrag);
  canvas.addEventListener('pointercancel', stopDrag);
  let resizeRaf = 0;
  window.addEventListener('resize', () => { cancelAnimationFrame(resizeRaf); resizeRaf = requestAnimationFrame(draw); });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { closeDownloadMenu(); if (!$('camera-modal').classList.contains('hidden')) closeCamera(); }
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || '');
    if (!typing && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && state.view === 'edit') { e.preventDefault(); undo(); }
  });
}

function moveSplit(e) {
  const rect = canvas.getBoundingClientRect();
  state.split = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(rect.width, 1)));
  draw();
}

function setMode(mode) {
  state.mode = mode;
  $('mode-include').setAttribute('aria-pressed', String(mode === 'pos'));
  $('mode-exclude').setAttribute('aria-pressed', String(mode === 'neg'));
}

// ---------------------------------------------------------------- masking
function resetSelection() {
  clearTimeout(state.maskTimer);
  state.pos = []; state.neg = []; state.order = [];
  state.maskBlob = null; state.overlayImg = null; state.maskSeq++;
  state.excludedSummary = {}; state.straightened = 0; state.coverage = 0;
  $('seg-busy').classList.replace('flex', 'hidden');
  updateMaskStatus();
}

function undo() {
  if (state.rendering) return;
  const last = state.order.pop();
  if (!last) return;
  state[last].pop();
  draw();
  scheduleMask();
}

function scheduleMask() {
  clearTimeout(state.maskTimer);
  if (!state.pos.length) {
    state.maskBlob = null; state.overlayImg = null; state.maskSeq++;
    $('seg-busy').classList.replace('flex', 'hidden');
    updateMaskStatus();
    draw();
    return;
  }
  // Invalidate the current mask right away: it no longer matches the clicks, so it must
  // not be rendered while the new one is on its way.
  state.maskBlob = null;
  state.maskSeq++;
  updateMaskStatus();
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
  fd.append('straighten', state.straighten);
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
    state.straightened = data.straightened_edges || 0;
    draw();
    // Once per photo: a small surface means tiny tiles and a noisy grid.
    if (data.distance_warning && state.distanceWarnedFor !== state.roomBlob) {
      state.distanceWarnedFor = state.roomBlob;
      toast(DISTANCE_WARNING, 7000);
    }
  } catch (e) {
    if (seq === state.maskSeq) toast(`Segmentation failed: ${e.message}`);
  } finally {
    if (seq === state.maskSeq) $('seg-busy').classList.replace('flex', 'hidden');
    updateMaskStatus();
  }
}

function describeExcluded() {
  const el = $('auto-excluded');
  if (!state.autoExclude) el.textContent = 'Off: only your Exclude taps are removed';
  else {
    const parts = Object.entries(state.excludedSummary || {}).map(([label, n]) => `${n} ${label}${n > 1 ? 's' : ''}`);
    el.textContent = parts.length ? `Cut out: ${parts.join(' · ')}` : 'Detected objects are outlined in rose';
  }
  $('straight-status').textContent = !state.straighten ? 'Off: edges follow the raw segmentation'
    : state.maskBlob && state.straightened ? `${state.straightened} edge${state.straightened > 1 ? 's' : ''} straightened`
      : 'Snaps ragged floor–wall junctions to clean lines';
}

function updateMaskStatus() {
  describeExcluded();
  const n = state.pos.length + state.neg.length;
  $('btn-undo').disabled = !n || state.rendering;
  $('btn-clear').disabled = !n || state.rendering;
  const status = $('mask-status');
  if (state.maskBlob) {
    status.textContent = `${Math.round((state.coverage || 0) * 100)}% of photo · ${state.pos.length}+ / ${state.neg.length}−`;
    status.className = 'text-xs text-accent-700 font-medium';
  } else {
    status.textContent = state.pos.length ? 'Segmenting…' : n ? 'Add an Include point' : 'No selection';
    status.className = 'text-xs text-stone-400';
  }
  const ready = Boolean(state.maskBlob && state.tile && !state.rendering);
  $('btn-render').disabled = !ready;
  $('render-hint').textContent = state.rendering ? 'Rendering…'
    : ready ? 'Depth, lighting and reflections are computed locally'
      : !state.maskBlob ? (state.pos.length ? 'Waiting for the selection…' : 'Select a surface to enable rendering')
        : 'Choose a tile to enable rendering';
}

// ---------------------------------------------------------------- tiles
async function selectTile(tile, card) {
  document.querySelectorAll('.tile-card').forEach((c) => c.setAttribute('aria-selected', 'false'));
  if (card) card.setAttribute('aria-selected', 'true');
  const want = tile.url;
  state.tileLoading = want;
  try {
    const blob = tile.blob || (await fetch(tile.url).then((r) => { if (!r.ok) throw new Error(); return r.blob(); }));
    if (state.tileLoading !== want) return;          // a later pick won
    state.tile = { ...tile, blob };
    $('tile-name').textContent = tile.name;
    updatePatternHint();
  } catch {
    toast(`Could not load tile "${tile.name}".`);
  }
  updateMaskStatus();
}

function addTileCard(tile, prepend = false) {
  const grid = $('tile-grid');
  const card = document.createElement('button');
  card.className = 'tile-card relative aspect-square rounded-lg overflow-hidden bg-stone-100 ring-1 ring-black/5 hover:ring-stone-400 transition';
  card.title = tile.name;
  if (tile.path) card.dataset.path = tile.path;
  card.setAttribute('aria-selected', 'false');
  card.setAttribute('aria-label', tile.name);
  const img = document.createElement('img');
  img.src = tile.url;
  img.alt = '';
  img.loading = 'lazy';
  img.className = 'h-full w-full object-cover';
  card.appendChild(img);
  card.addEventListener('click', () => selectTile(tile, card));
  prepend ? grid.prepend(card) : grid.appendChild(card);
  return card;
}

async function loadCatalog() {
  try {
    const res = await fetch(`${API}/api/catalog`);
    if (!res.ok) throw new Error(await apiError(res));
    const tiles = await res.json();
    if (!tiles.length) {
      $('tile-grid').innerHTML = '<p class="col-span-4 text-sm text-stone-500">No tiles in <code>catalog_tiles/</code> yet. Upload one above.</p>';
      return;
    }
    tiles.forEach((t) => addTileCard({ ...t, url: `${API}${t.url}` }));
    renderBestMatches(state.analysis?.palette || []);
    if (!state.tile) {
      const first = $('tile-grid').querySelector('.tile-card');
      selectTile({ ...tiles[0], url: `${API}${tiles[0].url}` }, first);
    }
  } catch {
    toast('Could not reach the API. Is uvicorn running on port 8000?');
  }
}

function tileDims() {
  return [Number($('tile-w').value), Number($('tile-h').value)];
}

function updatePatternHint() {
  const [w, h] = tileDims();
  const el = $('pattern-hint');
  let msg = '';
  if (state.pattern === 'herringbone' && w > 0 && h > 0 && Math.abs(w - h) <= 0.02 * Math.max(w, h)) {
    msg = 'Herringbone needs rectangular tiles (e.g. 600×1200 or 1200×200) to show its zig-zag.';
  } else if (state.pattern === 'checkerboard') {
    msg = 'Alternate tiles are turned a quarter (half for oblong tiles); best with veined or directional tiles.';
  }
  el.textContent = msg;
  el.classList.toggle('hidden', !msg);
}

function buildGroutSwatches() {
  const box = $('grout-swatches');
  GROUT_COLORS.forEach((g, i) => {
    const b = document.createElement('button');
    b.className = 'swatch h-8 w-8 rounded-full ring-1 ring-black/15 grid place-items-center text-[10px] font-semibold text-stone-600';
    b.title = g.name;
    b.setAttribute('aria-label', g.name);
    b.setAttribute('aria-pressed', String(i === 0));
    if (g.hex === 'auto') {
      b.style.background = 'conic-gradient(#f5f5f2 0 25%, #b8b8b4 0 50%, #d6c7b2 0 75%, #8b8680 0)';
      b.textContent = 'A';
    } else b.style.background = g.hex;
    b.addEventListener('click', () => pickGrout(g, b));
    box.appendChild(b);
  });
  // Custom colour: a native picker behind a swatch.
  const wrap = document.createElement('label');
  wrap.className = 'swatch relative h-8 w-8 rounded-full ring-1 ring-black/15 cursor-pointer overflow-hidden grid place-items-center';
  wrap.title = 'Custom colour';
  wrap.style.background = 'conic-gradient(#ef4444, #f59e0b, #84cc16, #06b6d4, #6366f1, #d946ef, #ef4444)';
  wrap.setAttribute('aria-pressed', 'false');
  wrap.innerHTML = '<span class="h-3.5 w-3.5 rounded-full bg-white/90 text-[10px] leading-[14px] text-center text-stone-700 font-bold">+</span>';
  const input = document.createElement('input');
  input.type = 'color';
  input.value = '#8a8f98';
  input.className = 'absolute inset-0 opacity-0 cursor-pointer';
  input.setAttribute('aria-label', 'Custom grout colour');
  input.addEventListener('input', () => {
    wrap.style.background = input.value;
    pickGrout({ name: `Custom ${input.value.toUpperCase()}`, hex: input.value.toUpperCase() }, wrap);
  });
  wrap.appendChild(input);
  box.appendChild(wrap);
}

function pickGrout(g, el) {
  state.grout = g;
  document.querySelectorAll('#grout-swatches .swatch').forEach((s) => s.setAttribute('aria-pressed', String(s === el)));
  $('grout-name').textContent = g.name;
}

function bindControls() {
  $('tile-input').addEventListener('change', (e) => {
    const f = e.target.files[0];
    e.target.value = '';
    if (!f) return;
    if (f.type && !f.type.startsWith('image/')) { toast('Choose an image file for the tile.'); return; }
    const tile = { name: f.name.replace(/\.[^.]+$/, ''), blob: f, url: URL.createObjectURL(f) };
    selectTile(tile, addTileCard(tile, true));
  });

  const presets = $('size-presets');
  SIZE_PRESETS.forEach(([w, h]) => {
    const b = document.createElement('button');
    b.className = 'px-2.5 py-1.5 rounded-full border border-stone-200 text-stone-600 hover:border-accent-500 hover:text-accent-700';
    b.textContent = `${w}×${h}`;
    b.addEventListener('click', () => { $('tile-w').value = w; $('tile-h').value = h; updatePatternHint(); });
    presets.appendChild(b);
  });
  $('btn-swap').addEventListener('click', () => {
    const w = $('tile-w').value;
    $('tile-w').value = $('tile-h').value;
    $('tile-h').value = w;
    updatePatternHint();
  });
  ['tile-w', 'tile-h'].forEach((id) => $(id).addEventListener('input', updatePatternHint));
  $('scale').addEventListener('input', (e) => { $('scale-val').textContent = `${Number(e.target.value).toFixed(2)}×`; });
  $('grout-mm').addEventListener('input', (e) => {
    state.groutMm = Number(e.target.value);
    $('grout-val').textContent = state.groutMm === 0 ? '0 mm · seamless' : `${state.groutMm.toFixed(1)} mm`;
  });
  buildGroutSwatches();

  $('auto-exclude').addEventListener('change', (e) => {
    state.autoExclude = e.target.checked;
    if (state.pos.length) scheduleMask(); else updateMaskStatus();
  });
  $('straighten').addEventListener('change', (e) => {
    state.straighten = e.target.checked;
    if (state.pos.length) scheduleMask(); else updateMaskStatus();
  });

  bindSeg('surface-seg', (v) => { state.surface = v; });
  bindSeg('orient-seg', (v) => { state.orientation = Number(v); });
  bindSeg('pattern-seg', (v) => { state.pattern = v; updatePatternHint(); });
  bindSeg('mood-seg', (v) => { state.mood = v; });
  $('mode-include').addEventListener('click', () => setMode('pos'));
  $('mode-exclude').addEventListener('click', () => setMode('neg'));

  $('btn-undo').addEventListener('click', undo);
  $('btn-clear').addEventListener('click', () => { if (!state.rendering) { resetSelection(); draw(); } });
  $('btn-view').addEventListener('click', () => setView(state.view === 'result' ? 'edit' : 'result'));
  $('btn-render').addEventListener('click', render);
  $('btn-home').addEventListener('click', showLanding);
  $('btn-new-room').addEventListener('click', showLanding);
  $('btn-keep').addEventListener('click', keepResult);
  $('btn-search').addEventListener('click', searchProducts);
  $('advisor-close').addEventListener('click', () => { $('advisor').classList.add('hidden'); requestAnimationFrame(draw); });

  $('btn-download').addEventListener('click', (e) => { e.stopPropagation(); toggleDownloadMenu(); });
  $('dl-image').addEventListener('click', () => closeDownloadMenu());
  $('dl-spec').addEventListener('click', () => { closeDownloadMenu(); downloadSpecSheet(); });
  $('dl-share').addEventListener('click', () => { closeDownloadMenu(); shareSpecSheet(); });
  document.addEventListener('click', (e) => { if (!$('download-menu').contains(e.target)) closeDownloadMenu(); });
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
  state.resultImg = null; state.resultUrl = null; state.renderParams = null;
  ['btn-keep', 'btn-download', 'render-info'].forEach((id) => $(id).classList.add('hidden'));
  closeDownloadMenu();
  setView('edit');
}

function startLoading() {
  const steps = $('loading-steps');
  steps.innerHTML = RENDER_STEPS.map((s) => `<li class="flex items-center gap-2"><span class="h-1.5 w-1.5 rounded-full bg-stone-300"></span>${esc(s)}</li>`).join('');
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
  if (!state.maskBlob || !state.tile || state.rendering) return;
  const [w, h] = tileDims();
  if (!(w >= 10 && h >= 10 && w <= 5000 && h <= 5000)) { toast('Enter the tile size in millimetres (10 to 5000).'); return; }

  const params = {
    tile: state.tile, w, h, scale: Number($('scale').value), glossy: $('glossy').checked,
    surface: state.surface, pattern: state.pattern, orientation: state.orientation,
    groutMm: state.groutMm, grout: state.grout, mood: state.mood,
  };
  const fd = new FormData();
  fd.append('room_image', state.roomBlob, 'room.jpg');
  fd.append('tile_image', params.tile.blob, 'tile');
  fd.append('mask_image', state.maskBlob, 'mask.png');
  fd.append('tile_width', w);
  fd.append('tile_height', h);
  fd.append('scale', params.scale);
  fd.append('is_glossy', params.glossy);
  fd.append('surface_type', params.surface);
  fd.append('pattern', params.pattern);
  fd.append('orientation', params.orientation);
  fd.append('grout_mm', params.groutMm);
  fd.append('grout_color', params.grout.hex);
  fd.append('lighting', params.mood);

  const seq = state.roomSeq;
  state.rendering = true;
  updateMaskStatus();
  const timer = startLoading();
  try {
    const res = await fetch(`${API}/api/apply-tile`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await apiError(res));
    const blob = await res.blob();
    // A different photo was opened (or a render kept) meanwhile: this result is stale.
    if (seq !== state.roomSeq) return;
    const url = URL.createObjectURL(blob);
    let img;
    try { img = await loadImage(url); } catch (e) { URL.revokeObjectURL(url); throw e; }
    if (seq !== state.roomSeq) { URL.revokeObjectURL(url); return; }
    clearResult();
    state.resultUrl = url;
    state.resultImg = img;
    state.renderParams = params;
    state.split = 0.5;
    showRenderInfo(res.headers.get('X-Render-Info'));
    $('dl-image').href = state.resultUrl;
    ['btn-keep', 'btn-download'].forEach((id) => $(id).classList.remove('hidden'));
    setView('result');
  } catch (e) {
    if (seq === state.roomSeq) toast(`Render failed: ${e.message}`);
  } finally {
    stopLoading(timer);
    state.rendering = false;
    updateMaskStatus();
  }
}

function showRenderInfo(raw) {
  const el = $('render-info');
  try {
    const info = JSON.parse(raw);
    const bits = [info.surface === 'wall' ? 'flat, level & plumb' : `perspective from ${esc(info.horizon_source)}`];
    if (info.grid_rotation_deg) bits.push(`grid ${esc(info.grid_rotation_deg)}°`);
    if (info.pattern && info.pattern !== 'standard') bits.push(esc(PATTERN_NAMES[info.pattern] || info.pattern));
    if (info.mood && info.mood !== 'natural') bits.push(esc(MOOD_NAMES[info.mood] || info.mood));
    el.innerHTML = `<span class="text-white font-medium capitalize">${esc(info.surface)}</span> · ${bits.join(' · ')}`;
    el.classList.remove('hidden');
    if (info.surface_override) toast('This selection is a wall, not a floor, so it was tiled as a wall.', 6000);
  } catch { el.classList.add('hidden'); }
}

// Use the render as the new base photo, e.g. to tile the walls after the floor.
async function keepResult() {
  if (!state.resultImg || state.rendering) return;
  const blob = await fetch(state.resultUrl).then((r) => r.blob());
  const url = URL.createObjectURL(blob);
  const img = await loadImage(url);
  if (state.roomUrl) URL.revokeObjectURL(state.roomUrl);
  state.roomSeq++;
  state.roomBlob = blob;
  state.roomImg = img;
  state.roomUrl = url;
  resetSelection();
  clearResult();
  toast('Render kept. Select the next surface.');
}

// ---------------------------------------------------------------- download / spec sheet / share
function toggleDownloadMenu() {
  const menu = $('download-menu');
  const open = menu.classList.contains('hidden');
  menu.classList.toggle('hidden', !open);
  $('btn-download').setAttribute('aria-expanded', String(open));
  if (open) $('dl-share').classList.toggle('hidden', !canShareFiles());
}

function closeDownloadMenu() {
  $('download-menu').classList.add('hidden');
  $('btn-download').setAttribute('aria-expanded', 'false');
}

function canShareFiles() {
  try {
    return Boolean(navigator.canShare && navigator.canShare({ files: [new File([new Blob(['x'], { type: 'image/jpeg' })], 'x.jpg', { type: 'image/jpeg' })] }));
  } catch { return false; }
}

function fitText(c, text, maxW) {
  if (c.measureText(text).width <= maxW) return text;
  let t = text;
  while (t.length > 1 && c.measureText(`${t}…`).width > maxW) t = t.slice(0, -1);
  return `${t}…`;
}

function roundRectPath(c, x, y, w, h, r) {
  c.beginPath();
  if (c.roundRect) c.roundRect(x, y, w, h, r); else c.rect(x, y, w, h);
}

// High-resolution presentation card: render, "before" inset, and a specification bar.
async function buildSpecSheet() {
  const p = state.renderParams;
  const after = state.resultImg;
  if (!p || !after) throw new Error('Render a surface first.');
  await document.fonts?.ready;
  const W = Math.max(1600, after.naturalWidth);
  const k = W / 1600;
  const imgH = Math.round(after.naturalHeight * (W / after.naturalWidth));
  const barH = Math.round(300 * k);
  const c = document.createElement('canvas');
  c.width = W;
  c.height = imgH + barH;
  const g = c.getContext('2d');
  g.imageSmoothingQuality = 'high';
  g.drawImage(after, 0, 0, W, imgH);

  // Before inset (bottom-left of the render).
  const before = state.originalImg || state.roomImg;
  const iw = Math.round(W * 0.24);
  const ih = Math.round(iw * before.naturalHeight / before.naturalWidth);
  const m = Math.round(28 * k);
  const ix = m;
  const iy = imgH - ih - m;
  g.save();
  g.shadowColor = 'rgba(0,0,0,.35)';
  g.shadowBlur = 24 * k;
  g.fillStyle = '#fff';
  roundRectPath(g, ix - 6 * k, iy - 6 * k, iw + 12 * k, ih + 12 * k, 14 * k);
  g.fill();
  g.restore();
  g.save();
  roundRectPath(g, ix, iy, iw, ih, 9 * k);
  g.clip();
  g.drawImage(before, ix, iy, iw, ih);
  g.restore();
  const tag = (text, x, y, align = 'left') => {
    g.font = `600 ${18 * k}px Inter, sans-serif`;
    const tw = g.measureText(text).width + 20 * k;
    const bx = align === 'left' ? x : x - tw;
    g.fillStyle = 'rgba(15,17,19,.78)';
    roundRectPath(g, bx, y, tw, 32 * k, 16 * k);
    g.fill();
    g.fillStyle = '#fff';
    g.textBaseline = 'middle';
    g.textAlign = 'left';
    g.fillText(text, bx + 10 * k, y + 16.5 * k);
  };
  tag('Before', ix + 10 * k, iy + 10 * k);
  tag('After', W - m, m, 'right');

  // Specification bar.
  const y0 = imgH;
  g.fillStyle = '#16191c';
  g.fillRect(0, y0, W, barH);
  g.fillStyle = '#14b8a6';
  g.fillRect(0, y0, W, Math.max(3, 4 * k));
  const pad = 48 * k;
  const sw = barH - 2 * pad + 8 * k;
  let tileImg = null;
  try { tileImg = await loadImage(p.tile.url); } catch { /* swatch is optional */ }
  g.fillStyle = '#fff';
  roundRectPath(g, pad - 5 * k, y0 + pad - 5 * k, sw + 10 * k, sw + 10 * k, 12 * k);
  g.fill();
  if (tileImg) {
    const s = Math.min(tileImg.naturalWidth, tileImg.naturalHeight);
    g.save();
    roundRectPath(g, pad, y0 + pad, sw, sw, 8 * k);
    g.clip();
    g.drawImage(tileImg, (tileImg.naturalWidth - s) / 2, (tileImg.naturalHeight - s) / 2, s, s, pad, y0 + pad, sw, sw);
    g.restore();
  }

  const tx = pad + sw + 44 * k;
  const brandW = 300 * k;
  const textW = W - tx - brandW - pad;
  g.textAlign = 'left';
  g.textBaseline = 'alphabetic';
  g.fillStyle = '#5eead4';
  g.font = `600 ${16 * k}px Inter, sans-serif`;
  g.fillText('SELECTED TILE', tx, y0 + pad + 14 * k);
  g.fillStyle = '#fff';
  g.font = `600 ${40 * k}px Inter, sans-serif`;
  g.fillText(fitText(g, p.tile.name || 'Custom tile', textW), tx, y0 + pad + 62 * k);

  const groutText = p.groutMm === 0 ? 'Seamless (0 mm)' : `${p.groutMm} mm · ${p.grout.name}`;
  const specs = [
    ['Size', `${p.w} × ${p.h} mm`],
    ['Pattern', `${PATTERN_NAMES[p.pattern]} · ${p.orientation}°`],
    ['Grout', groutText],
    ['Finish', p.glossy ? 'Glossy' : 'Matte'],
    ['Lighting', MOOD_NAMES[p.mood]],
    ['Surface', p.surface === 'auto' ? 'Auto-detected' : p.surface[0].toUpperCase() + p.surface.slice(1)],
  ];
  const cols = 3;
  const colW = textW / cols;
  specs.forEach(([key, val], i) => {
    const cx = tx + (i % cols) * colW;
    const cy = y0 + pad + 118 * k + Math.floor(i / cols) * 66 * k;
    g.fillStyle = '#a8a29e';
    g.font = `500 ${15 * k}px Inter, sans-serif`;
    g.fillText(key.toUpperCase(), cx, cy);
    g.fillStyle = '#f5f5f4';
    g.font = `500 ${22 * k}px Inter, sans-serif`;
    g.fillText(fitText(g, val, colW - 20 * k), cx, cy + 30 * k);
  });

  // Branding.
  const bx = W - pad;
  const by = y0 + barH / 2;
  const q = 15 * k;
  const gap = 4 * k;
  const mx = bx - 2 * q - gap;
  const my = by - 70 * k;
  [[0, 0, '#2dd4bf'], [1, 0, '#fff'], [0, 1, '#fff'], [1, 1, '#2dd4bf']].forEach(([cx, cy, col]) => {
    g.fillStyle = col;
    roundRectPath(g, mx + cx * (q + gap), my + cy * (q + gap), q, q, 3 * k);
    g.fill();
  });
  g.textAlign = 'right';
  g.fillStyle = '#fff';
  g.font = `600 ${30 * k}px Inter, sans-serif`;
  g.fillText('Smart Tile AI', bx, by + 6 * k);
  g.fillStyle = '#a8a29e';
  g.font = `400 ${16 * k}px Inter, sans-serif`;
  g.fillText('Photoreal room visualization', bx, by + 36 * k);
  g.fillText(new Date().toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }), bx, by + 62 * k);

  return toBlob(c, 'image/jpeg', 0.92);
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

function specFilename() {
  const base = (state.renderParams?.tile?.name || 'tile').replace(/[^\w-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'tile';
  return `smart-tile-${base}-spec-sheet.jpg`;
}

async function downloadSpecSheet() {
  try { saveBlob(await buildSpecSheet(), specFilename()); } catch (e) { toast(`Spec sheet failed: ${e.message}`); }
}

async function shareSpecSheet() {
  try {
    const file = new File([await buildSpecSheet()], specFilename(), { type: 'image/jpeg' });
    const p = state.renderParams;
    await navigator.share({ files: [file], title: 'Tile visualization', text: `${p.tile.name} · ${p.w}×${p.h} mm · ${PATTERN_NAMES[p.pattern]} — made with Smart Tile AI` });
  } catch (e) {
    if (e.name !== 'AbortError') toast(`Sharing failed: ${e.message}`);
  }
}

// ---------------------------------------------------------------- camera capture with level guide
const cam = { stream: null, onOrient: null, ok: false };

async function openCamera() {
  // iOS asks for motion access only from inside a user gesture, so request it first.
  let motion = Promise.resolve('granted');
  if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
    motion = DeviceOrientationEvent.requestPermission().catch(() => 'denied');
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext) {
    $('camera-input').click();                       // still inside the tap: native capture works
    return;
  }
  $('camera-modal').classList.remove('hidden');
  document.body.style.overflow = 'hidden';
  $('cam-error').classList.add('hidden');
  $('cam-shutter').disabled = true;
  $('cam-status').textContent = 'Starting camera…';
  $('cam-readout').textContent = '';
  $('cam-hint').textContent = '';
  try {
    cam.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1440 } }, audio: false,
    });
    if ($('camera-modal').classList.contains('hidden')) { stopStream(); return; }   // cancelled meanwhile
    const video = $('cam-video');
    video.srcObject = cam.stream;
    await video.play().catch(() => {});
    fitCamFrame();
    $('cam-shutter').disabled = false;
  } catch (e) {
    showCameraError(e.name === 'NotAllowedError' ? 'Camera access was blocked. You can still use your phone\'s camera app.' : 'The live camera isn\'t available on this device.');
    return;
  }
  const perm = await motion;
  if (perm !== 'granted') { levelUnavailable('Level guide needs motion access'); return; }
  startLevel();
}

// The video is letterboxed (object-contain shows exactly what gets captured); the grid,
// crosshair and level sit on the visible frame, not on the black bars.
function fitCamFrame() {
  const video = $('cam-video');
  const frame = $('cam-frame');
  const W = video.clientWidth;
  const H = video.clientHeight;
  if (!video.videoWidth || !W || !H) { frame.style.cssText = ''; return; }
  const k = Math.min(W / video.videoWidth, H / video.videoHeight);
  const w = video.videoWidth * k;
  const h = video.videoHeight * k;
  Object.assign(frame.style, { left: `${(W - w) / 2}px`, top: `${(H - h) / 2}px`, width: `${w}px`, height: `${h}px`, right: 'auto', bottom: 'auto' });
}

function showCameraError(text) {
  $('cam-error-text').textContent = text;
  $('cam-error').classList.remove('hidden');
  $('cam-status').textContent = 'Camera unavailable';
}

const FILL_HINT = 'Ensure the surface fills the frame';

function levelUnavailable(text) {
  $('cam-status').textContent = `${text} · frame the floor or wall at chest height`;
  $('cam-hint').textContent = FILL_HINT;
  $('cam-status').className = 'rounded-full bg-black/60 backdrop-blur px-4 py-2 text-sm font-medium text-center max-w-[92%]';
  $('level-line').style.background = 'rgba(255,255,255,.7)';
}

function startLevel() {
  cam.ok = false;
  cam.onOrient = (e) => {
    if (e.beta == null || e.gamma == null) return;
    cam.ok = true;
    updateLevel(e.beta, e.gamma);
  };
  window.addEventListener('deviceorientation', cam.onOrient);
  setTimeout(() => { if (cam.onOrient && !cam.ok) levelUnavailable('No motion sensor'); }, 1500);
}

// Gravity in device axes (x right, y up the screen, z out of the screen) from the
// beta/gamma Euler angles: g = (sin γ cos β, −sin β, −cos γ cos β). The camera looks
// along −z, so its pitch below the horizon is asin(g·(−z)); roll is gravity's angle in the
// screen plane, corrected for the current screen orientation.
function levelAngles(beta, gamma) {
  const rad = Math.PI / 180;
  const b = beta * rad;
  const g = gamma * rad;
  const gx = Math.sin(g) * Math.cos(b);
  const gy = -Math.sin(b);
  const pitchDown = Math.asin(Math.max(-1, Math.min(1, Math.cos(g) * Math.cos(b)))) / rad;
  const screenAngle = (screen.orientation && screen.orientation.angle) || window.orientation || 0;
  let roll = Math.atan2(gx, -gy) / rad + screenAngle;
  roll = ((roll + 540) % 360) - 180;
  return { pitchDown, roll };
}

function levelStatus(pitchDown, roll) {
  const r = Math.abs(roll);
  if (r <= 3 && pitchDown >= -10 && pitchDown <= 30) return 'good';
  if (r <= 6 && pitchDown >= -20 && pitchDown <= 40) return 'warn';
  return 'bad';
}

function updateLevel(beta, gamma) {
  const { pitchDown, roll } = levelAngles(beta, gamma);
  const status = levelStatus(pitchDown, roll);
  const colors = { good: '#34d399', warn: '#facc15', bad: '#f87171' };
  const line = $('level-line');
  line.style.transform = `rotate(${-roll}deg)`;
  line.style.background = colors[status];
  // Bubble: centre band is 0-30° down; clamp to the track.
  const t = Math.max(0, Math.min(1, (pitchDown + 30) / 90));
  $('pitch-bubble').style.top = `${t * 100}%`;
  $('pitch-bubble').style.background = colors[status];
  const s = $('cam-status');
  s.textContent = status === 'good' ? 'Perfect Angle — Ready to Capture' : 'Tilt Warning: Hold phone straight at chest height for best results';
  $('cam-hint').textContent = FILL_HINT;
  s.className = `rounded-full backdrop-blur px-4 py-2 text-sm font-medium text-center max-w-[92%] ${status === 'good' ? 'bg-emerald-500/85' : status === 'warn' ? 'bg-yellow-500/85 text-black' : 'bg-red-500/85'}`;
  $('cam-readout').textContent = `Tilt ${Math.round(pitchDown)}° ${pitchDown >= 0 ? 'down' : 'up'} · Roll ${Math.round(roll)}°`;
}

function stopStream() {
  cam.stream?.getTracks().forEach((t) => t.stop());
  cam.stream = null;
  $('cam-video').srcObject = null;
}

function closeCamera() {
  if (cam.onOrient) window.removeEventListener('deviceorientation', cam.onOrient);
  cam.onOrient = null;
  stopStream();
  $('camera-modal').classList.add('hidden');
  document.body.style.overflow = '';
}

async function capturePhoto() {
  const video = $('cam-video');
  if (!video.videoWidth) return;
  const c = document.createElement('canvas');
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext('2d').drawImage(video, 0, 0);
  closeCamera();
  try {
    const blob = await toBlob(c, 'image/jpeg', 0.94);
    const url = URL.createObjectURL(blob);
    await openRoom(url, 'Camera photo');
    URL.revokeObjectURL(url);
  } catch (e) {
    toast(`Capture failed: ${e.message}`);
  }
}

function bindCamera() {
  $('cam-cancel').addEventListener('click', closeCamera);
  $('cam-shutter').addEventListener('click', capturePhoto);
  $('cam-native').addEventListener('click', () => $('camera-input').click());
  $('cam-fallback').addEventListener('click', () => $('camera-input').click());
  $('cam-video').addEventListener('loadedmetadata', fitCamFrame);
  window.addEventListener('resize', () => { if (cam.stream) fitCamFrame(); });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && cam.stream) closeCamera();   // release the camera when the tab is hidden
  });
}

// ---------------------------------------------------------------- product search (/search-tiles/)
async function searchProducts() {
  if (!state.roomBlob) return;
  const out = $('search-results');
  out.innerHTML = '<div class="h-16 rounded-xl bg-stone-100 shimmer"></div>';
  const fd = new FormData();
  fd.append('query_image', state.roomBlob, 'room.jpg');
  const seq = state.roomSeq;
  try {
    const res = await fetch(`${API}/search-tiles/`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await apiError(res));
    const data = await res.json();
    if (seq !== state.roomSeq) return;
    const groups = [['Wall matches', data.wall_matches], ['Floor matches', data.floor_matches], ['Overall matches', data.overall_matches]]
      .filter(([, list]) => list && list.length);
    if (!groups.length) { out.innerHTML = '<p class="text-sm text-stone-500">No close matches in the catalog.</p>'; return; }
    out.innerHTML = '';
    groups.forEach(([title, list]) => {
      const sec = document.createElement('div');
      sec.innerHTML = `<p class="text-xs font-medium text-stone-500 mb-2">${esc(title)}</p><div class="space-y-2"></div>`;
      list.forEach((m) => {
        const price = typeof m.price_per_sqft === 'number' ? `₹${m.price_per_sqft}/sq.ft` : (m.price_per_sqft || '');
        const row = document.createElement('div');
        row.className = 'flex items-center gap-3 rounded-xl border border-stone-200 p-2';
        row.innerHTML = `
          <img src="${esc(m.image_url)}" class="h-12 w-12 rounded-lg object-cover bg-stone-100" alt="">
          <div class="min-w-0 flex-1">
            <p class="text-sm font-medium truncate">${esc(m.name)}</p>
            <p class="text-xs text-stone-500">${price ? `${esc(price)} · ` : ''}${esc(m.match_score)}% match</p>
          </div>
          <button class="text-xs font-medium text-accent-700 hover:underline shrink-0 px-2 py-1">Use</button>`;
        row.querySelector('button').addEventListener('click', () => {
          const tile = { name: m.name, url: m.image_url };
          selectTile(tile, addTileCard(tile, true));
        });
        sec.lastElementChild.appendChild(row);
      });
      out.appendChild(sec);
    });
  } catch (e) {
    if (seq === state.roomSeq) out.innerHTML = `<p class="text-sm text-rose-600">Search failed: ${esc(e.message)}</p>`;
  }
}

// ---------------------------------------------------------------- boot
buildLanding();
bindCanvas();
bindControls();
bindCamera();
loadCatalog();
updateMaskStatus();
