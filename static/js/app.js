/* ═══════════════════════════════════════════════════════════════════════════
   Geo RAG — frontend
   ═══════════════════════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────────────────────
let currentMode   = 'hybrid';
let currentFilter = '';
let docs          = {};         // id → doc
let citationMap   = {};         // "docId:page" → { url, file, page }
let graphNodes    = [];
let graphEdges    = [];
let graphAnim     = null;
let isDragging    = false;
let dragNode      = null;

const MODE_HINTS = {
  rag:      'только векторный поиск',
  hybrid:   'vector + BM25 (RRF)',
  graphrag: 'граф сущностей → векторный',
};
const MODE_BADGE = {
  rag:      { label: '🔍 RAG',     cls: 'bg-blue-900/60 text-blue-300' },
  hybrid:   { label: '⚡ Гибрид',  cls: 'bg-emerald-900/60 text-emerald-300' },
  graphrag: { label: '⬡ GraphRAG', cls: 'bg-purple-900/60 text-purple-300' },
};
const ENTITY_COLORS = {
  скважина:      '#f59e0b',
  пласт:         '#3b82f6',
  горизонт:      '#8b5cf6',
  свита:         '#ec4899',
  месторождение: '#10b981',
  формация:      '#f97316',
  default:       '#64748b',
};

// ── Doc type labels ────────────────────────────────────────────────────────
const DOC_TYPE_META = {
  text_pdf:   { icon: '📄', label: 'Текстовый PDF',  strategy: 'Гибрид: текст + VLM' },
  scan_pdf:   { icon: '🖼',  label: 'Скан-PDF',       strategy: 'Полный VLM OCR' },
  hybrid_pdf: { icon: '📄🖼', label: 'Смешанный PDF', strategy: 'Гибрид: текст + VLM' },
  docx:       { icon: '📝', label: 'DOCX',            strategy: 'Прямое извлечение' },
  doc:        { icon: '📝', label: 'DOC',             strategy: 'Прямое извлечение' },
  djvu:       { icon: '📚', label: 'DJVU',            strategy: 'djvutxt / VLM OCR' },
  unknown:    { icon: '❓', label: 'Неизвестный',     strategy: 'VLM OCR' },
};

// ── Pipeline step definitions ──────────────────────────────────────────────
const PIPELINE_STEPS = [
  { key: 'detect', label: 'Определение типа',  pattern: /тип:|text_pdf|scan_pdf|hybrid|docx|djvu|определен/i },
  { key: 'render', label: 'Рендер страниц',    pattern: /ренд|render|\[\d+\//i },
  { key: 'direct', label: 'Прямое извлечение', pattern: /прямое извлечени|direct|текстов[а-я]+ страниц/i },
  { key: 'vlm',    label: 'VLM анализ',        pattern: /vlm|qwen|vision/i },
  { key: 'embed',  label: 'Jina эмбеддинги',   pattern: /jina|эмбедд|embed/i },
  { key: 'chroma', label: 'ChromaDB',          pattern: /chroma/i },
  { key: 'graph',  label: 'Граф знаний',       pattern: /граф|graph/i },
];

function parsePipeline(progressText) {
  const done = new Set();
  for (const line of (progressText || '').split('\n')) {
    for (const step of PIPELINE_STEPS) {
      if (step.pattern.test(line)) done.add(step.key);
    }
  }
  return done;
}

function renderPipelineHTML(progressText, isActive) {
  const done = parsePipeline(progressText);
  let activeSet = false;
  return PIPELINE_STEPS.map(step => {
    const isDone = done.has(step.key);
    let cls = 'pip-step';
    if (isDone) {
      cls += ' done';
    } else if (!activeSet && isActive) {
      cls += ' active';
      activeSet = true;
    }
    return `<div class="${cls}"><div class="pip-dot"></div><span>${step.label}</span></div>`;
  }).join('');
}

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupTextarea();
  setupDropZone();
  loadDocs();
  refreshGraphStats();
  loadMetrics();

  // Attach canvas events ONCE — handlers use global graphNodes/dragNode
  _attachCanvasEventsOnce();

  document.querySelectorAll('.quick-q').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('query-input').value = btn.textContent;
      sendMessage();
    });
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { closeModal(); closeGraphPanel(); }
  });
});

// ── Textarea auto-resize ───────────────────────────────────────────────────
function setupTextarea() {
  const ta = document.getElementById('query-input');
  ta.addEventListener('input', () => {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
  });
  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
}

// ── Drop-zone ──────────────────────────────────────────────────────────────
function setupDropZone() {
  const zone  = document.getElementById('drop-zone');
  const input = document.getElementById('file-input');

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('border-blue-400');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('border-blue-400'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('border-blue-400');
    const allowed = ['.pdf', '.docx', '.doc', '.djvu'];
    [...e.dataTransfer.files].forEach(f => {
      const ext = f.name.toLowerCase().match(/\.[^.]+$/)?.[0] || '';
      if (allowed.includes(ext)) uploadFile(f);
    });
  });
  input.addEventListener('change', () => {
    [...input.files].forEach(f => uploadFile(f));
    input.value = '';
  });
}

// ── Mode ───────────────────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  ['rag', 'hybrid', 'graphrag'].forEach(m => {
    document.getElementById('mode-' + m).classList.toggle('active', m === mode);
  });
  document.getElementById('mode-hint').textContent = MODE_HINTS[mode] || '';
}

function setFilter(id) {
  currentFilter = id;
}

// ── Docs ───────────────────────────────────────────────────────────────────
async function loadDocs() {
  try {
    const r    = await fetch('/docs/status/');
    const data = await r.json();
    data.documents.forEach(d => { docs[d.id] = d; });
    renderDocs();
  } catch { /* ignore on startup */ }
}

function renderDocs() {
  const list       = document.getElementById('docs-list');
  const filterList = document.getElementById('filter-list');
  const docArr     = Object.values(docs).sort((a, b) => String(a.name).localeCompare(String(b.name)));

  if (docArr.length === 0) {
    list.innerHTML = '<p class="text-xs text-slate-600 text-center py-4">Нет документов</p>';
    filterList.innerHTML = `<label class="flex items-center gap-2 cursor-pointer py-0.5">
      <input type="radio" name="filter" value="" checked onchange="setFilter('')" class="accent-blue-500">
      <span class="text-sm text-slate-300">Все документы</span></label>`;
    return;
  }

  list.innerHTML = docArr.map(d => buildDocCard(d)).join('');

  const allChecked = currentFilter === '' ? 'checked' : '';
  filterList.innerHTML =
    `<label class="flex items-center gap-2 cursor-pointer py-0.5">
      <input type="radio" name="filter" value="" ${allChecked} onchange="setFilter('')" class="accent-blue-500">
      <span class="text-sm text-slate-300">Все</span></label>` +
    docArr.filter(d => d.status === 'indexed').map(d => {
      const checked = currentFilter === String(d.id) ? 'checked' : '';
      const name    = (d.name || '').replace(/\.pdf$/i, '').slice(0, 28);
      return `<label class="flex items-center gap-2 cursor-pointer py-0.5">
        <input type="radio" name="filter" value="${d.id}" ${checked} onchange="setFilter('${d.id}')" class="accent-blue-500">
        <span class="text-xs text-slate-400 truncate">${name}</span></label>`;
    }).join('');
}

function buildDocCard(d) {
  const isIndexing = d.status === 'indexing';
  const isIndexed  = d.status === 'indexed';
  const isError    = d.status === 'error';
  const isPending  = d.status === 'pending';

  const statusIcon = isIndexed ? '✅' : isIndexing ? '⏳' : isError ? '❌' : isPending ? '🕐' : '📄';
  const ext        = (d.name || '').match(/\.([^.]+)$/)?.[1]?.toLowerCase() || 'pdf';
  const name       = (d.name || '').replace(/\.[^.]+$/, '').slice(0, 30);

  // Doc type badge (shown when detected in progress)
  const docTypeBadge = _parseDocTypeBadge(d.progress || '');

  let pipelineHTML = '';
  if (isIndexing || isPending) {
    pipelineHTML = `<div class="mt-1.5 space-y-0.5 pipeline-steps" data-doc="${d.id}">
      ${renderPipelineHTML(d.progress || '', isIndexing)}
    </div>`;
  }

  let metaHTML = '';
  if (isIndexed) {
    metaHTML = `<div class="flex items-center gap-1.5 mt-0.5 flex-wrap">
      <span class="text-xs text-slate-600">${d.total_pages || 0} стр · ${d.text_chunks || 0} чанков</span>
      ${docTypeBadge}
    </div>`;
  } else if (isError) {
    metaHTML = `<p class="text-xs text-red-500 mt-0.5 truncate">${escHtml(d.error_msg || 'Ошибка')}</p>`;
  } else if (isPending || isIndexing) {
    metaHTML = docTypeBadge ? `<div class="mt-0.5">${docTypeBadge}</div>` : '';
  }

  const deleteBtn = `<button onclick="deleteDoc('${d.id}')" title="Удалить"
    class="text-slate-600 hover:text-red-400 transition-colors text-xs px-1">🗑</button>`;
  const indexBtn  = !isIndexing && !isPending
    ? `<button onclick="indexDoc('${d.id}')"
        class="text-slate-600 hover:text-blue-400 transition-colors text-xs px-1" title="Переиндексировать">↻</button>`
    : '';

  return `<div class="bg-slate-700/50 rounded-lg p-2.5 mb-2" data-doc-card="${d.id}">
    <div class="flex items-start justify-between gap-1">
      <div class="flex items-start gap-1.5 min-w-0">
        <span class="text-base flex-shrink-0">${statusIcon}</span>
        <div class="min-w-0">
          <p class="text-xs text-slate-300 font-medium truncate" title="${escAttr(d.name)}">${escHtml(name)}</p>
          ${metaHTML}
        </div>
      </div>
      <div class="flex items-center gap-0.5 flex-shrink-0">${indexBtn}${deleteBtn}</div>
    </div>
    ${pipelineHTML}
  </div>`;
}

function _parseDocTypeBadge(progress) {
  // Try to extract doc_type from progress text
  const m = progress.match(/тип[=:]?\s*(text_pdf|scan_pdf|hybrid_pdf|docx|doc|djvu)/i)
         || progress.match(/(text_pdf|scan_pdf|hybrid_pdf)/i);
  if (!m) return '';
  const meta = DOC_TYPE_META[m[1].toLowerCase()] || DOC_TYPE_META.unknown;
  const colors = {
    text_pdf:   'bg-emerald-900/50 text-emerald-300',
    scan_pdf:   'bg-amber-900/50 text-amber-300',
    hybrid_pdf: 'bg-blue-900/50 text-blue-300',
    docx:       'bg-sky-900/50 text-sky-300',
    doc:        'bg-sky-900/50 text-sky-300',
    djvu:       'bg-violet-900/50 text-violet-300',
    unknown:    'bg-slate-700 text-slate-400',
  };
  const cls = colors[m[1].toLowerCase()] || colors.unknown;
  return `<span class="text-xs px-1.5 py-0.5 rounded ${cls}">${meta.icon} ${meta.label}</span>`;
}

// ── Upload ─────────────────────────────────────────────────────────────────
async function uploadFile(file) {
  showUploadProgress(10, file.name);
  const form = new FormData();
  form.append('file', file);

  let resp;
  try {
    resp = await fetch('/upload/', {
      method:  'POST',
      headers: { 'X-CSRFToken': CSRF },
      body:    form,
    });
  } catch (e) {
    hideUploadProgress();
    alert('Ошибка сети: ' + e.message);
    return;
  }

  hideUploadProgress();
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    alert(err.error || 'Ошибка загрузки');
    return;
  }

  const doc = await resp.json();
  docs[doc.id] = doc;
  renderDocs();

  // Auto-start indexing
  const ir = await fetch(`/docs/${doc.id}/index/`, {
    method:  'POST',
    headers: { 'X-CSRFToken': CSRF },
  });
  if (ir.ok) {
    docs[doc.id].status = 'indexing';
    renderDocs();
    pollDocStatus(doc.id);
  }
}

function showUploadProgress(pct, label) {
  document.getElementById('upload-progress').classList.remove('hidden');
  document.getElementById('upload-bar').style.width = pct + '%';
  document.getElementById('upload-text').textContent = label;
}

function hideUploadProgress() {
  document.getElementById('upload-progress').classList.add('hidden');
}

// ── Poll doc status ────────────────────────────────────────────────────────
function pollDocStatus(docId) {
  let attempts = 0;
  const iv = setInterval(async () => {
    attempts++;
    if (attempts > 360) { clearInterval(iv); return; }
    try {
      const r = await fetch(`/docs/${docId}/status/`);
      const d = await r.json();
      docs[d.id] = d;

      // Update pipeline steps in-place without full re-render
      const stepsEl = document.querySelector(`.pipeline-steps[data-doc="${docId}"]`);
      if (stepsEl) {
        stepsEl.innerHTML = renderPipelineHTML(d.progress || '', d.status === 'indexing');
      }

      if (d.status === 'indexed' || d.status === 'error') {
        clearInterval(iv);
        renderDocs();
        if (d.status === 'indexed') {
          refreshGraphStats();
          loadMetrics();
        }
      }
    } catch { /* retry next tick */ }
  }, 2000);
}

async function indexDoc(docId) {
  const r = await fetch(`/docs/${docId}/index/`, {
    method:  'POST',
    headers: { 'X-CSRFToken': CSRF },
  });
  if (r.ok && docs[docId]) {
    docs[docId].status = 'indexing';
    renderDocs();
    pollDocStatus(docId);
  }
}

async function deleteDoc(docId) {
  if (!confirm('Удалить документ и его векторы из ChromaDB?')) return;
  await fetch(`/docs/${docId}/`, {
    method:  'DELETE',
    headers: { 'X-CSRFToken': CSRF },
  });
  delete docs[docId];
  if (currentFilter === String(docId)) currentFilter = '';
  renderDocs();
  refreshGraphStats();
  loadMetrics();
}

// ── Chat ───────────────────────────────────────────────────────────────────
async function sendMessage() {
  const ta    = document.getElementById('query-input');
  const query = ta.value.trim();
  if (!query) return;

  hideWelcome();
  ta.value = '';
  ta.style.height = 'auto';
  document.getElementById('send-btn').disabled = true;

  appendUserBubble(query);
  const botId = appendBotBubble();

  try {
    const r = await fetch('/chat/', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': CSRF },
      body:    JSON.stringify({ query, mode: currentMode, filter_doc_id: currentFilter || null }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'Ошибка сервера');

    // Build citation map: "docId:page" → { url, file, page }
    citationMap = {};
    (data.chunks || []).forEach(c => {
      if (c.document_id && c.page) {
        const key = `${c.document_id}:${c.page}`;
        if (!citationMap[key]) {
          citationMap[key] = {
            url:  c.image_url || '',
            file: c.source_file || '',
            page: c.page,
          };
        }
      }
    });

    updateBotBubble(botId, data);

    if (data.graph_data && (data.graph_data.nodes || []).length > 0) {
      openGraphPanel(data.graph_data);
    }
  } catch (e) {
    updateBotBubbleError(botId, e.message);
  } finally {
    document.getElementById('send-btn').disabled = false;
    scrollToBottom();
  }
}

function hideWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.remove();
}

function scrollToBottom() {
  const m = document.getElementById('messages');
  requestAnimationFrame(() => { m.scrollTop = m.scrollHeight; });
}

// ── Message bubbles ────────────────────────────────────────────────────────
function appendUserBubble(text) {
  const m   = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'flex justify-end';
  div.innerHTML = `<div class="max-w-xl bg-blue-600 rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm text-white">${escHtml(text)}</div>`;
  m.appendChild(div);
  scrollToBottom();
}

function appendBotBubble() {
  const id  = 'bot-' + Date.now();
  const m   = document.getElementById('messages');
  const div = document.createElement('div');
  div.id        = id;
  div.className = 'flex flex-col gap-3';
  div.innerHTML = `<div class="flex items-start gap-3">
    <div class="w-7 h-7 rounded-full bg-slate-700 flex items-center justify-center text-sm flex-shrink-0">🪨</div>
    <div class="bg-slate-800 border border-slate-700 rounded-2xl rounded-tl-sm px-4 py-3 text-sm">
      <span class="text-slate-500 flex items-center gap-2">
        <svg class="spinner w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <circle cx="12" cy="12" r="10" stroke-dasharray="31.4" stroke-dashoffset="10"/>
        </svg>
        Думаю…
      </span>
    </div>
  </div>`;
  m.appendChild(div);
  scrollToBottom();
  return id;
}

function updateBotBubble(id, data) {
  const wrap = document.getElementById(id);
  if (!wrap) return;

  const answerHtml = renderAnswer(data.answer || '');
  const mode       = data.mode || currentMode;
  const badge      = MODE_BADGE[mode] || MODE_BADGE.hybrid;
  const chunks     = data.chunks || [];
  const images     = data.images || [];

  wrap.innerHTML = `
    <div class="flex items-start gap-3">
      <div class="w-7 h-7 rounded-full bg-slate-700 flex items-center justify-center text-sm flex-shrink-0">🪨</div>
      <div class="flex-1 min-w-0">
        <div class="bg-slate-800 border border-slate-700 rounded-2xl rounded-tl-sm px-4 py-3">
          <div class="prose-dark text-sm leading-relaxed">${answerHtml}</div>
          <div class="mt-2 flex items-center gap-2 flex-wrap">
            <span class="text-xs px-2 py-0.5 rounded-full ${badge.cls}">${badge.label}</span>
            <span class="text-xs text-slate-600">${chunks.length} источников</span>
          </div>
        </div>
      </div>
    </div>
    ${images.length ? buildImagesRow(images) : ''}
    ${chunks.length ? buildSourcesPanel(chunks, mode) : ''}
  `;

  // Attach citation click handlers after HTML is set
  wrap.querySelectorAll('.geo-cite').forEach(el => {
    el.addEventListener('click', () => {
      const key  = el.dataset.key;
      const info = citationMap[key];
      if (info && info.url) {
        openModal(info.url, `${info.file || ''} · стр. ${info.page}`);
      } else {
        // Fall back: scroll to matching chunk card
        const card = wrap.querySelector(`[data-cite-key="${key}"]`);
        if (card) card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });
  });

  scrollToBottom();
}

function updateBotBubbleError(id, msg) {
  const wrap = document.getElementById(id);
  if (!wrap) return;
  wrap.innerHTML = `<div class="flex items-start gap-3">
    <div class="w-7 h-7 rounded-full bg-red-900/50 flex items-center justify-center text-sm flex-shrink-0">⚠️</div>
    <div class="bg-red-900/30 border border-red-800 rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-red-300">${escHtml(msg)}</div>
  </div>`;
}

// ── Answer renderer with clickable citations ───────────────────────────────
function renderAnswer(rawText) {
  // Step 1: replace [uuid:page] patterns BEFORE markdown parsing
  // Use placeholder tags that won't be mangled by marked
  const withCites = rawText.replace(
    /\[([0-9a-f-]{8,}):(\d+)\]/g,
    (_, docId, page) => `%%CITE:${docId}:${page}%%`
  );

  // Step 2: markdown → html
  let html = '';
  try {
    html = marked.parse(withCites, { breaks: true, gfm: true });
  } catch {
    html = `<p>${escHtml(rawText)}</p>`;
  }

  // Step 3: replace placeholders with styled spans
  html = html.replace(
    /%%CITE:([^:]+):(\d+)%%/g,
    (_, docId, page) => {
      const key = `${docId}:${page}`;
      return `<span class="geo-cite" data-key="${key}">📄 стр.${page}</span>`;
    }
  );

  return html;
}

// ── Images row ─────────────────────────────────────────────────────────────
function buildImagesRow(images) {
  const items = images.map(img => {
    const caption = escAttr((img.source_file || '') + ' · стр. ' + img.page);
    return `<div class="cursor-pointer group" onclick="openModal('${escAttr(img.url)}', '${caption}')">
      <div class="relative overflow-hidden rounded-lg border border-slate-700 hover:border-blue-500 transition-colors">
        <img src="${escAttr(img.url)}" class="w-full h-28 object-cover group-hover:opacity-90 transition-opacity" loading="lazy">
        <div class="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-2">
          <p class="text-xs text-white truncate">${escHtml(img.source_file || '')} · стр. ${img.page}</p>
        </div>
        <div class="absolute top-1 right-1 bg-black/60 text-white text-xs px-1.5 py-0.5 rounded">стр. ${img.page}</div>
      </div>
    </div>`;
  }).join('');

  return `<div class="ml-10 grid grid-cols-4 gap-2">${items}</div>`;
}

// ── Sources panel ──────────────────────────────────────────────────────────
function buildSourcesPanel(chunks, mode) {
  const badge = MODE_BADGE[mode] || MODE_BADGE.hybrid;
  const items = chunks.map((c, i) => {
    const citeKey  = c.document_id ? `${c.document_id}:${c.page}` : null;
    const hasImg   = !!c.image_url;
    const score    = typeof c.score === 'number' ? Math.round(c.score * 100) + '%' : '';
    const file     = (c.source_file || '').replace(/\.pdf$/i, '').slice(0, 30);
    const imgThumb = hasImg
      ? `<div class="mt-1.5 cursor-pointer" onclick="openModal('${escAttr(c.image_url)}', '${escAttr(c.source_file + ' · стр. ' + c.page)}')">
           <img src="${escAttr(c.image_url)}" class="w-full max-h-24 object-cover rounded-md border border-slate-700 hover:border-blue-400 transition-colors" loading="lazy">
         </div>`
      : '';

    return `<div class="bg-slate-700/40 rounded-xl p-3 hover:bg-slate-700/60 transition-colors"
                 ${citeKey ? `data-cite-key="${citeKey}"` : ''}>
      <div class="flex items-start justify-between gap-2 mb-1.5">
        <div class="flex items-center gap-1.5 flex-wrap">
          <span class="text-xs font-bold text-slate-500">#${i+1}</span>
          <span class="text-xs text-slate-500 truncate max-w-[120px]">${escHtml(file)}</span>
          <span class="text-xs bg-slate-600 text-slate-400 px-1.5 py-0.5 rounded-full flex-shrink-0">стр.${c.page}</span>
          ${hasImg ? '<span class="text-amber-400" title="Есть изображение">🖼</span>' : ''}
        </div>
        <span class="text-xs text-slate-600 flex-shrink-0">${score}</span>
      </div>
      <p class="text-xs text-slate-400 leading-relaxed line-clamp-3">${escHtml(c.text)}</p>
      ${imgThumb}
    </div>`;
  }).join('');

  return `<div class="ml-10">
    <div class="flex items-center gap-2 mb-2">
      <p class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Источники</p>
      <span class="text-xs px-2 py-0.5 rounded-full ${badge.cls}">${badge.label}</span>
    </div>
    <div class="grid gap-2">${items}</div>
  </div>`;
}

// ── Modal ──────────────────────────────────────────────────────────────────
function openModal(url, caption) {
  const modal = document.getElementById('img-modal');
  document.getElementById('modal-img').src = url;
  document.getElementById('modal-caption').textContent = caption || '';
  modal.classList.remove('hidden');
  modal.classList.add('flex');
}

function closeModal() {
  const modal = document.getElementById('img-modal');
  modal.classList.add('hidden');
  modal.classList.remove('flex');
  document.getElementById('modal-img').src = '';
}

// ── Graph panel ────────────────────────────────────────────────────────────
function openGraphPanel(data) {
  const panel = document.getElementById('graph-panel');
  panel.classList.remove('hidden');
  panel.classList.add('flex');

  graphNodes = (data.nodes || []).map(n => ({ ...n, x: 0, y: 0, vx: 0, vy: 0 }));
  graphEdges = data.edges || [];

  renderLegend(graphNodes);
  renderGraphTrace(data);

  if (graphAnim) { cancelAnimationFrame(graphAnim); graphAnim = null; }

  // Double rAF: first frame makes panel visible, second measures real dimensions
  requestAnimationFrame(() => requestAnimationFrame(() => {
    const canvas = document.getElementById('graph-canvas');
    const w = panel.offsetWidth  - 16;
    const h = Math.max(panel.offsetHeight - 270, 140);
    canvas.width  = w;
    canvas.height = h;

    const cx = w / 2, cy = h / 2;
    graphNodes.forEach((n, i) => {
      const angle = (i / Math.max(graphNodes.length, 1)) * 2 * Math.PI;
      const r = Math.min(cx, cy) * 0.65;
      n.x = cx + Math.cos(angle) * r + (Math.random() - .5) * 20;
      n.y = cy + Math.sin(angle) * r + (Math.random() - .5) * 20;
    });

    runForce(canvas);
  }));
}

// ── GraphRAG Trace panel ───────────────────────────────────────────────────
function renderGraphTrace(data) {
  const el    = document.getElementById('graph-trace');
  if (!el) return;
  const trace = data.trace || {};
  const terms = trace.extracted_terms || data.extracted_terms || [];
  const pages = trace.graph_pages     || data.pages_from_seeds || [];
  const seeds = data.seed_nodes       || [];
  const hop1  = data.hop1_count       || 0;
  const fromG = trace.chunks_from_graph  || 0;
  const fromV = trace.chunks_from_vector || 0;
  const total = trace.total_chunks       || (fromG + fromV) || (data.nodes || []).length;
  const cites = trace.citations_in_answer || 0;

  if (!terms.length && !seeds.length) {
    el.innerHTML = '<p class="text-xs text-slate-600 italic">Сущности не найдены в запросе</p>';
    return;
  }

  const termTags = terms.map(t => {
    // Try to find type from seed nodes
    const node = (data.nodes || []).find(n => n.label && n.label.toLowerCase() === t.toLowerCase());
    const type = node ? node.type : 'default';
    return `<span class="trace-tag tag-${type}">${escHtml(t)}</span>`;
  }).join('');

  const pageList = pages.slice(0, 8).map(p =>
    `<span class="text-slate-500 text-xs">стр.<strong class="text-slate-300">${p.page}</strong></span>`
  ).join(' · ');
  const pageMore = pages.length > 8 ? `<span class="text-slate-600"> +${pages.length - 8}</span>` : '';

  const graphBadge = fromG > 0
    ? `<span class="text-emerald-400 font-bold">${fromG}</span> из графа`
    : '<span class="text-amber-400">0</span> из графа';
  const vectorBadge = fromV > 0
    ? ` + <span class="text-blue-400">${fromV}</span> fallback` : '';

  el.scrollTop = 0;
  el.innerHTML = `
    <div class="trace-step ts-info">
      <span class="trace-icon">🔍</span>
      <div>
        <div class="trace-title">1. Запрос → сущности NER</div>
        <div class="trace-val">${termTags || '<span class="text-slate-600">не найдены</span>'}</div>
      </div>
    </div>

    <div class="trace-step ts-done">
      <span class="trace-icon">⬡</span>
      <div>
        <div class="trace-title">2. Поиск в графе знаний</div>
        <div class="trace-val">
          <span class="text-purple-400 font-bold">${seeds.length}</span> seed-узла
          → <span class="text-slate-300 font-bold">${hop1}</span> соседей (1-hop)
        </div>
      </div>
    </div>

    <div class="trace-step ${pages.length ? 'ts-done' : 'ts-warn'}">
      <span class="trace-icon">📄</span>
      <div>
        <div class="trace-title">3. Страницы из графа</div>
        <div class="trace-val">${pages.length ? (pageList + pageMore) : '<span class="text-amber-400">не найдены</span>'}</div>
      </div>
    </div>

    <div class="trace-step ts-done">
      <span class="trace-icon">📥</span>
      <div>
        <div class="trace-title">4. Чанки для контекста</div>
        <div class="trace-val">${graphBadge}${vectorBadge} = <strong>${total}</strong> итого</div>
      </div>
    </div>

    <div class="trace-step ts-done">
      <span class="trace-icon">💬</span>
      <div>
        <div class="trace-title">5. Ответ с цитатами</div>
        <div class="trace-val"><span class="text-blue-400 font-bold">${cites}</span> цитат[stр.N] → кликабельны</div>
      </div>
    </div>
  `;
}

function closeGraphPanel() {
  const panel = document.getElementById('graph-panel');
  panel.classList.add('hidden');
  panel.classList.remove('flex');
  if (graphAnim) { cancelAnimationFrame(graphAnim); graphAnim = null; }
}

function renderLegend(nodes) {
  const types  = [...new Set(nodes.map(n => n.type).filter(Boolean))];
  const legend = document.getElementById('graph-legend');
  legend.innerHTML = types.map(t => {
    const color = ENTITY_COLORS[t] || ENTITY_COLORS.default;
    return `<span class="flex items-center gap-1">
      <span style="background:${color}" class="inline-block w-2.5 h-2.5 rounded-full"></span>
      <span class="text-slate-400">${t}</span>
    </span>`;
  }).join('');
}

// ── Canvas force-directed graph ────────────────────────────────────────────
function runForce(canvas) {
  let iter      = 0;
  let revealTick = 0;          // for staggered node reveal animation
  const byId    = Object.fromEntries(graphNodes.map(n => [n.id, n]));

  // Phase 0 (iter 0-30):  only seeds visible
  // Phase 1 (iter 30-70): hop-1 neighbors fade in
  // Phase 2 (iter 70+):   everything visible, force settles
  graphNodes.forEach(n => { n.opacity = n.is_seed ? 1 : 0; });

  function tick() {
    iter++;
    const alpha = Math.max(0.004, 0.4 * Math.exp(-iter * 0.022));

    // Staggered reveal
    if (iter === 35) {
      graphNodes.forEach(n => { if (n.is_hop1) n._fadeTarget = 1; });
    }
    if (iter === 65) {
      graphNodes.forEach(n => { if (!n.is_seed && !n.is_hop1) n._fadeTarget = 1; });
    }
    graphNodes.forEach(n => {
      if (n._fadeTarget !== undefined && n.opacity < n._fadeTarget) {
        n.opacity = Math.min(n._fadeTarget, (n.opacity || 0) + 0.05);
      }
    });

    // Repulsion
    for (let i = 0; i < graphNodes.length; i++) {
      const a = graphNodes[i];
      if (!a.opacity) continue;
      for (let j = i + 1; j < graphNodes.length; j++) {
        const b = graphNodes[j];
        if (!b.opacity) continue;
        const dx = b.x - a.x, dy = b.y - a.y;
        const dist = Math.sqrt(dx*dx + dy*dy) || 0.1;
        const f = (900 / (dist * dist)) * alpha;
        a.vx -= dx / dist * f;  a.vy -= dy / dist * f;
        b.vx += dx / dist * f;  b.vy += dy / dist * f;
      }
    }

    // Spring attraction along edges
    for (const e of graphEdges) {
      const a = byId[e.source] || byId[e.from];
      const b = byId[e.target] || byId[e.to];
      if (!a || !b || !a.opacity || !b.opacity) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.sqrt(dx*dx + dy*dy) || 0.1;
      const f = (dist - 80) * 0.05 * alpha;
      a.vx += dx / dist * f;  a.vy += dy / dist * f;
      b.vx -= dx / dist * f;  b.vy -= dy / dist * f;
    }

    // Weak gravity
    const cx = canvas.width / 2, cy = canvas.height / 2;
    for (const n of graphNodes) {
      if (!n.opacity) continue;
      n.vx += (cx - n.x) * 0.008 * alpha;
      n.vy += (cy - n.y) * 0.008 * alpha;
    }

    // Integrate
    for (const n of graphNodes) {
      if (n === dragNode) { n.vx = 0; n.vy = 0; continue; }
      n.vx *= 0.85;  n.vy *= 0.85;
      n.x = Math.max(14, Math.min(canvas.width  - 14, n.x + n.vx));
      n.y = Math.max(14, Math.min(canvas.height - 14, n.y + n.vy));
    }

    draw(canvas, byId, iter);
    graphAnim = requestAnimationFrame(tick);
  }
  tick();
}

function draw(canvas, byId, iter) {
  const ctx  = canvas.getContext('2d');
  const time = iter * 0.06;   // for pulsing seed glow
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw edges
  for (const e of graphEdges) {
    const a = byId[e.source] || byId[e.from];
    const b = byId[e.target] || byId[e.to];
    if (!a || !b) continue;
    const edgeOp = Math.min(a.opacity || 0, b.opacity || 0);
    if (edgeOp <= 0) continue;

    const isSeedEdge = e.is_seed_edge;
    ctx.lineWidth   = isSeedEdge ? 1.5 : 0.8;
    ctx.strokeStyle = isSeedEdge
      ? `rgba(139,92,246,${0.7 * edgeOp})`
      : `rgba(71,85,105,${0.45 * edgeOp})`;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  // Draw nodes
  for (const n of graphNodes) {
    const op    = n.opacity || 0;
    if (op <= 0) continue;
    const color = ENTITY_COLORS[n.type] || ENTITY_COLORS.default;
    const r     = n.is_seed ? 9 : n.is_hop1 ? 6 : 4;

    // Seed glow pulse
    if (n.is_seed && op > 0.5) {
      const glowR = r + 4 + Math.sin(time) * 3;
      const grd   = ctx.createRadialGradient(n.x, n.y, r, n.x, n.y, glowR);
      grd.addColorStop(0, color.replace(')', `,${0.35 * op})`).replace('rgb', 'rgba'));
      grd.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.beginPath();
      ctx.arc(n.x, n.y, glowR, 0, 2 * Math.PI);
      ctx.fillStyle = grd;
      ctx.fill();
    }

    // Node circle
    ctx.globalAlpha = op;
    ctx.beginPath();
    ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    if (n.is_seed) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    } else if (n.is_hop1) {
      ctx.strokeStyle = 'rgba(139,92,246,0.6)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Label
    if (n.is_seed || (n.is_hop1 && graphNodes.length < 25) || graphNodes.length < 20) {
      ctx.globalAlpha = op;
      ctx.fillStyle = n.is_seed ? '#f8fafc' : n.is_hop1 ? '#c4b5fd' : '#64748b';
      ctx.font      = n.is_seed ? 'bold 9px sans-serif' : '8px sans-serif';
      ctx.fillText((n.label || '').slice(0, 18), n.x + r + 2, n.y + 3);
      ctx.globalAlpha = 1;
    }
  }
}

function _attachCanvasEventsOnce() {
  const c = document.getElementById('graph-canvas');
  if (!c || c._geoEventsOk) return;
  c._geoEventsOk = true;

  c.addEventListener('mousedown', e => {
    const rect = c.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    for (const n of graphNodes) {
      if (Math.hypot(n.x - mx, n.y - my) < 12) {
        isDragging = true; dragNode = n; break;
      }
    }
  });
  c.addEventListener('mousemove', e => {
    if (!isDragging || !dragNode) return;
    const rect = c.getBoundingClientRect();
    dragNode.x = e.clientX - rect.left;
    dragNode.y = e.clientY - rect.top;
  });
  c.addEventListener('mouseup',    () => { isDragging = false; dragNode = null; });
  c.addEventListener('mouseleave', () => { isDragging = false; dragNode = null; });
}

// ── Graph stats ────────────────────────────────────────────────────────────
async function refreshGraphStats() {
  try {
    const r = await fetch('/graph/stats/');
    const d = await r.json();

    const badge  = document.getElementById('graph-stats-badge');
    const nodeEl = document.getElementById('graph-nodes');
    const infoEl = document.getElementById('graph-info');

    if (d.nodes > 0) {
      badge.classList.remove('hidden');
      if (nodeEl) nodeEl.textContent = d.nodes;
    }

    if (infoEl) {
      if (d.available) {
        const typeStr = Object.entries(d.types || {})
          .sort((a, b) => b[1] - a[1]).slice(0, 4)
          .map(([k, v]) => `${k}:${v}`).join(' · ');
        infoEl.innerHTML =
          `<span class="text-slate-400">${d.nodes} узлов · ${d.edges} рёбер</span><br>` +
          `<span class="text-slate-600 text-xs">${typeStr}</span>`;
      } else {
        infoEl.textContent = 'Граф не построен';
      }
    }
  } catch { /* ignore */ }
}

async function rebuildGraph() {
  const btn = document.querySelector('button[onclick="rebuildGraph()"]');
  const orig = btn ? btn.textContent : '';
  if (btn) btn.textContent = '⏳ Строю…';

  try {
    const r = await fetch('/graph/rebuild/', {
      method:  'POST',
      headers: { 'X-CSRFToken': CSRF },
    });
    const d = await r.json();
    if (d.error) alert('Ошибка: ' + d.error);
    else await refreshGraphStats();
  } catch (e) {
    alert('Ошибка: ' + e.message);
  } finally {
    if (btn) btn.textContent = orig || '↻ Перестроить граф';
  }
}

// ── Metrics ────────────────────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const r  = await fetch('/metrics/');
    const d  = await r.json();
    const el = document.getElementById('global-stats');
    if (el) {
      el.textContent = `${d.index?.chunks || 0} чанков · ${d.documents?.indexed || 0} доков`;
    }
  } catch { /* ignore */ }
}

// ── Utilities ──────────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function escAttr(s) {
  return String(s ?? '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
}
