/* app.js — Geo RAG frontend */
'use strict';

// ─── Состояние ────────────────────────────────────────────────────────────────
let filterDocId  = null;
let pollingTimers = {};

// ─── Утилиты ─────────────────────────────────────────────────────────────────

function shortName(filename) {
  const stem = (filename || '').replace(/\.pdf$/i, '');
  return stem.length > 28 ? stem.slice(0, 28) + '…' : stem;
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function post(url, body) {
  return fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json', 'X-CSRFToken': CSRF },
    body:    JSON.stringify(body),
  });
}

// ─── Загрузка файлов ─────────────────────────────────────────────────────────

const dropZone  = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('border-blue-500', 'bg-slate-700/60');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('border-blue-500', 'bg-slate-700/60');
});
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('border-blue-500', 'bg-slate-700/60');
  [...e.dataTransfer.files].filter(f => f.name.toLowerCase().endsWith('.pdf'))
    .forEach(uploadFile);
});
fileInput.addEventListener('change', e => {
  [...e.target.files].forEach(uploadFile);
  fileInput.value = '';
});

async function uploadFile(file) {
  showUploadProgress(10, `Загрузка: ${file.name}`);

  const fd = new FormData();
  fd.append('file', file);

  try {
    const r1 = await fetch('/upload/', {
      method:  'POST',
      headers: { 'X-CSRFToken': CSRF },
      body:    fd,
    });
    if (!r1.ok) { const e = await r1.json(); throw new Error(e.error || 'Upload error'); }
    const doc = await r1.json();

    showUploadProgress(50, `Запуск индексации…`);

    const r2 = await post(`/docs/${doc.id}/index/`, {});
    if (!r2.ok) { const e = await r2.json(); throw new Error(e.error || 'Index error'); }

    hideUploadProgress();
    await refreshDocs();
    startPolling(doc.id);

  } catch (err) {
    hideUploadProgress();
    showToast(`Ошибка: ${err.message}`, 'error');
  }
}

function showUploadProgress(pct, text) {
  document.getElementById('upload-progress').classList.remove('hidden');
  document.getElementById('upload-bar').style.width = pct + '%';
  document.getElementById('upload-text').textContent = text;
}
function hideUploadProgress() {
  document.getElementById('upload-progress').classList.add('hidden');
}

// ─── Список документов ───────────────────────────────────────────────────────

async function refreshDocs() {
  const r = await fetch('/docs/status/');
  const data = await r.json();
  renderDocList(data.documents);
  updateGlobalStats(data.documents);
}

function renderDocList(docs) {
  const list       = document.getElementById('docs-list');
  const filterList = document.getElementById('filter-list');

  if (!docs.length) {
    list.innerHTML = '<p class="text-xs text-slate-600 text-center py-4">Нет документов</p>';
    filterList.innerHTML = `<label class="flex items-center gap-2 cursor-pointer py-0.5">
      <input type="radio" name="filter" value="" checked onchange="setFilter('')" class="accent-blue-500">
      <span class="text-sm text-slate-300">Все документы</span>
    </label>`;
    return;
  }

  list.innerHTML = docs.map(renderDocCard).join('');

  const indexedDocs = docs.filter(d => d.status === 'indexed');
  filterList.innerHTML = `
    <label class="flex items-center gap-2 cursor-pointer py-0.5">
      <input type="radio" name="filter" value="" ${!filterDocId ? 'checked' : ''} onchange="setFilter('')" class="accent-blue-500">
      <span class="text-sm text-slate-300">Все документы</span>
    </label>
    ${indexedDocs.map(d => `
    <label class="flex items-center gap-2 cursor-pointer py-0.5">
      <input type="radio" name="filter" value="${d.id}" ${filterDocId === d.id ? 'checked' : ''} onchange="setFilter('${d.id}')" class="accent-blue-500">
      <span class="text-xs text-slate-400 truncate" title="${escHtml(d.name)}">${escHtml(shortName(d.name))}</span>
    </label>`).join('')}`;
}

function renderDocCard(doc) {
  const badges = {
    pending:  '<span class="text-xs bg-yellow-900/60 text-yellow-400 px-2 py-0.5 rounded-full">Ожидает</span>',
    indexing: '<span class="text-xs bg-blue-900/60 text-blue-400 px-2 py-0.5 rounded-full flex items-center gap-1"><svg class="w-3 h-3 spinner" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4"/></svg>Индексируется</span>',
    indexed:  '<span class="text-xs bg-green-900/60 text-green-400 px-2 py-0.5 rounded-full">✓ Готов</span>',
    error:    '<span class="text-xs bg-red-900/60 text-red-400 px-2 py-0.5 rounded-full">✗ Ошибка</span>',
  };

  const extra = doc.status === 'indexed'
    ? `<p class="text-xs text-slate-500 mt-1">${doc.text_chunks} чанков · ${doc.images_count} изобр. · ${doc.total_pages} стр.</p>`
    : doc.status === 'indexing'
    ? `<p class="text-xs text-slate-500 mt-1 truncate">${escHtml(doc.progress || '…')}</p>`
    : doc.status === 'error'
    ? `<p class="text-xs text-red-500 mt-1 truncate" title="${escHtml(doc.error_msg)}">${escHtml((doc.error_msg || '').slice(0, 60))}</p>`
    : '';

  return `
  <div id="doc-${doc.id}" class="bg-slate-700/50 border border-slate-600/50 rounded-xl p-2.5 mb-2 hover:border-slate-500 transition-colors">
    <div class="flex items-start justify-between gap-1 mb-1.5">
      <p class="text-xs font-medium text-slate-200 truncate flex-1" title="${escHtml(doc.name)}">
        📄 ${escHtml(shortName(doc.name))}
      </p>
      <button onclick="deleteDoc('${doc.id}')"
              class="text-slate-600 hover:text-red-400 transition-colors flex-shrink-0 p-0.5"
              title="Удалить">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2">
          <path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
        </svg>
      </button>
    </div>
    ${badges[doc.status] || ''}
    ${extra}
  </div>`;
}

function updateGlobalStats(docs) {
  const ready = docs.filter(d => d.status === 'indexed').length;
  const total = docs.length;
  const el = document.getElementById('global-stats');
  if (total > 0) el.textContent = `${ready}/${total} документов готово`;
}

function setFilter(val) {
  filterDocId = val || null;
}

// ─── Поллинг статуса индексации ──────────────────────────────────────────────

function startPolling(docId) {
  if (pollingTimers[docId]) return;
  pollingTimers[docId] = setInterval(async () => {
    try {
      const r = await fetch(`/docs/${docId}/status/`);
      const doc = await r.json();

      const card = document.getElementById(`doc-${docId}`);
      if (card) {
        const tmp = document.createElement('div');
        tmp.innerHTML = renderDocCard(doc);
        card.replaceWith(tmp.firstElementChild);
      }

      if (doc.status === 'indexed' || doc.status === 'error') {
        clearInterval(pollingTimers[docId]);
        delete pollingTimers[docId];
        await refreshDocs();
        if (doc.status === 'indexed') {
          showToast(`✓ ${shortName(doc.name)} проиндексирован`, 'success');
        }
      }
    } catch (_) {}
  }, 2500);
}

// ─── Чат ─────────────────────────────────────────────────────────────────────

const queryInput = document.getElementById('query-input');
const sendBtn    = document.getElementById('send-btn');

queryInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
queryInput.addEventListener('input', () => {
  queryInput.style.height = 'auto';
  queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';
});

// Быстрые вопросы из приветствия
document.querySelectorAll('.quick-q').forEach(btn => {
  btn.addEventListener('click', () => {
    queryInput.value = btn.textContent;
    sendMessage();
  });
});

async function sendMessage() {
  const query = queryInput.value.trim();
  if (!query || sendBtn.disabled) return;

  queryInput.value = '';
  queryInput.style.height = 'auto';
  sendBtn.disabled = true;

  removeWelcome();
  appendUserMsg(query);
  const loadId = appendLoading();

  try {
    const r = await post('/chat/', { query, filter_doc_id: filterDocId });
    const data = await r.json();
    removeLoading(loadId);

    if (r.ok) {
      appendAssistantMsg(data.answer, data.sources || [], data.chunks || []);
    } else {
      appendErrorMsg(data.error || 'Неизвестная ошибка');
    }
  } catch (err) {
    removeLoading(loadId);
    appendErrorMsg(`Ошибка соединения: ${err.message}`);
  }

  sendBtn.disabled = false;
  queryInput.focus();
}

// ─── Рендеринг сообщений ─────────────────────────────────────────────────────

function removeWelcome() {
  document.getElementById('welcome')?.remove();
}

function scrollDown() {
  const m = document.getElementById('messages');
  m.scrollTop = m.scrollHeight;
}

function appendUserMsg(text) {
  const msgs = document.getElementById('messages');
  const div  = document.createElement('div');
  div.className = 'flex justify-end';
  div.innerHTML = `
    <div class="max-w-lg bg-blue-600 rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm leading-relaxed">
      ${escHtml(text).replace(/\n/g, '<br>')}
    </div>`;
  msgs.appendChild(div);
  scrollDown();
}

function appendAssistantMsg(text, sources, chunks) {
  const msgs = document.getElementById('messages');
  const div  = document.createElement('div');
  div.className = 'flex justify-start';

  const html  = marked.parse(text);
  const srcs  = buildSourceTags(sources);
  const panel = buildSourcesPanel(chunks);

  div.innerHTML = `
    <div class="max-w-2xl w-full">
      <div class="bg-slate-700 rounded-2xl rounded-tl-sm px-5 py-4 text-sm leading-relaxed prose-dark">
        ${html}
      </div>
      ${srcs}
      ${panel}
    </div>`;
  msgs.appendChild(div);
  scrollDown();
}

function appendErrorMsg(text) {
  const msgs = document.getElementById('messages');
  const div  = document.createElement('div');
  div.className = 'flex justify-start';
  div.innerHTML = `
    <div class="bg-red-900/50 border border-red-700/50 rounded-xl px-4 py-3 text-sm text-red-300 max-w-md">
      ⚠️ ${escHtml(text)}
    </div>`;
  msgs.appendChild(div);
  scrollDown();
}

function appendLoading() {
  const id   = 'ld-' + Date.now();
  const msgs = document.getElementById('messages');
  const div  = document.createElement('div');
  div.id = id;
  div.className = 'flex justify-start';
  div.innerHTML = `
    <div class="bg-slate-700 rounded-2xl rounded-tl-sm px-5 py-4 flex gap-1.5 items-center">
      <span class="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style="animation-delay:0ms"></span>
      <span class="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style="animation-delay:160ms"></span>
      <span class="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style="animation-delay:320ms"></span>
    </div>`;
  msgs.appendChild(div);
  scrollDown();
  return id;
}

function removeLoading(id) {
  document.getElementById(id)?.remove();
}

// ─── Источники и изображения ─────────────────────────────────────────────────

function buildSourceTags(sources) {
  if (!sources?.length) return '';
  // Дедуплицировать — показать уникальные книги
  const books = [...new Set(sources.map(([src]) => src))];
  const tags = books.map(src => `
    <span class="inline-flex items-center gap-1 bg-slate-800 border border-slate-600/70 rounded-full px-2.5 py-0.5 text-xs">
      <span class="text-blue-400">📖</span>
      <span class="text-slate-300 font-medium">${escHtml(shortName(src))}</span>
    </span>`).join('');
  const pages = sources.map(([src, pg]) => `
    <span class="inline-flex items-center gap-1 bg-slate-800/60 border border-slate-700 rounded-full px-2 py-0.5 text-xs text-slate-400">
      стр. ${pg}
    </span>`).join('');
  return `<div class="mt-2 flex flex-wrap gap-1.5 items-center">${tags}${pages}</div>`;
}

function buildSourcesPanel(chunks) {
  if (!chunks?.length) return '';

  const panelId = 'src-' + Date.now();

  const cards = chunks.map((c, i) => {
    const pct   = Math.round((c.score || 0) * 100);
    const color = pct >= 80 ? 'text-green-400' : pct >= 60 ? 'text-yellow-400' : 'text-slate-500';
    const thumb = c.image_url ? `
      <button class="flex-shrink-0 w-28 rounded-lg overflow-hidden border border-slate-600
                     hover:border-blue-400 transition-colors focus:outline-none"
              onclick="openModal('${escHtml(c.image_url)}', '${escHtml(shortName(c.source_file))} · стр. ${c.page}')"
              title="Открыть страницу">
        <img src="${escHtml(c.image_url)}" alt="стр. ${c.page}"
             class="w-full object-contain max-h-36 bg-slate-900">
        <p class="text-xs text-slate-500 text-center py-1">стр. ${c.page}</p>
      </button>` : '';

    return `
      <div class="bg-slate-800/70 border border-slate-700/80 rounded-xl overflow-hidden">
        <div class="flex items-center justify-between px-3 py-2 bg-slate-800 border-b border-slate-700/60">
          <div class="flex items-center gap-2 min-w-0">
            <span class="text-xs font-semibold text-blue-400 truncate" title="${escHtml(c.source_file)}">
              ${escHtml(shortName(c.source_file))}
            </span>
            <span class="text-slate-600">·</span>
            <span class="text-xs text-green-400 flex-shrink-0">стр. ${c.page}</span>
          </div>
          <span class="text-xs ${color} bg-slate-700 rounded-full px-2 py-0.5 flex-shrink-0 ml-2">
            ${pct}%
          </span>
        </div>
        <div class="flex gap-3 p-3">
          <p class="text-xs text-slate-400 flex-1 leading-relaxed" style="overflow:hidden;display:-webkit-box;-webkit-line-clamp:5;-webkit-box-orient:vertical">
            ${escHtml(c.text)}
          </p>
          ${thumb}
        </div>
      </div>`;
  }).join('');

  return `
    <div class="mt-3">
      <button onclick="togglePanel('${panelId}')"
              class="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors select-none">
        <svg id="${panelId}-icon" class="w-3 h-3 transition-transform duration-200"
             viewBox="0 0 20 20" fill="currentColor">
          <path fill-rule="evenodd"
                d="M7.293 4.707a1 1 0 011.414 0L14 10l-5.293 5.293a1 1 0 01-1.414-1.414L11.586 10 6.586 5.707a1 1 0 010-1.414z"
                clip-rule="evenodd"/>
        </svg>
        📚 Источники &nbsp;·&nbsp; ${chunks.length} фрагментов
      </button>
      <div id="${panelId}" class="hidden mt-2 space-y-2">
        ${cards}
      </div>
    </div>`;
}

function togglePanel(id) {
  const el   = document.getElementById(id);
  const icon = document.getElementById(id + '-icon');
  const open = el.classList.toggle('hidden');
  icon.style.transform = open ? '' : 'rotate(90deg)';
  scrollDown();
}

// ─── Модалка изображения ─────────────────────────────────────────────────────

function openModal(url, caption) {
  document.getElementById('modal-img').src        = url;
  document.getElementById('modal-caption').textContent = caption;
  const m = document.getElementById('img-modal');
  m.classList.remove('hidden');
  m.classList.add('flex');
}

function closeModal() {
  const m = document.getElementById('img-modal');
  m.classList.add('hidden');
  m.classList.remove('flex');
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});

// ─── Удаление документа ──────────────────────────────────────────────────────

async function deleteDoc(docId) {
  if (!confirm('Удалить документ и его данные из индекса?')) return;
  try {
    await fetch(`/docs/${docId}/delete/`, {
      method: 'DELETE',
      headers: { 'X-CSRFToken': CSRF },
    });
    if (filterDocId === docId) { filterDocId = null; }
    await refreshDocs();
  } catch (err) {
    showToast(`Ошибка удаления: ${err.message}`, 'error');
  }
}

// ─── Toast уведомления ───────────────────────────────────────────────────────

function showToast(msg, type = 'info') {
  const colors = { success: 'bg-green-800 border-green-600 text-green-200',
                   error:   'bg-red-900 border-red-700 text-red-200',
                   info:    'bg-slate-700 border-slate-600 text-slate-200' };
  const toast = document.createElement('div');
  toast.className = `fixed bottom-6 right-6 z-50 ${colors[type]} border rounded-xl px-4 py-3 text-sm shadow-xl
                     transition-all duration-300 translate-y-0 opacity-100 max-w-sm`;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; toast.style.transform = 'translateY(8px)'; }, 2500);
  setTimeout(() => toast.remove(), 2900);
}

// ─── Инициализация ───────────────────────────────────────────────────────────

(async () => {
  await refreshDocs();

  // Запустить поллинг для документов которые уже в процессе индексации
  const r    = await fetch('/docs/status/');
  const data = await r.json();
  data.documents
    .filter(d => d.status === 'indexing' || d.status === 'pending')
    .forEach(d => startPolling(d.id));
})();
