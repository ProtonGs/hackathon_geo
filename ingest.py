"""
ingest.py — индексация документов: PDF (текст/скан/гибрид), DOCX, DJVU.

Пайплайн:
  1. detect_source_type()  — определяет тип и выбирает стратегию
  2. Стратегия «direct»    — PyMuPDF прямое извлечение текста (текстовые PDF, DOCX)
  3. Стратегия «vlm»       — рендер страниц → VLM OCR (сканы, DJVU)
  4. Стратегия «hybrid»    — прямой текст для насыщенных страниц + VLM для рисунков
  5. Jina эмбеддинги → ChromaDB
  6. Перестройка графа знаний
"""

import os
import sys
import json
import time
import random
import base64
import shutil
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import fitz
import requests
import chromadb
from PIL import Image

from config import (
    IMAGES_DIR, CHROMA_DIR, COLLECTION_NAME,
    OPENROUTER_API_KEY, OPENROUTER_BASE, VLM_MODEL,
    JINA_API_KEY, JINA_BASE, EMBED_MODEL, EMBED_DIM,
    OCR_DPI, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LEN,
)
from abbreviations import ABBR_TEXT

MAX_VLM_WORKERS = int(os.getenv("VLM_WORKERS", "2"))

# Порог символов на страницу: выше → прямое извлечение
TEXT_PAGE_THRESHOLD   = 150   # chars: страница считается «текстовой»
HYBRID_RATIO_THRESHOLD = 0.35  # если >35% страниц текстовые → гибрид/прямая

_DOC_TYPE_LABELS = {
    "text_pdf":   "📄 Текстовый PDF",
    "scan_pdf":   "🖼 Скан-PDF",
    "hybrid_pdf": "📄🖼 Смешанный PDF",
    "docx":       "📝 DOCX",
    "doc":        "📝 DOC",
    "djvu":       "📚 DJVU",
    "unknown":    "❓ Неизвестный формат",
}
_STRATEGY_LABELS = {
    "direct":    "прямое извлечение текста (быстро, без VLM)",
    "vlm":       "VLM OCR пайплайн (полный анализ страниц)",
    "hybrid":    "гибрид: текст напрямую + VLM для рисунков",
    "text_only": "извлечение текста",
}

# ─── VLM кэш ─────────────────────────────────────────────────────────────────

_VLM_CACHE_FILE = Path(__file__).parent / "vlm_cache.json"
_vlm_cache: dict[str, str] = {}
_cache_lock = threading.Lock()

_RATE_MIN_INTERVAL = float(os.getenv("VLM_RATE_INTERVAL", "2.0"))
_rate_lock = threading.Lock()
_rate_last_time = [0.0]


def _rate_throttle():
    with _rate_lock:
        now  = time.time()
        wait = _RATE_MIN_INTERVAL - (now - _rate_last_time[0])
        if wait > 0:
            time.sleep(wait)
        _rate_last_time[0] = time.time()


def _load_cache():
    global _vlm_cache
    if _VLM_CACHE_FILE.exists():
        with open(_VLM_CACHE_FILE, encoding="utf-8") as f:
            _vlm_cache = json.load(f)
        print(f"  Кэш VLM загружен: {len(_vlm_cache)} страниц")


def _save_cache(img_path: str, text: str):
    with _cache_lock:
        _vlm_cache[img_path] = text
        with open(_VLM_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_vlm_cache, f, ensure_ascii=False)


_load_cache()


# ═══════════════════════════════════════════════════════════════════════════════
#  1. ДЕТЕКТОР ТИПА ДОКУМЕНТА
# ═══════════════════════════════════════════════════════════════════════════════

def detect_source_type(file_path: str) -> dict:
    """
    Определяет тип документа и рекомендует стратегию обработки.

    Возвращает:
        doc_type:       'text_pdf' | 'scan_pdf' | 'hybrid_pdf' | 'docx' | 'doc' | 'djvu' | 'unknown'
        strategy:       'direct' | 'vlm' | 'hybrid' | 'text_only'
        extension:      расширение файла
        text_coverage:  доля текстовых страниц (0.0–1.0), для PDF
        avg_chars:      среднее число символов на странице
        sample_pages:   сколько страниц проверено
        details:        строка-объяснение для лога
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    # ── DOCX ──────────────────────────────────────────────────────────────────
    if ext in (".docx", ".doc"):
        return {
            "doc_type":      ext.lstrip("."),
            "strategy":      "text_only",
            "extension":     ext,
            "text_coverage": 1.0,
            "avg_chars":     0,
            "sample_pages":  0,
            "details":       "DOCX: прямое извлечение текста через python-docx",
        }

    # ── DJVU ──────────────────────────────────────────────────────────────────
    if ext == ".djvu":
        has_djvutxt = shutil.which("djvutxt") is not None
        return {
            "doc_type":      "djvu",
            "strategy":      "direct" if has_djvutxt else "vlm",
            "extension":     ext,
            "text_coverage": 0.0,
            "avg_chars":     0,
            "sample_pages":  0,
            "details": (
                "DJVU: djvutxt найден — прямое извлечение текста"
                if has_djvutxt else
                "DJVU: djvutxt не найден — рендер страниц + VLM OCR"
            ),
        }

    # ── PDF ───────────────────────────────────────────────────────────────────
    if ext != ".pdf":
        return {
            "doc_type":      "unknown",
            "strategy":      "vlm",
            "extension":     ext,
            "text_coverage": 0.0,
            "avg_chars":     0,
            "sample_pages":  0,
            "details":       f"Неизвестный формат '{ext}' — попытка VLM",
        }

    try:
        doc     = fitz.open(file_path)
        total   = len(doc)
        step    = max(1, total // 10)
        indices = list(range(0, total, step))[:10]

        char_counts = []
        for i in indices:
            text = doc[i].get_text("text").strip()
            char_counts.append(len(text))
        doc.close()

        avg_chars      = sum(char_counts) / len(char_counts) if char_counts else 0
        text_pages     = sum(1 for c in char_counts if c >= TEXT_PAGE_THRESHOLD)
        text_coverage  = text_pages / len(char_counts) if char_counts else 0.0

        if text_coverage >= 0.70:
            doc_type = "text_pdf"
            strategy = "hybrid"   # прямой текст + VLM для страниц с рисунками
            details  = (
                f"Текстовый PDF: {text_coverage:.0%} страниц с текстом "
                f"(ср. {avg_chars:.0f} симв.) → гибридная стратегия"
            )
        elif text_coverage >= HYBRID_RATIO_THRESHOLD:
            doc_type = "hybrid_pdf"
            strategy = "hybrid"
            details  = (
                f"Смешанный PDF: {text_coverage:.0%} страниц с текстом "
                f"(ср. {avg_chars:.0f} симв.) → гибридная стратегия"
            )
        else:
            doc_type = "scan_pdf"
            strategy = "vlm"
            details  = (
                f"Скан-PDF: {text_coverage:.0%} страниц с текстом "
                f"(ср. {avg_chars:.0f} симв.) → полный VLM OCR"
            )

        return {
            "doc_type":      doc_type,
            "strategy":      strategy,
            "extension":     ext,
            "text_coverage": round(text_coverage, 3),
            "avg_chars":     round(avg_chars, 1),
            "sample_pages":  len(char_counts),
            "details":       details,
        }

    except Exception as exc:
        return {
            "doc_type":      "scan_pdf",
            "strategy":      "vlm",
            "extension":     ext,
            "text_coverage": 0.0,
            "avg_chars":     0,
            "sample_pages":  0,
            "details":       f"Ошибка анализа PDF ({exc}) — используем VLM",
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  2. VLM OCR ПАЙПЛАЙН (скан-PDF)
# ═══════════════════════════════════════════════════════════════════════════════

VLM_PROMPT_TEMPLATE = """\
Анализируй страницу {page_num} из геологической книги «{source}».

Выполни три задачи:

1. OCR — извлеки ВЕСЬ текст со страницы дословно, сохраняя структуру \
(заголовки, абзацы, нумерованные списки, формулы, подписи к рисункам).

2. Визуальные элементы — если на странице есть схемы, карты, графики, \
разрезы, диаграммы, таблицы — опиши каждый элемент:
   • Тип: сейсмический разрез / структурная карта / каротажная диаграмма / \
стратиграфическая колонка / палеогеографическая карта / график / таблица / другое
   • Что изображено: горизонты, тектонические структуры, залежи УВ, скважины
   • Подписи осей, легенды, условные обозначения, единицы измерения

3. Аббревиатуры — расшифруй ВСЕ аббревиатуры встреченные на странице.
   Геологический словарь аббревиатур:
{abbr}

Отвечай строго в формате:
=== ТЕКСТ ===
[весь извлечённый текст]

=== ДИАГРАММЫ ===
[описание каждого визуального элемента, или "отсутствуют"]

=== АББРЕВИАТУРЫ ===
[список встреченных: ABBR — расшифровка, или "нет"]"""


def _analyze_page_vlm(img_path: str, page_num: int, source_name: str) -> str:
    if img_path in _vlm_cache:
        return _vlm_cache[img_path]

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    prompt  = VLM_PROMPT_TEMPLATE.format(page_num=page_num, source=source_name, abbr=ABBR_TEXT)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost:8000",
        "X-Title":       "Geo RAG",
    }
    payload = {
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text",      "text": prompt},
        ]}],
        "max_tokens": 2000, "temperature": 0.1,
    }
    for attempt in range(8):
        try:
            _rate_throttle()
            r = requests.post(f"{OPENROUTER_BASE}/chat/completions",
                              headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            _save_cache(img_path, text)
            return text
        except requests.HTTPError as e:
            code = e.response.status_code if e.response else 0
            if code == 402:
                raise RuntimeError("OpenRouter: недостаточно кредитов (402). Пополните баланс.") from e
            wait = min((2 ** attempt) * 5 + random.uniform(0, 5), 120)
            print(f"  HTTP {code} стр.{page_num} попытка {attempt+1}/8, ждём {wait:.0f}с…")
            time.sleep(wait)
        except (requests.Timeout, requests.ConnectionError) as e:
            print(f"  Network err стр.{page_num}: {e}")
            time.sleep(10 + random.uniform(0, 5))
        except Exception as e:
            print(f"  WARN стр.{page_num}: {e}")
            time.sleep(5)
    print(f"  SKIP стр.{page_num}: не удалось обработать")
    return ""


def _parse_vlm_output(raw: str) -> str:
    sections: dict[str, list[str]] = {"ТЕКСТ": [], "ДИАГРАММЫ": [], "АББРЕВИАТУРЫ": []}
    current: str | None = None
    for line in raw.splitlines():
        s = line.strip()
        if "=== ТЕКСТ ===" in s:            current = "ТЕКСТ"
        elif "=== ДИАГРАММЫ ===" in s:      current = "ДИАГРАММЫ"
        elif "=== АББРЕВИАТУРЫ ===" in s:   current = "АББРЕВИАТУРЫ"
        elif current:                        sections[current].append(line)

    text_body = "\n".join(sections["ТЕКСТ"]).strip()
    diag_body = "\n".join(sections["ДИАГРАММЫ"]).strip()
    abbr_body = "\n".join(sections["АББРЕВИАТУРЫ"]).strip()

    parts = [text_body]
    if diag_body and "отсутствуют" not in diag_body.lower()[:60]:
        parts.append("Визуальные элементы страницы:\n" + diag_body)
    if abbr_body and abbr_body.lower().strip() != "нет":
        parts.append("Аббревиатуры:\n" + abbr_body)
    return "\n\n".join(p for p in parts if p)


def _make_chunks(raw_text: str, page_num: int, source_name: str,
                 doc_id: str, img_path: str) -> list[dict]:
    text = _parse_vlm_output(raw_text) if "=== ТЕКСТ ===" in raw_text else raw_text
    if len(text) < MIN_CHUNK_LEN:
        return [{"text": f"[Страница {page_num} из {source_name}]",
                 "page": page_num, "source_file": source_name,
                 "document_id": doc_id, "type": "image", "image_path": img_path}]

    words = text.split()
    step  = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    chunks = []
    for i in range(0, len(words), step):
        body = " ".join(words[i:i + CHUNK_SIZE])
        if len(body) >= MIN_CHUNK_LEN:
            chunks.append({"text": body, "page": page_num,
                            "source_file": source_name, "document_id": doc_id,
                            "type": "text", "image_path": img_path})
    return chunks


def _render_pages(pdf_path: str, safe_stem: str) -> list[tuple[int, str]]:
    """Рендерит все страницы PDF → JPG. Пропускает уже готовые."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    doc   = fitz.open(pdf_path)
    pages = []
    for idx, page in enumerate(doc):
        pn       = idx + 1
        img_path = IMAGES_DIR / f"{safe_stem}_p{pn:04d}.jpg"
        if not img_path.exists():
            scale = OCR_DPI / 72.0
            pix   = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB)
            pil   = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pil.save(img_path, "JPEG", quality=75, optimize=True)
        pages.append((pn, str(img_path)))
    doc.close()
    return pages


def _vlm_parallel(pages_data: list[tuple[int, str]], source_name: str,
                  doc_id: str, cb=None) -> list[dict]:
    total     = len(pages_data)
    completed = [0]
    lock      = threading.Lock()
    results: dict[int, list[dict]] = {}

    def process_one(args: tuple[int, str]) -> tuple[int, list[dict]]:
        pn, img_path = args
        try:
            text = _analyze_page_vlm(img_path, pn, source_name) if OPENROUTER_API_KEY else \
                   f"[Страница {pn}: OPENROUTER_API_KEY не задан]"
        except RuntimeError:
            raise
        except Exception as e:
            print(f"  WORKER ERR стр.{pn}: {e}"); text = ""

        chunks = _make_chunks(text, pn, source_name, doc_id, img_path)
        with lock:
            completed[0] += 1
            done = completed[0]
            if done == 1 or done % 10 == 0 or done == total:
                _log(f"  VLM: {done}/{total} стр. обработано", cb)
        return pn, chunks

    with ThreadPoolExecutor(max_workers=MAX_VLM_WORKERS) as ex:
        for pn, chunks in ex.map(process_one, pages_data):
            results[pn] = chunks

    all_chunks: list[dict] = []
    for pn in sorted(results):
        all_chunks.extend(results[pn])
    return all_chunks


def extract_pdf_vlm(pdf_path: str, doc_id: str = "", cb=None) -> list[dict]:
    """Стратегия 'vlm': рендер всех страниц → VLM (для сканов)."""
    source_name = Path(pdf_path).name
    safe_stem   = doc_id[:8] if doc_id else Path(pdf_path).stem[:20].replace(" ", "_")

    _log(f"  Открыт: {source_name}  (VLM: {VLM_MODEL}, потоков: {MAX_VLM_WORKERS})", cb)
    _log("  [1/2] Рендер страниц → JPG…", cb)
    pages_data = _render_pages(pdf_path, safe_stem)
    _log(f"  Рендер готов: {len(pages_data)} страниц", cb)

    _log(f"  [2/2] Параллельный VLM-анализ ({MAX_VLM_WORKERS} потоков)…", cb)
    t0         = time.time()
    all_chunks = _vlm_parallel(pages_data, source_name, doc_id, cb)
    elapsed    = time.time() - t0

    text_cnt  = sum(1 for c in all_chunks if c["type"] == "text")
    image_cnt = sum(1 for c in all_chunks if c["type"] == "image")
    _log(f"  VLM готов: {text_cnt} текст. чанков, {image_cnt} без текста — {elapsed:.0f}с", cb)
    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  3. ГИБРИДНАЯ СТРАТЕГИЯ (текстовый/смешанный PDF)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf_hybrid(pdf_path: str, doc_id: str = "", cb=None) -> list[dict]:
    """
    Стратегия 'hybrid':
    - страницы с текстом ≥ TEXT_PAGE_THRESHOLD символов → PyMuPDF прямое извлечение
    - остальные страницы (рисунки, сканированные вставки) → VLM
    Рендерит все страницы в JPG для единообразного отображения в UI.
    """
    source_name = Path(pdf_path).name
    safe_stem   = doc_id[:8] if doc_id else Path(pdf_path).stem[:20].replace(" ", "_")

    _log(f"  Открыт: {source_name}  (гибрид: текст + VLM для рисунков)", cb)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    doc         = fitz.open(pdf_path)
    total       = len(doc)
    direct_data: list[tuple[int, str, str]] = []  # (page_num, img_path, text)
    vlm_data:   list[tuple[int, str]]       = []  # (page_num, img_path)

    _log(f"  [1/3] Анализ и рендер {total} страниц…", cb)
    for idx, page in enumerate(doc):
        pn       = idx + 1
        img_path = str(IMAGES_DIR / f"{safe_stem}_p{pn:04d}.jpg")

        # Рендер (всегда, для отображения в UI)
        if not Path(img_path).exists():
            scale = OCR_DPI / 72.0
            pix   = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB)
            pil   = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pil.save(img_path, "JPEG", quality=75, optimize=True)

        # Определяем стратегию для страницы
        raw_text = page.get_text("text").strip()
        if len(raw_text) >= TEXT_PAGE_THRESHOLD:
            direct_data.append((pn, img_path, raw_text))
        else:
            vlm_data.append((pn, img_path))

    doc.close()
    _log(f"  Прямое извлечение: {len(direct_data)} стр.  VLM: {len(vlm_data)} стр.", cb)

    # ── Прямое извлечение ─────────────────────────────────────────────────────
    _log("  [2/3] Прямое извлечение текстовых страниц…", cb)
    all_chunks: list[dict] = []
    for pn, img_path, raw_text in direct_data:
        chunks = _make_chunks(raw_text, pn, source_name, doc_id, img_path)
        all_chunks.extend(chunks)
    _log(f"  Прямое: {len(all_chunks)} чанков", cb)

    # ── VLM для страниц с рисунками ───────────────────────────────────────────
    if vlm_data:
        _log(f"  [3/3] VLM-анализ {len(vlm_data)} страниц с рисунками…", cb)
        vlm_chunks = _vlm_parallel(vlm_data, source_name, doc_id, cb)
        all_chunks.extend(vlm_chunks)
    else:
        _log("  [3/3] VLM: нет страниц с рисунками — пропускаем", cb)

    all_chunks.sort(key=lambda c: c["page"])

    text_cnt  = sum(1 for c in all_chunks if c["type"] == "text")
    image_cnt = sum(1 for c in all_chunks if c["type"] == "image")
    _log(f"  Гибрид готов: {text_cnt} текст. чанков, {image_cnt} изображений", cb)
    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  4. DOCX СТРАТЕГИЯ
# ═══════════════════════════════════════════════════════════════════════════════

def extract_docx(docx_path: str, doc_id: str = "", cb=None) -> list[dict]:
    """Стратегия 'text_only': python-docx → чанки (без рендера страниц)."""
    try:
        from docx import Document as DocxDoc
    except ImportError:
        _log("  ОШИБКА: pip install python-docx", cb)
        return []

    source_name = Path(docx_path).name
    _log(f"  DOCX: {source_name}", cb)

    doc    = DocxDoc(docx_path)
    blocks: list[str] = []

    # Параграфы
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            blocks.append(t)

    # Таблицы
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
            if row_text:
                blocks.append(row_text)

    full_text = "\n".join(blocks)
    _log(f"  DOCX: {len(blocks)} блоков, {len(full_text)} символов", cb)

    if not full_text.strip():
        _log("  DOCX: документ пустой или защищён", cb)
        return []

    WORDS_PER_FAKE_PAGE = 300
    words  = full_text.split()
    step   = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    chunks = []
    for i in range(0, len(words), step):
        body = " ".join(words[i:i + CHUNK_SIZE])
        if len(body) < MIN_CHUNK_LEN:
            continue
        page_num = (i // WORDS_PER_FAKE_PAGE) + 1
        chunks.append({
            "text":        body,
            "page":        page_num,
            "source_file": source_name,
            "document_id": doc_id,
            "type":        "text",
            "image_path":  "",
        })

    _log(f"  DOCX готов: {len(chunks)} чанков", cb)
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  5. DJVU СТРАТЕГИЯ
# ═══════════════════════════════════════════════════════════════════════════════

def extract_djvu(djvu_path: str, doc_id: str = "", cb=None) -> list[dict]:
    """Стратегия для DJVU: djvutxt (если есть) → иначе рендер страниц + VLM."""
    source_name = Path(djvu_path).name
    _log(f"  DJVU: {source_name}", cb)

    # ── Вариант А: djvutxt ─────────────────────────────────────────────────────
    if shutil.which("djvutxt"):
        _log("  DJVU: используем djvutxt для извлечения текста…", cb)
        try:
            result = subprocess.run(
                ["djvutxt", djvu_path],
                capture_output=True, text=True, encoding="utf-8", timeout=120,
            )
            if result.returncode == 0 and result.stdout.strip():
                text  = result.stdout
                words = text.split()
                step  = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
                chunks = []
                WORDS_PER_PAGE = 300
                for i in range(0, len(words), step):
                    body = " ".join(words[i:i + CHUNK_SIZE])
                    if len(body) >= MIN_CHUNK_LEN:
                        chunks.append({
                            "text":        body,
                            "page":        (i // WORDS_PER_PAGE) + 1,
                            "source_file": source_name,
                            "document_id": doc_id,
                            "type":        "text",
                            "image_path":  "",
                        })
                _log(f"  DJVU (djvutxt): {len(chunks)} чанков", cb)
                return chunks
        except Exception as e:
            _log(f"  djvutxt ошибка: {e}", cb)

    # ── Вариант Б: конвертация ddjvu → PDF → VLM ─────────────────────────────
    if shutil.which("ddjvu"):
        _log("  DJVU: конвертация в PDF через ddjvu…", cb)
        pdf_tmp = Path(djvu_path).with_suffix(".tmp.pdf")
        try:
            result = subprocess.run(
                ["ddjvu", "-format=pdf", djvu_path, str(pdf_tmp)],
                capture_output=True, timeout=180,
            )
            if result.returncode == 0 and pdf_tmp.exists():
                _log("  DJVU: конвертирован в PDF, запуск VLM пайплайна…", cb)
                chunks = extract_pdf_vlm(str(pdf_tmp), doc_id, cb)
                pdf_tmp.unlink(missing_ok=True)
                return chunks
        except Exception as e:
            _log(f"  ddjvu ошибка: {e}", cb)
        finally:
            pdf_tmp.unlink(missing_ok=True)

    # ── Вариант В: нет инструментов ───────────────────────────────────────────
    _log("  ⚠️  DJVU: нет djvutxt/ddjvu.", cb)
    _log("  Установите: brew install djvulibre (Mac) / apt install djvulibre-bin (Linux)", cb)
    _log("  Или конвертируйте вручную: ddjvu -format=pdf input.djvu output.pdf", cb)
    return []


# ═══════════════════════════════════════════════════════════════════════════════
#  6. JINA + CHROMADB
# ═══════════════════════════════════════════════════════════════════════════════

def _embed_jina(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    BATCH   = 50
    all_emb: list[list[float]] = []
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        for attempt in range(6):
            try:
                r = requests.post(
                    f"{JINA_BASE}/embeddings",
                    headers=headers,
                    json={"model": EMBED_MODEL, "input": batch,
                          "task": task, "dimensions": EMBED_DIM},
                    timeout=60,
                )
                r.raise_for_status()
                all_emb.extend(d["embedding"] for d in r.json()["data"])
                break
            except requests.HTTPError as e:
                code = e.response.status_code if e.response else 0
                if code == 429:
                    wait = min((2 ** attempt) * 10 + random.uniform(0, 5), 120)
                    print(f"  Jina 429 батч {start//BATCH+1}, ждём {wait:.0f}с…")
                    time.sleep(wait)
                    continue
                raise
            except requests.Timeout:
                if attempt >= 5: raise
                time.sleep(10)
    return all_emb


def _embed_and_store(chunks: list[dict], doc_id: str, cb=None):
    if not chunks:
        _log("  Нет данных для индексации.", cb); return

    _log(f"  Jina эмбеддинги: {len(chunks)} чанков…", cb)
    texts      = [c["text"] for c in chunks]
    embeddings = _embed_jina(texts)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        col = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    if doc_id:
        try: col.delete(where={"document_id": doc_id})
        except Exception: pass

    ids   = [f"{doc_id}_c{i}" for i in range(len(chunks))]
    metas = [{"page":        c["page"],
               "type":        c["type"],
               "image_path":  c.get("image_path", ""),
               "source_file": c["source_file"],
               "document_id": c["document_id"]} for c in chunks]

    for start in range(0, len(chunks), 50):
        end = min(start + 50, len(chunks))
        col.add(ids=ids[start:end], documents=texts[start:end],
                embeddings=embeddings[start:end], metadatas=metas[start:end])

    _log(f"  ChromaDB: добавлено {len(chunks)} чанков.", cb)

    _log("  Граф знаний: перестройка…", cb)
    try:
        from graph import rebuild_graph_from_chroma, invalidate_graph
        invalidate_graph()
        stats = rebuild_graph_from_chroma()
        _log(f"  Граф готов: {stats.get('nodes', 0)} узлов, {stats.get('edges', 0)} рёбер", cb)
    except Exception as e:
        _log(f"  Граф (warn): {e}", cb)


# ═══════════════════════════════════════════════════════════════════════════════
#  7. ПУБЛИЧНЫЙ ИНТЕРФЕЙС
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_document(file_path: str, doc_id: str = "", cb=None) -> dict:
    """
    Основная точка входа. Определяет тип документа, выбирает стратегию,
    индексирует и сохраняет в ChromaDB.
    """
    path = Path(file_path)
    _log(f"=== Индексация: {path.name} ===", cb)

    # ── Детекция типа ──────────────────────────────────────────────────────────
    _log("  🔍 Определение типа документа…", cb)
    doc_info = detect_source_type(file_path)
    label    = _DOC_TYPE_LABELS.get(doc_info["doc_type"], doc_info["doc_type"])
    strategy = _STRATEGY_LABELS.get(doc_info["strategy"], doc_info["strategy"])
    _log(f"  {label}  →  {strategy}", cb)
    _log(f"  {doc_info['details']}", cb)

    # ── Извлечение по стратегии ────────────────────────────────────────────────
    ext = path.suffix.lower()
    if ext in (".docx", ".doc"):
        chunks = extract_docx(file_path, doc_id, cb)
    elif ext == ".djvu":
        chunks = extract_djvu(file_path, doc_id, cb)
    elif ext == ".pdf":
        if doc_info["strategy"] == "vlm":
            chunks = extract_pdf_vlm(file_path, doc_id, cb)
        else:
            chunks = extract_pdf_hybrid(file_path, doc_id, cb)
    else:
        _log(f"  Неизвестный формат '{ext}', пробуем VLM…", cb)
        chunks = extract_pdf_vlm(file_path, doc_id, cb)

    if not chunks:
        _log("  ⚠️  Чанки не извлечены — возможно файл пуст или не поддерживается.", cb)
        return {"text": 0, "images": 0, "pages": 0,
                "doc_type": doc_info["doc_type"], "strategy": doc_info["strategy"]}

    # ── Эмбеддинги + ChromaDB ──────────────────────────────────────────────────
    _log(f"  Запись {len(chunks)} чанков в ChromaDB…", cb)
    _embed_and_store(chunks, doc_id, cb)

    pages = max((c["page"] for c in chunks), default=0)
    saved = sum(1 for c in chunks if c.get("image_path") and Path(c["image_path"]).exists())
    stats = {
        "text":     sum(1 for c in chunks if c["type"] == "text"),
        "images":   saved,
        "pages":    pages,
        "doc_type": doc_info["doc_type"],
        "strategy": doc_info["strategy"],
    }
    _log(f"=== Готово: тип={doc_info['doc_type']}, стратегия={doc_info['strategy']}, {stats} ===", cb)
    return stats


def ingest_multiple(file_paths: list[str]) -> None:
    print(f"\n=== INGESTION: {len(file_paths)} файлов ===")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:   client.delete_collection(COLLECTION_NAME)
    except Exception: pass
    client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    for p in file_paths:
        ingest_document(p, Path(p).stem)
    print("=== ГОТОВО ===")


def _log(msg: str, cb=None):
    try:    print(msg)
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(msg.encode("utf-8", errors="replace").decode("ascii", errors="replace"))
    if cb:
        try: cb(msg[:500])
        except Exception: pass


if __name__ == "__main__":
    paths = sys.argv[1:] or [str(p) for p in Path(".").glob("*.pdf")]
    if paths: ingest_multiple(paths)
    else: print("Использование: python ingest.py file1.pdf file2.pdf file.docx file.djvu")
