"""
ingest.py — индексация PDF через VLM API (OpenRouter) + Jina эмбеддинги.

Пайплайн:
  1. PyMuPDF рендерит все страницы → JPG (быстро, CPU)
  2. Параллельные VLM-запросы (MAX_VLM_WORKERS потоков) → текст + диаграммы + аббревиатуры
  3. Jina Embeddings API → ChromaDB
"""

import os
import sys
import json
import time
import random
import base64
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

# Кэш результатов VLM — позволяет продолжить с места остановки
_VLM_CACHE_FILE = Path(__file__).parent / "vlm_cache.json"
_vlm_cache: dict[str, str] = {}
_cache_lock = threading.Lock()

# Глобальный ограничитель запросов: не более 1 запроса в N секунд
_RATE_MIN_INTERVAL = float(os.getenv("VLM_RATE_INTERVAL", "2.0"))
_rate_lock = threading.Lock()
_rate_last_time = [0.0]

def _rate_throttle():
    with _rate_lock:
        now = time.time()
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

# ─── VLM промпт ──────────────────────────────────────────────────────────────

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


# ─── VLM: один запрос на страницу ────────────────────────────────────────────

def _analyze_page_vlm(img_path: str, page_num: int, source_name: str) -> str:
    if img_path in _vlm_cache:
        return _vlm_cache[img_path]

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    prompt = VLM_PROMPT_TEMPLATE.format(
        page_num=page_num, source=source_name, abbr=ABBR_TEXT
    )
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
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": 2000,
        "temperature": 0.1,
    }
    max_attempts = 8
    for attempt in range(max_attempts):
        try:
            _rate_throttle()
            r = requests.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers=headers, json=payload, timeout=120,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            _save_cache(img_path, text)
            return text
        except requests.HTTPError as e:
            code = e.response.status_code if e.response else 0
            if code == 402:
                raise RuntimeError(
                    "OpenRouter: недостаточно кредитов (402). "
                    "Пополните баланс на openrouter.ai/credits"
                ) from e
            if code == 429:
                wait = min((2 ** attempt) * 5 + random.uniform(0, 5), 120)
                print(f"  429 стр.{page_num} попытка {attempt+1}/{max_attempts}, ждём {wait:.0f}с…")
                time.sleep(wait)
                continue
            # другие HTTP-ошибки — подождём и повторим
            wait = 10 + random.uniform(0, 5)
            print(f"  HTTP {code} стр.{page_num} попытка {attempt+1}/{max_attempts}, ждём {wait:.0f}с…")
            time.sleep(wait)
            continue
        except (requests.Timeout, requests.ConnectionError) as e:
            wait = 10 + random.uniform(0, 5)
            print(f"  Network err стр.{page_num} попытка {attempt+1}/{max_attempts}: {e}")
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"  WARN стр.{page_num} попытка {attempt+1}/{max_attempts}: {e}")
            if attempt >= max_attempts - 1:
                break
            time.sleep(5)
            continue
    print(f"  SKIP стр.{page_num}: не удалось обработать после {max_attempts} попыток")
    return ""


# ─── Парсинг VLM-вывода ──────────────────────────────────────────────────────

def _parse_vlm_output(raw: str) -> str:
    """Извлекает секции из структурированного ответа VLM и склеивает в чистый текст."""
    sections: dict[str, list[str]] = {"ТЕКСТ": [], "ДИАГРАММЫ": [], "АББРЕВИАТУРЫ": []}
    current: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if "=== ТЕКСТ ===" in stripped:
            current = "ТЕКСТ"
        elif "=== ДИАГРАММЫ ===" in stripped:
            current = "ДИАГРАММЫ"
        elif "=== АББРЕВИАТУРЫ ===" in stripped:
            current = "АББРЕВИАТУРЫ"
        elif current:
            sections[current].append(line)

    text_body = "\n".join(sections["ТЕКСТ"]).strip()
    diag_body = "\n".join(sections["ДИАГРАММЫ"]).strip()
    abbr_body = "\n".join(sections["АББРЕВИАТУРЫ"]).strip()

    parts = [text_body]
    if diag_body and "отсутствуют" not in diag_body.lower()[:60]:
        parts.append("Визуальные элементы страницы:\n" + diag_body)
    if abbr_body and abbr_body.lower().strip() != "нет":
        parts.append("Аббревиатуры:\n" + abbr_body)

    return "\n\n".join(p for p in parts if p)


# ─── Текст → чанки ───────────────────────────────────────────────────────────

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


# ─── Рендер всех страниц PDF → JPG (быстро, последовательно) ─────────────────

def _render_pages(pdf_path: str, safe_stem: str) -> list[tuple[int, str]]:
    """Возвращает [(page_num, img_path), ...] — рендерит только новые страницы."""
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


# ─── Параллельные VLM-запросы ─────────────────────────────────────────────────

def _vlm_parallel(pages_data: list[tuple[int, str]], source_name: str,
                  doc_id: str, cb=None) -> list[dict]:
    total     = len(pages_data)
    completed = [0]
    lock      = threading.Lock()
    results: dict[int, list[dict]] = {}

    def process_one(args: tuple[int, str]) -> tuple[int, list[dict]]:
        pn, img_path = args
        try:
            if OPENROUTER_API_KEY:
                text = _analyze_page_vlm(img_path, pn, source_name)
            else:
                text = f"[Страница {pn}: OPENROUTER_API_KEY не задан]"
        except RuntimeError:
            raise  # 402 — пробрасываем, нужен явный экшн пользователя
        except Exception as e:
            print(f"  WORKER ERR стр.{pn}: {e}")
            text = ""

        chunks = _make_chunks(text, pn, source_name, doc_id, img_path)

        with lock:
            completed[0] += 1
            done = completed[0]
            if done == 1 or done % 10 == 0 or done == total:
                _log(f"  VLM: {done}/{total} страниц обработано", cb)

        return pn, chunks

    with ThreadPoolExecutor(max_workers=MAX_VLM_WORKERS) as executor:
        for pn, chunks in executor.map(process_one, pages_data):
            results[pn] = chunks

    # Возвращаем в порядке страниц
    all_chunks: list[dict] = []
    for pn in sorted(results):
        all_chunks.extend(results[pn])
    return all_chunks


# ─── Jina эмбеддинги ─────────────────────────────────────────────────────────

def _embed_jina(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    BATCH = 50  # меньший батч — реже 429 от Jina
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
                if attempt >= 5:
                    raise
                time.sleep(10)
    return all_emb


# ─── Полный обход PDF ────────────────────────────────────────────────────────

def extract_pdf_vlm(pdf_path: str, doc_id: str = "", cb=None) -> list[dict]:
    source_name = Path(pdf_path).name
    safe_stem   = doc_id[:8] if doc_id else Path(pdf_path).stem[:20].replace(" ", "_")

    _log(f"  Открыт: {source_name}  (VLM: {VLM_MODEL}, потоков: {MAX_VLM_WORKERS})", cb)

    _log("  [1/2] Рендер страниц → JPG…", cb)
    pages_data = _render_pages(pdf_path, safe_stem)
    total      = len(pages_data)
    _log(f"  Рендер готов: {total} страниц", cb)

    _log(f"  [2/2] Параллельный VLM-анализ ({MAX_VLM_WORKERS} потоков)…", cb)
    t0         = time.time()
    all_chunks = _vlm_parallel(pages_data, source_name, doc_id, cb)
    elapsed    = time.time() - t0

    text_cnt  = sum(1 for c in all_chunks if c["type"] == "text")
    image_cnt = sum(1 for c in all_chunks if c["type"] == "image")
    _log(f"  VLM готов: {text_cnt} текст. чанков, {image_cnt} страниц без текста — {elapsed:.0f}с", cb)
    return all_chunks


# ─── Эмбеддинги + ChromaDB ───────────────────────────────────────────────────

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
        try:
            col.delete(where={"document_id": doc_id})
        except Exception:
            pass

    ids   = [f"{doc_id}_c{i}" for i in range(len(chunks))]
    metas = [{"page": c["page"], "type": c["type"], "image_path": c["image_path"],
               "source_file": c["source_file"], "document_id": c["document_id"]} for c in chunks]

    for start in range(0, len(chunks), 50):
        end = min(start + 50, len(chunks))
        col.add(ids=ids[start:end], documents=texts[start:end],
                embeddings=embeddings[start:end], metadatas=metas[start:end])

    _log(f"  Добавлено {len(chunks)} чанков в ChromaDB.", cb)


# ─── Публичные функции ────────────────────────────────────────────────────────

def ingest_document(pdf_path: str, doc_id: str = "", cb=None) -> dict:
    _log(f"=== Индексация: {Path(pdf_path).name} ===", cb)
    chunks = extract_pdf_vlm(pdf_path, doc_id, cb)
    _log(f"  Запись {len(chunks)} чанков в ChromaDB…", cb)
    _embed_and_store(chunks, doc_id, cb)

    pages = max((c["page"] for c in chunks), default=0)
    saved = sum(1 for c in chunks if c.get("image_path") and Path(c["image_path"]).exists())
    stats = {"text": sum(1 for c in chunks if c["type"] == "text"),
             "images": saved, "pages": pages}
    _log(f"=== Готово: {stats} ===", cb)
    return stats


def ingest_multiple(pdf_paths: list[str]) -> None:
    print(f"\n=== INGESTION: {len(pdf_paths)} файлов ===")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try: client.delete_collection(COLLECTION_NAME)
    except Exception: pass
    client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    for p in pdf_paths:
        ingest_document(p, Path(p).stem)
    print("=== ГОТОВО ===")


def _log(msg: str, cb=None):
    try:
        print(msg)
    except (UnicodeEncodeError, UnicodeDecodeError):
        print(msg.encode("utf-8", errors="replace").decode("ascii", errors="replace"))
    if cb:
        try: cb(msg[:500])
        except Exception: pass


if __name__ == "__main__":
    paths = sys.argv[1:] or [str(p) for p in Path(".").glob("*.pdf")]
    if paths: ingest_multiple(paths)
    else: print("Использование: python ingest.py file1.pdf file2.pdf")
