# -*- coding: utf-8 -*-
"""
test_10pages.py — тест VLM+Jina пайплайна на первых 10 страницах.

Проверяет:
  1. Скорость VLM (параллельные запросы)
  2. Качество OCR русского текста
  3. Обнаружение и описание диаграмм
  4. Расшифровку аббревиатур
  5. Качество поиска через Jina эмбеддинги
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv; load_dotenv()
from pathlib import Path
import chromadb

from config import CHROMA_DIR, JINA_API_KEY, JINA_BASE, EMBED_MODEL, EMBED_DIM, VLM_MODEL
from ingest import _render_pages, _analyze_page_vlm, _make_chunks, _embed_jina
import requests

TEST_COLLECTION = "test_10pages"
PDF = next(Path(".").glob("*.img.pdf"), None)  # первый PDF
TEST_PAGES = 10

print("=" * 65)
print(f"ТЕСТ: первые {TEST_PAGES} страниц из {PDF.name if PDF else '???'}")
print(f"VLM:  {VLM_MODEL}")
print("=" * 65)

if not PDF:
    print("PDF не найден"); sys.exit(1)

# ── 1. Рендер страниц ────────────────────────────────────────────────────────
print("\n[1] Рендер страниц → JPG…")
all_pages = _render_pages(str(PDF), "test10")
pages = all_pages[:TEST_PAGES]
print(f"  Рендер готов: {len(pages)} страниц")

# ── 2. Параллельный VLM-анализ ───────────────────────────────────────────────
print(f"\n[2] VLM-анализ {TEST_PAGES} страниц (параллельно)…")
from concurrent.futures import ThreadPoolExecutor

t0 = time.time()
vlm_results: dict[int, str] = {}

def analyze(args):
    pn, img_path = args
    text = _analyze_page_vlm(img_path, pn, PDF.name)
    return pn, text

with ThreadPoolExecutor(max_workers=5) as ex:
    for pn, text in ex.map(analyze, pages):
        vlm_results[pn] = text
        print(f"  ✓ Страница {pn} ({len(text)} символов)")

elapsed = time.time() - t0
print(f"\n  Время: {elapsed:.1f}с на {TEST_PAGES} страниц = {elapsed/TEST_PAGES:.1f}с/стр")
print(f"  Прогноз для 461 страниц: {elapsed/TEST_PAGES*461/60:.0f} мин")

# ── 3. Качество VLM — показать ключевые страницы ─────────────────────────────
print("\n[3] Качество VLM-вывода:")
print("-" * 65)
for pn in [1, 5, 10]:
    if pn not in vlm_results: continue
    text = vlm_results[pn]
    print(f"\n>>> Страница {pn} (первые 600 символов):")
    print(text[:600])
    print()

# Подсчёт находок
has_diagrams = sum(1 for t in vlm_results.values()
                   if "=== ДИАГРАММЫ ===" in t and "отсутствуют" not in t.lower()[:600].split("=== ДИАГРАММЫ ===")[-1][:200])
has_abbr     = sum(1 for t in vlm_results.values()
                   if "=== АББРЕВИАТУРЫ ===" in t and "нет" not in t.split("=== АББРЕВИАТУРЫ ===")[-1][:100])
print(f"  Страниц с диаграммами: {has_diagrams}/{TEST_PAGES}")
print(f"  Страниц с аббревиатурами: {has_abbr}/{TEST_PAGES}")

# ── 4. Создаём тестовую коллекцию в ChromaDB ─────────────────────────────────
print("\n[4] Создание тестового индекса (ChromaDB)…")
chunks = []
for pn, text in vlm_results.items():
    img_path = str(next(Path("media/images").glob(f"test10_p{pn:04d}.jpg"), ""))
    chunks.extend(_make_chunks(text, pn, PDF.name, "test_doc", img_path))

print(f"  Чанков: {len(chunks)}")

texts = [c["text"] for c in chunks]
embeddings = _embed_jina(texts, task="retrieval.passage")
print(f"  Jina эмбеддинги: {len(embeddings)} векторов × {len(embeddings[0])} dim")

client = chromadb.PersistentClient(path=CHROMA_DIR)
try: client.delete_collection(TEST_COLLECTION)
except: pass
col = client.create_collection(TEST_COLLECTION, metadata={"hnsw:space": "cosine"})
col.add(
    ids=[f"t_c{i}" for i in range(len(chunks))],
    documents=texts,
    embeddings=embeddings,
    metadatas=[{"page": c["page"], "type": c["type"]} for c in chunks],
)
print(f"  Добавлено в ChromaDB: {col.count()} чанков")

# ── 5. Тест поиска ────────────────────────────────────────────────────────────
print("\n[5] Тест поиска по {TEST_PAGES} страницам:")
print("-" * 65)

test_queries = [
    "геология нефть газ углеводороды",
    "ВНК водонефтяной контакт залежь",
    "каустобиолиты органическое вещество",
    "диаграмма схема рисунок",
]

def embed_query(q):
    r = requests.post(f"{JINA_BASE}/embeddings",
        headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": [q], "task": "retrieval.query", "dimensions": EMBED_DIM},
        timeout=20)
    return r.json()["data"][0]["embedding"]

for q in test_queries:
    emb = embed_query(q)
    res = col.query(query_embeddings=[emb], n_results=2,
                    include=["documents", "metadatas", "distances"])
    print(f"\n  Запрос: «{q}»")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = round(1.0 - dist, 3)
        print(f"    [{score:.3f}] стр.{meta['page']} | {doc[:120]}…")

# ── 6. Итог ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ИТОГ ТЕСТА:")
print(f"  Скорость:    {elapsed:.1f}с / {TEST_PAGES} стр = {elapsed/TEST_PAGES:.1f}с на страницу")
print(f"  Прогноз 461: {elapsed/TEST_PAGES*461/60:.0f} мин (при {min(TEST_PAGES,10)} потоках)")
print(f"  OCR:         {'✓ работает' if any(len(t) > 200 for t in vlm_results.values()) else '✗ проблемы'}")
print(f"  Диаграммы:   {has_diagrams}/{TEST_PAGES} страниц с описанием")
print(f"  Аббревиатуры:{has_abbr}/{TEST_PAGES} страниц с расшифровкой")
print(f"  Поиск:       {'✓ ChromaDB + Jina' if col.count() > 0 else '✗ ошибка'}")
print("=" * 65)

# Удаляем тест-коллекцию
client.delete_collection(TEST_COLLECTION)
print("Тестовый индекс удалён. Запусти run_indexing.py для полной индексации.")
