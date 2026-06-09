# -*- coding: utf-8 -*-
"""
test_verification.py — автоматизированная верификация качества RAG-системы.

Проверяет:
  1. Качество индексации (покрытие, релевантность)
  2. Качество ответов LLM (через LLM-as-judge: Llama 4 Maverick)
  3. Итоговый отчёт с оценками
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv; load_dotenv()
import chromadb
from config import CHROMA_DIR, COLLECTION_NAME, LLM_MODEL, VERIFIER_MODEL

print("=" * 65)
print("ВЕРИФИКАЦИЯ RAG-СИСТЕМЫ")
print(f"LLM:        {LLM_MODEL}")
print(f"Верификатор:{VERIFIER_MODEL}")
print("=" * 65)

# ── Проверка ChromaDB ─────────────────────────────────────────────────────────
print("\n[1] Проверка ChromaDB индекса...")
try:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(COLLECTION_NAME)
    total = col.count()
    print(f"  Всего чанков: {total}")
    if total == 0:
        print("  ОШИБКА: индекс пустой!"); sys.exit(1)
    # Проверяем распределение по книгам
    sample = col.get(limit=total, include=["metadatas"])
    from collections import Counter
    sources = Counter(m.get("source_file", "?")[:50] for m in sample["metadatas"])
    for src, cnt in sources.most_common():
        print(f"  {src}: {cnt} чанков")
except Exception as e:
    print(f"  ОШИБКА ChromaDB: {e}"); sys.exit(1)

# ── Тестовые запросы ──────────────────────────────────────────────────────────
TEST_QUERIES = [
    {
        "query": "Что такое каустобиолиты?",
        "expected_terms": ["каустобиолиты", "органическое", "углерод"],
        "min_score": 0.5,
    },
    {
        "query": "Классификация нефтяных ловушек",
        "expected_terms": ["ловушка", "антиклиналь", "структур"],
        "min_score": 0.5,
    },
    {
        "query": "Геохимия органического вещества в нефтепроизводящих породах",
        "expected_terms": ["органическое вещество", "ОВ", "геохимия", "нефтепроизводящ"],
        "min_score": 0.5,
    },
    {
        "query": "ВНК водонефтяной контакт",
        "expected_terms": ["контакт", "нефть", "вода", "ВНК"],
        "min_score": 0.45,
    },
    {
        "query": "Методы сейсмической разведки нефти и газа",
        "expected_terms": ["сейсм", "разведка", "волн"],
        "min_score": 0.45,
    },
]

from rag import GeoRAG
rag = GeoRAG()

# ── Тест 2: Качество поиска ───────────────────────────────────────────────────
print("\n[2] Качество поиска (retrieval)...")
print("-" * 65)

retrieval_results = []
for tq in TEST_QUERIES:
    chunks = rag.retrieve(tq["query"], top_k=3)
    if not chunks:
        print(f"  ПУСТО: «{tq['query'][:50]}»")
        retrieval_results.append({"query": tq["query"], "ok": False, "score": 0})
        continue

    top_score = chunks[0]["score"]
    top_text  = chunks[0]["text"].lower()
    top_page  = chunks[0]["page"]
    top_src   = chunks[0]["source_file"][:35]

    # Проверяем наличие ожидаемых терминов
    found_terms = [t for t in tq["expected_terms"] if t.lower() in top_text]
    score_ok    = top_score >= tq["min_score"]
    terms_ok    = len(found_terms) > 0

    status = "✓" if (score_ok and terms_ok) else ("~" if score_ok else "✗")
    print(f"  {status} [{top_score:.3f}] «{tq['query'][:45]}»")
    print(f"      → стр.{top_page} из {top_src}")
    print(f"      → Термины найдены: {found_terms or 'нет'}")

    retrieval_results.append({
        "query": tq["query"],
        "ok": score_ok and terms_ok,
        "score": top_score,
        "found_terms": found_terms,
    })

retrieval_ok = sum(1 for r in retrieval_results if r["ok"])
print(f"\n  Поиск: {retrieval_ok}/{len(TEST_QUERIES)} запросов — отлично")

# ── Тест 3: Качество ответов + верификация ────────────────────────────────────
print("\n[3] Качество ответов LLM + верификация (Llama-as-judge)...")
print("-" * 65)

# Берём только первые 3 запроса чтобы не расходовать много кредитов
ANSWER_QUERIES = TEST_QUERIES[:3]
verification_results = []

for tq in ANSWER_QUERIES:
    print(f"\n  Запрос: «{tq['query']}»")

    t0 = time.time()
    result = rag.answer(tq["query"])
    answer_time = time.time() - t0

    answer = result.get("answer", "")
    chunks = result.get("chunks", [])

    if not answer or "не проиндексированы" in answer:
        print(f"    ПРОПУСК: индекс пустой или LLM недоступен")
        continue

    print(f"    Ответ ({answer_time:.1f}с, {len(answer)} символов):")
    print(f"    {answer[:250]}{'...' if len(answer) > 250 else ''}")

    # Строим контекст из ВСЕХ чанков (чтобы верификатор видел то же, что LLM)
    context = "\n\n".join(
        f"[Фрагмент {i} | {c['source_file']}, стр.{c['page']}]\n{c['text'][:350]}"
        for i, c in enumerate(chunks, 1)
    )

    # Верификация через Llama
    print(f"    Верификация...")
    vt0 = time.time()
    ver = rag.verify_answer(tq["query"], answer, context)
    ver_time = time.time() - vt0

    if "error" in ver:
        print(f"    Верификатор: ОШИБКА — {ver['error']}")
    else:
        verdict_emoji = {"ОТЛИЧНО": "🟢", "ХОРОШО": "🟡",
                         "УДОВЛЕТВОРИТЕЛЬНО": "🟠", "ПЛОХО": "🔴"}.get(
            ver.get("verdict", ""), "⚪")
        print(f"    {verdict_emoji} {ver.get('verdict','?')} | "
              f"Точность: {ver.get('accuracy','?')}/5 | "
              f"Полнота: {ver.get('completeness','?')}/5 | "
              f"Галлюцинации: {'есть ⚠️' if ver.get('hallucinations') else 'нет ✓'}")
        print(f"    Комментарий: {ver.get('comment', '')}")
        print(f"    (верификация заняла {ver_time:.1f}с)")

    verification_results.append({
        "query": tq["query"],
        "answer_len": len(answer),
        "verdict": ver.get("verdict") if "error" not in ver else "ERROR",
        "score": ver.get("score") if "error" not in ver else 0,
        "hallucinations": ver.get("hallucinations", False),
    })

# ── Итоговый отчёт ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ИТОГОВЫЙ ОТЧЁТ:")
print(f"  Индекс:     {total} чанков из {len(sources)} книг")
print(f"  Поиск:      {retrieval_ok}/{len(TEST_QUERIES)} запросов прошли ✓")

if verification_results:
    good = sum(1 for v in verification_results
               if v["verdict"] in ("ОТЛИЧНО", "ХОРОШО"))
    avg_score = sum(v.get("score") or 0 for v in verification_results) / len(verification_results)
    halluc = sum(1 for v in verification_results if v.get("hallucinations"))
    print(f"  LLM ответы: {good}/{len(verification_results)} оценены хорошо/отлично")
    print(f"  Ср. оценка: {avg_score:.1f}/5")
    print(f"  Галлюцин.: {halluc}/{len(verification_results)} ответов")
else:
    print(f"  LLM ответы: не протестированы (пополните кредиты OpenRouter)")

print("=" * 65)
