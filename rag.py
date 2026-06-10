"""
rag.py — поиск и генерация ответов.

Режимы поиска:
  rag      — чистый векторный поиск (Jina + ChromaDB)
  hybrid   — векторный + BM25 с Reciprocal Rank Fusion (RRF)
  graphrag — граф сущностей → страницы → векторный поиск по тем страницам

Эмбеддинги: Jina AI API (jina-embeddings-v3)
LLM:        OpenRouter — open-weight модели
"""

import re
import requests
from pathlib import Path, PureWindowsPath

import chromadb

from config import IMAGES_DIR


def _resolve_image_path(stored_path: str) -> Path | None:
    """
    Resolve a stored image path to a local Path that actually exists.
    Handles Windows absolute paths stored from another machine by extracting
    just the filename and looking for it in the local IMAGES_DIR.
    """
    if not stored_path:
        return None

    # Try as-is first (local relative path)
    p = Path(stored_path)
    if p.exists():
        return p

    # Extract filename from stored path (may be a Windows or Unix absolute path)
    try:
        filename = PureWindowsPath(stored_path).name
    except Exception:
        filename = Path(stored_path).name

    if not filename:
        return None

    local = Path(IMAGES_DIR) / filename
    if local.exists():
        return local

    return None

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

from config import (
    CHROMA_DIR, COLLECTION_NAME,
    OPENROUTER_API_KEY, OPENROUTER_BASE, LLM_MODEL, VERIFIER_MODEL,
    JINA_API_KEY, JINA_BASE, EMBED_MODEL, EMBED_DIM,
    TOP_K,
)
from abbreviations import GEO_ABBREVIATIONS

SYSTEM_PROMPT = """Ты — эксперт-геолог. Отвечай ТОЛЬКО по фрагментам из книг, приведённым ниже в разделе КОНТЕКСТ.

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
1. Используй ИСКЛЮЧИТЕЛЬНО информацию из раздела КОНТЕКСТ. Никаких фактов из собственных знаний.
2. Для ссылок используй формат [doc_id:страница] — именно такой, как указано перед каждым фрагментом.
3. Если нужной информации нет в контексте — пиши: «В предоставленных фрагментах данная информация отсутствует.»
4. Не добавляй авторов, даты, формулы или цифры, которых нет в контексте.
5. Расшифровывай аббревиатуры при первом упоминании."""


class GeoRAG:
    def __init__(self) -> None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            self._col = client.get_collection(COLLECTION_NAME)
            print(f"RAG готов. Чанков в индексе: {self._col.count()}")
        except Exception:
            self._col = None
            print("RAG: коллекция пустая, загрузите документы.")

        # BM25 index (rebuild from ChromaDB)
        self._bm25        = None
        self._bm25_ids:   list[str]  = []
        self._bm25_metas: list[dict] = []
        self._bm25_texts: list[str]  = []
        if HAS_BM25 and self._col is not None and self._col.count() > 0:
            self._build_bm25()

    # ── BM25 ─────────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[а-яёА-ЯЁa-zA-Z0-9]+", text.lower())

    def _build_bm25(self) -> None:
        try:
            all_data = self._col.get(include=["documents", "metadatas", "ids"])
            self._bm25_texts = all_data["documents"]
            self._bm25_ids   = all_data["ids"]
            self._bm25_metas = all_data["metadatas"]
            corpus      = [self._tokenize(t) for t in self._bm25_texts]
            self._bm25  = BM25Okapi(corpus)
            print(f"BM25 готов: {len(self._bm25_texts)} документов")
        except Exception as e:
            print(f"BM25 init warn: {e}")

    def _bm25_retrieve(self, query: str, top_k: int) -> list[dict]:
        if not HAS_BM25 or self._bm25 is None:
            return []
        import numpy as np
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k * 2]
        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            meta = self._bm25_metas[idx]
            results.append({
                "_bm25_id":    self._bm25_ids[idx],
                "text":        self._bm25_texts[idx],
                "page":        meta.get("page", 0),
                "type":        meta.get("type", "text"),
                "image_path":  meta.get("image_path", ""),
                "source_file": meta.get("source_file", ""),
                "document_id": meta.get("document_id", ""),
                "score":       round(float(scores[idx]), 4),
            })
        return results[:top_k]

    def _rrf_merge(self, vec: list[dict], bm25: list[dict], top_k: int, k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion of vector + BM25 results."""
        def uid(c: dict) -> str:
            return c.get("_bm25_id") or f"{c['source_file']}|{c['page']}|{c['text'][:40]}"

        rrf: dict[str, float] = {}
        by_uid: dict[str, dict] = {}

        for rank, c in enumerate(vec):
            u = uid(c)
            rrf[u] = rrf.get(u, 0.0) + 1.0 / (k + rank + 1)
            by_uid[u] = c

        for rank, c in enumerate(bm25):
            u = uid(c)
            rrf[u] = rrf.get(u, 0.0) + 1.0 / (k + rank + 1)
            if u not in by_uid:
                by_uid[u] = c

        ranked = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:top_k]
        max_score = max(rrf.values()) if rrf else 1.0
        result = []
        for u in ranked:
            c = dict(by_uid[u])
            c["score"] = round(rrf[u] / max_score, 4)
            c.pop("_bm25_id", None)
            result.append(c)
        return result

    # ── Аббревиатуры ─────────────────────────────────────────────────────────

    def _expand_abbreviations(self, text: str) -> str:
        result = text
        for abbr, expansion in GEO_ABBREVIATIONS.items():
            pattern = r"(?<![А-ЯЁA-Zа-яёa-z])" + re.escape(abbr) + r"(?![А-ЯЁA-Zа-яёa-z])"
            if re.search(pattern, result):
                result = re.sub(pattern, f"{abbr} ({expansion})", result, count=1)
        return result

    # ── Jina эмбеддинги ──────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> list[float]:
        resp = requests.post(
            f"{JINA_BASE}/embeddings",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":      EMBED_MODEL,
                "input":      [query],
                "task":       "retrieval.query",
                "dimensions": EMBED_DIM,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    # ── Vector search ─────────────────────────────────────────────────────────

    def _vec_retrieve(self, query: str, top_k: int,
                      where: dict | None = None) -> list[dict]:
        if self._col is None or self._col.count() == 0:
            return []
        expanded = self._expand_abbreviations(query)
        emb      = self._embed_query(expanded)
        n        = min(top_k, self._col.count())
        res      = self._col.query(
            query_embeddings=[emb],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            chunks.append({
                "text":        doc,
                "page":        meta["page"],
                "type":        meta["type"],
                "image_path":  meta.get("image_path", ""),
                "source_file": meta.get("source_file", ""),
                "document_id": meta.get("document_id", ""),
                "score":       round(1.0 - dist, 4),
            })
        return chunks

    # ── Public retrieve ───────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K,
                 where: dict | None = None,
                 mode: str = "hybrid") -> list[dict]:
        """
        mode:
          rag      — vector only
          hybrid   — vector + BM25 (RRF)
          graphrag — graph-expanded vector search
        """
        if self._col is None or self._col.count() == 0:
            return []

        if mode == "graphrag":
            return self._graphrag_retrieve(query, top_k)

        vec = self._vec_retrieve(query, top_k, where)

        if mode == "hybrid" and HAS_BM25 and self._bm25 is not None:
            expanded = self._expand_abbreviations(query)
            bm25     = self._bm25_retrieve(expanded, top_k)
            return self._rrf_merge(vec, bm25, top_k)

        return vec

    def _graphrag_retrieve(self, query: str, top_k: int) -> list[dict]:
        """Public wrapper — discards trace, returns only chunks."""
        chunks, _ = self._graphrag_retrieve_with_trace(query, top_k)
        return chunks

    def _graphrag_retrieve_with_trace(self, query: str, top_k: int) -> tuple:
        """
        GraphRAG: query → extract entities → graph pages → vector search.
        Returns (chunks, trace_dict) where trace_dict has full traversal evidence.
        """
        trace: dict = {
            "extracted_terms":    [],
            "graph_pages":        [],
            "chunks_from_graph":  0,
            "chunks_from_vector": 0,
        }

        try:
            from graph import extract_query_entities, get_graph_pages
            terms = extract_query_entities(query)
            trace["extracted_terms"] = terms

            if terms:
                pages = get_graph_pages(terms)  # [(source_file, page), ...]
                trace["graph_pages"] = [
                    {"source": s.replace(".pdf", "").replace(".img", "")[-28:], "page": p}
                    for s, p in pages
                ]

                if pages:
                    all_chunks: list[dict] = []
                    seen_texts: set[str]   = set()

                    for src, pg in pages[:6]:
                        where = {"$and": [
                            {"source_file": src},
                            {"page": pg},
                        ]}
                        for c in self._vec_retrieve(query, top_k=3, where=where):
                            if c["text"] not in seen_texts:
                                seen_texts.add(c["text"])
                                all_chunks.append(c)

                    trace["chunks_from_graph"] = len(all_chunks)

                    # Fill remaining slots with standard vector search
                    for c in self._vec_retrieve(query, top_k):
                        if c["text"] not in seen_texts and len(all_chunks) < top_k:
                            seen_texts.add(c["text"])
                            all_chunks.append(c)

                    trace["chunks_from_vector"] = len(all_chunks) - trace["chunks_from_graph"]

                    return (
                        sorted(all_chunks, key=lambda x: x["score"], reverse=True)[:top_k],
                        trace,
                    )
        except Exception:
            pass

        # Fallback to hybrid
        fallback = self.retrieve(query, top_k, mode="hybrid")
        trace["chunks_from_vector"] = len(fallback)
        return fallback, trace

    # ── Answer ────────────────────────────────────────────────────────────────

    def answer(self, query: str,
               filter_source: str | None = None,
               filter_doc_id: str | None = None,
               mode: str = "hybrid") -> dict:

        if self._col is None or self._col.count() == 0:
            return {
                "answer":  "Документы ещё не проиндексированы. Загрузите PDF.",
                "images":  [],
                "sources": [],
                "chunks":  [],
                "mode":    mode,
            }

        where = None
        if filter_doc_id:
            where = {"document_id": filter_doc_id}
        elif filter_source:
            where = {"source_file": filter_source}

        # GraphRAG: retrieve with full traversal trace
        graph_data: dict = {}
        traversal_trace: dict = {}
        if mode == "graphrag":
            chunks, traversal_trace = self._graphrag_retrieve_with_trace(query, TOP_K)
            if where:
                chunks = [c for c in chunks if (
                    (filter_doc_id and c.get("document_id") == filter_doc_id) or
                    (filter_source and c.get("source_file") == filter_source)
                )] or chunks
        else:
            chunks = self.retrieve(query, where=where, mode=mode)

        if not chunks:
            return {
                "answer":  "По данному запросу ничего не найдено.",
                "images":  [],
                "sources": [],
                "chunks":  chunks,
                "mode":    mode,
                "graph_data": {},
            }

        context_parts: list[str] = []
        images: list[dict]       = []
        seen_images: set[str]    = set()

        for i, c in enumerate(chunks, 1):
            src    = c["source_file"]
            doc_id = c.get("document_id", src)
            prefix = f"[{doc_id}:{c['page']}]"
            context_parts.append(f"{prefix}\n{c['text']}")

            ip       = c["image_path"]
            resolved = _resolve_image_path(ip)
            if resolved and str(resolved) not in seen_images:
                seen_images.add(str(resolved))
                images.append({
                    "path":        str(resolved),
                    "page":        c["page"],
                    "source_file": src,
                    "caption":     f"Страница {c['page']} из {src}",
                })

        context  = "\n\n---\n\n".join(context_parts)
        user_msg = (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== КОНТЕКСТ ===\n{context}\n\n"
            f"=== ВОПРОС ===\n{query}"
        )

        answer_text = self._call_llm(user_msg, query, context)

        # GraphRAG: build subgraph + attach traversal trace
        if mode == "graphrag":
            try:
                from graph import extract_query_entities, query_graph_neighborhood
                terms      = traversal_trace.get("extracted_terms") or extract_query_entities(query)
                graph_data = query_graph_neighborhood(terms)
                graph_data["trace"] = {
                    "extracted_terms":    traversal_trace.get("extracted_terms", []),
                    "graph_pages":        traversal_trace.get("graph_pages", []),
                    "chunks_from_graph":  traversal_trace.get("chunks_from_graph", 0),
                    "chunks_from_vector": traversal_trace.get("chunks_from_vector", 0),
                    "total_chunks":       len(chunks),
                    "citations_in_answer": len([
                        m for m in __import__("re").finditer(
                            r'\[[0-9a-f-]{8,}:\d+\]', answer_text
                        )
                    ]),
                }
            except Exception:
                graph_data = {"trace": traversal_trace}

        return {
            "answer":     answer_text,
            "images":     images,
            "sources":    sorted({(c["source_file"], c["page"]) for c in chunks}),
            "chunks":     chunks,
            "mode":       mode,
            "graph_data": graph_data,
        }

    # ── LLM ──────────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, query: str, context: str) -> str:
        if not OPENROUTER_API_KEY:
            return self._format_raw_context(query, context)
        try:
            resp = requests.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type":  "application/json",
                    "HTTP-Referer":  "http://localhost:8000",
                    "X-Title":       "Geo RAG",
                },
                json={
                    "model":       LLM_MODEL,
                    "messages":    [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens":  2048,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            err = str(e).lower()
            if "401" in err or "403" in err:
                return "⚠️ Ошибка авторизации OpenRouter. Проверьте OPENROUTER_API_KEY в .env"
            return self._format_raw_context(query, context)

    def _format_raw_context(self, query: str, context: str) -> str:
        return "\n".join([
            "⚠️ **OPENROUTER_API_KEY не задан** — показываю найденные фрагменты.\n",
            f"**Запрос:** {query}\n",
            "---",
            "**Найденные отрывки:**\n",
            context[:3000] + ("…" if len(context) > 3000 else ""),
        ])

    # ── Верификация (LLM-as-judge) ────────────────────────────────────────────

    VERIFIER_PROMPT = """\
Ты — независимый эксперт-геолог и верификатор качества RAG-системы.
Проверь ответ системы на основе контекста из книг.

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

КОНТЕКСТ ИЗ КНИГ (источник истины):
{context}

ОТВЕТ СИСТЕМЫ:
{answer}

Оцени строго по критериям:
1. Точность (1-5): всё ли в ответе подтверждается контекстом?
2. Полнота (1-5): охвачены ли все ключевые аспекты вопроса?
3. Галлюцинации: есть ли факты/цифры в ответе, которых НЕТ в контексте?
4. Источники: корректно ли указаны ссылки [doc_id:страница]?

Отвечай строго в формате:
ТОЧНОСТЬ: X/5
ПОЛНОТА: X/5
ГАЛЛЮЦИНАЦИИ: есть/нет
ИСТОЧНИКИ: корректны/некорректны/отсутствуют
ИТОГ: ОТЛИЧНО/ХОРОШО/УДОВЛЕТВОРИТЕЛЬНО/ПЛОХО
КОММЕНТАРИЙ: [одно-два предложения на русском]"""

    def verify_answer(self, query: str, answer: str, context: str) -> dict:
        if not OPENROUTER_API_KEY or not answer or not context:
            return {"error": "Верификация недоступна"}
        prompt = self.VERIFIER_PROMPT.format(
            query=query, context=context[:4000], answer=answer[:2000]
        )
        try:
            resp = requests.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type":  "application/json",
                    "HTTP-Referer":  "http://localhost:8000",
                    "X-Title":       "Geo RAG Verifier",
                },
                json={
                    "model":       VERIFIER_MODEL,
                    "messages":    [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens":  300,
                },
                timeout=45,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            return self._parse_verification(raw)
        except Exception as e:
            return {"error": str(e)[:150], "raw": ""}

    @staticmethod
    def _parse_verification(raw: str) -> dict:
        result = {"raw": raw, "score": None, "verdict": None,
                  "hallucinations": None, "comment": ""}
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("ТОЧНОСТЬ:"):
                try:
                    result["accuracy"] = int(line.split(":")[1].strip()[0])
                except Exception:
                    pass
            elif line.startswith("ПОЛНОТА:"):
                try:
                    result["completeness"] = int(line.split(":")[1].strip()[0])
                except Exception:
                    pass
            elif line.startswith("ГАЛЛЮЦИНАЦИИ:"):
                result["hallucinations"] = "есть" in line.split(":", 1)[1].lower()
            elif line.startswith("ИТОГ:"):
                result["verdict"] = line.split(":", 1)[1].strip()
            elif line.startswith("КОММЕНТАРИЙ:"):
                result["comment"] = line.split(":", 1)[1].strip()
        acc = result.get("accuracy", 3)
        cmp = result.get("completeness", 3)
        result["score"] = round((acc + cmp) / 2, 1)
        return result
