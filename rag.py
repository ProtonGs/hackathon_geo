"""
rag.py — поиск и генерация ответов.

Эмбеддинги: Jina AI API (jina-embeddings-v3, задача retrieval.query)
LLM:        OpenRouter — DeepSeek-V3 или другой open-weight
Аббревиатуры: автоматически расшифровываются в запросах перед поиском
"""

import re
import requests
from pathlib import Path
import chromadb

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
2. Для ссылок используй ТОЛЬКО теги вида [файл, стр. N] которые уже стоят перед каждым фрагментом в контексте. Не придумывай другие страницы.
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

    # ── Расширение аббревиатур в запросе ─────────────────────────────────────

    def _expand_abbreviations(self, text: str) -> str:
        """Добавляет расшифровку геологических аббревиатур в текст запроса."""
        result = text
        for abbr, expansion in GEO_ABBREVIATIONS.items():
            pattern = r'(?<![А-ЯЁA-Zа-яёa-z])' + re.escape(abbr) + r'(?![А-ЯЁA-Zа-яёa-z])'
            if re.search(pattern, result):
                result = re.sub(pattern, f"{abbr} ({expansion})", result, count=1)
        return result

    # ── Jina эмбеддинги для запроса ──────────────────────────────────────────

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

    # ── Поиск ────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K,
                 where: dict | None = None) -> list[dict]:
        if self._col is None or self._col.count() == 0:
            return []

        expanded = self._expand_abbreviations(query)
        emb = self._embed_query(expanded)
        n   = min(top_k, self._col.count())

        res = self._col.query(
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

    # ── Ответ ─────────────────────────────────────────────────────────────────

    def answer(self, query: str,
               filter_source: str | None = None,
               filter_doc_id: str | None = None) -> dict:

        if self._col is None or self._col.count() == 0:
            return {
                "answer":  "Документы ещё не проиндексированы. Загрузите PDF и дождитесь завершения индексации.",
                "images":  [],
                "sources": [],
                "chunks":  [],
            }

        where = None
        if filter_doc_id:
            where = {"document_id": filter_doc_id}
        elif filter_source:
            where = {"source_file": filter_source}

        chunks = self.retrieve(query, where=where)
        if not chunks:
            return {
                "answer":  "По данному запросу ничего не найдено в выбранных документах.",
                "images":  [],
                "sources": [],
                "chunks":  chunks,
            }

        context_parts: list[str] = []
        images: list[dict] = []
        seen_images: set[str] = set()

        for i, c in enumerate(chunks, 1):
            src    = c["source_file"]
            prefix = f"[Фрагмент {i} | {src}, стр. {c['page']}]"
            context_parts.append(f"{prefix}\n{c['text']}")

            ip = c["image_path"]
            if ip and ip not in seen_images and Path(ip).exists():
                seen_images.add(ip)
                images.append({
                    "path":        ip,
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

        return {
            "answer":  answer_text,
            "images":  images,
            "sources": sorted({(c["source_file"], c["page"]) for c in chunks}),
            "chunks":  chunks,
        }

    # ── LLM через OpenRouter ──────────────────────────────────────────────────

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
                    "model":      LLM_MODEL,
                    "messages":   [{"role": "user", "content": prompt}],
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
                return "⚠️ Ошибка авторизации OpenRouter. Проверьте OPENROUTER_API_KEY в файле .env"
            return self._format_raw_context(query, context)

    def _format_raw_context(self, query: str, context: str) -> str:
        lines = [
            "⚠️ **OPENROUTER_API_KEY не задан** — показываю найденные фрагменты.\n",
            f"**Запрос:** {query}\n",
            "---",
            "**Найденные отрывки из книг:**\n",
            context[:3000] + ("…" if len(context) > 3000 else ""),
            "\n---",
            "_Заполните `.env` файл: `OPENROUTER_API_KEY=sk-or-...`_",
        ]
        return "\n".join(lines)

    # ── Верификация ответа (LLM-as-judge) ────────────────────────────────────

    VERIFIER_PROMPT = """\
Ты — независимый эксперт-геолог и верификатор качества RAG-системы.
Проверь ответ системы по геологии на основе предоставленного контекста из книг.

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
4. Источники: корректно ли указаны ссылки на книги/страницы?

Отвечай строго в формате (ничего лишнего):
ТОЧНОСТЬ: X/5
ПОЛНОТА: X/5
ГАЛЛЮЦИНАЦИИ: есть/нет
ИСТОЧНИКИ: корректны/некорректны/отсутствуют
ИТОГ: ОТЛИЧНО/ХОРОШО/УДОВЛЕТВОРИТЕЛЬНО/ПЛОХО
КОММЕНТАРИЙ: [одно-два предложения на русском]"""

    def verify_answer(self, query: str, answer: str, context: str) -> dict:
        """Второй LLM-вызов: проверяет точность ответа на основе контекста."""
        if not OPENROUTER_API_KEY or not answer or not context:
            return {"error": "Верификация недоступна (нет ключа или данных)"}

        prompt = self.VERIFIER_PROMPT.format(
            query=query,
            context=context[:4000],
            answer=answer[:2000],
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
                val = line.split(":", 1)[1].strip().lower()
                result["hallucinations"] = "есть" in val
            elif line.startswith("ИТОГ:"):
                result["verdict"] = line.split(":", 1)[1].strip()
            elif line.startswith("КОММЕНТАРИЙ:"):
                result["comment"] = line.split(":", 1)[1].strip()
        acc = result.get("accuracy", 3)
        cmp = result.get("completeness", 3)
        result["score"] = round((acc + cmp) / 2, 1)
        return result
