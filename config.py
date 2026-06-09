import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).parent

# ─── Хранилище ────────────────────────────────────────────────────────────────
CHROMA_DIR      = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "book_rag"
IMAGES_DIR      = BASE_DIR / "media" / "images"

# ─── API ключи (из .env или переменных среды) ─────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
JINA_API_KEY       = os.getenv("JINA_API_KEY", "")

# ─── Модели OpenRouter (только open-weight) ───────────────────────────────────
# VLM: OCR страниц + понимание геологических диаграмм/карт
VLM_MODEL = os.getenv("VLM_MODEL", "qwen/qwen3-vl-32b-instruct")
# LLM: генерация ответов на вопросы
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-4-maverick")
# Верификатор: проверяет точность ответов (второй вызов LLM)
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "meta-llama/llama-4-maverick")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# ─── Эмбеддинги: Jina AI API ──────────────────────────────────────────────────
# jina-embeddings-v3: 89 языков включая русский, 1024 размерность
EMBED_MODEL = "jina-embeddings-v3"
EMBED_DIM   = 1024
JINA_BASE   = "https://api.jina.ai/v1"

# ─── Параметры индексации ─────────────────────────────────────────────────────
OCR_DPI       = 150   # DPI рендера страниц PDF → качество OCR
CHUNK_SIZE    = 400   # слов в одном чанке
CHUNK_OVERLAP = 50    # перекрытие между чанками
MIN_CHUNK_LEN = 30    # минимум символов чтобы не выбросить чанк
TOP_K         = 7     # количество релевантных фрагментов при поиске
