"""
tasks.py — фоновая индексация PDF в отдельном потоке.
"""
import threading
import django
from django.utils import timezone

_rag_cache = {'instance': None}
_rag_lock  = threading.Lock()


def get_rag():
    """Возвращает singleton GeoRAG, создаёт при первом вызове или после инвалидации."""
    with _rag_lock:
        if _rag_cache['instance'] is None:
            from rag import GeoRAG
            _rag_cache['instance'] = GeoRAG()
    return _rag_cache['instance']


def invalidate_rag():
    """Сбросить кэш RAG — вызывается после завершения индексации."""
    with _rag_lock:
        _rag_cache['instance'] = None


def _run_indexing(doc_id: str) -> None:
    """Тело фонового потока индексации."""
    from rag_app.models import Document

    try:
        doc = Document.objects.get(id=doc_id)
        doc.status   = 'indexing'
        doc.progress = 'Начало обработки…'
        doc.save()

        def cb(msg: str):
            Document.objects.filter(id=doc_id).update(progress=msg)

        from ingest import ingest_document
        stats = ingest_document(doc.file.path, str(doc_id), cb)

        Document.objects.filter(id=doc_id).update(
            status       = 'indexed',
            progress     = 'Индексация завершена',
            text_chunks  = stats.get('text', 0),
            images_count = stats.get('images', 0),
            total_pages  = stats.get('pages', 0),
            indexed_at   = timezone.now(),
        )

        invalidate_rag()

    except Exception as exc:
        import traceback
        traceback.print_exc()
        Document.objects.filter(id=doc_id).update(
            status   = 'error',
            error_msg = str(exc)[:1000],
        )


def start_indexing(doc_id: str) -> None:
    t = threading.Thread(target=_run_indexing, args=(doc_id,), daemon=True)
    t.start()
