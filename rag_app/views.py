import json
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.http import require_POST, require_http_methods

from rag_app.models import Document
from rag_app.tasks import start_indexing, get_rag, invalidate_rag


# ─── Страницы ─────────────────────────────────────────────────────────────────

def index(request):
    return render(request, 'rag_app/index.html')


# ─── Документы ────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.djvu'}


@require_POST
def upload(request):
    f = request.FILES.get('file')
    if not f:
        return JsonResponse({'error': 'Файл не передан'}, status=400)

    ext = Path(f.name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JsonResponse(
            {'error': f'Поддерживаются: {", ".join(sorted(ALLOWED_EXTENSIONS))}'},
            status=400,
        )

    doc = Document.objects.create(name=f.name, file=f)

    # Быстрое определение типа (без тяжёлого чтения)
    try:
        from ingest import detect_source_type
        doc_info = detect_source_type(doc.file.path)
        doc.progress = (
            f"Тип: {doc_info['doc_type']} · "
            f"стратегия: {doc_info['strategy']} · "
            f"{doc_info['details']}"
        )
        doc.save(update_fields=['progress'])
    except Exception:
        pass

    return JsonResponse(doc.to_dict(), status=201)


def detect_type_view(request):
    """Quick doc-type detection endpoint (no indexing)."""
    doc_id = request.GET.get('doc_id')
    if not doc_id:
        return JsonResponse({'error': 'doc_id required'}, status=400)
    doc = get_object_or_404(Document, id=doc_id)
    try:
        from ingest import detect_source_type
        return JsonResponse(detect_source_type(doc.file.path))
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_POST
def doc_index(request, doc_id):
    doc = get_object_or_404(Document, id=doc_id)
    if doc.status == 'indexing':
        return JsonResponse({'error': 'Уже индексируется'}, status=400)
    doc.status    = 'pending'
    doc.progress  = ''
    doc.error_msg = ''
    doc.save()
    start_indexing(str(doc.id))
    return JsonResponse({'status': 'started'})


def doc_status(request, doc_id):
    doc = get_object_or_404(Document, id=doc_id)
    return JsonResponse(doc.to_dict())


def docs_status_all(request):
    docs = Document.objects.all()
    return JsonResponse({'documents': [d.to_dict() for d in docs]})


@require_http_methods(['DELETE'])
def doc_delete(request, doc_id):
    doc = get_object_or_404(Document, id=doc_id)

    try:
        import chromadb
        from config import CHROMA_DIR, COLLECTION_NAME
        col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
        col.delete(where={'document_id': str(doc_id)})
    except Exception:
        pass

    if doc.file:
        doc.file.delete(save=False)
    doc.delete()
    invalidate_rag()

    # Rebuild graph after deletion
    try:
        from graph import rebuild_graph_from_chroma, invalidate_graph
        invalidate_graph()
        rebuild_graph_from_chroma()
    except Exception:
        pass

    return JsonResponse({'status': 'deleted'})


# ─── Чат ─────────────────────────────────────────────────────────────────────

@require_POST
def chat(request):
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Неверный JSON'}, status=400)

    query = body.get('query', '').strip()
    if not query:
        return JsonResponse({'error': 'Пустой запрос'}, status=400)

    filter_doc_id = body.get('filter_doc_id') or None
    mode          = body.get('mode', 'hybrid')
    if mode not in ('rag', 'hybrid', 'graphrag'):
        mode = 'hybrid'

    try:
        rag    = get_rag()
        result = rag.answer(query, filter_doc_id=filter_doc_id, mode=mode)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    from django.conf import settings as dj_settings
    media_root = Path(dj_settings.MEDIA_ROOT)

    def _to_url(path_str: str) -> str:
        if not path_str:
            return ''
        from rag import _resolve_image_path
        resolved = _resolve_image_path(path_str)
        if resolved is None:
            return ''
        try:
            rel = resolved.relative_to(media_root)
            return dj_settings.MEDIA_URL + str(rel).replace('\\', '/')
        except ValueError:
            return ''

    images_out = []
    for img in result['images'][:4]:
        url = _to_url(img['path'])
        if url:
            images_out.append({
                'url':         url,
                'page':        img['page'],
                'source_file': img.get('source_file', ''),
                'caption':     img.get('caption', '')[:200],
            })

    chunks_out = []
    for c in result.get('chunks', []):
        chunks_out.append({
            'text':        c['text'][:700],
            'page':        c['page'],
            'score':       c.get('score', 0),
            'source_file': c.get('source_file', ''),
            'document_id': c.get('document_id', ''),
            'image_url':   _to_url(c.get('image_path', '')),
        })

    return JsonResponse({
        'answer':     result['answer'],
        'sources':    result['sources'],
        'images':     images_out,
        'chunks':     chunks_out,
        'mode':       result.get('mode', mode),
        'graph_data': result.get('graph_data', {}),
    })


# ─── Граф знаний ─────────────────────────────────────────────────────────────

def graph_stats_view(request):
    """Return knowledge graph statistics."""
    try:
        from graph import graph_stats
        return JsonResponse(graph_stats())
    except Exception as e:
        return JsonResponse({'error': str(e), 'nodes': 0, 'edges': 0}, status=200)


@require_POST
def graph_rebuild(request):
    """Rebuild knowledge graph from all indexed chunks."""
    try:
        from graph import rebuild_graph_from_chroma, invalidate_graph
        invalidate_graph()
        stats = rebuild_graph_from_chroma()
        invalidate_rag()
        return JsonResponse(stats)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def graph_query(request):
    """Return graph neighborhood for a query string."""
    query = request.GET.get('q', '').strip()
    if not query:
        return JsonResponse({'nodes': [], 'edges': [], 'seed_nodes': []})
    try:
        from graph import extract_query_entities, query_graph_neighborhood
        terms = extract_query_entities(query)
        if not terms:
            terms = query.split()[:3]
        data = query_graph_neighborhood(terms)
        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({'error': str(e), 'nodes': [], 'edges': []})


# ─── Метрики ─────────────────────────────────────────────────────────────────

def metrics(request):
    """System metrics endpoint."""
    docs  = Document.objects.all()
    ready = [d for d in docs if d.status == 'indexed']

    total_chunks = sum(d.text_chunks for d in ready)
    total_pages  = sum(d.total_pages for d in ready)

    try:
        import chromadb
        from config import CHROMA_DIR, COLLECTION_NAME
        col          = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
        chroma_count = col.count()
    except Exception:
        chroma_count = 0

    try:
        from graph import graph_stats
        g_stats = graph_stats()
    except Exception:
        g_stats = {'nodes': 0, 'edges': 0, 'available': False}

    return JsonResponse({
        'documents': {
            'total':   len(docs),
            'indexed': len(ready),
        },
        'index': {
            'chunks':      chroma_count,
            'text_chunks': total_chunks,
            'pages':       total_pages,
        },
        'graph':     g_stats,
        'models': {
            'vlm':      __import__('config').VLM_MODEL,
            'llm':      __import__('config').LLM_MODEL,
            'embedder': __import__('config').EMBED_MODEL,
        },
    })
