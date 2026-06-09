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

@require_POST
def upload(request):
    f = request.FILES.get('file')
    if not f:
        return JsonResponse({'error': 'Файл не передан'}, status=400)
    if not f.name.lower().endswith('.pdf'):
        return JsonResponse({'error': 'Только PDF файлы'}, status=400)

    doc = Document.objects.create(name=f.name, file=f)
    return JsonResponse(doc.to_dict(), status=201)


@require_POST
def doc_index(request, doc_id):
    doc = get_object_or_404(Document, id=doc_id)
    if doc.status == 'indexing':
        return JsonResponse({'error': 'Уже индексируется'}, status=400)
    doc.status   = 'pending'
    doc.progress = ''
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

    # Удалить из ChromaDB
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

    try:
        rag = get_rag()
        result = rag.answer(query, filter_doc_id=filter_doc_id)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    # Конвертируем абсолютные пути изображений в URL
    images_out = []
    from django.conf import settings as dj_settings
    media_root = Path(dj_settings.MEDIA_ROOT)

    for img in result['images'][:4]:
        p = Path(img['path'])
        try:
            rel = p.relative_to(media_root)
            url = dj_settings.MEDIA_URL + str(rel).replace('\\', '/')
        except ValueError:
            url = ''
        if url:
            images_out.append({
                'url':         url,
                'page':        img['page'],
                'source_file': img.get('source_file', ''),
                'caption':     img.get('caption', '')[:200],
            })

    def _img_path_to_url(path_str):
        if not path_str:
            return ''
        try:
            rel = Path(path_str).relative_to(media_root)
            return dj_settings.MEDIA_URL + str(rel).replace('\\', '/')
        except ValueError:
            return ''

    chunks_out = []
    for c in result.get('chunks', []):
        chunks_out.append({
            'text':        c['text'][:700],
            'page':        c['page'],
            'score':       c.get('score', 0),
            'source_file': c.get('source_file', ''),
            'image_url':   _img_path_to_url(c.get('image_path', '')),
        })

    return JsonResponse({
        'answer':  result['answer'],
        'sources': result['sources'],
        'images':  images_out,
        'chunks':  chunks_out,
    })
