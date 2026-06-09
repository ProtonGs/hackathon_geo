# -*- coding: utf-8 -*-
"""Direct indexing script - bypasses HTTP, calls ingest directly."""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'geo_project.settings')
import django
django.setup()

from pathlib import Path
from rag_app.models import Document
from ingest import ingest_document

BASE = Path(__file__).parent
PDFS = sorted(BASE.glob("*.pdf"))

print(f"Found {len(PDFS)} PDFs:")
for p in PDFS:
    print(f"  {p.name}  ({p.stat().st_size/1e6:.1f} MB)")

for pdf_path in PDFS:
    # Create or get document record
    doc, created = Document.objects.get_or_create(
        name=pdf_path.name,
        defaults={"file": f"pdfs/{pdf_path.name}", "status": "pending"}
    )

    if not created and doc.status == "indexed":
        print(f"\n{pdf_path.name}: already indexed ({doc.text_chunks} chunks), skipping.")
        continue

    # Copy PDF to media/pdfs/ if not there
    media_pdf = BASE / "media" / "pdfs" / pdf_path.name
    media_pdf.parent.mkdir(parents=True, exist_ok=True)
    if not media_pdf.exists():
        import shutil
        shutil.copy2(pdf_path, media_pdf)
        doc.file = f"pdfs/{pdf_path.name}"
        doc.save()

    doc.status = "indexing"
    doc.progress = "Starting..."
    doc.save()

    def cb(msg):
        Document.objects.filter(id=doc.id).update(progress=msg[:500])

    try:
        print(f"\n{'='*60}")
        print(f"Indexing: {pdf_path.name}")
        print(f"{'='*60}")
        stats = ingest_document(str(pdf_path), str(doc.id), cb)

        from django.utils import timezone
        Document.objects.filter(id=doc.id).update(
            status="indexed",
            progress="Done",
            text_chunks=stats["text"],
            images_count=stats["images"],
            total_pages=stats["pages"],
            indexed_at=timezone.now(),
        )
        print(f"\nDone: {stats}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        Document.objects.filter(id=doc.id).update(
            status="error",
            error_msg=str(e)[:500],
        )
        print(f"ERROR: {e}")

print("\n\nAll done! Open http://127.0.0.1:8000 to chat.")
