# -*- coding: utf-8 -*-
"""
Script to upload both PDFs and start indexing via Django API.
"""
import os, sys, time, requests
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
BOOKS_DIR = Path(__file__).parent

def get_csrf(session):
    session.get(f"{BASE_URL}/")
    return session.cookies.get("csrftoken", "")

def upload_pdf(session, csrf, pdf_path):
    with open(pdf_path, "rb") as f:
        r = session.post(
            f"{BASE_URL}/upload/",
            files={"file": (pdf_path.name, f, "application/pdf")},
            headers={"X-CSRFToken": csrf},
        )
    r.raise_for_status()
    return r.json()["id"]

def start_index(session, csrf, doc_id):
    r = session.post(
        f"{BASE_URL}/docs/{doc_id}/index/",
        headers={"X-CSRFToken": csrf, "Content-Type": "application/json"},
        data="{}",
    )
    r.raise_for_status()
    return r.json()

def check_status(session, doc_id):
    r = session.get(f"{BASE_URL}/docs/{doc_id}/status/")
    return r.json()

def main():
    pdfs = sorted(BOOKS_DIR.glob("*.pdf"))
    if not pdfs:
        print("PDF не найдены в директории.")
        sys.exit(1)

    print(f"Найдено {len(pdfs)} PDF файла:")
    for p in pdfs:
        print(f"  {p.name}  ({p.stat().st_size/1e6:.1f} MB)")

    s = requests.Session()
    csrf = get_csrf(s)
    print(f"\nCSRF: {csrf[:12]}...")

    doc_ids = []
    for pdf in pdfs:
        print(f"\nЗагрузка: {pdf.name}...")
        doc_id = upload_pdf(s, csrf, pdf)
        print(f"  ID: {doc_id}")
        print(f"  Запуск индексации...")
        start_index(s, csrf, doc_id)
        doc_ids.append(doc_id)
        print(f"  Индексация запущена в фоне!")
        time.sleep(0.5)

    print(f"\n{'='*60}")
    print("Оба файла загружены и индексация запущена!")
    print("Прогресс можно отслеживать на сайте: http://127.0.0.1:8000")
    print(f"{'='*60}")
    print("\nОтслеживание прогресса (Ctrl+C чтобы остановить мониторинг):\n")

    try:
        while True:
            all_done = True
            for doc_id in doc_ids:
                st = check_status(s, doc_id)
                status = st["status"]
                prog   = st.get("progress", "")
                name   = st["name"][:40]
                print(f"  [{status:10}] {name}: {prog[:60]}")
                if status in ("indexing", "pending"):
                    all_done = False
            if all_done:
                print("\nВсе документы проиндексированы!")
                break
            print()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nМониторинг остановлен. Индексация продолжается в фоне.")

if __name__ == "__main__":
    main()
