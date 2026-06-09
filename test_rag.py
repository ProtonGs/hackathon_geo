# -*- coding: utf-8 -*-
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from rag import GeoRAG

rag = GeoRAG()
print("Chunks in index:", rag._col.count() if rag._col else 0)
print()

queries = [
    "что такое каустобиолиты и нефть",
    "нефтяные ловушки классификация",
    "геохимия органического вещества",
]

for q in queries:
    print("="*60)
    print("QUERY:", q)
    chunks = rag.retrieve(q, top_k=2)
    for i, c in enumerate(chunks, 1):
        score = c["score"]
        page  = c["page"]
        src   = c["source_file"][:35]
        text  = c["text"][:200]
        print(f"  [{i}] score={score} p.{page} | {src}")
        print(f"       {text}")
    print()
