# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv; load_dotenv()
import os, requests, base64
from pathlib import Path
from config import OPENROUTER_API_KEY, JINA_API_KEY, VLM_MODEL, LLM_MODEL, OPENROUTER_BASE, JINA_BASE, EMBED_MODEL, EMBED_DIM

print(f"VLM: {VLM_MODEL}")
print(f"LLM: {LLM_MODEL}")
print()

# 1. Тест LLM (DeepSeek)
print("=== LLM (DeepSeek) ===")
r = requests.post(f"{OPENROUTER_BASE}/chat/completions",
    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
    json={"model": LLM_MODEL,
          "messages": [{"role":"user","content":"Что такое ВНК в геологии нефти? Отвечай кратко 1 предложение."}],
          "max_tokens": 100, "temperature": 0.1},
    timeout=30)
print("Status:", r.status_code)
if r.ok:
    print("Ответ:", r.json()["choices"][0]["message"]["content"])
else:
    print("Ошибка:", r.text[:300])

print()

# 2. Тест VLM с реальной страницей
print("=== VLM (Qwen3-VL) ===")
imgs = list(Path("media/images").glob("*.jpg"))
if not imgs:
    print("Нет изображений страниц. Сначала запусти индексацию хотя бы одной страницы через OCR.")
else:
    img_path = imgs[0]
    print(f"Тест на: {img_path.name}")
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    r2 = requests.post(f"{OPENROUTER_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json",
                 "HTTP-Referer": "http://localhost:8000"},
        json={"model": VLM_MODEL,
              "messages": [{"role":"user","content":[
                  {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}},
                  {"type":"text","text":"Извлеки весь текст с этой страницы. Опиши кратко что на ней изображено."}
              ]}],
              "max_tokens": 500, "temperature": 0.1},
        timeout=60)
    print("Status:", r2.status_code)
    if r2.ok:
        content = r2.json()["choices"][0]["message"]["content"]
        print("Ответ VLM:")
        print(content[:800])
    else:
        print("Ошибка:", r2.text[:400])
