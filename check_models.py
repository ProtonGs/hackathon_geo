import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from dotenv import load_dotenv; load_dotenv()
import os, requests

or_key = os.getenv('OPENROUTER_API_KEY', '')
r = requests.get('https://openrouter.ai/api/v1/models',
    headers={'Authorization': f'Bearer {or_key}'}, timeout=20)
models = r.json().get('data', [])

print('=== DeepSeek ===')
for m in models:
    mid = m.get('id','')
    if 'deepseek' in mid.lower():
        pricing = m.get('pricing', {})
        print(f'  {mid}   prompt={pricing.get("prompt","?")}')

print()
print('=== Qwen VL / Vision ===')
for m in models:
    mid = m.get('id','')
    if 'qwen' in mid.lower() and ('vl' in mid.lower() or 'vision' in mid.lower() or 'qvq' in mid.lower()):
        pricing = m.get('pricing', {})
        print(f'  {mid}   prompt={pricing.get("prompt","?")}')

print()
print('=== All Qwen ===')
for m in models:
    mid = m.get('id','')
    if 'qwen' in mid.lower():
        pricing = m.get('pricing', {})
        print(f'  {mid}   prompt={pricing.get("prompt","?")}')
