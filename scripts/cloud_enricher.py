"""
🚀 Archi V2: Gemini Free Tier VLM Enrichment
Google Gemini-2.5-Flash API'sini kullanarak ücretsiz ve kusursuz etiketleme yapar.
Dakikada 15 istek sınırına takılmamak için her istek arasına 4 saniye bekleme süresi koyar.
"""
import os
import sys
import json
import base64
import logging
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indexer.chroma_store import ChromaStore
from config import get_settings, MetadataSchema

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("data/cloud_enrichment.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GeminiEnricher")

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

PROMPT = (
    "Analyze this 3D model image. Return a raw JSON object (NO markdown tags, NO code blocks, ONLY valid JSON). "
    "Use this exact schema:\n"
    "{\n"
    '  "style": "string (e.g. Modern, Classic, Industrial, Minimalist, etc.)",\n'
    '  "materials": ["string array (e.g. Wood, Leather, Metal, Glass)"],\n'
    '  "colors": ["string array (e.g. Black, White, Oak, Gold)"],\n'
    '  "objects": ["string array (e.g. Sofa, Table, Lamp, Bed)"]\n'
    "}"
)

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_gemini(base64_img: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [
                {"text": PROMPT},
                {"inlineData": {"mimeType": "image/jpeg", "data": base64_img}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 150
        }
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    content = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)

def main(args):
    if not GEMINI_KEY:
        logger.error("❌ GEMINI_API_KEY .env dosyasında bulunamadı!")
        return

    settings = get_settings()
    store = ChromaStore(
        db_path=settings.chroma_db_path,
        collection_name=settings.get_collection_name(),
    )
    
    logger.info("💎 Archi Gemini Enrichment Başlıyor... (Hız Sınırı: 15/dakika)")
    
    unprocessed_ids = []
    id_to_image = {}
    id_to_meta = {}
    
    total_in_db = store.get_count()
    
    for offset in range(0, total_in_db, 5000):
        results = store.collection.get(include=["metadatas"], limit=5000, offset=offset)
        if not results or not results["ids"]: break
            
        for doc_id, meta in zip(results["ids"], results["metadatas"]):
            if not meta.get(MetadataSchema.DEEP_TAGS):
                img_path = meta.get(MetadataSchema.IMAGE_PATH)
                if img_path and Path(img_path).exists():
                    unprocessed_ids.append(doc_id)
                    id_to_image[doc_id] = img_path
                    id_to_meta[doc_id] = meta
                    if len(unprocessed_ids) >= args.limit: break
        if len(unprocessed_ids) >= args.limit: break

    total = len(unprocessed_ids)
    if total == 0:
        logger.info("✅ Tüm görseller zaten etiketlenmiş!")
        return

    logger.info(f"📊 İşlenecek: {total} görsel. Güvenli indirme için 4 saniye aralıklarla çalışacak.")
    
    success_count = 0
    error_count = 0
    
    for doc_id in tqdm(unprocessed_ids, desc="Gemini Analizi"):
        try:
            base64_img = encode_image_base64(id_to_image[doc_id])
            data = call_gemini(base64_img)
            
            orig_meta = id_to_meta[doc_id]
            new_meta = orig_meta.copy()
            new_meta[MetadataSchema.DEEP_TAGS] = json.dumps(data, ensure_ascii=False)
            new_meta[MetadataSchema.AI_ENRICHED] = True
            desc = f"{data.get('style', '')} " + " ".join(data.get('objects', [])) + " " + " ".join(data.get('materials', []))
            new_meta[MetadataSchema.AI_DESCRIPTION] = desc.strip()
            
            store.collection.update(ids=[doc_id], metadatas=[new_meta])
            success_count += 1
            
        except Exception as e:
            logger.error(f"Hata ({doc_id}): {str(e)}")
            error_count += 1
            
        # Free Tier Koruması: Dakikada 15 istek (1 istek = 4 saniye)
        time.sleep(4.1)

    logger.info(f"\n🎉 Görev tamamlandı! Başarılı: {success_count}, Hatalı: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    main(args)
