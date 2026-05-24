"""
🚀 Archi V2: Deep VLM Enrichment (Faz 1)
Bu script, arşivdeki görselleri tek tek yerel Vision LLM (LM Studio) ile analiz eder.
Çok uzun süreceği için kesintilere dayanıklıdır ve "limit" ile parça parça çalıştırılabilir.
Kullanım:
    python scripts/vlm_enricher.py --limit 1000
"""
import argparse
import base64
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indexer.chroma_store import ChromaStore
from config import get_settings, MetadataSchema

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("data/vlm_enrichment.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VLM_Enricher")

def analyze_image(image_path: str, model_name: str) -> dict:
    """Görseli Vision LLM'e gönderip yapılandırılmış JSON döner."""
    url = "http://localhost:1234/v1/chat/completions"
    
    if not Path(image_path).exists():
        return {"error": "File not found"}
        
    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        
        prompt = (
            "Analyze this 3D model image. Return a raw JSON object (NO markdown tags, NO code blocks, ONLY valid JSON). "
            "Use this exact schema:\n"
            "{\n"
            '  "style": "string (e.g. Modern, Classic, Industrial, Minimalist, etc.)",\n'
            '  "materials": ["string array (e.g. Wood, Leather, Metal, Glass)"],\n'
            '  "colors": ["string array (e.g. Black, White, Oak, Gold)"],\n'
            '  "objects": ["string array (e.g. Sofa, Table, Lamp, Bed)"]\n'
            "}"
        )

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 200,
            "temperature": 0.1
        }
        
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Temizlik (Markdown işaretleri varsa kaldır)
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        else:
            return {"error": f"HTTP {resp.status_code}"}
            
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from LLM"}
    except Exception as e:
        return {"error": str(e)}

def main(args):
    settings = get_settings()
    store = ChromaStore(
        db_path=settings.chroma_db_path,
        collection_name=settings.get_collection_name(),
    )
    
    logger.info("🤖 Archi V2: Deep VLM Enrichment Başlıyor...")
    logger.info(f"   Kullanılacak Vision Model: {settings.llm_vision_model}")
    
    # 1. Eksik kayıtları bul
    unprocessed_ids = []
    id_to_image = {}
    id_to_meta = {}
    
    batch_scan_size = 5000
    total_in_db = store.get_count()
    
    logger.info("🔍 Veritabanı taranıyor (Etiketlenmemiş görseller bulunuyor)...")
    
    for offset in range(0, total_in_db, batch_scan_size):
        results = store.collection.get(
            include=["metadatas"],
            limit=batch_scan_size,
            offset=offset
        )
        
        if not results or not results["ids"]:
            break
            
        for doc_id, meta in zip(results["ids"], results["metadatas"]):
            # Eğer deep_tags alanı yoksa işlenmemiştir
            if not meta.get(MetadataSchema.DEEP_TAGS):
                img_path = meta.get(MetadataSchema.IMAGE_PATH)
                if img_path and Path(img_path).exists():
                    unprocessed_ids.append(doc_id)
                    id_to_image[doc_id] = img_path
                    id_to_meta[doc_id] = meta
                    
                    if len(unprocessed_ids) >= args.limit:
                        break
        
        if len(unprocessed_ids) >= args.limit:
            break

    total = len(unprocessed_ids)
    if total == 0:
        logger.info("✅ Tüm görseller zaten VLM ile etiketlenmiş!")
        return

    logger.info(f"📊 Toplam {total} model analiz edilecek (Limit: {args.limit}).")
    
    # 2. Paralel İşleme (LM Studio'yu çok yormamak için max_workers düşük tutulmalı)
    batch_size = 10
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i in range(0, total, batch_size):
            batch_ids = unprocessed_ids[i:i+batch_size]
            
            future_to_id = {
                executor.submit(analyze_image, id_to_image[doc_id], settings.llm_vision_model): doc_id 
                for doc_id in batch_ids
            }
            
            current_batch_metas = []
            current_batch_ids = []
            
            for future in tqdm(as_completed(future_to_id), total=len(batch_ids), desc=f"VLM Analiz ({i}/{total})", leave=False):
                doc_id = future_to_id[future]
                try:
                    result_json = future.result()
                    
                    if "error" not in result_json:
                        orig_meta = id_to_meta[doc_id]
                        new_meta = orig_meta.copy()
                        
                        # Yapılandırılmış veriyi metadata'ya ekle
                        new_meta[MetadataSchema.DEEP_TAGS] = json.dumps(result_json, ensure_ascii=False)
                        new_meta[MetadataSchema.AI_ENRICHED] = True
                        
                        # Klasik ai_description alanını da uyumluluk için dolduralım
                        desc = f"{result_json.get('style', '')} " + " ".join(result_json.get('objects', [])) + " " + " ".join(result_json.get('materials', []))
                        new_meta[MetadataSchema.AI_DESCRIPTION] = desc.strip()
                        
                        current_batch_metas.append(new_meta)
                        current_batch_ids.append(doc_id)
                        success_count += 1
                    else:
                        logger.error(f"ID {doc_id} hata: {result_json['error']}")
                        error_count += 1
                        
                except Exception as exc:
                    logger.error(f"Beklenmeyen Hata (ID {doc_id}): {exc}")
                    error_count += 1

            # Toplu Güncelleme
            if current_batch_ids:
                try:
                    store.collection.update(
                        ids=current_batch_ids,
                        metadatas=current_batch_metas
                    )
                except Exception as e:
                    logger.error(f"Veritabanı güncelleme hatası: {e}")

    logger.info(f"\n🎉 Görev tamamlandı!")
    logger.info(f"   Başarılı: {success_count}")
    logger.info(f"   Hatalı/Atlanan: {error_count}")
    logger.info(f"   Log dosyası: data/vlm_enrichment.log")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Metadata Zenginleştirici")
    parser.add_argument("--limit", type=int, default=1000, help="Kaç görsel işlenecek?")
    parser.add_argument("--workers", type=int, default=2, help="Eşzamanlı LLM isteği sayısı")
    args = parser.parse_args()
    main(args)
