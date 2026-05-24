import os
import sys
import json
import logging
import requests
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# Ana dizini path'e ekle (modülleri import edebilmek için)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indexer.chroma_store import ChromaStore
from config import get_settings

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/ai_enrichment.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def analyze_for_indexing(image_path):
    """Görseli analiz eder ve anahtar kelimeler döndürür."""
    import base64
    
    url = "http://localhost:1234/v1/chat/completions"
    model = get_settings().llm_vision_model
    
    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Bu bir 3D model görselidir. Bu modeli şu kategorilere göre analiz et: Kategori (koltuk, lamba vb.), Tarz (modern, minimalist, klasik vb.), Materyal (ahşap, metal, mermer vb.) ve Renk. SADECE virgülle ayrılmış anahtar kelimeler döndür. Cümle kurma."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": 100,
            "temperature": 0.2
        }
        
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Görsel analiz hatası ({image_path}): {e}")
    return ""

def main():
    settings = get_settings()
    store = ChromaStore(
        db_path=settings.chroma_db_path,
        collection_name=settings.get_collection_name(),
    )
    from config import MetadataSchema
    
    logger.info("🚀 AI Arşiv Akıllandırma İşlemi Başlıyor (PARALEL MOD: 4)...")
    
    logger.info("🔍 Veritabanı parça parça taranıyor (5000'lik bloklar)...")
    
    unprocessed_ids = []
    id_to_image = {}
    id_to_meta = {}
    
    batch_scan_size = 5000
    total_in_db = store.get_count()
    
    for offset in range(0, total_in_db, batch_scan_size):
        logger.info(f"  🔎 Tarama: {offset}/{total_in_db}...")
        results = store.collection.get(
            include=["metadatas"],
            limit=batch_scan_size,
            offset=offset
        )
        
        if not results or not results["ids"]:
            break
            
        for doc_id, meta in zip(results["ids"], results["metadatas"]):
            if not meta.get(MetadataSchema.AI_ENRICHED) or meta.get(MetadataSchema.AI_DESCRIPTION) == "":
                img_path = meta.get("image_path")
                if img_path and Path(img_path).exists():
                    unprocessed_ids.append(doc_id)
                    id_to_image[doc_id] = img_path
                    id_to_meta[doc_id] = meta

    total = len(unprocessed_ids)
    if total == 0:
        logger.info("✅ Analiz edilecek yeni model bulunamadı. Tüm arşiv zaten akıllı!")
        return

    logger.info(f"📊 Toplam {total} model analiz edilecek.")
    
    # Batch işleme (Paralel İşleme)
    batch_size = 40 
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(0, total, batch_size):
            batch_ids = unprocessed_ids[i:i+batch_size]
            
            # Paralel olarak analiz yap
            future_to_id = {executor.submit(analyze_for_indexing, id_to_image[doc_id]): doc_id for doc_id in batch_ids}
            
            current_batch_metas = []
            current_batch_ids = []
            
            for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(batch_ids), desc=f"Paralel Analiz ({i}/{total})", leave=False):
                doc_id = future_to_id[future]
                try:
                    description = future.result()
                    if description:
                        orig_meta = id_to_meta[doc_id]
                        new_meta = orig_meta.copy()
                        new_meta[MetadataSchema.AI_DESCRIPTION] = description
                        new_meta[MetadataSchema.AI_ENRICHED] = True
                        
                        current_batch_metas.append(new_meta)
                        current_batch_ids.append(doc_id)
                except Exception as exc:
                    logger.error(f"ID {doc_id} için analiz hatası: {exc}")

            # Toplu Güncelleme
            if current_batch_ids:
                try:
                    store.collection.update(
                        ids=current_batch_ids,
                        metadatas=current_batch_metas
                    )
                except Exception as e:
                    logger.error(f"Güncelleme hatası: {e}")

    logger.info(f"🎉 İşlem bitti! Toplam {total} model başarıyla akıllandırıldı.")

if __name__ == "__main__":
    main()
