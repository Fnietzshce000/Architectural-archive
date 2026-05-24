import os
import sys
import json
import logging
import requests
import time
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Ana dizini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indexer.chroma_store import ChromaStore
from config import get_settings

# Loglama
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/ai_enrichment.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

QUEUE_FILE = Path("data/enrich_queue.json")

def analyze_for_indexing(image_path):
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
                        {"type": "text", "text": "Bu bir 3D model görselidir. Kategori, Tarz, Materyal ve Renk bilgilerini SADECE virgülle ayrılmış anahtar kelimeler olarak döndür."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 100,
            "temperature": 0.2
        }
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except: pass
    return ""

def create_queue():
    store = ChromaStore()
    logger.info("🔍 Kuyruk oluşturuluyor (Bu işlem bir kez yapılır ve uzun sürebilir)...")
    
    unprocessed = []
    batch_scan_size = 10000
    total_in_db = store.get_count()
    
    for offset in range(0, total_in_db, batch_scan_size):
        logger.info(f"  🔎 Tarama: {offset}/{total_in_db}")
        results = store.collection.get(include=["metadatas"], limit=batch_scan_size, offset=offset)
        if not results or not results["ids"]: break
        for doc_id, meta in zip(results["ids"], results["metadatas"]):
            if not meta.get("ai_enriched"):
                unprocessed.append({"id": doc_id, "path": meta.get("image_path"), "meta": meta})
    
    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(unprocessed, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Kuyruk oluşturuldu: {len(unprocessed)} model işlenecek.")
    return unprocessed

def main():
    if not QUEUE_FILE.exists():
        queue = create_queue()
    else:
        with open(QUEUE_FILE, "r", encoding="utf-8") as f:
            queue = json.load(f)
        logger.info(f"📂 Mevcut kuyruk yüklendi: {len(queue)} model kaldı.")

    if not queue:
        logger.info("🎉 İşlenecek model kalmadı!")
        return

    store = ChromaStore()
    batch_size = 40
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        while queue:
            current_batch = queue[:batch_size]
            remaining_queue = queue[batch_size:]
            
            # Paralel Analiz
            future_to_item = {executor.submit(analyze_for_indexing, item["path"]): item for item in current_batch if Path(item["path"]).exists()}
            
            current_batch_metas = []
            current_batch_ids = []
            
            for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(future_to_item), desc="Analiz", leave=False):
                item = future_to_item[future]
                res = future.result()
                if res:
                    new_meta = item["meta"].copy()
                    new_meta["ai_description"] = res
                    new_meta["ai_enriched"] = True
                    current_batch_metas.append(new_meta)
                    current_batch_ids.append(item["id"])

            # DB Güncelle
            if current_batch_ids:
                store.collection.update(ids=current_batch_ids, metadatas=current_batch_metas)
            
            # Kuyruğu Güncelle ve Kaydet
            queue = remaining_queue
            with open(QUEUE_FILE, "w", encoding="utf-8") as f:
                json.dump(queue, f, ensure_ascii=False)
            
            logger.info(f"✅ {len(current_batch_ids)} model tamamlandı. Kalan: {len(queue)}")

if __name__ == "__main__":
    main()
