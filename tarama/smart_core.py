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
        logging.FileHandler("data/smart_enrichment.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CLUSTER_FILE = Path("data/clusters.json")
QUEUE_FILE = Path("data/smart_queue.json")

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

def create_smart_queue():
    if not CLUSTER_FILE.exists():
        logger.error("❌ clusters.json bulunamadı! Önce kümeleme yapılmalı.")
        return []
        
    with open(CLUSTER_FILE, "r", encoding="utf-8") as f:
        clusters = json.load(f)
    
    store = ChromaStore()
    logger.info("🔍 Akıllı Kuyruk hazırlanıyor...")
    
    # Veritabanında zaten analiz edilmiş olanları kontrol et (Vakit kaybetmemek için)
    # Not: Hız için şimdilik tüm liderleri kuyruğa alalım, core içinde kontrol ederiz.
    queue = []
    for rep_id, data in clusters.items():
        queue.append({
            "rep_id": rep_id,
            "path": data["image_path"],
            "members": data["members"]
        })
    
    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue, f, ensure_ascii=False, indent=2)
    return queue

def main():
    if not QUEUE_FILE.exists():
        queue = create_smart_queue()
    else:
        with open(QUEUE_FILE, "r", encoding="utf-8") as f:
            queue = json.load(f)
            
    if not queue: return

    store = ChromaStore()
    
    # 4B modeli kullandığımız için paralel işçi sayısını artıralım!
    # LM Studio "Parallel 4" diyordu ama 4B model küçüktür, 6-8 deneyebiliriz.
    # Güvenli tarafta kalmak için 6 yapalım.
    max_parallel = 6 
    batch_size = max_parallel * 5
    
    logger.info(f"🚀 SMART ENRICHMENT BAŞLADI (Paralel: {max_parallel})")
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        while queue:
            current_batch = queue[:batch_size]
            remaining_queue = queue[batch_size:]
            
            # 1. Liderleri Analiz Et
            future_to_item = {}
            for item in current_batch:
                if item["path"] and Path(item["path"]).exists():
                    future_to_item[executor.submit(analyze_for_indexing, item["path"])] = item
            
            # 2. Sonuçları Topla ve Tüm Üyelere Yay
            updates_ids = []
            updates_metas = []
            
            for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(future_to_item), desc="Lider Analizi", leave=False):
                item = future_to_item[future]
                description = future.result()
                
                if description:
                    # Grubun tüm üyeleri için metadata hazırla
                    # (Her bir üye için orijinal metadatayı çekip güncellemek gerek)
                    # Hız için: Toplu bir get yapalım
                    member_ids = item["members"]
                    member_data = store.collection.get(ids=member_ids, include=["metadatas"])
                    
                    if member_data and member_data["metadatas"]:
                        for mid, mmeta in zip(member_data["ids"], member_data["metadatas"]):
                            new_meta = mmeta.copy()
                            new_meta["ai_description"] = description
                            new_meta["ai_enriched"] = True
                            updates_ids.append(mid)
                            updates_metas.append(new_meta)
            
            # 3. Veritabanını Toplu Güncelle (Devasa hız artışı!)
            if updates_ids:
                # ChromaDB update batch limitine dikkat (500-1000 iyidir)
                for start in range(0, len(updates_ids), 500):
                    end = start + 500
                    store.collection.update(
                        ids=updates_ids[start:end],
                        metadatas=updates_metas[start:end]
                    )
                logger.info(f"✨ {len(updates_ids)} model AI bilgisiyle güncellendi.")

            # 4. Kuyruğu Kaydet
            queue = remaining_queue
            with open(QUEUE_FILE, "w", encoding="utf-8") as f:
                json.dump(queue, f, ensure_ascii=False)

    logger.info("🎉 ARŞİVİN TAMAMI AKILLANDIRILDI!")

if __name__ == "__main__":
    main()
