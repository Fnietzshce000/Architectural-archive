import os
import sys
import json
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Ana dizini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indexer.chroma_store import ChromaStore
from indexer.clip_encoder import encode_text
from config import get_settings

# Loglama
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ÇOK DİLLİ TASARIM KATALOĞU (TR - EN - RU) ---
CATEGORIES = [
    "sofa", "koltuk", "диван", "armchair", "berjer", "кресло", 
    "chair", "sandalye", "стул", "table", "masa", "стол", 
    "bed", "yatak", "кровать", "lamp", "lamba", "лампа", 
    "lighting", "aydinlatma", "освещение", "kitchen", "mutfak", "кухня", 
    "bathroom", "banyo", "ванная", "closet", "dolap", "шкаф", 
    "decor", "dekorasyon", "декор", "plant", "bitki", "растение", 
    "rug", "hali", "ковер", "wall panel", "duvar paneli", "стеновая панель"
]
STYLES = [
    "modern", "modern", "современный", "classic", "klasik", "классический", 
    "minimalist", "minimalist", "минималист", "industrial", "endüstriyel", "индустриальный", 
    "loft", "loft", "лофт", "vintage", "vintage", "винтаж", 
    "luxury", "lüks", "роскошь", "scandinavian", "iskandinav", "скандинавский"
]
MATERIALS = [
    "wood", "ahsap", "дерево", "metal", "metal", "металл", 
    "marble", "mermer", "мрамор", "glass", "cam", "стекло", 
    "fabric", "kumas", "ткань", "leather", "deri", "кожа", 
    "velvet", "kadife", "бархат", "gold", "altin", "золото"
]
COLORS = ["white", "black", "grey", "brown", "beige", "blue", "green", "red", "gold"]

ALL_LABELS = CATEGORIES + STYLES + MATERIALS + COLORS

def main():
    store = ChromaStore()
    settings = get_settings()
    
    logger.info("🚀 ULTRA HIZLI CLIP ETİKETLEME BAŞLIYOR...")
    
    # 1. Etiketleri Vektörleştir (Sadece bir kez yapılır)
    logger.info(f"🏷️ {len(ALL_LABELS)} etiket vektörleştiriliyor...")
    label_vectors = []
    for label in tqdm(ALL_LABELS, desc="Etiket Hazırlığı"):
        vec = encode_text(label, settings.clip_model_name, settings.clip_pretrained)
        label_vectors.append(vec)
    
    label_vectors = np.array(label_vectors) # (N, 512)
    
    # 2. Veritabanını Tara ve Etiketle
    total_in_db = store.get_count()
    batch_size = 5000 # Büyük batch'lerle çok daha hızlı
    
    logger.info(f"🔎 {total_in_db} model için AI eşleştirmesi yapılıyor...")
    
    for offset in range(0, total_in_db, batch_size):
        # Batch veriyi çek
        results = store.collection.get(
            include=["embeddings", "metadatas"],
            limit=batch_size,
            offset=offset
        )
        
        if not results or not results["ids"]:
            break
            
        current_batch_ids = results["ids"]
        current_batch_embeddings = np.array(results["embeddings"]) # (Batch, 512)
        current_batch_metas = results["metadatas"]
        
        # Matematiksel Eşleştirme (Cosine Similarity)
        # (Batch, 512) x (512, N) -> (Batch, N)
        similarities = np.dot(current_batch_embeddings, label_vectors.T)
        
        updated_metas = []
        updated_ids = []
        
        for i in range(len(current_batch_ids)):
            # En iyi eşleşen 6 etiketi al (TR-EN-RU karışık gelecek)
            top_indices = np.argsort(similarities[i])[-8:][::-1]
            top_labels = [ALL_LABELS[idx] for idx in top_indices if similarities[i][idx] > 0.16]
            
            if top_labels:
                new_description = ", ".join(top_labels)
                
                # Mevcut metadatayı güncelle
                meta = current_batch_metas[i].copy()
                meta["ai_description"] = new_description
                meta["ai_enriched"] = True
                
                updated_metas.append(meta)
                updated_ids.append(current_batch_ids[i])
        
        # Veritabanına Toplu Yaz (Işık Hızı!)
        if updated_ids:
            store.collection.update(
                ids=updated_ids,
                metadatas=updated_metas
            )
            
        logger.info(f"✅ İlerleme: {offset + len(current_batch_ids)} / {total_in_db} (+{len(updated_ids)} etiketlendi)")

    logger.info("🎉 TÜM ARŞİV BAŞARIYLA AKILLANDIRILDI!")

if __name__ == "__main__":
    main()
