import logging
import time
import numpy as np
from indexer.chroma_store import ChromaStore
from config import get_settings
from indexer.clip_encoder import load_model, encode_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# En elit 3D model ve iç mimari etiketleri (İngilizce olarak, çünkü ViT-L-14'ün ana dili İngilizce)
TAGS = [
    # Mimari Tarzlar
    "modern", "classic", "minimalist", "scandinavian", "industrial", 
    "vintage", "retro", "bohemian", "luxury", "rustic", "contemporary",
    "baroque", "art deco", "mid-century", "futuristic",
    # Materyaller & Dokular
    "wood", "metal", "glass", "marble", "leather", "fabric", "plastic", 
    "concrete", "stone", "velvet", "ceramic", "gold", "silver", "brass",
    # Odalar & Konseptler
    "living room", "bedroom", "kitchen", "bathroom", "office", "outdoor",
    "garden", "restaurant", "cafe", "store",
    # Eşya Türleri
    "sofa", "chair", "table", "bed", "lamp", "cabinet", "plant", "rug", 
    "door", "window", "armchair", "bookshelf", "decor", "lighting", "desk"
]

def run_zero_shot():
    settings = get_settings()
    store = ChromaStore(settings.chroma_db_path, settings.get_collection_name())
    
    logger.info("🧠 ViT-L-14 Model belleğe alınıyor...")
    # Sadece modeli yükle (Cache için)
    load_model(settings.clip_model_name, settings.clip_pretrained)
    
    logger.info("🏷️ Etiketlerin (Tags) vektör şifreleri çıkarılıyor...")
    tag_vectors = {}
    for tag in TAGS:
        # Daha doğru isabet için "a photo of a {tag} 3d model" formatını kullanabiliriz, ama direkt kelime de çok güçlü
        vec = encode_text(tag, settings.clip_model_name, settings.clip_pretrained)
        if vec is not None:
            tag_vectors[tag] = vec

    tag_names = list(tag_vectors.keys())
    tag_matrix = np.array([tag_vectors[t] for t in tag_names]) # Şekil: (N_tags, 768)
    
    # Vektörleri normalize et ki kosinüs benzerliği doğru çıksın
    tag_matrix_norm = np.linalg.norm(tag_matrix, axis=1, keepdims=True)
    tag_matrix = tag_matrix / (tag_matrix_norm + 1e-8)

    logger.info("📦 ChromaDB'den tüm ID listesi çekiliyor...")
    
    # Sadece ID'leri çek (hafif işlem)
    all_ids_data = store.collection.get(include=[])
    all_ids = all_ids_data["ids"]
    total = len(all_ids)
    
    logger.info(f"🚀 Toplam {total} adet modelin üzerine ZERO-SHOT Derin Etiketleme başlıyor!")
    
    batch_size = 500
    start_time = time.time()
    processed = 0
    
    for i in range(0, total, batch_size):
        batch_ids = all_ids[i:i+batch_size]
        
        def process_batch_data(b_data):
            b_embs = np.array(b_data["embeddings"])
            b_metas = b_data["metadatas"]
            
            emb_n = np.linalg.norm(b_embs, axis=1, keepdims=True)
            b_embs = b_embs / (emb_n + 1e-8)
            
            s = np.dot(b_embs, tag_matrix.T)
            t4_idx = np.argsort(s, axis=1)[:, -4:][:, ::-1] 
            
            u_metas = []
            for j, t_idx in enumerate(t4_idx):
                m = b_metas[j] if b_metas[j] else {}
                t_tags = [tag_names[idx] for idx in t_idx]
                m["deep_tags"] = ", ".join(t_tags)
                u_metas.append(m)
                
            store.collection.update(ids=b_data["ids"], metadatas=u_metas)
            return len(b_data["ids"])

        try:
            # Sadece bu paketin vektörlerini çek
            batch_data = store.collection.get(
                ids=batch_ids,
                include=["embeddings", "metadatas"]
            )
            processed += process_batch_data(batch_data)
        except Exception as e:
            # Hayalet ID hatası! Bu paketi tek tek işleyerek bozuk olanı atla.
            for single_id in batch_ids:
                try:
                    single_data = store.collection.get(ids=[single_id], include=["embeddings", "metadatas"])
                    processed += process_batch_data(single_data)
                except Exception as inner_e:
                    logger.warning(f"Hayalet ID atlandı: {single_id}")
                    pass
        
        elapsed = time.time() - start_time
        logger.info(f"⚡ İşlendi: {processed} / {total} | Geçen Süre: {elapsed:.1f} sn")

        
    logger.info("🎉 TÜM ARŞİVE ELİT ETİKETLER BAŞARIYLA BASILDI! (Zero-Shot Tamamlandı)")

if __name__ == "__main__":
    run_zero_shot()
