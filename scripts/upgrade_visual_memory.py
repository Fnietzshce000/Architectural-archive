import os
import sys
import json
import logging
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

# Ana dizini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indexer.chroma_store import ChromaStore
from indexer.clip_encoder import load_model, encode_image
from config import get_settings

# Loglama
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    settings = get_settings()
    
    # 1. ESKİ Koleksiyonu Hazırla (Eski 512 Boyutlu Veriler)
    logger.info("📡 Eski koleksiyon bağlanıyor: models_3d")
    old_store = ChromaStore(collection_name="models_3d")
    
    # 2. YENİ Koleksiyonu Hazırla (Yeni 768 Boyutlu Veriler)
    logger.info("🎯 Yeni koleksiyon hazırlanıyor: models_3d_v2")
    new_store = ChromaStore(collection_name="models_3d_v2")
    
    # 💾 RESUME DESTEĞİ
    try:
        existing_ids = set(new_store.collection.get(include=[])["ids"])
        if existing_ids:
            logger.info(f"💾 {len(existing_ids)} model zaten taşınmış, atlanacak.")
    except Exception as e:
        logger.info(f"🆕 Yeni koleksiyon sıfırdan oluşturuluyor.")
        existing_ids = set()
    
    # 3. YENİ Modeli Yükle (ViT-L-14 - FP16 Full Performance)
    logger.info(f"💎 YENİ MODEL YÜKLENİYOR (Full Performance): {settings.clip_model_name}...")
    model, preprocess, _, device = load_model(settings.clip_model_name, settings.clip_pretrained)
    
    # FP16 (Yarı Hassasiyet) ile hızı ikiye katlayalım
    if device == "cuda":
        model = model.half()
    
    # 4. Eski verileri al
    total_models = old_store.collection.count()
    batch_size = 128 # Full Performance sınırı
    
    logger.info(f"🚀 {total_models} model yeni nesil görsel zekaya taşınıyor...")
    
    SKIP_TO = 193000  # Zaten yükseltilmiş kayıtları atla
    logger.info(f"⏩ İlk {SKIP_TO} kayıt atlanıyor (zaten yükseltilmiş)...")
    
    for offset in range(SKIP_TO, total_models, batch_size):
        results = old_store.collection.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset
        )
        
        if not results or not results["ids"]:
            break
            
        current_batch_ids = results["ids"]
        current_batch_metas = results["metadatas"]
        
        new_embeddings = []
        valid_ids = []
        valid_metas = []
        
        for i, doc_id in enumerate(current_batch_ids):
            img_path = Path(f"data/images/{doc_id}.jpg")
            
            if img_path.exists():
                try:
                    # Resmi yükle ve modele hazırla
                    image = Image.open(img_path).convert("RGB")
                    img_tensor = preprocess(image).unsqueeze(0).to(device)
                    
                    # Yeni modelle vektörleştir
                    with torch.no_grad():
                        if device == "cuda":
                            with torch.amp.autocast("cuda"):
                                features = model.encode_image(img_tensor)
                        else:
                            features = model.encode_image(img_tensor)
                        
                        # Normalize et
                        features /= features.norm(dim=-1, keepdim=True)
                        emb = features.cpu().numpy().flatten()
                        
                        new_embeddings.append(emb.tolist())
                        valid_ids.append(doc_id)
                        valid_metas.append(current_batch_metas[i])
                except Exception as e:
                    logger.warning(f"Resim işlenemedi {doc_id}: {e}")
            else:
                logger.warning(f"Resim bulunamadı, atlanıyor: {img_path}")
        
        # YENİ Koleksiyona Ekle
        if valid_ids:
            new_store.collection.add(
                ids=valid_ids,
                embeddings=new_embeddings,
                metadatas=valid_metas
            )
            
        logger.info(f"✅ İlerleme: {offset + len(current_batch_ids)} / {total_models}")

    logger.info("🎉 GÖRSEL HAFIZA BAŞARIYLA YÜKSELTİLDİ!")
    logger.info(f"🚀 Artık {settings.get_collection_name()} (ViT-L-14) devrede.")

if __name__ == "__main__":
    main()
