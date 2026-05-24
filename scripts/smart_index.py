"""
Akıllı İndeksleme — Sadece son 48 saatte indirilen dosyaları bulur ve indeksler.
400K+ dosyayı tek tek taramak yerine sadece yeni dosyalara odaklanır.
"""
import os
import time
import logging
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from indexer.clip_encoder import load_model, encode_images_batch, unload_model
from indexer.chroma_store import ChromaStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smart_index")

def main():
    settings = get_settings()
    images_dir = str(settings.get_images_path())
    
    logger.info("🚀 Akıllı İndeksleme başlatılıyor (Sadece yeni dosyalar)...")
    
    # 1. Son 48 saatte indirilen görselleri bul (ÇOK HIZLI)
    logger.info("🔍 Son 48 saatte indirilen dosyalar aranıyor...")
    new_images = []
    now = time.time()
    two_days_ago = now - (2 * 24 * 60 * 60)
    
    try:
        for entry in os.scandir(images_dir):
            if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                if entry.stat().st_mtime > two_days_ago:
                    new_images.append(entry.path)
    except FileNotFoundError:
        logger.error(f"Görsel dizini bulunamadı: {images_dir}")
        return

    logger.info(f"   ✅ Son 48 saatte indirilen {len(new_images)} görsel bulundu.")
    
    if not new_images:
        logger.info("   İndekslenecek yeni görsel yok. İşlem tamam.")
        return

    # 2. ChromaDB Hazırla ve Zaten İndeksli Olanları Filtrele
    logger.info("📦 Veritabanı kontrol ediliyor...")
    store = ChromaStore(
        db_path=str(settings.chroma_db_path),
        collection_name=settings.get_collection_name(),
    )
    existing_ids = set(store.get_all_ids())
    
    images_to_index = []
    for img_path in new_images:
        img_id = Path(img_path).stem
        if img_id not in existing_ids:
            images_to_index.append(img_path)
            
    logger.info(f"   Gerçekten yeni olan (veritabanında olmayan): {len(images_to_index)} görsel.")
    
    if not images_to_index:
        logger.info("   Tüm yeni görseller zaten veritabanında. İşlem tamam.")
        return

    # 3. Metadata Yükle (Sadece yeni dosyalar için hızlı sözlük araması)
    logger.info("📋 Metadata yükleniyor...")
    metadata_file = settings.get_data_path() / "messages_metadata.jsonl"
    metadata_map = {}
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "image_path" in data:
                        metadata_map[Path(data["image_path"]).name] = data
                except:
                    pass

    # 4. CLIP Encode
    logger.info(f"\n🧠 CLIP ile {len(images_to_index)} görsel vektörleştiriliyor...")
    load_model(
        model_name=settings.clip_model_name,
        pretrained=settings.clip_pretrained,
    )
    
    batch_size = 32
    all_ids = []
    all_embeddings = []
    all_metadatas = []
    errors = 0
    
    for i in range(0, len(images_to_index), batch_size):
        batch = images_to_index[i:i+batch_size]
        embeddings = encode_images_batch(
            batch,
            model_name=settings.clip_model_name,
            pretrained=settings.clip_pretrained,
        )
        
        for j, embedding in enumerate(embeddings):
            if embedding is None:
                errors += 1
                continue
            
            img_path = batch[j]
            img_name = Path(img_path).name
            img_id = Path(img_path).stem
            meta = metadata_map.get(img_name, {})
            
            from config import MetadataSchema
            all_ids.append(img_id)
            all_embeddings.append(embedding)
            all_metadatas.append({
                MetadataSchema.IMAGE_PATH: str(img_path),
                MetadataSchema.CHANNEL_TITLE: meta.get("channel_title", ""),
                MetadataSchema.CHANNEL_USERNAME: meta.get("channel_username", ""),
                MetadataSchema.CAPTION: meta.get("caption", "")[:500],
                MetadataSchema.DEEP_LINK: meta.get("deep_link", ""),
                MetadataSchema.MESSAGE_ID: str(meta.get("message_id", "")),
                MetadataSchema.TIMESTAMP: meta.get("timestamp", ""),
                MetadataSchema.SOURCE: "SmartIndex",
                MetadataSchema.MODEL_NAME: settings.clip_model_name,
                MetadataSchema.SCHEMA_VERSION_KEY: MetadataSchema.CURRENT_SCHEMA_VERSION,
            })
        
        done = min(i + batch_size, len(images_to_index))
        logger.info(f"   📊 {done}/{len(images_to_index)} tamamlandı...")
    
    # 5. ChromaDB'ye kaydet
    logger.info(f"\n💾 {len(all_ids)} vektör ChromaDB'ye kaydediliyor...")
    if all_ids:
        store.add_batch(
            doc_ids=all_ids,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
            documents=[""] * len(all_ids),
        )
    
    unload_model()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"🎉 Akıllı indeksleme tamamlandı!")
    logger.info(f"   Eklenen: {len(all_ids)} | Hata: {errors}")
    logger.info(f"   Toplam Veritabanı: {len(existing_ids) + len(all_ids)}")

if __name__ == "__main__":
    main()
