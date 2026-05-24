"""
Hızlı İndeksleme — Deduplikasyonu atlayıp direkt CLIP ile indeksler.
Çok sayıda görsel için deduplikasyon çok yavaş, bu script onu bypass eder.
"""
import logging
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from preprocessing.image_processor import ImageProcessor
from indexer.clip_encoder import load_model, encode_images_batch, unload_model
from indexer.chroma_store import ChromaStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quick_index")


def main():
    settings = get_settings()
    images_dir = settings.get_images_path()
    
    logger.info("🚀 Hızlı indeksleme başlatılıyor (deduplikasyon atlanıyor)...")
    
    # 1. Metadata yükle
    metadata_file = settings.get_data_path() / "messages_metadata.jsonl"
    metadata_map = {}
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                if "image_path" in data:
                    metadata_map[Path(data["image_path"]).name] = data
    logger.info(f"   {len(metadata_map)} mesaj metadata'sı bulundu")
    
    # 2. Görselleri tara  
    processor = ImageProcessor()
    valid_images = processor.scan_directory(str(images_dir))
    
    # 3. ChromaDB hazırla
    store = ChromaStore(
        db_path=str(settings.chroma_db_path),
        collection_name=settings.get_collection_name(),
    )
    existing_ids = set(store.get_all_ids())
    
    # Yeni görselleri filtrele
    new_images = []
    for img_path in valid_images:
        img_id = Path(img_path).stem
        if img_id not in existing_ids:
            new_images.append(img_path)
    
    logger.info(f"   {len(existing_ids)} zaten indeksli, {len(new_images)} yeni görsel")
    
    if not new_images:
        logger.info("✅ Tüm görseller zaten indeksli!")
        return
    
    # 4. CLIP encode
    logger.info(f"\n🧠 CLIP ile {len(new_images)} görsel vektörleştiriliyor...")
    load_model(
        model_name=settings.clip_model_name,
        pretrained=settings.clip_pretrained,
    )
    
    batch_size = 32
    all_ids = []
    all_embeddings = []
    all_metadatas = []
    errors = 0
    
    for i in range(0, len(new_images), batch_size):
        batch = new_images[i:i+batch_size]
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
                MetadataSchema.SOURCE: "QuickIndex",
                MetadataSchema.MODEL_NAME: settings.clip_model_name,
                MetadataSchema.SCHEMA_VERSION_KEY: MetadataSchema.CURRENT_SCHEMA_VERSION,
            })
        
        done = min(i + batch_size, len(new_images))
        logger.info(f"   📊 {done}/{len(new_images)} encode edildi ({errors} hata)")
    
    # 5. ChromaDB'ye kaydet
    logger.info(f"\n💾 {len(all_ids)} vektör ChromaDB'ye kaydediliyor...")
    store.add_batch(
        doc_ids=all_ids,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
        documents=[""] * len(all_ids),
    )
    
    # CLIP modelini bellekten temizle
    unload_model()
    
    total = len(existing_ids) + len(all_ids)
    logger.info(f"\n{'='*50}")
    logger.info(f"🎉 Hızlı indeksleme tamamlandı!")
    logger.info(f"   Yeni: {len(all_ids)} | Hata: {errors}")
    logger.info(f"   Toplam indeks: {total}")
    logger.info(f"   Streamlit'i yeniden başlat: streamlit run ui/app.py")


if __name__ == "__main__":
    main()
