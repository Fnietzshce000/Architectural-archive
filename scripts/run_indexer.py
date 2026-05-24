"""
İndeksleme Pipeline — Dedublikasyon + CLIP vektörleştirme + ChromaDB kayıt.

Kullanım:
    python scripts/run_indexer.py                 # Tam pipeline
    python scripts/run_indexer.py --skip-dedup    # Dedublikasyonu atla
    python scripts/run_indexer.py --reset         # İndeksi sıfırla ve yeniden oluştur
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from preprocessing.deduplicator import ImageDeduplicator
from preprocessing.image_processor import ImageProcessor
from indexer.clip_encoder import encode_images_batch, load_model
from indexer.chroma_store import ChromaStore
from scraper.models import TelegramMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_indexer")


def load_metadata(data_dir: Path) -> dict:
    """
    Metadata dosyasından mesaj bilgilerini yükler.
    Returns: {image_path: TelegramMessage}
    """
    metadata_file = data_dir / "messages_metadata.jsonl"
    metadata_map = {}
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        msg = TelegramMessage.from_dict(data)
                        if msg.image_path:
                            metadata_map[msg.image_path] = msg
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"Metadata satırı okunamadı: {e}")
    return metadata_map


def main(args):
    settings = get_settings()
    data_dir = settings.get_data_path()
    images_dir = settings.get_images_path()

    logger.info("🚀 İndeksleme pipeline başlatılıyor...")
    logger.info(f"   Görsel dizini: {images_dir}")
    logger.info(f"   ChromaDB: {settings.chroma_db_path}")

    # ── Adım 1: Metadata yükleme ──
    logger.info("\n📋 Adım 1: Metadata yükleniyor...")
    metadata_map = load_metadata(data_dir)
    logger.info(f"   {len(metadata_map)} mesaj metadata'sı bulundu")

    # ── Adım 2: Görsel tarama ve doğrulama ──
    logger.info("\n📁 Adım 2: Görseller taranıyor...")
    processor = ImageProcessor()
    all_images = processor.scan_directory(images_dir)

    if not all_images:
        logger.error(
            "❌ Hiç görsel bulunamadı! Önce scraper'ı çalıştırın:\n"
            "   python scripts/run_scraper.py"
        )
        return

    # ── Adım 3: Dedublikasyon ──
    if not args.skip_dedup:
        logger.info("\n🔍 Adım 3: Dedublikasyon yapılıyor...")
        dedup = ImageDeduplicator(
            hash_db_path=data_dir / "hash_db.json",
            dhash_threshold=5,
            phash_threshold=8,
        )
        unique_images, duplicate_images = dedup.process_directory(images_dir)
        logger.info(f"   ✅ {len(unique_images)} benzersiz, {len(duplicate_images)} duplikat")
    else:
        logger.info("\n⏭️ Adım 3: Dedublikasyon atlandı")
        unique_images = all_images

    # ── Adım 4: ChromaDB hazırlık ──
    logger.info(f"\n📦 Adım 4: ChromaDB hazırlanıyor...")
    store = ChromaStore(
        db_path=settings.chroma_db_path,
        collection_name=settings.get_collection_name(),
    )

    if args.reset:
        logger.info("   🔄 İndeks sıfırlanıyor...")
        store.reset()

    # Zaten indekslenmiş görselleri filtrele
    existing_ids = set(store.get_all_ids())
    images_to_index = []
    for img_path in unique_images:
        doc_id = Path(img_path).stem  # ör: "123_456"
        if doc_id not in existing_ids:
            images_to_index.append(img_path)

    if not images_to_index:
        logger.info("   ✅ Tüm görseller zaten indekslenmiş!")
        logger.info(f"   Toplam kayıt: {store.get_count()}")
        return

    logger.info(f"   {len(images_to_index)} yeni görsel indekslenecek")

    # ── Adım 5: CLIP vektörleştirme ──
    logger.info(f"\n🧠 Adım 5: CLIP ile vektörleştirme ({settings.clip_model_name})...")

    # Batch encode
    vectors = encode_images_batch(
        image_paths=images_to_index,
        batch_size=32,
        model_name=settings.clip_model_name,
        pretrained=settings.clip_pretrained,
    )

    # ── Adım 6: ChromaDB'ye yazma ──
    logger.info("\n💾 Adım 6: ChromaDB'ye kaydediliyor...")

    doc_ids = []
    embeddings = []
    metadatas = []
    documents = []
    skipped = 0

    from config import MetadataSchema

    for img_path, vector in tqdm(
        zip(images_to_index, vectors),
        total=len(images_to_index),
        desc="ChromaDB kayıt",
    ):
        if vector is None:
            skipped += 1
            continue

        doc_id = Path(img_path).stem
        meta = metadata_map.get(img_path)

        if meta:
            metadata = {
                MetadataSchema.CHANNEL_ID: meta.channel_id,
                MetadataSchema.CHANNEL_TITLE: meta.channel_title,
                MetadataSchema.CHANNEL_USERNAME: meta.channel_username or "",
                MetadataSchema.MESSAGE_ID: meta.message_id,
                MetadataSchema.CAPTION: meta.caption,
                MetadataSchema.IMAGE_PATH: meta.image_path,
                MetadataSchema.DEEP_LINK: meta.deep_link,
                MetadataSchema.TIMESTAMP: meta.timestamp,
                MetadataSchema.SOURCE: "Indexer",
                MetadataSchema.MODEL_NAME: settings.clip_model_name,
                MetadataSchema.SCHEMA_VERSION_KEY: MetadataSchema.CURRENT_SCHEMA_VERSION,
            }
            # file_names ek alanı (ChromaDB list kabul etmez, string'e çevir)
            file_names = getattr(meta, "file_names", [])
            if file_names:
                import json as _json
                metadata["file_names"] = _json.dumps(file_names, ensure_ascii=False)
            document_text = meta.caption
        else:
            # Metadata yoksa minimal kayıt
            metadata = {
                MetadataSchema.IMAGE_PATH: img_path,
                MetadataSchema.DEEP_LINK: "",
                MetadataSchema.CAPTION: "",
                MetadataSchema.CHANNEL_TITLE: "",
                MetadataSchema.SOURCE: "Indexer",
                MetadataSchema.MODEL_NAME: settings.clip_model_name,
                MetadataSchema.SCHEMA_VERSION_KEY: MetadataSchema.CURRENT_SCHEMA_VERSION,
            }
            document_text = Path(img_path).stem

        doc_ids.append(doc_id)
        embeddings.append(vector)
        metadatas.append(metadata)
        documents.append(document_text)

    # Batch yazma
    if doc_ids:
        store.add_batch(
            doc_ids=doc_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            batch_size=500,
        )

    logger.info(f"\n{'='*50}")
    logger.info(f"🎉 İndeksleme tamamlandı!")
    logger.info(f"   Yeni kayıt: {len(doc_ids)}")
    logger.info(f"   Atlanan: {skipped}")
    logger.info(f"   Toplam indeks boyutu: {store.get_count()}")
    logger.info(f"   Sonraki adım: streamlit run ui/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Model İndeksleme Pipeline")
    parser.add_argument(
        "--skip-dedup", action="store_true",
        help="Dedublikasyon adımını atla"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="ChromaDB indeksini sıfırla ve yeniden oluştur"
    )
    args = parser.parse_args()

    main(args)
