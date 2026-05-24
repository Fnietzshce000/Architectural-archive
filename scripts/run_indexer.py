"""
Indexing Pipeline — Deduplication + CLIP vectorization + ChromaDB storage.

Usage:
    python scripts/run_indexer.py                 # Full pipeline
    python scripts/run_indexer.py --skip-dedup    # Skip deduplication
    python scripts/run_indexer.py --reset         # Reset index and rebuild
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# Add project root to path
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


# tqdm wrapper that also logs progress milestones to the log file
class LoggingTqdm(tqdm):
    """tqdm subclass that logs progress at regular intervals."""
    def __init__(self, *args, log_interval: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_interval = log_interval
        self._last_logged_pct = -1

    def update(self, n=1):
        super().update(n)
        if self.total:
            pct = int(100 * self.n / self.total)
            if pct >= self._last_logged_pct + self._log_interval:
                self._last_logged_pct = pct
                elapsed = self.format_dict.get("elapsed", 0)
                rate = self.format_dict.get("rate", 0)
                eta = (self.total - self.n) / rate if rate else 0
                logger.info(
                    f"Progress: {pct}% ({self.n:,}/{self.total:,}) "
                    f"| Speed: {rate:.1f} it/s | ETA: {eta:.0f}s"
                )


def load_metadata(data_dir: Path) -> dict:
    """
    Load message metadata from JSONL file.
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
                        logger.warning(f"Failed to parse metadata line: {e}")
    return metadata_map


def main(args):
    settings = get_settings()
    data_dir = settings.get_data_path()
    images_dir = settings.get_images_path()

    logger.info("🚀 Starting indexing pipeline...")
    logger.info(f"   Image directory: {images_dir}")
    logger.info(f"   ChromaDB path:   {settings.chroma_db_path}")

    # ── Step 1: Load metadata ──
    logger.info("\n📋 Step 1: Loading metadata...")
    metadata_map = load_metadata(data_dir)
    logger.info(f"   Found {len(metadata_map):,} message metadata entries")

    # ── Step 2: Scan and validate images (with cache) ──
    logger.info("\n📁 Step 2: Scanning images (cached)...")
    processor = ImageProcessor(cache_dir=data_dir)
    all_images = processor.scan_directory(images_dir)

    if not all_images:
        logger.error(
            "❌ No images found! Run the scraper first:\n"
            "   python scripts/run_scraper.py"
        )
        return

    # ── Step 3: Deduplication ──
    if not args.skip_dedup:
        logger.info("\n🔍 Step 3: Running deduplication...")
        dedup = ImageDeduplicator(
            hash_db_path=data_dir / "hash_db.json",
            dhash_threshold=5,
            phash_threshold=8,
        )
        unique_images, duplicate_images = dedup.process_directory(images_dir)
        logger.info(f"   ✅ {len(unique_images):,} unique, {len(duplicate_images):,} duplicates")
    else:
        logger.info("\n⏭️ Step 3: Deduplication skipped")
        unique_images = all_images

    # ── Step 4: ChromaDB setup ──
    logger.info("\n📦 Step 4: Preparing ChromaDB...")
    store = ChromaStore(
        db_path=settings.chroma_db_path,
        collection_name=settings.get_collection_name(),
    )

    if args.reset:
        logger.info("   🔄 Resetting index...")
        store.reset()

    # Filter out already-indexed images
    existing_ids = set(store.get_all_ids())
    images_to_index = []
    for img_path in unique_images:
        doc_id = Path(img_path).stem
        if doc_id not in existing_ids:
            images_to_index.append(img_path)

    if not images_to_index:
        logger.info("   ✅ All images already indexed!")
        logger.info(f"   Total records: {store.get_count():,}")
        return

    logger.info(f"   {len(images_to_index):,} new images to index")

    # ── Step 5: CLIP vectorization ──
    logger.info(f"\n🧠 Step 5: Encoding with CLIP ({settings.clip_model_name})...")

    vectors = encode_images_batch(
        image_paths=images_to_index,
        batch_size=32,
        model_name=settings.clip_model_name,
        pretrained=settings.clip_pretrained,
    )

    # ── Step 6: Write to ChromaDB ──
    logger.info("\n💾 Step 6: Saving to ChromaDB...")

    doc_ids = []
    embeddings = []
    metadatas = []
    documents = []
    skipped = 0

    from config import MetadataSchema

    for img_path, vector in LoggingTqdm(
        zip(images_to_index, vectors),
        total=len(images_to_index),
        desc="ChromaDB write",
        log_interval=10,
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
            file_names = getattr(meta, "file_names", [])
            if file_names:
                import json as _json
                metadata["file_names"] = _json.dumps(file_names, ensure_ascii=False)
            document_text = meta.caption
        else:
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

    # Batch write
    if doc_ids:
        store.add_batch(
            doc_ids=doc_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            batch_size=500,
        )

    logger.info(f"\n{'='*50}")
    logger.info(f"🎉 Indexing complete!")
    logger.info(f"   New records:      {len(doc_ids):,}")
    logger.info(f"   Skipped (errors): {skipped:,}")
    logger.info(f"   Total index size: {store.get_count():,}")
    logger.info(f"   Next step: streamlit run ui/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Model Indexing Pipeline")
    parser.add_argument(
        "--skip-dedup", action="store_true",
        help="Skip the deduplication step"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Reset ChromaDB index and rebuild from scratch"
    )
    args = parser.parse_args()

    main(args)
