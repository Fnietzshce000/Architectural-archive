import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from indexer.chroma_store import ChromaStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MegaSync kaynakli kopyalari temizle")
    parser.add_argument("--confirm", action="store_true", help="Gercek DB/disk silme islemini baslat")
    args = parser.parse_args()

    settings = get_settings()
    store = ChromaStore(
        db_path=settings.chroma_db_path,
        collection_name=settings.get_collection_name(),
    )

    logger.info("MegaSync kaynakli kopyalar tespit ediliyor...")
    ids_to_delete = []
    batch_size = 5000
    total = store.get_count()
    for offset in range(0, total, batch_size):
        results = store.collection.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset,
        )
        if not results or not results.get("ids"):
            break
        for doc_id, meta in zip(results["ids"], results["metadatas"]):
            if (meta or {}).get("source") == "MegaSync":
                ids_to_delete.append(doc_id)

    images_dir = Path(settings.data_dir) / "images"
    files_to_delete = []
    total_size = 0
    for doc_id in ids_to_delete:
        img_path = images_dir / f"{doc_id}.jpg"
        if img_path.exists():
            files_to_delete.append(img_path)
            total_size += img_path.stat().st_size

    logger.info("Analiz sonucu:")
    logger.info(f"  Silinecek DB kaydi: {len(ids_to_delete)}")
    logger.info(f"  Silinecek dosya: {len(files_to_delete)}")
    logger.info(f"  Tahmini alan kazanci: {total_size / (1024 * 1024):.2f} MB")

    if not ids_to_delete:
        logger.info("Silinecek kopya bulunamadi.")
        return

    if not args.confirm:
        logger.warning("DRY-RUN modu: hicbir DB kaydi veya dosya silinmedi.")
        logger.warning("Gercek temizlik icin: python scripts/clean_duplicates.py --confirm")
        return

    store.delete(ids_to_delete)

    deleted_files = 0
    for img_path in files_to_delete:
        try:
            img_path.unlink()
            deleted_files += 1
        except Exception as exc:
            logger.warning(f"Dosya silinemedi: {img_path} ({exc})")

    logger.info(f"Temizlik tamamlandi. Diskten silinen dosya: {deleted_files}")


if __name__ == "__main__":
    main()
