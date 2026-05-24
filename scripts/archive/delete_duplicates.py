import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# Ana dizini path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from indexer.chroma_store import ChromaStore

# Loglama
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Mükerrer Modelleri Temizleme Scripti")
    parser.add_argument("--confirm", action="store_true", help="Gerçekten silme işlemini başlat")
    args = parser.parse_args()

    settings = get_settings()
    cluster_file = settings.get_data_path() / "clusters.json"

    if not cluster_file.exists():
        logger.error("❌ clusters.json bulunamadı! Önce cluster_archive.py çalıştırılmalı.")
        return

    with open(cluster_file, "r", encoding="utf-8") as f:
        clusters = json.load(f)

    store = ChromaStore(
        db_path=settings.chroma_db_path,
        collection_name=settings.get_collection_name(),
    )

    # Silinecek ID'leri ve Dosyaları Belirle
    to_delete_ids = []
    to_delete_files = []
    total_saved_space = 0
    images_dir = settings.get_images_path()

    logger.info("🔍 Mükerrer kopyalar analiz ediliyor...")
    for rep_id, data in clusters.items():
        members = data["members"]
        # Lideri (rep_id) koru, diğerlerini sil
        for member_id in members:
            if member_id != rep_id:
                to_delete_ids.append(member_id)
                img_path = images_dir / f"{member_id}.jpg"
                if img_path.exists():
                    to_delete_files.append(img_path)
                    total_saved_space += img_path.stat().st_size

    total_to_delete = len(to_delete_ids)

    if total_to_delete == 0:
        logger.info("✅ Silinecek kopya bulunamadı! Arşiv zaten temiz.")
        return

    # --- RAPORLAMA ---
    space_mb = total_saved_space / (1024 * 1024)
    logger.info(f"\n{'='*40}")
    logger.info(f"📊 ANALİZ SONUCU:")
    logger.info(f"   Silinecek Kayıt (DB): {total_to_delete}")
    logger.info(f"   Silinecek Dosya (Disk): {len(to_delete_files)}")
    logger.info(f"   Tahmini Kazanılacak Alan: {space_mb:.2f} MB")
    logger.info(f"{'='*40}\n")

    if not args.confirm:
        logger.warning("⚠️ DİKKAT: Şu an DRY-RUN (Simülasyon) modundasınız.")
        logger.warning("   Hiçbir işlem YAPILMADI. Gerçekten silmek için --confirm ekleyin.")
        logger.warning("   Örn: python scripts/delete_duplicates.py --confirm")
        return

    # --- GERÇEK SİLME İŞLEMİ ---
    logger.info("🚀 SİLME İŞLEMİ BAŞLATILIYOR...")

    # 1. Veritabanından Sil (500'lük batch'lerle — timeout koruması)
    logger.info("🗑️ ChromaDB temizleniyor...")
    for i in range(0, len(to_delete_ids), 500):
        batch = to_delete_ids[i:i + 500]
        store.delete(batch)

    # 2. Diskten Sil
    logger.info("📂 Disk temizleniyor...")
    deleted_count = 0
    for img_path in tqdm(to_delete_files, desc="Resimler Siliniyor"):
        try:
            img_path.unlink()
            deleted_count += 1
        except Exception as e:
            logger.debug(f"Dosya silinemedi: {img_path} ({e})")

    logger.info(f"\n✨ TEMİZLİK BAŞARIYLA TAMAMLANDI!")
    logger.info(f"✅ ChromaDB'den uçurulan: {total_to_delete}")
    logger.info(f"✅ Diskten silinen resim: {deleted_count}")
    logger.info(f"🎉 Arşivin artık %100 benzersiz modellerden oluşuyor!")


if __name__ == "__main__":
    main()
