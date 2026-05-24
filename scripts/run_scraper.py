"""
Scraper Çalıştırıcı — Telegram kanallarını tarar ve görselleri indirir.

Kullanım:
    python scripts/run_scraper.py                    # Tüm kanalları tara
    python scripts/run_scraper.py --limit 100        # Kanal başına 100 mesaj
    python scripts/run_scraper.py --no-resume        # Sıfırdan başla
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from scraper.client import TelegramClientManager
from scraper.channel_scraper import ChannelScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_scraper")


async def main(args):
    settings = get_settings()
    channels = settings.get_channel_list()

    if not channels:
        logger.error(
            "❌ Hedef kanal bulunamadı! .env dosyasında TARGET_CHANNELS ayarlayın.\n"
            "   Örnek: TARGET_CHANNELS=@channel1,@channel2,-1001234567890"
        )
        return

    logger.info(f"🚀 Scraper başlatılıyor — {len(channels)} kanal hedefleniyor")
    logger.info(f"   Kanallar: {', '.join(channels)}")

    # Telethon istemcisini başlat
    manager = TelegramClientManager(
        api_id=settings.telegram_api_id,
        api_hash=settings.telegram_api_hash,
        phone=settings.telegram_phone,
        session_dir=str(settings.get_data_path()),
    )

    async with manager as client:
        scraper = ChannelScraper(
            client=client,
            images_dir=str(settings.get_images_path()),
            data_dir=str(settings.get_data_path()),
            max_concurrent=150,
            download_delay=0,
        )

        # Tüm kanalları tara
        results = await scraper.scrape_all_channels(
            channels=channels,
            limit_per_channel=args.limit,
            resume=not args.no_resume,
            parallel_channels=32,
        )

        logger.info(f"\n{'='*50}")
        logger.info(f"🎉 Tarama tamamlandı!")
        logger.info(f"   Toplam indirilen görsel: {len(results)}")
        logger.info(f"   Görsel dizini: {settings.get_images_path()}")
        logger.info(f"   Sonraki adım: python scripts/run_indexer.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telegram Kanal Tarayıcı")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Kanal başına maximum mesaj sayısı (test için)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Sıfırdan başla (state'i görmezden gel)"
    )
    args = parser.parse_args()

    asyncio.run(main(args))
