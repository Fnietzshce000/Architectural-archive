"""
Telegram'da uye oldugun TUM kanallari listeler.
Mevcut TARGET_CHANNELS ile karsilastirip eksikleri gosterir.
"""
import asyncio
import sys
import os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from scraper.client import TelegramClientManager
from telethon.tl.types import Channel

async def main():
    settings = get_settings()
    current_channels = set(settings.get_channel_list())
    current_usernames = set()
    for ch in current_channels:
        current_usernames.add(ch.replace("@", "").lower())

    manager = TelegramClientManager(
        api_id=settings.telegram_api_id,
        api_hash=settings.telegram_api_hash,
        phone=settings.telegram_phone,
        session_dir=str(settings.get_data_path()),
    )

    async with manager as client:
        all_channels = []
        async for dialog in client.iter_dialogs():
            if isinstance(dialog.entity, Channel):
                username = getattr(dialog.entity, "username", None)
                title = dialog.entity.title
                channel_id = dialog.entity.id
                all_channels.append({
                    "title": title,
                    "username": username,
                    "id": channel_id,
                })

        # Sonuclari dosyaya yaz (encoding sorunu olmaz)
        out = Path(PROJECT_ROOT / "data" / "channel_report.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"Telegram'da {len(all_channels)} kanal/grup bulundu\n")
            f.write(f"Mevcut TARGET_CHANNELS'da {len(current_channels)} kanal var\n")
            f.write("=" * 60 + "\n\n")

            new_channels = []
            for ch in all_channels:
                uname = (ch["username"] or "").lower()
                if uname and uname not in current_usernames:
                    new_channels.append(ch)

            if new_channels:
                f.write(f"EKLENMEMIS {len(new_channels)} KANAL:\n\n")
                for i, ch in enumerate(new_channels, 1):
                    f.write(f"  {i:3d}. @{ch['username']}  --  {ch['title']}\n")
                
                new_tags = ",".join([f"@{ch['username']}" for ch in new_channels])
                f.write(f"\n{'='*60}\n")
                f.write(f"Kopyala-yapistir formati (.env'ye ekle):\n")
                f.write(f"{'='*60}\n")
                f.write(f",{new_tags}\n")
            else:
                f.write("Tum kanallar zaten TARGET_CHANNELS'a eklenmis!\n")

            private = [ch for ch in all_channels if not ch["username"]]
            if private:
                f.write(f"\n{len(private)} adet private kanal (username yok):\n")
                for ch in private[:15]:
                    f.write(f"  - {ch['title']} (ID: -{ch['id']})\n")

        print(f"Rapor yazildi: {out}")

if __name__ == "__main__":
    asyncio.run(main())
