"""Kanalların toplam mesaj sayısını ve scraper ilerleme durumunu gösterir."""
import asyncio
import json
import sys
import os
from pathlib import Path

# Windows encoding fix
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from telethon import TelegramClient

async def main():
    settings = get_settings()
    channels = settings.get_channel_list()
    
    state_path = settings.get_data_path() / "scraper_state.json"
    state = {}
    if state_path.exists():
        with open(state_path, "r") as f:
            state = json.load(f)
    
    session_path = str(settings.get_data_path() / "list_session")
    client = TelegramClient(
        session_path,
        settings.telegram_api_id,
        settings.telegram_api_hash,
    )
    client.flood_sleep_threshold = 120
    
    await client.start(phone=settings.telegram_phone)
    
    total_messages = 0
    total_scraped = 0
    results = []
    
    for i, ch in enumerate(channels):
        try:
            entity = await client.get_entity(ch)
            msgs = await client.get_messages(entity.id, limit=1)
            last_msg_id = msgs[0].id if msgs else 0
            
            channel_id = str(entity.id)
            scraped_up_to = state.get(channel_id, 0)
            remaining = max(0, last_msg_id - scraped_up_to)
            
            # ASCII-safe isim
            name = getattr(entity, "title", ch)
            safe_name = name.encode('ascii', 'replace').decode('ascii')
            
            results.append({
                "name": safe_name,
                "total": last_msg_id,
                "scraped_to": scraped_up_to,
                "remaining": remaining,
            })
            total_messages += last_msg_id
            total_scraped += scraped_up_to
            
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(channels)} kanal kontrol edildi...")
                
        except Exception as e:
            print(f"HATA {ch}: {e}")
            await asyncio.sleep(3)
    
    await client.disconnect()
    
    print(f"\n{'='*72}")
    print(f"{'Kanal':<40} {'Toplam':>8} {'Taranan':>8} {'Kalan':>8} {'%':>5}")
    print(f"{'='*72}")
    
    results.sort(key=lambda x: x["remaining"], reverse=True)
    for r in results:
        pct = (r["scraped_to"] / r["total"] * 100) if r["total"] > 0 else 100
        print(f"{r['name'][:39]:<40} {r['total']:>8} {r['scraped_to']:>8} {r['remaining']:>8}  {pct:>4.0f}%")
    
    print(f"{'='*72}")
    print(f"{'TOPLAM':<40} {total_messages:>8} {total_scraped:>8} {total_messages - total_scraped:>8}")
    remaining_total = total_messages - total_scraped
    speed_per_min = 300
    eta_min = remaining_total / speed_per_min if speed_per_min > 0 else 0
    print(f"\nTahmini kalan sure (~{speed_per_min} gorsel/dk): {eta_min:.0f} dakika ({eta_min/60:.1f} saat)")

if __name__ == "__main__":
    asyncio.run(main())
