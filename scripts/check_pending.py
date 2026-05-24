import asyncio
import json
from telethon import TelegramClient
from config import get_settings
from pathlib import Path

async def check_pending():
    settings = get_settings()
    state_path = Path("data/scraper_state.json")
    
    if not state_path.exists():
        print("Scraper state file not found!")
        return

    with open(state_path, 'r') as f:
        state = json.load(f)

    client = TelegramClient('data/scraper_session', settings.telegram_api_id, settings.telegram_api_hash)
    await client.start(phone=settings.telegram_phone)
    
    total_pending = 0
    print("-" * 50)
    print(f"{'Kanal ID':<15} | {'Bizdeki':<10} | {'Güncel':<10} | {'Bekleyen':<10}")
    print("-" * 50)

    for channel_id, last_id in state.items():
        try:
            # Handle string IDs like @username
            target = channel_id
            if str(channel_id).replace("-", "").isdigit():
                target = int(channel_id)
            
            entity = await client.get_entity(target)
            # En son mesajı al
            async for message in client.iter_messages(entity, limit=1):
                current_id = message.id
                pending = max(0, current_id - last_id)
                total_pending += pending
                
                # Sadece bekleyen mesajı olanları bas (kalabalık yapmasın)
                if pending > 0:
                    print(f"{str(channel_id):<15} | {last_id:<10} | {current_id:<10} | {pending:<10}")
        except Exception as e:
            # print(f"Error checking {channel_id}: {e}")
            pass

    print("-" * 50)
    print(f"BOMBA HABER! Toplam bekleyen mesaj sayısı: {total_pending}")
    print("-" * 50)
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(check_pending())
