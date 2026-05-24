
import asyncio
import json
import os
from pathlib import Path
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from dotenv import load_dotenv

async def main():
    load_dotenv()
    api_id = int(os.getenv("TELEGRAM_API_ID"))
    api_hash = os.getenv("TELEGRAM_API_HASH")
    phone = os.getenv("TELEGRAM_PHONE")
    target_channels = os.getenv("TARGET_CHANNELS").split(",")
    data_dir = Path("data")
    
    # Load state
    state_file = data_dir / "scraper_state.json"
    state = {}
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
            
    # Load map
    map_file = data_dir / "channel_map.json"
    channel_map = {}
    if map_file.exists():
        with open(map_file, "r", encoding="utf-8") as f:
            channel_map = json.load(f)

    client = TelegramClient(str(data_dir / "report_session"), api_id, api_hash)
    await client.connect()
    if not await client.is_user_authorized():
        print("Lütfen önce giriş yapın.")
        return

    print(f"{'Kanal':<30} | {'İşlenen ID':<10} | {'Son Mesaj ID':<12} | {'Kalan'}")
    print("-" * 70)

    for channel_name in target_channels:
        clean_name = channel_name.replace("@", "")
        # Try to get numeric ID from map
        channel_id = channel_map.get(clean_name, channel_name)
        
        # Get last processed from state
        # State keys are strings of numeric IDs usually
        processed_id = 0
        if str(channel_id) in state:
            processed_id = state[str(channel_id)]
        elif channel_name in state:
            processed_id = state[channel_name]

        try:
            # Get latest message ID
            history = await client(GetHistoryRequest(
                peer=channel_id,
                offset_id=0,
                offset_date=None,
                add_offset=0,
                limit=1,
                max_id=0,
                min_id=0,
                hash=0
            ))
            
            if history.messages:
                latest_id = history.messages[0].id
                remaining = max(0, latest_id - processed_id)
                print(f"{channel_name:<30} | {processed_id:<10} | {latest_id:<12} | {remaining}")
            else:
                print(f"{channel_name:<30} | {processed_id:<10} | {'?':<12} | {'?'}")
        except Exception as e:
            print(f"{channel_name:<30} | {processed_id:<10} | {'HATA':<12} | {str(e)[:20]}")
        
        # Slight delay to avoid flood
        await asyncio.sleep(0.5)

    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
