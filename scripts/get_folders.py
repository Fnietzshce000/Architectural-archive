import asyncio
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
from telethon.tl.functions.messages import GetDialogFiltersRequest
from telethon.tl.types import DialogFilter

async def main():
    settings = get_settings()
    session_path = str(settings.get_data_path() / "list_session")
    client = TelegramClient(
        session_path,
        settings.telegram_api_id,
        settings.telegram_api_hash,
    )
    
    await client.start(phone=settings.telegram_phone)
    
    try:
        dialog_filters_result = await client(GetDialogFiltersRequest())
        
        for dialog_filter in dialog_filters_result.filters:
            if not isinstance(dialog_filter, DialogFilter):
                continue
                
            print(f"Folder: {dialog_filter.title}")
            print(f"  Includes: {len(dialog_filter.include_peers)} peers")
            
            for peer in dialog_filter.include_peers:
                try:
                    entity = await client.get_entity(peer)
                    name = getattr(entity, 'title', getattr(entity, 'username', 'Unknown'))
                    username = getattr(entity, 'username', None)
                    safe_name = name.encode('ascii', 'replace').decode('ascii')
                    print(f"    - {safe_name} (@{username}) ID: {entity.id}")
                except Exception as e:
                    print(f"    - Unknown Peer: {peer}")
    except Exception as e:
        print(f"Error: {e}")
            
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
