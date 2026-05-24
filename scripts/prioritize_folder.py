import asyncio
import sys
import re
from pathlib import Path

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
    
    dialog_filters_result = await client(GetDialogFiltersRequest())
    
    target_usernames = []
    
    for dialog_filter in dialog_filters_result.filters:
        if isinstance(dialog_filter, DialogFilter):
            if dialog_filter.title == "3D Model":
                print(f"Found '3D Model' folder with {len(dialog_filter.include_peers)} peers.")
                for peer in dialog_filter.include_peers:
                    try:
                        entity = await client.get_entity(peer)
                        username = getattr(entity, 'username', None)
                        if username:
                            target_usernames.append(f"@{username}")
                        else:
                            target_usernames.append(f"@{getattr(entity, 'title', str(entity.id)).replace(' ', '')}") # fallback
                    except Exception as e:
                        print(f"Could not fetch peer {peer}")
                        
    await client.disconnect()
    
    if target_usernames:
        # Join with comma
        new_targets = ",".join(target_usernames)
        print(f"Channels: {new_targets}")
        
        # update .env
        env_path = PROJECT_ROOT / ".env"
        with open(env_path, "r", encoding="utf-8") as f:
            env_content = f.read()
            
        new_env_content = re.sub(r"TARGET_CHANNELS=(.*)", f"TARGET_CHANNELS={new_targets}", env_content)
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(new_env_content)
            
        print("Updated .env successfully.")
    else:
        print("No channels found.")

if __name__ == "__main__":
    asyncio.run(main())
