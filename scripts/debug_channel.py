import asyncio
from scraper.client import TelegramClientManager
from config import get_settings

async def debug_channel():
    settings = get_settings()
    manager = TelegramClientManager(
        api_id=settings.telegram_api_id,
        api_hash=settings.telegram_api_hash,
        phone=settings.telegram_phone,
        session_dir=str(settings.get_data_path())
    )
    async with manager as client:
        entity = await client.get_entity('@Model_Library_Free')
        print(f"--- CHANNEL ANALYSIS: {entity.title} ---")
        async for m in client.iter_messages(entity, limit=10):
            txt = m.text if m.text else "[EMPTY]"
            media_type = "NONE"
            if m.photo: media_type = "PHOTO"
            elif m.document: media_type = "DOC"
            
            print(f"ID: {m.id} | TYPE: {media_type} | TEXT: {txt[:100].replace('\n', ' ')}")

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(debug_channel())
    except Exception as e:
        print(f"Error: {e}")
