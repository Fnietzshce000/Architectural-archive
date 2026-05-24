"""
Telegram kanallarını listeler — 3D model kanallarını otomatik bulur.
"""
import asyncio
import sys
import io
from pathlib import Path

# Windows terminal Unicode fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

from telethon import TelegramClient
from telethon.tl.types import Channel, Chat

API_ID = 26541243
API_HASH = "5db64a68fdd4f0ad376d31900c97b0d0"

# 3D model ile ilgili anahtar kelimeler
KEYWORDS_3D = [
    "3d", "model", "3ds", "max", "blender", "archviz", "interior",
    "furniture", "decor", "render", "vray", "corona", "cgtrader",
    "turbosquid", "sketchup", "autocad", "revit", "архитектур",
    "модел", "интерьер", "мебел", "визуализ", "design", "arch",
    "panel", "texture", "material", "hdri", "asset", "obj", "fbx",
    "mobilya", "dekor", "iç mimar", "duvar", "zemin", "kitchen",
    "bathroom", "bedroom", "living", "sofa", "chair", "table", "lamp",
]


async def main():
    client = TelegramClient("data/list_session", API_ID, API_HASH)
    await client.start()
    
    me = await client.get_me()
    print(f"\nGiris yapildi: {me.first_name} ({me.phone})\n")
    
    all_channels = []
    matched_channels = []
    
    print("Kanallar taraniyor...\n")
    
    async for dialog in client.iter_dialogs():
        entity = dialog.entity
        if isinstance(entity, Channel):
            title = entity.title or ""
            username = entity.username or ""
            channel_id = entity.id
            
            info = {
                "id": channel_id,
                "title": title,
                "username": username,
                "members": getattr(entity, "participants_count", 0) or 0,
            }
            all_channels.append(info)
            
            # 3D anahtar kelime kontrolu
            check_text = f"{title} {username}".lower()
            if any(kw in check_text for kw in KEYWORDS_3D):
                matched_channels.append(info)
    
    # Sonuclari yazdir
    print(f"{'='*70}")
    print(f"  TOPLAM KANAL: {len(all_channels)}")
    print(f"  3D MODEL KANALLARI (otomatik tespit): {len(matched_channels)}")
    print(f"{'='*70}\n")
    
    if matched_channels:
        print("--- 3D MODEL KANALLARI ---\n")
        for i, ch in enumerate(matched_channels, 1):
            uname = f"@{ch['username']}" if ch['username'] else f"ID:{ch['id']}"
            print(f"  {i:2d}. {ch['title'][:45]:<45s} {uname}")
    
    print(f"\n--- TUM KANALLAR ({len(all_channels)}) ---\n")
    for i, ch in enumerate(all_channels, 1):
        uname = f"@{ch['username']}" if ch['username'] else f"ID:{ch['id']}"
        marker = " [3D]" if ch in matched_channels else ""
        print(f"  {i:2d}. {ch['title'][:45]:<45s} {uname}{marker}")
    
    # Eslesen kanallari .env formatinda kaydet
    if matched_channels:
        channels_str = ",".join(
            f"@{ch['username']}" if ch['username'] else str(-100*1 - ch['id'] + ch['id'])
            for ch in matched_channels
        )
        env_entries = []
        for ch in matched_channels:
            if ch['username']:
                env_entries.append(f"@{ch['username']}")
            else:
                env_entries.append(str(ch['id']))
        
        print(f"\n{'='*70}")
        print(f"  .env icin TARGET_CHANNELS degeri:")
        print(f"  TARGET_CHANNELS={','.join(env_entries)}")
        print(f"{'='*70}")
        
        # Dosyaya da kaydet
        with open("data/detected_channels.txt", "w", encoding="utf-8") as f:
            f.write(",".join(env_entries))
            f.write("\n\n# Detay:\n")
            for ch in matched_channels:
                uname = f"@{ch['username']}" if ch['username'] else f"ID:{ch['id']}"
                f.write(f"# {ch['title']} — {uname}\n")
        print("  Ayrica data/detected_channels.txt dosyasina kaydedildi.")
    
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
