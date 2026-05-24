"""
Kanal Tarayıcı — Telegram kanallarından görselleri ve metadata'yı çeker.
telegram_media_downloader'dan esinlenilmiş Telethon tabanlı scraper.
"""
import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Set

from telethon import TelegramClient
from telethon.errors import FloodWaitError, ChannelPrivateError
from telethon.tl.types import (
    Channel,
    Chat,
    Message,
    MessageMediaDocument,
    MessageMediaPhoto,
    Photo,
    Document,
)
from tqdm import tqdm

from scraper.models import TelegramMessage, build_deep_link

logger = logging.getLogger(__name__)

# 3D model dosya uzantıları
MODEL_EXTENSIONS = {
    ".max", ".blend", ".obj", ".fbx", ".3ds", ".stl",
    ".glb", ".gltf", ".ply", ".c4d", ".skp", ".dwg",
    ".zip", ".rar", ".7z",
}

# Görsel uzantıları
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# State dosyası
STATE_FILE = "scraper_state.json"


class ChannelScraper:
    """Telegram kanallarını tarar ve görselleri yerel diske indirir."""

    def __init__(
        self,
        client: TelegramClient,
        images_dir: str | Path,
        data_dir: str | Path,
        max_concurrent: int = 3,
        download_delay: float = 1.0,
    ):
        self.client = client
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.download_delay = download_delay

        # Kanal başına son taranan mesaj ID
        self.state: Dict[str, int] = self._load_state()
        # Kanal @username -> numerical_id haritası
        self.channel_map: Dict[str, int] = self._load_channel_map()
        # İndirilen mesajların metadata'sı
        self.metadata_file = self.data_dir / "messages_metadata.jsonl"

    # ──────────────────────────────────────────────
    # State Yönetimi
    # ──────────────────────────────────────────────

    def _load_state(self) -> Dict[str, int]:
        """Son taranan mesaj ID'lerini yükler."""
        state_path = self.data_dir / STATE_FILE
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_state(self):
        """State'i diske atomik olarak kaydeder."""
        state_path = self.data_dir / STATE_FILE
        temp_path = state_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        temp_path.replace(state_path)

    def _load_channel_map(self) -> Dict[str, int]:
        """Kullanıcı adı -> ID eşleşmelerini yükler."""
        map_path = self.data_dir / "channel_map.json"
        if map_path.exists():
            with open(map_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_channel_map(self):
        """Map'i diske atomik olarak kaydeder."""
        map_path = self.data_dir / "channel_map.json"
        temp_path = map_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.channel_map, f, indent=2)
        temp_path.replace(map_path)

    # ──────────────────────────────────────────────
    # Kanal Bilgisi
    # ──────────────────────────────────────────────

    async def _get_channel_info(self, channel_id) -> dict:
        """Kanal bilgisini çeker (username, title, private/public)."""
        # Eğer username verildiyse ve map'imizde varsa direkt ID'yi kullan (Limitleri aşmak için)
        original_request = channel_id
        if isinstance(channel_id, str):
            clean_username = channel_id.replace("@", "")
            if clean_username in self.channel_map:
                channel_id = int(self.channel_map[clean_username])

        try:
            entity = await self.client.get_entity(channel_id)
            info = {
                "id": entity.id,
                "title": getattr(entity, "title", str(entity.id)),
                "username": getattr(entity, "username", None),
                "is_private": getattr(entity, "username", None) is None,
            }
            # Eğer map'te yoksa ekle (gelecek sefer için)
            if info["username"] and info["username"] not in self.channel_map:
                self.channel_map[info["username"]] = entity.id
                self._save_channel_map()
            return info
        except FloodWaitError as e:
            logger.warning(f"⏳ FloodWait in resolver: {e.seconds}s. Falling back to ID/Username.")
            # Fallback for FloodWait
            return {
                "id": channel_id,
                "title": str(original_request),
                "username": original_request if isinstance(original_request, str) else None,
                "is_private": True,
            }
        except Exception as e:
            logger.error(f"Kanal bilgisi alınamadı ({original_request}): {e}")
            return {
                "id": channel_id,
                "title": str(original_request),
                "username": original_request if isinstance(original_request, str) else None,
                "is_private": True,
            }

    # ──────────────────────────────────────────────
    # Mesaj İşleme
    # ──────────────────────────────────────────────

    def _extract_file_names(self, message: Message) -> List[str]:
        """Mesajdaki dosya isimlerini çıkarır."""
        names = []
        if message.media and isinstance(message.media, MessageMediaDocument):
            doc = message.media.document
            if doc and hasattr(doc, "attributes"):
                for attr in doc.attributes:
                    if hasattr(attr, "file_name") and attr.file_name:
                        names.append(attr.file_name)
        # Caption'da dosya adı varsa
        if message.text:
            for word in message.text.split():
                ext = os.path.splitext(word)[-1].lower()
                if ext in MODEL_EXTENSIONS:
                    names.append(word)
        return names

    def _has_downloadable_photo(self, message: Message) -> bool:
        """Mesajda indirilebilir fotoğraf/thumbnail var mı?"""
        if isinstance(message.media, MessageMediaPhoto):
            return True
        if isinstance(message.media, MessageMediaDocument):
            doc = message.media.document
            if doc:
                # Thumbnail varsa kullan
                if hasattr(doc, "thumbs") and doc.thumbs:
                    return True
                # Veya doğrudan image dosyasıysa
                mime = getattr(doc, "mime_type", "") or ""
                if mime.startswith("image/"):
                    return True
        return False

    async def _download_preview(
        self, message: Message, channel_id: int | str
    ) -> Optional[str]:
        """Mesajdaki görseli yerel diske indirir."""
        # channel_id'den güvenli bir sayısal ID üret (abs() hatasını önlemek için)
        if isinstance(channel_id, int):
            safe_id = abs(channel_id)
        elif isinstance(channel_id, str) and channel_id.lstrip('-').isdigit():
            safe_id = abs(int(channel_id))
        else:
            # Sayısal değilse (ör: @username), string'den bir hash üretip onu ID gibi kullan
            # Bu, abs() hatasını önler ve tutarlı bir isim şeması sağlar
            import hashlib
            safe_id = int(hashlib.md5(str(channel_id).encode()).hexdigest(), 16) % (10**10)

        filename = f"{safe_id}_{message.id}.jpg"
        filepath = self.images_dir / filename

        if filepath.exists():
            return str(filepath)

        try:
            if isinstance(message.media, MessageMediaPhoto):
                # Fotoğrafı indir
                result = await self.client.download_media(
                    message.media,
                    file=str(filepath),
                )
                if result:
                    return str(filepath)

            elif isinstance(message.media, MessageMediaDocument):
                doc = message.media.document
                mime = getattr(doc, "mime_type", "") or ""

                if mime.startswith("image/"):
                    # Image document'ı indir
                    result = await self.client.download_media(
                        message.media,
                        file=str(filepath),
                    )
                    if result:
                        return str(filepath)

                elif hasattr(doc, "thumbs") and doc.thumbs:
                    # Thumbnail'ı indir
                    result = await self.client.download_media(
                        message.media,
                        file=str(filepath),
                        thumb=-1,  # En büyük thumbnail
                    )
                    if result:
                        return str(filepath)

        except FloodWaitError as e:
            logger.warning(f"⏳ FloodWait: {e.seconds}s bekleniyor...")
            await asyncio.sleep(e.seconds)
            # Recursive yerine tek retry (stack overflow koruması)
            try:
                if isinstance(message.media, MessageMediaPhoto):
                    result = await self.client.download_media(message.media, file=str(filepath))
                    if result:
                        return str(filepath)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Görsel indirilemedi (msg {message.id}): {e}")

        return None

    # ──────────────────────────────────────────────
    # Ana Tarama
    # ──────────────────────────────────────────────

    async def scrape_channel(
        self,
        channel,
        limit: Optional[int] = None,
        resume: bool = True,
    ) -> List[TelegramMessage]:
        """
        Tek bir kanalı tarar.

        Args:
            channel: Kanal username veya ID
            limit: Maximum mesaj sayısı (None = tümü)
            resume: True ise kaldığı yerden devam eder
        """
        info = await self._get_channel_info(channel)
        channel_id = info["id"]
        channel_key = str(channel_id)
        logger.info(f"📡 Kanal taranıyor: {info['title']} (ID: {channel_id})")

        # Resume: son taranan mesajdan devam et
        min_id = 0
        if resume and channel_key in self.state:
            min_id = self.state[channel_key]
            logger.info(f"  ↪ Mesaj #{min_id}'den devam ediliyor...")

        results: List[TelegramMessage] = []
        count = 0
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Mesajları tara
        async for message in self.client.iter_messages(
            channel_id,
            min_id=min_id,
            reverse=True,
            limit=limit,
        ):
            if not message.media:
                continue

            if not self._has_downloadable_photo(message):
                continue

            async with semaphore:
                # Rate limiting
                await asyncio.sleep(self.download_delay)

                # Görseli indir
                image_path = await self._download_preview(message, channel_id)
                if not image_path:
                    continue

                # Metadata oluştur
                caption = message.text or ""
                file_names = self._extract_file_names(message)
                deep_link = build_deep_link(
                    channel_id, info["username"], message.id
                )

                msg_data = TelegramMessage(
                    channel_id=channel_id,
                    channel_title=info["title"],
                    channel_username=info["username"],
                    is_private=info["is_private"],
                    message_id=message.id,
                    caption=caption,
                    timestamp=message.date.isoformat() if message.date else "",
                    image_path=image_path,
                    file_names=file_names,
                    deep_link=deep_link,
                )
                results.append(msg_data)
                count += 1

                # State güncelle
                self.state[channel_key] = message.id
                if count % 200 == 0:
                    self._save_state()
                    self._append_metadata(results[-200:])
                    logger.info(f"  📊 {count} mesaj işlendi...")

        # Final state kaydet
        self._save_state()
        if results:
            remaining = results[-(count % 200):] if count % 200 != 0 else []
            if remaining:
                self._append_metadata(remaining)

        logger.info(f"✅ {info['title']}: toplam {count} görsel indirildi.")
        return results

    async def scrape_all_channels(
        self,
        channels: List[str],
        limit_per_channel: Optional[int] = None,
        resume: bool = True,
        parallel_channels: int = 4,
    ) -> List[TelegramMessage]:
        """Kanalları paralel tarar (varsayılan 4 kanal aynı anda)."""
        all_results: List[TelegramMessage] = []
        channel_sem = asyncio.Semaphore(parallel_channels)
        completed = 0
        total = len(channels)

        async def _scrape_one(channel, idx):
            nonlocal completed
            async with channel_sem:
                logger.info(f"\n{'='*50}")
                logger.info(f"Kanal {idx}/{total}: {channel}")
                logger.info(f"{'='*50}")

                try:
                    results = await self.scrape_channel(
                        channel, limit=limit_per_channel, resume=resume
                    )
                    all_results.extend(results)
                except ChannelPrivateError:
                    logger.error(f"❌ Kanal erişilemiyor (private/banned): {channel}")
                except FloodWaitError as e:
                    logger.warning(f"⏳ FloodWait: {e.seconds}s bekleniyor...")
                    await asyncio.sleep(e.seconds)
                    try:
                        results = await self.scrape_channel(
                            channel, limit=limit_per_channel, resume=resume
                        )
                        all_results.extend(results)
                    except Exception as e2:
                        logger.error(f"❌ Tekrar deneme başarısız: {e2}")
                except Exception as e:
                    logger.error(f"❌ Kanal taranamadı ({channel}): {e}")
                finally:
                    completed += 1
                    logger.info(f"📈 İlerleme: {completed}/{total} kanal tamamlandı")

        # Tüm kanalları paralel başlat
        tasks = [_scrape_one(ch, i) for i, ch in enumerate(channels, 1)]
        await asyncio.gather(*tasks)

        logger.info(f"\n🎉 Toplam {len(all_results)} görsel indirildi.")
        return all_results

    # ──────────────────────────────────────────────
    # Metadata Yazma
    # ──────────────────────────────────────────────

    def _append_metadata(self, messages: List[TelegramMessage]):
        """Metadata'yı JSONL dosyasına ekler."""
        with open(self.metadata_file, "a", encoding="utf-8") as f:
            for msg in messages:
                f.write(msg.to_json() + "\n")

    def load_all_metadata(self) -> List[TelegramMessage]:
        """Tüm kaydedilmiş metadata'yı yükler."""
        results = []
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        results.append(TelegramMessage.from_dict(data))
        return results
