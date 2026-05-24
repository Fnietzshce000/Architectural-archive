"""
Telethon İstemci Yönetimi.
Oturum açma, bağlantı yönetimi ve FloodWait koruma.
"""
import asyncio
import logging
from pathlib import Path
from telethon import TelegramClient
from telethon.errors import FloodWaitError

logger = logging.getLogger(__name__)


class TelegramClientManager:
    """Telethon istemcisini yönetir."""

    def __init__(self, api_id: int, api_hash: str, phone: str,
                 session_dir: str = "./data"):
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.session_path = str(Path(session_dir) / "scraper_session")
        self.client: TelegramClient | None = None

    async def start(self) -> TelegramClient:
        """İstemciyi başlatır ve oturum açar."""
        Path(self.session_path).parent.mkdir(parents=True, exist_ok=True)

        self.client = TelegramClient(
            self.session_path,
            self.api_id,
            self.api_hash,
            request_retries=10,
            connection_retries=10,
            retry_delay=2,
            auto_reconnect=True,
        )
        # 300 saniyenin altındaki FloodWait'lerde otomatik bekleme
        self.client.flood_sleep_threshold = 300

        await self.client.start(phone=self.phone)
        me = await self.client.get_me()
        logger.info(f"✅ Telegram'a giriş yapıldı: {me.first_name} ({me.phone})")
        return self.client

    async def stop(self):
        """İstemciyi kapatır."""
        if self.client:
            await self.client.disconnect()
            logger.info("🔌 Telegram bağlantısı kapatıldı.")

    async def __aenter__(self) -> TelegramClient:
        return await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
