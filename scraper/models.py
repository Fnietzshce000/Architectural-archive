"""
Telegram mesaj veri modeli.
Her indirilen mesajın metadata'sını tutar.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional
import json


@dataclass
class TelegramMessage:
    """Telegram'dan çekilen bir mesajın tüm bilgilerini tutar."""

    # ── Kanal Bilgisi ──
    channel_id: int = 0
    channel_title: str = ""
    channel_username: Optional[str] = None  # Public kanallar için @username
    is_private: bool = False

    # ── Mesaj Bilgisi ──
    message_id: int = 0
    caption: str = ""
    timestamp: str = ""  # ISO format

    # ── Dosya Bilgisi ──
    image_path: str = ""  # Yerel diskteki görsel yolu
    file_names: List[str] = field(default_factory=list)  # İlişkili 3D dosyalar
    file_size: int = 0

    # ── Deep Link ──
    deep_link: str = ""

    def to_dict(self) -> dict:
        """Sözlüğe çevir."""
        return asdict(self)

    def to_json(self) -> str:
        """JSON string'e çevir."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "TelegramMessage":
        """Sözlükten oluştur."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def build_deep_link(channel_id: int, channel_username: Optional[str], message_id: int) -> str:
    """
    Telegram deep link oluşturur.

    Public kanal:  https://t.me/kanaladi/mesaj_id
    Private kanal: https://t.me/c/saf_id/mesaj_id
    """
    if channel_username:
        # Public kanal — username ile link
        username = channel_username.lstrip("@")
        return f"https://t.me/{username}/{message_id}"
    else:
        # Private kanal — -100 öneki kırpılarak saf ID elde edilir
        raw_id = str(abs(channel_id))
        if raw_id.startswith("100"):
            raw_id = raw_id[3:]  # -100 önekini kırp
        return f"https://t.me/c/{raw_id}/{message_id}"
