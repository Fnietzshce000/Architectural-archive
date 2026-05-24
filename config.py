"""
Merkezi KonfigÃ¼rasyon â€” .env dosyasÄ±ndan tÃ¼m ayarlarÄ± okur.
"""
import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class MetadataSchema:
    """Sistem genelinde kullanÄ±lacak metadata anahtarlarÄ± (Contract)."""
    IMAGE_PATH = "image_path"
    DEEP_LINK = "deep_link"
    CAPTION = "caption"
    CHANNEL_TITLE = "channel_title"
    CHANNEL_ID = "channel_id"
    CHANNEL_USERNAME = "channel_username"
    MESSAGE_ID = "message_id"
    TIMESTAMP = "timestamp"
    TAGS = "tags"            # Genel etiketler
    DEEP_TAGS = "deep_tags"  # AI tarafÄ±ndan Ã¼retilen detaylÄ± etiketler
    AI_DESCRIPTION = "ai_description"
    AESTHETIC_SCORE = "aesthetic_score"
    SOURCE = "source"        # 'MegaSync', 'Manual', 'Scraper' vb.
    OBJECT_TYPE = "object_type"
    STYLE = "style"
    MATERIAL = "material"
    ROOM = "room"
    COLOR_FAMILY = "color_family"
    RENDER_TYPE = "render_type"
    CATEGORY_SOURCE = "category_source"
    CATEGORY_CONFIDENCE = "category_confidence"
    QUALITY_TAGS = "quality_tags"
    AI_ENRICHED = "ai_enriched"  # Vision LLM ile zenginlestirildi mi
    SCHEMA_VERSION_KEY = "schema_version"
    CURRENT_SCHEMA_VERSION = "v2"    # Sema versiyon takibi
    MODEL_NAME = "model_name" # Hangi CLIP modeliyle vektorlesti



class Settings(BaseSettings):
    """Uygulama ayarlarÄ± â€” .env dosyasÄ±ndan otomatik yÃ¼klenir."""

    # â”€â”€ Telegram API â”€â”€
    telegram_api_id: int = Field(..., description="Telegram API ID")
    telegram_api_hash: str = Field(..., description="Telegram API Hash")
    telegram_phone: str = Field("+905551234567", description="Telefon numarasÄ±")

    # â”€â”€ Hedef Kanallar â”€â”€
    target_channels: str = Field("", description="VirgÃ¼lle ayrÄ±lmÄ±ÅŸ kanal listesi")

    # â”€â”€ Veri Dizini â”€â”€
    data_dir: str = Field("./data", description="Veri depolama dizini")

    # â”€â”€ CLIP Model â”€â”€
    clip_model_name: str = Field("ViT-L-14", description="OpenCLIP model adÄ±")
    clip_pretrained: str = Field("laion2b_s32b_b82k", description="Pretrained aÄŸÄ±rlÄ±k")

    # â”€â”€ ChromaDB â”€â”€
    chroma_collection_base: str = Field("models_3d", description="Temel koleksiyon adÄ±")
    chroma_collection_name: Optional[str] = Field(None, validation_alias="CHROMA_COLLECTION")
    chroma_db_path: str = Field("./data/chroma_db", description="ChromaDB dizini")

    # -- Arama Harmanlama --
    image_text_blend_ratio: float = Field(0.7, description="Gorsel arama: gorsel vs metin vektoru harmanlama orani (0.0-1.0)")

    # ── LM Studio Modelleri (Dual Model Swapping) ──
    llm_text_model: str = Field("qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2", description="Yazı/metin işlemleri için LM Studio model id")
    llm_vision_model: str = Field("qwen/qwen3-vl-8b", description="Görsel analizleri için LM Studio vision model id")

    # ── Telegram Bot ──
    telegram_bot_token: str = Field("", description="Telegram Bot API Token (@BotFather'dan)")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def get_collection_name(self) -> str:
        """Model adÄ±na ve versiyona gÃ¶re gÃ¼venli koleksiyon ismi Ã¼retir."""
        if self.chroma_collection_name:
            return self.chroma_collection_name
        # Ã–rn: models_3d_ViT_L_14_v2
        safe_model = self.clip_model_name.replace("-", "_")
        return f"{self.chroma_collection_base}_{safe_model}_{MetadataSchema.CURRENT_SCHEMA_VERSION}"

    @property
    def chroma_collection(self) -> str:
        """Eski kodlar icin geriye uyumlu koleksiyon adi."""
        return self.get_collection_name()

    def get_channel_list(self) -> List[str]:
        """Kanal listesini ayrÄ±ÅŸtÄ±rÄ±r."""
        if not self.target_channels:
            return []
        return [ch.strip() for ch in self.target_channels.split(",") if ch.strip()]

    def get_data_path(self) -> Path:
        """Veri dizini yolunu dÃ¶ndÃ¼rÃ¼r, yoksa oluÅŸturur."""
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_images_path(self) -> Path:
        """GÃ¶rsel depolama dizini yolunu dÃ¶ndÃ¼rÃ¼r."""
        p = self.get_data_path() / "images"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_chroma_path(self) -> Path:
        """ChromaDB dizin yolunu dÃ¶ndÃ¼rÃ¼r."""
        p = Path(self.chroma_db_path)
        p.mkdir(parents=True, exist_ok=True)
        return p


# Singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Tek bir Settings nesnesi dÃ¶ndÃ¼rÃ¼r."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
