"""
Gerçek Zamanlı Dinleyici — Yeni mesajları anında yakalar ve indeksler.
Scraper tamamlandıktan sonra arka planda çalışır.
"""
import asyncio
import logging
from pathlib import Path
from typing import Callable, List, Optional

from telethon import TelegramClient, events
from telethon.tl.types import MessageMediaDocument, MessageMediaPhoto

from scraper.channel_scraper import ChannelScraper
from scraper.models import TelegramMessage, build_deep_link

logger = logging.getLogger(__name__)


async def _default_index_callback(msg_data: TelegramMessage) -> None:
    """Yeni görseli SigLIP 2 ile encode edip, zero-shot etiketleyip, estetik puanlayıp ChromaDB'ye kaydeder."""
    try:
        from config import get_settings, MetadataSchema
        from indexer.clip_encoder import load_model, encode_image
        from indexer.chroma_store import ChromaStore
        import torch
        import numpy as np
        from PIL import Image

        settings = get_settings()
        store = ChromaStore(settings.chroma_db_path, settings.get_collection_name())

        img_path = msg_data.image_path
        if not img_path or not Path(img_path).exists():
            return

        doc_id = Path(img_path).stem
        # Zaten indekslenmişse atla
        existing = store.collection.get(ids=[doc_id], include=[])
        if existing and existing.get("ids"):
            return

        # SigLIP 2 ile vektörleştir
        img_vec = encode_image(
            img_path,
            model_name=settings.clip_model_name,
            pretrained=settings.clip_pretrained,
        )
        if img_vec is None:
            return

        # ── Zero-Shot Etiketleme ──
        clip_tags = {}
        try:
            from scripts.run_siglip_indexer import TAXONOMY
            model, preprocess, tokenizer, device = load_model(
                settings.clip_model_name, settings.clip_pretrained
            )
            img_tensor = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                if device == "cuda":
                    with torch.amp.autocast("cuda"):
                        features = model.encode_image(img_tensor)
                else:
                    features = model.encode_image(img_tensor)
                features /= features.norm(dim=-1, keepdim=True)

            all_tags_list = []
            for category, labels in TAXONOMY.items():
                prompts_map = {
                    "furniture_type": "a 3D model render of a {}",
                    "style": "a {} style interior design",
                    "material": "an object made of {} material",
                    "color": "a {} colored item",
                    "room": "a 3D render of a {} space",
                    "category": "a 3D asset of {}",
                }
                template = prompts_map.get(category, "a {}")
                best = []
                for label in labels:
                    tokens = tokenizer([template.format(label)]).to(device)
                    with torch.no_grad():
                        txt_vec = model.encode_text(tokens)
                        txt_vec /= txt_vec.norm(dim=-1, keepdim=True)
                    score = float((features @ txt_vec.T).squeeze())
                    best.append((label, score))
                best.sort(key=lambda x: -x[1])
                top_labels = [l for l, s in best[:3] if s > 0.02]
                clip_tags[category] = top_labels
                all_tags_list.extend(top_labels)
            clip_tags_str = ", ".join(all_tags_list)
        except Exception as e:
            logger.warning(f"Live etiketleme başarısız (fallback): {e}")
            clip_tags_str = ""

        # ── Estetik Puanlama (Aesthetic Predictor V2.5) ──
        aesthetic_score = 0.0
        try:
            global _aesthetic_vision_model, _aesthetic_mlp_head, _aesthetic_processor
            if "_aesthetic_vision_model" not in dir() or _aesthetic_vision_model is None:
                from transformers import SiglipVisionModel, SiglipConfig, AutoProcessor
                enc_name = "google/siglip-so400m-patch14-384"
                full_cfg = SiglipConfig.from_pretrained(enc_name)
                _aesthetic_vision_model = SiglipVisionModel.from_pretrained(
                    enc_name, config=full_cfg.vision_config, low_cpu_mem_usage=True,
                ).to(torch.bfloat16).cuda().eval()
                _aesthetic_processor = AutoProcessor.from_pretrained(enc_name)

                import torch.nn as nn
                class _AesHead(nn.Module):
                    def __init__(self, d=1152):
                        super().__init__()
                        self.scoring_head = nn.Sequential(
                            nn.Linear(d, 1024), nn.Dropout(0.2),
                            nn.Linear(1024, 128), nn.Dropout(0.2),
                            nn.Linear(128, 64), nn.Dropout(0.1),
                            nn.Linear(64, 16), nn.Dropout(0.1),
                            nn.Linear(16, 1),
                        )
                    def forward(self, x): return self.scoring_head(x)

                url = "https://github.com/discus0434/aesthetic-predictor-v2-5/raw/refs/heads/main/models/aesthetic_predictor_v2_5.pth"
                sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
                _aesthetic_mlp_head = _AesHead(1152)
                _aesthetic_mlp_head.load_state_dict(sd)
                _aesthetic_mlp_head = _aesthetic_mlp_head.to(torch.bfloat16).cuda().eval()
                logger.info("🎨 Live Aesthetic Model yüklendi")

            pil_img = Image.open(img_path).convert("RGB")
            inputs = _aesthetic_processor(images=pil_img, return_tensors="pt")
            pv = inputs.pixel_values.to(torch.bfloat16).cuda()
            with torch.inference_mode():
                out = _aesthetic_vision_model(pixel_values=pv)
                emb = out.pooler_output
                emb = emb / emb.norm(dim=-1, keepdim=True)
                aesthetic_score = float(_aesthetic_mlp_head(emb).squeeze().float().cpu().numpy())
                aesthetic_score = max(1.0, min(10.0, aesthetic_score))
        except Exception as e:
            logger.warning(f"Live estetik puanlama başarısız: {e}")

        metadata = {
            MetadataSchema.CHANNEL_ID: msg_data.channel_id,
            MetadataSchema.MESSAGE_ID: msg_data.message_id,
            MetadataSchema.IMAGE_PATH: img_path,
            MetadataSchema.DEEP_LINK: msg_data.deep_link,
            MetadataSchema.CHANNEL_TITLE: msg_data.channel_title,
            MetadataSchema.CHANNEL_USERNAME: msg_data.channel_username or "",
            MetadataSchema.TIMESTAMP: msg_data.timestamp,
            MetadataSchema.CAPTION: (msg_data.caption or "")[:500],
            MetadataSchema.SOURCE: "LiveListener",
            MetadataSchema.MODEL_NAME: settings.clip_model_name,
            MetadataSchema.SCHEMA_VERSION_KEY: MetadataSchema.CURRENT_SCHEMA_VERSION,
            "clip_tagged": True,
        }
        # Etiketleri metadata'ya ekle
        for cat, labels in clip_tags.items():
            metadata[f"clip_{cat}"] = ", ".join(labels)
        if clip_tags_str:
            metadata["clip_tags"] = clip_tags_str
        # Estetik skor
        if aesthetic_score > 0:
            metadata["aesthetic_score"] = round(aesthetic_score, 4)
            metadata["aesthetic_scorer_model"] = "aesthetic_predictor_v2.5"

        document = f"{msg_data.caption or ''} | {clip_tags_str}"
        store.add(doc_id=doc_id, embedding=img_vec, metadata=metadata, document=document)
        aes_str = f" | AES:{aesthetic_score:.1f}" if aesthetic_score > 0 else ""
        logger.info(f"⚡ Canlı indeksleme: {doc_id} → {clip_tags_str[:60]}{aes_str}")
    except Exception as exc:
        logger.error(f"Canlı indeksleme hatası: {exc}")

# Global aesthetic model refs (lazy loaded)
_aesthetic_vision_model = None
_aesthetic_mlp_head = None
_aesthetic_processor = None


class LiveListener:
    """Hedef kanallardaki yeni mesajları gerçek zamanlı dinler ve indeksler."""

    def __init__(
        self,
        client: TelegramClient,
        scraper: ChannelScraper,
        channels: List,
        on_new_message: Optional[Callable] = None,
    ):
        self.client = client
        self.scraper = scraper
        self.channels = channels
        # Callback verilmediyse varsayılan olarak gerçek zamanlı indeksleme yap
        self.on_new_message = on_new_message or _default_index_callback
        self._running = False

    async def start(self):
        """Dinleyiciyi başlatır."""
        if not self.channels:
            logger.warning("Dinlenecek kanal bulunamadı.")
            return

        # Kanal entity'lerini çöz
        resolved_channels = []
        for ch in self.channels:
            try:
                entity = await self.client.get_entity(ch)
                resolved_channels.append(entity)
                logger.info(f"👂 Dinleniyor: {getattr(entity, 'title', ch)}")
            except Exception as e:
                logger.error(f"Kanal çözümlenemedi: {ch} — {e}")

        if not resolved_channels:
            logger.error("Hiçbir kanal çözümlenemedi.")
            return

        @self.client.on(events.NewMessage(chats=resolved_channels))
        async def handler(event):
            """Yeni mesaj geldiğinde tetiklenir."""
            message = event.message

            # Medya kontrolü
            if not message.media:
                return

            has_photo = isinstance(message.media, MessageMediaPhoto)
            has_doc_image = False
            if isinstance(message.media, MessageMediaDocument):
                doc = message.media.document
                mime = getattr(doc, "mime_type", "") or ""
                has_doc_image = mime.startswith("image/") or (
                    hasattr(doc, "thumbs") and doc.thumbs
                )

            if not has_photo and not has_doc_image:
                return

            # Kanal bilgisi
            chat = await event.get_chat()
            channel_id = chat.id
            channel_username = getattr(chat, "username", None)
            channel_title = getattr(chat, "title", str(channel_id))

            logger.info(
                f"🆕 Yeni mesaj: {channel_title} / #{message.id}"
            )

            # Görseli indir
            image_path = await self.scraper._download_preview(message, channel_id)
            if not image_path:
                return

            # Metadata oluştur
            msg_data = TelegramMessage(
                channel_id=channel_id,
                channel_title=channel_title,
                channel_username=channel_username,
                is_private=channel_username is None,
                message_id=message.id,
                caption=message.text or "",
                timestamp=message.date.isoformat() if message.date else "",
                image_path=image_path,
                file_names=self.scraper._extract_file_names(message),
                deep_link=build_deep_link(channel_id, channel_username, message.id),
            )

            # Metadata kaydet
            self.scraper._append_metadata([msg_data])
            self.scraper.state[str(channel_id)] = message.id
            self.scraper._save_state()

            # Callback (indeksleme dahil)
            if self.on_new_message:
                try:
                    await self.on_new_message(msg_data)
                except Exception as e:
                    logger.error(f"Callback hatası: {e}")

        self._running = True
        logger.info("🎧 Canlı dinleyici aktif — yeni mesajlar bekleniyor ve anında indeksleniyor...")

    async def run_forever(self):
        """İstemciyi sonsuza kadar çalıştırır."""
        await self.start()
        await self.client.run_until_disconnected()

    def stop(self):
        """Dinleyiciyi durdurur."""
        self._running = False
        logger.info("🛑 Canlı dinleyici durduruldu.")

