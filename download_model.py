import open_clip
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelDownloader")

logger.info("ViT-L-14 modeli (1.7 GB) indiriliyor... Lütfen bekleyin.")
try:
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    logger.info("✅ İndirme başarıyla tamamlandı! Artık arayüzü açabilirsiniz.")
except Exception as e:
    logger.error(f"❌ İndirme hatası: {e}")
