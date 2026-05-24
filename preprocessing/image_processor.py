"""
Görüntü İşleme — CLIP için görselleri normalize eder.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Görselleri CLIP modeline uygun boyut ve formata dönüştürür."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def validate_image(self, image_path: str) -> bool:
        """Görselin açılabilir ve geçerli olduğunu doğrular."""
        try:
            img = Image.open(image_path)
            img.verify()
            return True
        except Exception:
            return False

    def load_and_preprocess(self, image_path: str) -> Optional[Image.Image]:
        """
        Görseli yükler ve CLIP'e uygun şekle getirir.
        OpenCLIP kendi preprocessing'ini yapacak, biz sadece RGB'ye çeviriyoruz.
        """
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except Exception as e:
            logger.warning(f"Görsel yüklenemedi ({image_path}): {e}")
            return None

    def get_valid_images(self, image_paths: List[str]) -> List[str]:
        """Geçerli görsellerin listesini döndürür."""
        valid = []
        for path in image_paths:
            if self.validate_image(path):
                valid.append(path)
            else:
                logger.warning(f"❌ Geçersiz görsel atlandı: {path}")
        return valid

    def scan_directory(self, directory: str | Path,
                       extensions: set = {".jpg", ".jpeg", ".png", ".webp"}) -> List[str]:
        """Dizindeki tüm geçerli görselleri tarar."""
        directory = Path(directory)
        images = []
        for f in sorted(directory.iterdir()):
            if f.is_file() and f.suffix.lower() in extensions:
                if self.validate_image(str(f)):
                    images.append(str(f))
        logger.info(f"📁 {len(images)} geçerli görsel bulundu: {directory}")
        return images
