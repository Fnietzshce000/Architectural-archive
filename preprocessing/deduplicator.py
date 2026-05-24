"""
Görüntü Dedublikasyon — dHash + pHash ile tekrarlanan görselleri tespit eder.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)


class ImageDeduplicator:
    """
    İki aşamalı görsel tekilleştirme:
    1. dHash ile hızlı ön eleme — dict-tabanlı O(1) lookup
    2. pHash ile şüpheli grupları ikinci kontrol
    """

    def __init__(
        self,
        hash_db_path: str | Path,
        dhash_threshold: int = 5,
        phash_threshold: int = 8,
        hash_size: int = 16,
    ):
        self.hash_db_path = Path(hash_db_path)
        self.dhash_threshold = dhash_threshold
        self.phash_threshold = phash_threshold
        self.hash_size = hash_size

        # Hash veritabanı: {image_path: {"dhash": str, "phash": str}}
        self.hash_db: Dict[str, dict] = self._load_hash_db()
        # ⚡ O(1) hash lookup indeksi: {dhash_str: [image_paths]}
        self._dhash_index: Dict[str, List[str]] = self._build_hash_index()

    def _load_hash_db(self) -> Dict[str, dict]:
        """Hash veritabanını diskten yükler."""
        if self.hash_db_path.exists():
            with open(self.hash_db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _build_hash_index(self) -> Dict[str, List[str]]:
        """Mevcut hash DB'den hızlı lookup indeksi oluşturur."""
        index: Dict[str, List[str]] = {}
        for path, hashes in self.hash_db.items():
            dhash = hashes.get("dhash", "")
            if dhash:
                index.setdefault(dhash, []).append(path)
        return index

    def _save_hash_db(self):
        """Hash veritabanını diske kaydeder."""
        self.hash_db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hash_db_path, "w", encoding="utf-8") as f:
            json.dump(self.hash_db, f, indent=1)

    def _compute_hashes(self, image_path: str) -> Optional[Tuple[str, str]]:
        """Bir görselin dHash ve pHash değerlerini hesaplar."""
        try:
            img = Image.open(image_path)
            img.verify()
            img = Image.open(image_path)  # verify sonrası yeniden aç

            dhash = str(imagehash.dhash(img, hash_size=self.hash_size))
            phash = str(imagehash.phash(img, hash_size=self.hash_size))
            return dhash, phash
        except Exception as e:
            logger.warning(f"Hash hesaplanamadı ({image_path}): {e}")
            return None

    def is_duplicate(self, image_path: str) -> bool:
        """
        Görselin mevcut veritabanında duplikatı var mı kontrol eder.
        O(1) hash index lookup kullanır (eski O(n²) döngü yerine).

        Returns:
            True ise duplikat (silinebilir), False ise benzersiz (korunmalı)
        """
        hashes = self._compute_hashes(image_path)
        if hashes is None:
            return False  # Bozuk dosyalar için duplikat sayma

        new_dhash_str = hashes[0]
        new_dhash_obj = imagehash.hex_to_hash(new_dhash_str)
        new_phash_obj = imagehash.hex_to_hash(hashes[1])

        # ⚡ Exact match: O(1) — aynı hash varsa kesin duplikat
        if new_dhash_str in self._dhash_index:
            for existing_path in self._dhash_index[new_dhash_str]:
                if existing_path != image_path:
                    return True

        # ⚡ Yakın eşleşme: Sadece tüm benzersiz hash'leri tara (hash başına bir kez)
        for existing_dhash_str, paths in self._dhash_index.items():
            if existing_dhash_str == new_dhash_str:
                continue  # Zaten kontrol ettik

            existing_dhash_obj = imagehash.hex_to_hash(existing_dhash_str)
            dhash_distance = new_dhash_obj - existing_dhash_obj

            if dhash_distance < self.dhash_threshold:
                return True  # Kesin duplikat

            # Aşama 2: Yakın olanlar için pHash kontrolü
            if dhash_distance < self.dhash_threshold * 2:
                # paths'teki ilk elemanın phash'ini kontrol et (hepsi aynı grupta)
                sample_path = paths[0]
                existing_hashes = self.hash_db.get(sample_path, {})
                existing_phash_str = existing_hashes.get("phash", "")
                if existing_phash_str:
                    existing_phash_obj = imagehash.hex_to_hash(existing_phash_str)
                    if new_phash_obj - existing_phash_obj < self.phash_threshold:
                        return True

        return False

    def add_image(self, image_path: str) -> bool:
        """
        Görseli veritabanına ekler.

        Returns:
            True ise eklendi (benzersiz), False ise duplikat (eklenmedi)
        """
        if image_path in self.hash_db:
            return True  # Zaten var

        hashes = self._compute_hashes(image_path)
        if hashes is None:
            return False

        # Duplikat kontrolü
        if self.is_duplicate(image_path):
            logger.debug(f"🔄 Duplikat tespit edildi: {Path(image_path).name}")
            return False

        # Benzersiz — ekle
        self.hash_db[image_path] = {"dhash": hashes[0], "phash": hashes[1]}
        # İndeksi güncelle
        self._dhash_index.setdefault(hashes[0], []).append(image_path)
        return True

    def process_directory(
        self, directory: str | Path, extensions: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
    ) -> Tuple[List[str], List[str]]:
        """
        Bir dizindeki tüm görselleri işler.

        Returns:
            (benzersiz_dosyalar, duplikat_dosyalar)
        """
        directory = Path(directory)
        unique_files = []
        duplicate_files = []

        all_images = sorted(
            [f for f in directory.iterdir()
             if f.is_file() and f.suffix.lower() in extensions]
        )

        logger.info(f"🔍 {len(all_images)} görsel dedublikasyon için taranıyor...")

        for img_path in all_images:
            path_str = str(img_path)
            if self.add_image(path_str):
                unique_files.append(path_str)
            else:
                duplicate_files.append(path_str)

        self._save_hash_db()

        logger.info(
            f"✅ Dedublikasyon tamamlandı: "
            f"{len(unique_files)} benzersiz, {len(duplicate_files)} duplikat"
        )
        return unique_files, duplicate_files

    def get_unique_count(self) -> int:
        """Veritabanındaki benzersiz görsel sayısı."""
        return len(self.hash_db)
