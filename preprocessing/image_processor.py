"""
Image Processor — Validates and scans images for the indexing pipeline.
Includes a file scan cache to skip re-validation of unchanged files.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# Cache file for scan results — avoids re-opening 500K+ images every run
_DEFAULT_CACHE_FILE = "scan_cache.json"


class ImageProcessor:
    """Validates images and manages a scan cache for fast re-indexing."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        cache_dir: Optional[Path] = None,
    ):
        self.target_size = target_size
        self._cache_dir = cache_dir

    def validate_image(self, image_path: str) -> bool:
        """Check if the image file can be opened and is not corrupted."""
        try:
            img = Image.open(image_path)
            img.verify()
            return True
        except Exception:
            return False

    def load_and_preprocess(self, image_path: str) -> Optional[Image.Image]:
        """Load image and convert to RGB for CLIP processing."""
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except Exception as e:
            logger.warning(f"Failed to load image ({image_path}): {e}")
            return None

    def get_valid_images(self, image_paths: List[str]) -> List[str]:
        """Filter and return only valid image paths."""
        valid = []
        for path in image_paths:
            if self.validate_image(path):
                valid.append(path)
            else:
                logger.warning(f"Skipped invalid image: {path}")
        return valid

    def _get_cache_path(self, directory: Path) -> Path:
        """Get the cache file path for a given directory."""
        cache_dir = self._cache_dir or directory.parent
        return cache_dir / _DEFAULT_CACHE_FILE

    def _load_cache(self, cache_path: Path) -> Dict[str, Dict]:
        """Load the scan cache from disk."""
        if not cache_path.exists():
            return {}
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"Scan cache corrupted, rebuilding: {e}")
        return {}

    def _save_cache(self, cache_path: Path, cache: Dict[str, Dict]) -> None:
        """Save the scan cache to disk."""
        try:
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to save scan cache: {e}")

    def scan_directory(
        self,
        directory: str | Path,
        extensions: Set[str] = {".jpg", ".jpeg", ".png", ".webp"},
    ) -> List[str]:
        """
        Scan a directory for valid images with smart caching.

        On first run: validates every file and builds a cache.
        On subsequent runs: only validates NEW or MODIFIED files.
        Result: 15 min scan → ~30 seconds.
        """
        directory = Path(directory)
        cache_path = self._get_cache_path(directory)
        cache = self._load_cache(cache_path)

        # Collect all candidate files
        all_files = sorted(
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )

        total = len(all_files)
        cached_hits = 0
        new_validated = 0
        invalid = 0
        valid_images = []

        for f in all_files:
            fname = f.name
            try:
                stat = f.stat()
                file_key = f"{stat.st_size}_{int(stat.st_mtime)}"
            except OSError:
                continue

            # Check cache: same name + same size + same mtime = skip validation
            cached = cache.get(fname)
            if cached and cached.get("key") == file_key:
                if cached.get("valid"):
                    valid_images.append(str(f))
                    cached_hits += 1
                else:
                    invalid += 1
                continue

            # New or modified file — validate it
            is_valid = self.validate_image(str(f))
            cache[fname] = {"key": file_key, "valid": is_valid}

            if is_valid:
                valid_images.append(str(f))
                new_validated += 1
            else:
                invalid += 1

        # Save updated cache
        self._save_cache(cache_path, cache)

        logger.info(
            f"📁 Scan complete: {len(valid_images):,} valid images "
            f"({cached_hits:,} cached, {new_validated:,} new, {invalid:,} invalid)"
        )
        return valid_images
