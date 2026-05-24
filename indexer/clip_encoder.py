"""
CLIP Encoder — OpenCLIP ile görselleri ve metinleri vektörleştirir.
ViT-B-32 modeli kullanır (~400 MB VRAM).
"""
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-load modüller (import sırasında GPU belleği kullanmamak için)
_model = None
_preprocess = None
_tokenizer = None
_device = None
_model_name = None  # Yüklü olan modelin adını tutar

def _get_device() -> str:
    """Uygun cihazı seçer."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_model(
    model_name: str = "ViT-L-14",
    pretrained: str = "laion2b_s32b_b82k",
) -> tuple:
    """
    CLIP modelini yükler. Eğer farklı bir model istenirse eskisini boşaltıp yenisini yükler.
    """
    global _model, _preprocess, _tokenizer, _device, _model_name

    # Eğer istenen model şu an yüklü olandan farklıysa, önce eskisini temizle
    if _model is not None and _model_name != model_name:
        logger.info(f"🔄 Model değişimi: {_model_name} -> {model_name}. Eski model boşaltılıyor...")
        unload_model()

    if _model is not None:
        return _model, _preprocess, _tokenizer, _device
    
    _model_name = model_name

    import open_clip

    _device = _get_device()
    if model_name.startswith("hf-hub:"):
        pretrained = None
    logger.info(f"🧠 CLIP modeli yükleniyor: {model_name} ({pretrained}) → {_device}")

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    _tokenizer = open_clip.get_tokenizer(model_name)
    _model = _model.to(_device)
    _model.eval()

    # VRAM durumu
    if _device == "cuda":
        mem = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"  ✅ Model yüklendi — VRAM: {mem:.0f} MB")
    else:
        logger.info("  ✅ Model CPU'da yüklendi")

    return _model, _preprocess, _tokenizer, _device


def unload_model():
    """CLIP modelini bellekten temizler (LLaMA ile sıralı çalışma için)."""
    global _model, _preprocess, _tokenizer, _device, _model_name
    if _model is not None:
        del _model
        _model = None
        _preprocess = None
        _tokenizer = None
        _model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🗑️ CLIP modeli bellekten temizlendi.")


def encode_image(image_path: str, model_name: str = "ViT-L-14",
                 pretrained: str = "laion2b_s32b_b82k") -> Optional[np.ndarray]:
    """
    Tek bir görseli CLIP vektörüne çevirir.

    Returns:
        512 boyutlu numpy array (normalized)
    """
    model, preprocess, _, device = load_model(model_name, pretrained)

    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast("cuda"):
                    features = model.encode_image(img_tensor)
            else:
                features = model.encode_image(img_tensor)
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten()

    except Exception as e:
        logger.error(f"Görsel encode edilemedi ({image_path}): {e}")
        return None


def encode_images_batch(
    image_paths: List[str],
    batch_size: int = 32,
    model_name: str = "ViT-L-14",
    pretrained: str = "laion2b_s32b_b82k",
) -> List[Optional[np.ndarray]]:
    """
    Görselleri batch halinde vektörleştirir (GPU optimizasyonu).

    Args:
        image_paths: Görsel dosya yolları listesi
        batch_size: Batch boyutu (varsayılan 32, 8GB VRAM için uygun)

    Returns:
        Her görsel için 512 boyutlu vektör listesi (hata durumunda None)
    """
    model, preprocess, _, device = load_model(model_name, pretrained)
    results: List[Optional[np.ndarray]] = [None] * len(image_paths)

    for start in range(0, len(image_paths), batch_size):
        end = min(start + batch_size, len(image_paths))
        batch_paths = image_paths[start:end]

        # Geçerli görselleri yükle
        valid_indices = []
        tensors = []
        for i, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                tensor = preprocess(img)
                tensors.append(tensor)
                valid_indices.append(start + i)
            except Exception as e:
                logger.warning(f"Görsel atlandı ({path}): {e}")

        if not tensors:
            continue

        # Batch encode
        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast("cuda"):
                    features = model.encode_image(batch_tensor)
            else:
                features = model.encode_image(batch_tensor)
            features /= features.norm(dim=-1, keepdim=True)

        features_np = features.cpu().numpy()

        for idx, global_idx in enumerate(valid_indices):
            results[global_idx] = features_np[idx]

    return results


def encode_text(text: str, model_name: str = "ViT-L-14",
                pretrained: str = "laion2b_s32b_b82k") -> Optional[np.ndarray]:
    """
    Metni CLIP vektörüne çevirir (arama sorgusu için).

    Returns:
        512 boyutlu numpy array (normalized)
    """
    model, _, tokenizer, device = load_model(model_name, pretrained)

    try:
        tokens = tokenizer([text]).to(device)

        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast("cuda"):
                    features = model.encode_text(tokens)
            else:
                features = model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten()

    except Exception as e:
        logger.error(f"Metin encode edilemedi ({text}): {e}")
        return None
