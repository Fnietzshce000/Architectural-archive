"""
🎨 Archi Estetik Puanlama — Aesthetic Predictor V2.5 (SigLIP Tabanlı)

Mevcut ChromaDB'deki 465K görseli gerçek bir estetik model ile puanlar.
Transformers 5.x uyumlu — modeli manuel olarak yükler.

Kullanım:
    python scripts/aesthetic_scorer.py
    python scripts/aesthetic_scorer.py --batch-size 16 --limit 1000
"""
import argparse
import gc
import logging
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# Proje kök dizini
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indexer.chroma_store import ChromaStore
from config import get_settings

# ── Loglama ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("data/aesthetic_scoring.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("AestheticScorer")

# ── MLP Head (orijinal LAION/V2.5 mimarisi) ──
PREDICTOR_URL = "https://github.com/discus0434/aesthetic-predictor-v2-5/raw/refs/heads/main/models/aesthetic_predictor_v2_5.pth"


class AestheticMLPHead(nn.Module):
    """Aesthetic Predictor V2.5 MLP kafası — SigLIP pooler output → skor."""

    def __init__(self, input_dim: int = 1152):
        super().__init__()
        self.scoring_head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.scoring_head(x)


def load_aesthetic_model(device: str = "cuda"):
    """SigLIP Vision + Aesthetic MLP Head — Transformers 5.x uyumlu."""
    from transformers import SiglipVisionModel, SiglipConfig, AutoProcessor

    encoder_name = "google/siglip-so400m-patch14-384"

    logger.info(f"🎨 SigLIP Vision Encoder yükleniyor: {encoder_name}")

    # Transformers 5.x fix: vision_config'i ayrı çek
    full_config = SiglipConfig.from_pretrained(encoder_name)
    vision_config = full_config.vision_config

    vision_model = SiglipVisionModel.from_pretrained(
        encoder_name,
        config=vision_config,
        low_cpu_mem_usage=True,
    )
    vision_model = vision_model.to(torch.bfloat16).to(device).eval()

    processor = AutoProcessor.from_pretrained(encoder_name)

    # MLP Head ağırlıklarını indir
    logger.info("🧠 Aesthetic MLP Head yükleniyor...")
    state_dict = torch.hub.load_state_dict_from_url(PREDICTOR_URL, map_location="cpu")

    # input_dim otomatik tespit
    first_key = [k for k in state_dict.keys() if "weight" in k][0]
    input_dim = state_dict[first_key].shape[1]
    logger.info(f"   MLP input dim: {input_dim}")

    mlp_head = AestheticMLPHead(input_dim=input_dim)
    mlp_head.load_state_dict(state_dict)
    mlp_head = mlp_head.to(torch.bfloat16).to(device).eval()

    logger.info(f"✅ Model yüklendi → {device} (bfloat16)")
    return vision_model, mlp_head, processor


def score_images_batch(
    vision_model,
    mlp_head,
    preprocessor,
    image_paths: list[str],
    device: str = "cuda",
) -> list[float]:
    """Bir batch görseli puanlar. Bozuk görseller -1.0 ile döner."""
    valid_images = []
    valid_indices = []
    scores = [-1.0] * len(image_paths)

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            valid_images.append(img)
            valid_indices.append(i)
        except Exception:
            continue

    if not valid_images:
        return scores

    try:
        inputs = preprocessor(
            images=valid_images, return_tensors="pt"
        )
        pixel_values = inputs.pixel_values.to(torch.bfloat16).to(device)

        with torch.inference_mode():
            outputs = vision_model(pixel_values=pixel_values)
            embeds = outputs.pooler_output
            embeds_norm = embeds / embeds.norm(dim=-1, keepdim=True)
            logits = mlp_head(embeds_norm).squeeze(-1).float().cpu().numpy()

        # Tek görsel durumunda numpy scalar dönebilir
        if logits.ndim == 0:
            logits = np.array([float(logits)])

        for idx, score in zip(valid_indices, logits):
            scores[idx] = float(np.clip(score, 1.0, 10.0))

    except Exception as e:
        logger.warning(f"Batch scoring hatası: {e}")

    for img in valid_images:
        try:
            img.close()
        except Exception:
            pass

    return scores


def main(args):
    settings = get_settings()
    store = ChromaStore(
        db_path=settings.chroma_db_path,
        collection_name=settings.get_collection_name(),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info("🎨 ESTETİK PUANLAMA BAŞLIYOR — Aesthetic Predictor V2.5")
    logger.info("=" * 60)

    # 1. Model yükle
    vision_model, mlp_head, preprocessor = load_aesthetic_model(device)

    # 2. Puanlanmamış kayıtları bul
    total_in_db = store.get_count()
    logger.info(f"📦 Veritabanı: {total_in_db} kayıt ({settings.get_collection_name()})")

    unscored_ids = []
    unscored_paths = []
    unscored_metas = []

    scan_batch = 5000
    logger.info("🔍 Puanlanmamış kayıtlar taranıyor...")

    for offset in range(0, total_in_db, scan_batch):
        batch = store.collection.get(
            include=["metadatas"],
            limit=scan_batch,
            offset=offset,
        )
        if not batch or not batch["ids"]:
            break

        for doc_id, meta in zip(batch["ids"], batch["metadatas"]):
            scorer_model = meta.get("aesthetic_scorer_model", "")
            if scorer_model == "aesthetic_predictor_v2.5":
                continue

            img_path = meta.get("image_path", "")
            if img_path and Path(img_path).exists():
                unscored_ids.append(doc_id)
                unscored_paths.append(img_path)
                unscored_metas.append(meta)

                if args.limit and len(unscored_ids) >= args.limit:
                    break

        if args.limit and len(unscored_ids) >= args.limit:
            break

    total = len(unscored_ids)
    if total == 0:
        logger.info("✅ Tüm görseller zaten puanlanmış!")
        return

    logger.info(f"📊 {total} görsel puanlanacak (batch_size={args.batch_size})")

    # 3. Batch puanlama
    scored_count = 0
    error_count = 0
    all_scores = []
    start_time = time.time()

    for batch_start in tqdm(range(0, total, args.batch_size), desc="Estetik Puanlama"):
        batch_end = min(batch_start + args.batch_size, total)
        batch_paths = unscored_paths[batch_start:batch_end]
        batch_ids = unscored_ids[batch_start:batch_end]
        batch_metas = unscored_metas[batch_start:batch_end]

        batch_scores = score_images_batch(
            vision_model, mlp_head, preprocessor, batch_paths, device
        )

        update_ids = []
        update_metas = []

        for i, score in enumerate(batch_scores):
            if score < 0:
                error_count += 1
                continue

            meta = batch_metas[i].copy()
            meta["aesthetic_score"] = round(score, 4)
            meta["aesthetic_scorer_model"] = "aesthetic_predictor_v2.5"
            update_ids.append(batch_ids[i])
            update_metas.append(meta)
            all_scores.append(score)
            scored_count += 1

        if update_ids:
            try:
                store.collection.update(ids=update_ids, metadatas=update_metas)
            except Exception as e:
                logger.error(f"ChromaDB güncelleme hatası: {e}")

        # Bellek temizliği
        if (batch_start // args.batch_size) % 50 == 0 and batch_start > 0:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            elapsed = time.time() - start_time
            speed = scored_count / elapsed if elapsed > 0 else 0
            remaining = (total - batch_end) / speed if speed > 0 else 0
            logger.info(
                f"📈 İlerleme: {batch_end}/{total} | "
                f"Hız: {speed:.1f} img/sn | "
                f"Kalan: {remaining/60:.1f} dk"
            )

    # 4. Sonuç raporu
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("🎉 ESTETİK PUANLAMA TAMAMLANDI!")
    logger.info("=" * 60)
    logger.info(f"   Puanlanan: {scored_count}")
    logger.info(f"   Hatalı/Atlanan: {error_count}")
    logger.info(f"   Süre: {elapsed/60:.1f} dakika")

    if all_scores:
        scores_arr = np.array(all_scores)
        logger.info(f"\n   📊 Puan Dağılımı:")
        logger.info(f"      Ortalama : {scores_arr.mean():.2f}")
        logger.info(f"      Medyan   : {np.median(scores_arr):.2f}")
        logger.info(f"      Min      : {scores_arr.min():.2f}")
        logger.info(f"      Max      : {scores_arr.max():.2f}")
        logger.info(f"      Std      : {scores_arr.std():.2f}")

        elite = np.sum(scores_arr >= 7.0)
        good = np.sum((scores_arr >= 5.5) & (scores_arr < 7.0))
        avg = np.sum((scores_arr >= 4.0) & (scores_arr < 5.5))
        low = np.sum(scores_arr < 4.0)
        logger.info(f"\n   🏆 Kalite Sınıfları:")
        logger.info(f"      Elite (7+)     : {elite:,} ({elite/len(scores_arr)*100:.1f}%)")
        logger.info(f"      İyi (5.5-7)    : {good:,} ({good/len(scores_arr)*100:.1f}%)")
        logger.info(f"      Orta (4-5.5)   : {avg:,} ({avg/len(scores_arr)*100:.1f}%)")
        logger.info(f"      Düşük (<4)     : {low:,} ({low/len(scores_arr)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aesthetic Predictor V2.5")
    parser.add_argument("--batch-size", type=int, default=16, help="GPU batch boyutu")
    parser.add_argument("--limit", type=int, default=0, help="Kaç görsel (0 = tümü)")
    args = parser.parse_args()
    if args.limit == 0:
        args.limit = None
    main(args)
