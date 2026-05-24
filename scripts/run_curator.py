"""
Reklam & Logo Avcisi Pipeline
================================
Amac: 485K gorsel icinden sadece reklam, logo, yazi ve afisleri bul ve tasI.
3D modellere asla dokunma!

Strateji:
  Asama 1: CLIP ile "logo, reklam, yazi, afis" benzerlerini bul (~%5-10)
  Asama 2: Qwen3-VL ile sadece suphelileri dogrula

Kullanim:
    python scripts/run_curator.py                  # Tam pipeline
    python scripts/run_curator.py --clip-only      # Sadece CLIP asama (onizleme)
    python scripts/run_curator.py --resume         # Kaldigi yerden devam
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from indexer.chroma_store import ChromaStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("curator")

# -- Sabitler --
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_ID = "qwen/qwen3-vl-8b"
MAX_WORKERS = 4           # RTX 4060 icin 4 esanli istek
BATCH_LOG_EVERY = 50
STATE_SAVE_EVERY = 100

# Qwen prompt: ters mantik - sadece cop olanlari yakala
SYSTEM_PROMPT = (
    "You classify images. Answer ONLY with one word: TRASH or KEEP.\n"
    "TRASH = advertisement, logo, banner, flyer, text-heavy image, "
    "screenshot, watermark-only image, channel promotion, or any "
    "non-3D-model content.\n"
    "KEEP = 3D model render, architectural visualization, product render, "
    "furniture render, interior/exterior design, or any 3D artwork.\n"
    "When in doubt, answer KEEP."
)

USER_PROMPT = "Classify this image. Answer TRASH or KEEP only."

# CLIP: "reklam/logo" benzerlik esigi
# Bu esik uzerindekiler Qwen'e gidecek
JUNK_SIMILARITY_THRESHOLD = 0.22


# -- Yardimci Fonksiyonlar --

def image_to_base64(image_path: str) -> str | None:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def query_qwen(image_path: str, timeout: int = 30) -> str | None:
    b64 = image_to_base64(image_path)
    if b64 is None:
        return None

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        "max_tokens": 3,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
        if "TRASH" in answer:
            return "TRASH"
        if "KEEP" in answer:
            return "KEEP"
        return "KEEP"  # emin degilse koru
    except Exception:
        return None


def load_state(state_file: Path) -> dict:
    if state_file.exists():
        with open(state_file, "r") as f:
            return json.load(f)
    return {"processed": {}, "stats": {"kept": 0, "trashed": 0, "errors": 0}}


def save_state_atomic(state: dict, state_file: Path):
    """Dosyayı önce geçici bir yere yazar, sonra üzerine taşır (Atomic Write)."""
    temp_file = state_file.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=4)
    if temp_file.exists():
        temp_file.replace(state_file)


# -- Asama 1: CLIP ile Reklam/Logo Avcisi --

def clip_find_junk(store: ChromaStore) -> list:
    """
    CLIP vektorlerini kullanarak reklam, logo ve yazi iceren
    gorselleri bulur. TERS mantik: sadece cop olanlari yakala.
    """
    logger.info("Asama 1: CLIP ile reklam/logo taramasi basliyor...")

    try:
        import torch
        from indexer.clip_encoder import load_model

        settings = get_settings()
        model, preprocess, tokenizer, device = load_model(
            model_name=settings.clip_model_name,
            pretrained=settings.clip_pretrained,
        )

        # SADECE cop tanimlamalari - 3D model degil!
        junk_texts = [
            "advertisement banner flyer poster with text",
            "company logo brand icon watermark",
            "screenshot of website or chat application",
            "text overlay promotional image sale discount",
            "social media channel promotion subscribe",
        ]

        with torch.no_grad():
            junk_tokens = tokenizer(junk_texts).to(device)
            junk_features = model.encode_text(junk_tokens)
            junk_features /= junk_features.norm(dim=-1, keepdim=True)
            junk_vec = junk_features.mean(dim=0).cpu().numpy()

        junk_vec_norm = junk_vec / (np.linalg.norm(junk_vec) + 1e-8)
        logger.info("   CLIP reklam/logo referans vektoru olusturuldu")

    except Exception as e:
        logger.error(f"CLIP yuklenemedi: {e}")
        raise

    # ChromaDB'den vektorleri tara
    collection = store.collection
    total = store.get_count()
    logger.info(f"   Toplam: {total:,} gorsel taranacak")

    suspects = []
    clean = 0
    all_ids = store.get_all_ids()
    batch_size = 5000

    def iter_valid_batches(ids):
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            try:
                batch = collection.get(
                    ids=batch_ids,
                    include=["embeddings", "metadatas"],
                )
                yield start, batch
            except Exception as exc:
                logger.warning(f"   Batch okunamadi ({start}-{start + len(batch_ids)}): {exc}. Tek tek deneniyor...")
                safe_batch = {"ids": [], "embeddings": [], "metadatas": []}
                for doc_id in batch_ids:
                    try:
                        single = collection.get(ids=[doc_id], include=["embeddings", "metadatas"])
                        if single and single["ids"]:
                            safe_batch["ids"].extend(single["ids"])
                            safe_batch["embeddings"].extend(single["embeddings"])
                            safe_batch["metadatas"].extend(single["metadatas"])
                    except Exception:
                        logger.debug(f"   Hayalet/bozuk ID atlandi: {doc_id}")
                yield start, safe_batch

    for offset, batch in iter_valid_batches(all_ids):

        ids = batch["ids"]
        embeddings = batch["embeddings"]
        metadatas = batch["metadatas"]

        if not ids:
            logger.warning(f"   Okunabilir kayit bulunamadi, batch atlandi: {offset}-{offset + batch_size}")
            processed = min(offset + batch_size, total)
            logger.info(
                f"   Islendi: {processed:,}/{total:,} | "
                f"Temiz: {clean:,} | Supheli: {len(suspects):,}"
            )
            continue

        for doc_id, emb, meta in zip(ids, embeddings, metadatas):
            img_path = meta.get("image_path", "")
            if not img_path or not Path(img_path).exists():
                continue

            vec = np.array(emb)
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            junk_sim = float(np.dot(vec_norm, junk_vec_norm))

            if junk_sim >= JUNK_SIMILARITY_THRESHOLD:
                suspects.append(img_path)
            else:
                clean += 1

        processed = min(offset + batch_size, total)
        logger.info(
            f"   Islendi: {processed:,}/{total:,} | "
            f"Temiz: {clean:,} | Supheli: {len(suspects):,}"
        )

    logger.info(f"\nCLIP Asama Sonucu:")
    logger.info(f"   Temiz (3D Model - dokunulmayacak): {clean:,}")
    logger.info(f"   Supheli (Qwen'e gidecek): {len(suspects):,}")
    logger.info(f"   Oran: %{len(suspects)*100/(clean+len(suspects)):.1f} supheli")

    return suspects


# -- Asama 2: Qwen-VL ile Dogrulama --

def qwen_verify(suspects: list, trash_dir: Path, state_file: Path, store: ChromaStore, confirm: bool = False, resume: bool = False) -> dict:
    logger.info(f"\nAsama 2: Qwen3-VL dogrulama basliyor...")
    logger.info(f"   Supheli gorsel: {len(suspects):,}")
    
    if not confirm:
        logger.warning("⚠️ DRY-RUN MODU: Hiçbir dosya taşınmayacak ve DB'den silinmeyecek.")
        logger.warning("   Gerçekten temizlik yapmak için --confirm ekleyin.")

    state = load_state(state_file) if resume else {
        "processed": {}, "stats": {"kept": 0, "trashed": 0, "errors": 0}
    }
    stats = state["stats"] if confirm else {"kept": 0, "trashed": 0, "errors": 0}

    to_process = [p for p in suspects if p not in state["processed"]]
    logger.info(f"   Islenecek: {len(to_process):,}")

    if not to_process:
        logger.info("   Hepsi zaten islenmis!")
        return stats

    trash_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(query_qwen, path): path
            for path in to_process
        }

        for future in as_completed(future_map):
            img_path = future_map[future]
            completed += 1

            try:
                result = future.result()
            except Exception:
                result = None

            if result == "TRASH":
                src = Path(img_path)
                if confirm and src.exists():
                    # 1. Dosyayı Taşı
                    dst = trash_dir / src.name
                    src.rename(dst)
                    
                    # 2. DB'den Sil (ID genelde dosya ismidir)
                    doc_id = src.stem # "123_456.jpg" -> "123_456"
                    try:
                        store.delete([doc_id])
                    except Exception as e:
                        logger.error(f"   DB silme hatası ({doc_id}): {e}")
                    state["processed"][img_path] = "trash"
                elif confirm:
                    state["processed"][img_path] = "trash"

                stats["trashed"] += 1
            elif result == "KEEP":
                if confirm:
                    state["processed"][img_path] = "keep"
                stats["kept"] += 1
            else:
                if confirm:
                    state["processed"][img_path] = "error"
                stats["errors"] += 1

            if completed % BATCH_LOG_EVERY == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = len(to_process) - completed
                eta_min = (remaining / rate / 60) if rate > 0 else 0

                logger.info(
                    f"   [{completed:,}/{len(to_process):,}] "
                    f"KEEP: {stats['kept']:,} | "
                    f"TRASH: {stats['trashed']:,} | "
                    f"Hiz: {rate:.1f}/sn | "
                    f"ETA: {eta_min:.0f} dk"
                )

            if confirm and completed % STATE_SAVE_EVERY == 0:
                save_state_atomic(state, state_file)

    if confirm:
        save_state_atomic(state, state_file)
    return stats


# -- Ana Pipeline --

def main(args):
    settings = get_settings()
    data_dir = settings.get_data_path()
    trash_dir = data_dir / "trash_images"
    state_file = data_dir / "curator_state.json"

    logger.info("=" * 55)
    logger.info("REKLAM & LOGO AVCISI PIPELINE")
    logger.info("=" * 55)

    store = ChromaStore(
        db_path=settings.chroma_db_path,
        collection_name=settings.get_collection_name(),
    )

    # Asama 1: CLIP
    suspects = clip_find_junk(store)

    if args.clip_only:
        logger.info("\nSadece CLIP modu - bitti!")
        logger.info(f"Supheliler listesi: {len(suspects):,} gorsel")
        # Suphelileri dosyaya kaydet
        with open(data_dir / "suspects_list.json", "w") as f:
            json.dump(suspects, f)
        logger.info(f"Liste kaydedildi: {data_dir / 'suspects_list.json'}")
        return

    if not suspects:
        logger.info("\nHic supheli gorsel bulunamadi! Arsiviniz tertemiz.")
        return

    # LM Studio kontrol
    try:
        r = requests.get("http://localhost:1234/v1/models", timeout=3)
        r.raise_for_status()
        logger.info("LM Studio baglantisi OK")
    except Exception:
        logger.error("LM Studio erisilemedi! Modeli baslatin.")
        return

    # Asama 2: Qwen
    stats = qwen_verify(suspects, trash_dir, state_file, store, confirm=args.confirm, resume=args.resume)

    logger.info(f"\n{'=' * 55}")
    logger.info(f"KURASYON TAMAMLANDI!")
    logger.info(f"   KEEP (korunan):  {stats['kept']:,}")
    logger.info(f"   TRASH (silinen): {stats['trashed']:,}")
    logger.info(f"   Hata:            {stats['errors']:,}")
    logger.info(f"   Cop klasoru: {trash_dir}")
    logger.info(f"{'=' * 55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reklam & Logo Avcisi")
    parser.add_argument("--clip-only", action="store_true", help="Sadece CLIP taramasi yap")
    parser.add_argument("--resume", action="store_true", help="Kaldigi yerden devam et")
    parser.add_argument("--confirm", action="store_true", help="Gerçek silme/taşıma işlemini başlat")
    args = parser.parse_args()
    main(args)
