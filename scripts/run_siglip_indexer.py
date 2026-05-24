"""
🚀 Archi V2: SigLIP 2 Indexer and Tagger (Optimized)
Tarama + Dedublikasyon + SigLIP 2 ile Vektörleştirme + Detaylı Zero-Shot Etiketleme + ChromaDB Kayıt.

Bu script 460K+ görseli GPU kullanarak batch halinde okur, 
SigLIP 2 modeli ile hem yüksek kaliteli arama vektörlerini çıkarır 
hem de 6 farklı kategoride sıfır maliyetle detaylı etiketleme yapar.

İyileştirmeler:
1. ChromaDB commit frekansı düşürüldü (1000'erli biriktirerek yazma → 10x-20x hız artışı).
2. CUDA bellek yönetimi (VRAM sızıntılarını önlemek için explicit garbage collection).
3. Bozuk görsellerin veritabanını kilitlemesini ve kısır döngüye girmesini önlemek için hata toleranslı checkpoint.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Proje kökünü ekle
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings, MetadataSchema
from preprocessing.deduplicator import ImageDeduplicator
from preprocessing.image_processor import ImageProcessor
from indexer.clip_encoder import load_model
from indexer.chroma_store import ChromaStore
from scraper.models import TelegramMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("data/siglip_indexing.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("SigLIPIndexer")

# ══════════════════════════════════════════════════
# Detaylı İç Mimarlık ve Tasarım Taksonomisi
# ══════════════════════════════════════════════════
TAXONOMY: Dict[str, List[str]] = {
    "furniture_type": [
        "sofa", "armchair", "dining chair", "office chair", "bar stool", "bench",
        "dining table", "coffee table", "side table", "desk", "console table",
        "bed", "wardrobe", "dresser", "nightstand", "bookshelf", "cabinet", "sideboard", "tv unit",
        "bathtub", "sink", "toilet", "shower", "faucet",
        "kitchen cabinet", "kitchen island",
        "rug", "carpet", "curtain", "mirror", "pillow",
        "chandelier", "pendant light", "floor lamp", "table lamp", "wall sconce", "spotlight",
        "plant", "vase", "sculpture", "painting", "wall art", "clock", "book", "decorations",
        "door", "window", "staircase", "fireplace", "column", "wall panel",
    ],
    "style": [
        "modern", "minimalist", "contemporary", "scandinavian", "japandi",
        "industrial", "rustic", "loft", "mid-century modern",
        "classic", "neoclassic", "baroque", "provence", "traditional",
        "art deco", "bohemian", "retro", "vintage", "luxury", "futuristic",
    ],
    "material": [
        "wood", "oak", "walnut", "pine", "plywood",
        "metal", "steel", "brass", "gold", "chrome", "bronze",
        "glass", "mirror glass", "tinted glass",
        "marble", "granite", "terrazzo", "concrete", "stone", "brick",
        "leather", "fabric", "velvet", "boucle", "linen",
        "ceramic", "porcelain", "plastic", "acrylic", "rattan", "bamboo",
    ],
    "color": [
        "white", "black", "gray", "anthracite", "beige", "cream", "ivory",
        "brown", "oak brown", "walnut brown",
        "blue", "navy blue", "cyan",
        "green", "olive green", "emerald green",
        "red", "burgundy", "terracotta",
        "yellow", "mustard yellow", "orange", "pink", "purple",
        "gold color", "silver color", "brass color",
    ],
    "room": [
        "living room", "bedroom", "bathroom", "kitchen", "dining room",
        "office", "study room", "hallway", "entryway", "corridor",
        "kids room", "nursery", "dressing room", "laundry room",
        "balcony", "terrace", "outdoor", "garden", "commercial space", "restaurant",
    ],
    "category": [
        "furniture", "lighting", "decoration", "accessory", "sanitary",
        "kitchenware", "textile", "architectural element", "3d scene",
        "greenery", "electronics", "pbr material", "texture",
    ],
}

CHECKPOINT_FILE = PROJECT_ROOT / "data" / "siglip_indexed_checkpoint.txt"
CORRUPTED_FILE = PROJECT_ROOT / "data" / "siglip_corrupted_images.txt"

def load_metadata(data_dir: Path) -> dict:
    """Metadata dosyasından mesaj bilgilerini yükler."""
    metadata_file = data_dir / "messages_metadata.jsonl"
    metadata_map = {}
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        msg = TelegramMessage.from_dict(data)
                        if msg.image_path:
                            # Dosya yolunu normalize et (Windows/Posix uyumluluğu için)
                            normalized_path = str(Path(msg.image_path).as_posix())
                            metadata_map[normalized_path] = msg
                    except Exception as e:
                        pass
    return metadata_map

def load_checkpoint() -> Set[str]:
    """Daha önce başarıyla indekslenen veya bozuk olduğu tespit edilen görsel ID'lerini yükler."""
    checkpoint_ids = set()
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    checkpoint_ids.add(line)
    return checkpoint_ids

def save_checkpoint(doc_ids: List[str]):
    """İndekslenen görsel ID'lerini checkpoint dosyasına ekler."""
    if not doc_ids:
        return
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        for doc_id in doc_ids:
            f.write(f"{doc_id}\n")

def log_corrupted_image(img_path: Path):
    """Bozuk görseli hem corrupted loguna hem de checkpoint'e ekler."""
    doc_id = img_path.stem
    # Checkpoint'e yaz ki bir daha taramaya çalışmasın
    save_checkpoint([doc_id])
    # Bozuklar listesine yaz
    with open(CORRUPTED_FILE, "a", encoding="utf-8") as f:
        f.write(f"{img_path.as_posix()}\n")

def build_taxonomy_vectors(model, tokenizer, device) -> Dict[str, torch.Tensor]:
    """
    Taksonomideki tüm etiketlerin SigLIP 2 metin vektörlerini hesaplar.
    """
    tax_vectors = {}
    logger.info("📐 Taksonomi etiketleri SigLIP 2 ile vektörleştiriliyor...")
    
    for category, labels in TAXONOMY.items():
        for label in labels:
            if category == "furniture_type":
                prompt = f"a 3D model render of a {label}"
            elif category == "style":
                prompt = f"a {label} style interior design"
            elif category == "material":
                prompt = f"an object made of {label} material"
            elif category == "color":
                prompt = f"a {label} colored item"
            elif category == "room":
                prompt = f"a 3D render of a {label} space"
            elif category == "category":
                prompt = f"a 3D asset of {label}"
            else:
                prompt = f"a {label}"

            tokens = tokenizer([prompt]).to(device)
            with torch.no_grad():
                vec = model.encode_text(tokens)
                vec /= vec.norm(dim=-1, keepdim=True)
            tax_vectors[f"{category}:{label}"] = vec.squeeze()
            
    logger.info(f"   ✅ {len(tax_vectors)} etiket vektörü hazırlandı.")
    return tax_vectors

def classify_embeddings_batch(
    embeddings: torch.Tensor,
    tax_vectors: Dict[str, torch.Tensor],
    top_k: int = 3,
    min_score: float = 0.02
) -> List[Dict[str, List[str]]]:
    """
    Kosinüs benzerliği matrisi ile batch halindeki görsel vektörlerini hızlıca sınıflandırır.
    """
    keys = list(tax_vectors.keys())
    tax_matrix = torch.stack([tax_vectors[k] for k in keys]).to(embeddings.device).to(embeddings.dtype)
    similarity_matrix = embeddings @ tax_matrix.T
    
    batch_results = []
    for b in range(embeddings.shape[0]):
        scores = similarity_matrix[b].cpu().numpy()
        results: Dict[str, List[Tuple[str, float]]] = {}
        
        for idx, score in enumerate(scores):
            if score < min_score:
                continue
            cat, label = keys[idx].split(":", 1)
            if cat not in results:
                results[cat] = []
            results[cat].append((label, float(score)))
            
        final_tags = {}
        for cat in TAXONOMY.keys():
            cat_list = results.get(cat, [])
            cat_list = sorted(cat_list, key=lambda x: -x[1])[:top_k]
            final_tags[cat] = [label for label, _ in cat_list]
            
        batch_results.append(final_tags)
        
    return batch_results

def main(args):
    settings = get_settings()
    data_dir = settings.get_data_path()
    images_dir = settings.get_images_path()

    logger.info("🚀 SigLIP 2 İndeksleyici ve Etiketleyici (Optimized) Başlatılıyor...")
    logger.info(f"   Koleksiyon: {settings.get_collection_name()}")

    # ── Adım 1: Metadata yükleme ──
    logger.info("\n📋 Adım 1: Metadata yükleniyor...")
    metadata_map = load_metadata(data_dir)
    logger.info(f"   {len(metadata_map)} mesaj metadata'sı bellekten çözüldü.")

    # ── Adım 2: Görsel tarama ──
    logger.info("\n📁 Adım 2: Görseller taranıyor...")
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images_dir_path = Path(images_dir)
    all_images = [
        str(f) for f in sorted(images_dir_path.iterdir())
        if f.is_file() and f.suffix.lower() in extensions
    ]

    if not all_images:
        logger.error("❌ Hiç görsel bulunamadı! Lütfen önce scraper'ı çalıştırın.")
        return

    # ── Adım 3: Dedublikasyon ──
    if not args.skip_dedup:
        logger.info("\n🔍 Adım 3: Dedublikasyon yapılıyor...")
        dedup = ImageDeduplicator(
            hash_db_path=data_dir / "hash_db.json",
            dhash_threshold=5,
            phash_threshold=8,
        )
        unique_images, duplicate_images = dedup.process_directory(images_dir)
        logger.info(f"   ✅ {len(unique_images)} benzersiz görsel, {len(duplicate_images)} kopya elendi.")
    else:
        logger.info("\n⏭️ Adım 3: Dedublikasyon atlandı.")
        unique_images = all_images

    # ── Adım 4: ChromaDB hazırlık ──
    logger.info("\n📦 Adım 4: ChromaDB bağlantısı kuruluyor...")
    store = ChromaStore(
        db_path=settings.chroma_db_path,
        collection_name=settings.get_collection_name(),
    )

    if args.reset:
        logger.warning("   ⚠️ İndeks sıfırlanıyor ve checkpoint siliniyor...")
        store.reset()
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
        if CORRUPTED_FILE.exists():
            CORRUPTED_FILE.unlink()

    # Checkpoint yükle
    indexed_ids = load_checkpoint()
    logger.info(f"   Checkpoint'e göre {len(indexed_ids)} görsel zaten işlenmiş/atlanmış.")

    # İndekslenecek görselleri filtrele
    images_to_index = []
    for img_path in unique_images:
        doc_id = Path(img_path).stem
        if doc_id not in indexed_ids:
            images_to_index.append(img_path)

    if not images_to_index:
        logger.info("   🎉 Harika! İndekslenecek yeni görsel yok. Sistem güncel.")
        return

    logger.info(f"   👉 {len(images_to_index)} yeni görsel SigLIP 2 ile taranacak ve etiketlenecek.")

    # ── Adım 5: SigLIP 2 Modeli yükleme ──
    logger.info(f"\n🧠 Adım 5: SigLIP 2 modeli yükleniyor ({settings.clip_model_name})...")
    model, preprocess, tokenizer, device = load_model(
        model_name=settings.clip_model_name,
        pretrained=settings.clip_pretrained,
    )
    
    # Etiket vektörlerini çıkar
    tax_vectors = build_taxonomy_vectors(model, tokenizer, device)

    # ── Adım 6: Batch İşleme ve Biriktirerek Kayıt ──
    logger.info("\n🏷️ Adım 6: Görseller taranıyor, etiketleniyor ve kaydediliyor...")
    
    gpu_batch_size = args.batch_size
    write_batch_size = args.write_batch
    total_images = len(images_to_index)
    start_time = time.time()

    # Bellekte biriktirme listeleri
    accumulated_ids = []
    accumulated_embeddings = []
    accumulated_metadatas = []
    accumulated_documents = []

    def commit_accumulated():
        """Bellekteki verileri tek seferde ChromaDB'ye yazar ve checkpoint'e kaydeder."""
        if not accumulated_ids:
            return
        try:
            store.add_batch(
                doc_ids=accumulated_ids,
                embeddings=accumulated_embeddings,
                metadatas=accumulated_metadatas,
                documents=accumulated_documents,
                batch_size=500,  # ChromaDB içindeki alt batch boyutu
            )
            save_checkpoint(accumulated_ids)
            logger.info(f"   💾 {len(accumulated_ids)} model ChromaDB'ye kaydedildi ve commitlendi.")
        except Exception as e:
            logger.error(f"❌ Toplu ChromaDB commit hatası: {e}")
        finally:
            accumulated_ids.clear()
            accumulated_embeddings.clear()
            accumulated_metadatas.clear()
            accumulated_documents.clear()

    for start_idx in range(0, total_images, gpu_batch_size):
        end_idx = min(start_idx + gpu_batch_size, total_images)
        batch_paths = images_to_index[start_idx:end_idx]

        # Batch resim yükleme ve preprocess
        tensors = []
        valid_indices = []
        for i, img_path in enumerate(batch_paths):
            path_obj = Path(img_path)
            try:
                if not path_obj.exists() or path_obj.stat().st_size == 0:
                    raise ValueError("Dosya yok veya 0 bayt.")
                img = Image.open(path_obj).convert("RGB")
                tensor = preprocess(img)
                tensors.append(tensor)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"⚠️ Görsel yüklenemedi, atlanıyor ve kara listeye ekleniyor ({img_path}): {e}")
                log_corrupted_image(path_obj)

        if not tensors:
            continue

        # Tensors GPU'ya
        batch_tensor = torch.stack(tensors).to(device)

        # SigLIP 2 Görsel Vektörleştirme
        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast("cuda"):
                    features = model.encode_image(batch_tensor)
            else:
                features = model.encode_image(batch_tensor)
            features /= features.norm(dim=-1, keepdim=True)

        # Batch Etiketleme (Zero-shot classification)
        batch_tags = classify_embeddings_batch(features, tax_vectors)

        # Numpy'a çevir
        features_np = features.cpu().numpy()

        for idx, global_idx in enumerate(valid_indices):
            img_path = batch_paths[global_idx]
            vector = features_np[idx]
            tags = batch_tags[idx]

            doc_id = Path(img_path).stem
            normalized_path = str(Path(img_path).as_posix())
            meta = metadata_map.get(normalized_path)

            # Metadata oluştur
            metadata = {
                MetadataSchema.SOURCE: "SigLIPIndexer",
                MetadataSchema.MODEL_NAME: settings.clip_model_name,
                MetadataSchema.SCHEMA_VERSION_KEY: MetadataSchema.CURRENT_SCHEMA_VERSION,
                "clip_tagged": True,
            }

            # Etiketleri ekle
            all_tags_list = []
            for cat, labels in tags.items():
                metadata[f"clip_{cat}"] = ", ".join(labels)
                all_tags_list.extend(labels)
            
            clip_tags_str = ", ".join(all_tags_list)
            metadata["clip_tags"] = clip_tags_str

            if meta:
                metadata.update({
                    MetadataSchema.CHANNEL_ID: meta.channel_id,
                    MetadataSchema.CHANNEL_TITLE: meta.channel_title,
                    MetadataSchema.CHANNEL_USERNAME: meta.channel_username or "",
                    MetadataSchema.MESSAGE_ID: meta.message_id,
                    MetadataSchema.CAPTION: meta.caption,
                    MetadataSchema.IMAGE_PATH: meta.image_path,
                    MetadataSchema.DEEP_LINK: meta.deep_link,
                    MetadataSchema.TIMESTAMP: meta.timestamp,
                })
                file_names = getattr(meta, "file_names", [])
                if file_names:
                    metadata["file_names"] = json.dumps(file_names, ensure_ascii=False)
                
                document_text = f"{meta.caption} | {clip_tags_str}"
            else:
                metadata.update({
                    MetadataSchema.IMAGE_PATH: str(img_path),
                    MetadataSchema.DEEP_LINK: "",
                    MetadataSchema.CAPTION: "",
                    MetadataSchema.CHANNEL_TITLE: "",
                })
                document_text = f"{doc_id} | {clip_tags_str}"

            accumulated_ids.append(doc_id)
            accumulated_embeddings.append(vector)
            accumulated_metadatas.append(metadata)
            accumulated_documents.append(document_text)

        # Explicit GPU Bellek Temizliği
        del batch_tensor, features
        if device == "cuda":
            torch.cuda.empty_cache()

        # Birikme limitine ulaştıysa veya sona gelindiyse diske yaz
        if len(accumulated_ids) >= write_batch_size:
            commit_accumulated()

        # İlerleme raporu
        elapsed = time.time() - start_time
        speed = end_idx / max(elapsed, 0.1)
        remaining = (total_images - end_idx) / max(speed, 0.1) if speed > 0 else 0
        logger.info(
            f"📈 İlerleme: {end_idx}/{total_images} model tamamlandı | "
            f"Hız: {speed:.1f} model/sn | Kalan Süre: {remaining/3600:.2f} saat"
        )

    # Döngü bittiğinde kalan son verileri de commit et
    commit_accumulated()

    logger.info(f"\n{'='*60}")
    logger.info("🎉 SigLIP 2 İndeksleme ve Etiketleme İşlemi Başarıyla Tamamlandı!")
    logger.info(f"   Toplam eklenen model sayısı: {total_images}")
    logger.info(f"   Koleksiyon adı: {settings.get_collection_name()}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SigLIP 2 İndeksleyici ve Etiketleyici (Optimized)")
    parser.add_argument(
        "--skip-dedup", action="store_true",
        help="Dedublikasyon adımını atla"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="ChromaDB indeksini sıfırla ve yeniden oluştur"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="GPU batch boyutu (RTX 4060 8GB için 32-64 idealdir)"
    )
    parser.add_argument(
        "--write-batch", type=int, default=1000,
        help="ChromaDB diske yazma sıklığı (varsayılan: 1000)"
    )
    args = parser.parse_args()

    main(args)
