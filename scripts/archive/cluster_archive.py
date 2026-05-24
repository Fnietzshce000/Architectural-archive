import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# Ana dizini path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from indexer.chroma_store import ChromaStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Benzerlik Eşiği: 0.04 (Cosine distance). 1.0 - 0.04 = 0.96 benzerlik demektir.
SIMILARITY_THRESHOLD = 0.04


def save_clusters_atomic(clusters: dict, file_path: Path):
    """Kümeleme sonuçlarını atomik olarak kaydeder."""
    temp_path = file_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    temp_path.replace(file_path)

def main():
    parser = argparse.ArgumentParser(description="Gelişmiş Mükerrer Tespit Sistemi (Deduplicator)")
    parser.add_argument("--confirm", action="store_true", help="Tespit edilenleri clusters.json'a yaz")
    parser.add_argument("--batch-size", type=int, default=100, help="Vektör çekme batch boyutu")
    args = parser.parse_args()

    settings = get_settings()
    cluster_file = settings.get_data_path() / "clusters.json"
    store = ChromaStore(
        db_path=settings.chroma_db_path,
        collection_name=settings.get_collection_name(),
    )
    total_models = store.get_count()
    from config import MetadataSchema
    
    logger.info(f"🔍 {total_models} model üzerinde akıllı deduplication başlatılıyor...")
    
    clusters = {}
    processed_ids = set()
    
    if cluster_file.exists():
        try:
            with open(cluster_file, "r", encoding="utf-8") as f:
                clusters = json.load(f)
            for data in clusters.values():
                processed_ids.update(data["members"])
            logger.info(f"💾 {len(clusters)} mevcut grup yüklendi, {len(processed_ids)} ID atlanacak.")
        except: pass
        
    all_ids = store.get_all_ids()
    unprocessed_ids = sorted(list(set(all_ids) - processed_ids))
    total_to_process = len(unprocessed_ids)
    
    if total_to_process == 0:
        logger.info("✅ Tüm modeller zaten analiz edilmiş!")
        return
        
    logger.info(f"🚀 İşlem başlıyor: {total_to_process} yeni model analiz edilecek.")
    pbar = tqdm(total=total_to_process, desc="Analiz")
    
    idx = 0
    while idx < len(unprocessed_ids):
        rep_id = unprocessed_ids[idx]
        if rep_id in processed_ids:
            idx += 1
            continue
            
        # Temsilcinin verisini al
        rep_data = store.collection.get(ids=[rep_id], include=["embeddings", "metadatas"])
        if not rep_data or not rep_data["ids"]:
            idx += 1
            pbar.update(1)
            continue
            
        rep_vector = rep_data["embeddings"][0]
        rep_meta = rep_data["metadatas"][0]
        
        # Akıllı Sorgu: Benzerlerini bul
        results = store.collection.query(
            query_embeddings=[rep_vector],
            n_results=50,
            include=["distances"]
        )
        
        members = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                member_id = results["ids"][0][i]
                distance = results["distances"][0][i]
                
                # %96 ve üzeri benzerlik (Threshold: 0.04 distance)
                if distance <= SIMILARITY_THRESHOLD:
                    if member_id not in processed_ids:
                        members.append(member_id)
                        processed_ids.add(member_id)
        
        if rep_id not in members: # Güvenlik
            members.append(rep_id)
            processed_ids.add(rep_id)

        clusters[rep_id] = {
            "image_path": rep_meta.get(MetadataSchema.IMAGE_PATH),
            "members": members
        }
        
        pbar.update(len(members))
        
        # Ara kayıt
        if len(clusters) % 500 == 0 and args.confirm:
            save_clusters_atomic(clusters, cluster_file)
        
        idx += 1

    if args.confirm:
        save_clusters_atomic(clusters, cluster_file)
        logger.info(f"✅ Sonuçlar kaydedildi: {cluster_file}")
    else:
        logger.warning("⚠️ DRY-RUN: Sonuçlar kaydedilmedi. Yazmak için --confirm ekleyin.")

    logger.info(f"📊 Sonuç: {total_models} model -> {len(clusters)} benzersiz grup.")

if __name__ == "__main__":
    main()
