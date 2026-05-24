"""Hızlı ChromaDB çıktı analizi"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings
from indexer.chroma_store import ChromaStore

settings = get_settings()
store = ChromaStore(db_path=settings.chroma_db_path, collection_name=settings.get_collection_name())
col = store.collection

print(f"=== ChromaDB Koleksiyon: {settings.get_collection_name()} ===")
print(f"Toplam kayit: {col.count()}")
print()

# Ilk 8 kayit cek
results = col.get(limit=8, include=["metadatas", "documents"])

for i in range(len(results["ids"])):
    doc_id = results["ids"][i]
    meta = results["metadatas"][i]
    doc = results["documents"][i]
    print(f"--- Model #{i+1}: {doc_id} ---")
    print(f"  Furniture: {meta.get('clip_furniture_type', 'N/A')}")
    print(f"  Style:     {meta.get('clip_style', 'N/A')}")
    print(f"  Material:  {meta.get('clip_material', 'N/A')}")
    print(f"  Color:     {meta.get('clip_color', 'N/A')}")
    print(f"  Room:      {meta.get('clip_room', 'N/A')}")
    print(f"  Category:  {meta.get('clip_category', 'N/A')}")
    print(f"  Channel:   {meta.get('channel_title', 'N/A')}")
    cap = (meta.get("caption", "") or "")[:100]
    print(f"  Caption:   {cap}")
    print()

# Checkpoint ve corrupted dosyalarini kontrol et
cp = Path("data/siglip_indexed_checkpoint.txt")
cr = Path("data/siglip_corrupted_images.txt")
if cp.exists():
    lines = cp.read_text(encoding="utf-8").strip().split("\n")
    print(f"=== Checkpoint: {len(lines)} gorsel islendi ===")
else:
    print("=== Checkpoint dosyasi yok ===")

if cr.exists():
    clines = cr.read_text(encoding="utf-8").strip().split("\n")
    print(f"=== Bozuk gorseller: {len(clines)} adet ===")
    for c in clines[:5]:
        print(f"  - {c}")
else:
    print("=== Bozuk gorsel dosyasi yok ===")
