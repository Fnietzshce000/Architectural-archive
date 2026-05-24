"""ChromaDB'den 100 rastgele kayıt çekip etiket kalitesini analiz et."""
import sys, random, json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_settings
from indexer.chroma_store import ChromaStore

settings = get_settings()
store = ChromaStore(db_path=settings.chroma_db_path, collection_name=settings.get_collection_name())
col = store.collection

total = col.count()
print(f"=== Koleksiyon: {settings.get_collection_name()} | Toplam: {total} kayit ===\n")

# 100 rastgele kayit cek (offset ile dagitarak)
sample_size = 100
step = max(1, total // sample_size)
offsets = [i * step for i in range(sample_size)]

all_records = []
for off in offsets:
    batch = col.get(limit=1, offset=off, include=["metadatas"])
    if batch and batch["ids"]:
        all_records.append({"id": batch["ids"][0], "meta": batch["metadatas"][0]})

print(f"Incelenen kayit: {len(all_records)}\n")

# Etiket doluluk analizi
tag_fields = ["clip_furniture_type", "clip_style", "clip_material", "clip_color", "clip_room", "clip_category", "clip_tags"]
field_filled = {f: 0 for f in tag_fields}
field_empty = {f: 0 for f in tag_fields}

# Etiket dagilimi
furniture_counter = Counter()
style_counter = Counter()
material_counter = Counter()
color_counter = Counter()
room_counter = Counter()

for rec in all_records:
    meta = rec["meta"]
    for f in tag_fields:
        val = meta.get(f, "")
        if val and str(val).strip():
            field_filled[f] += 1
        else:
            field_empty[f] += 1
    
    # Ilk etiketleri say
    furn = meta.get("clip_furniture_type", "")
    if furn:
        first = furn.split(",")[0].strip()
        furniture_counter[first] += 1
    
    style = meta.get("clip_style", "")
    if style:
        first = style.split(",")[0].strip()
        style_counter[first] += 1

    mat = meta.get("clip_material", "")
    if mat:
        first = mat.split(",")[0].strip()
        material_counter[first] += 1

    color = meta.get("clip_color", "")
    if color:
        first = color.split(",")[0].strip()
        color_counter[first] += 1

    room = meta.get("clip_room", "")
    if room:
        first = room.split(",")[0].strip()
        room_counter[first] += 1

# Doluluk raporu
print("=" * 50)
print("ETIKET DOLULUK ORANI (100 kayit)")
print("=" * 50)
for f in tag_fields:
    pct = (field_filled[f] / len(all_records)) * 100
    print(f"  {f:25s} : {field_filled[f]:3d}/100  ({pct:.0f}%)")

# Dagilim raporu
print(f"\n{'=' * 50}")
print("EN SIK MOBILYA TIPLERI (ilk etiket)")
print("=" * 50)
for item, count in furniture_counter.most_common(15):
    print(f"  {item:25s} : {count}")

print(f"\n{'=' * 50}")
print("EN SIK STILLER")
print("=" * 50)
for item, count in style_counter.most_common(15):
    print(f"  {item:25s} : {count}")

print(f"\n{'=' * 50}")
print("EN SIK MALZEMELER")
print("=" * 50)
for item, count in material_counter.most_common(15):
    print(f"  {item:25s} : {count}")

print(f"\n{'=' * 50}")
print("EN SIK RENKLER")
print("=" * 50)
for item, count in color_counter.most_common(15):
    print(f"  {item:25s} : {count}")

print(f"\n{'=' * 50}")
print("EN SIK ODA TIPLERI")
print("=" * 50)
for item, count in room_counter.most_common(15):
    print(f"  {item:25s} : {count}")

# 10 ornek kayit goster
print(f"\n{'=' * 50}")
print("10 ORNEK KAYIT (Detayli)")
print("=" * 50)
samples = random.sample(all_records, min(10, len(all_records)))
for i, rec in enumerate(samples, 1):
    meta = rec["meta"]
    print(f"\n--- #{i}: {rec['id']} ---")
    print(f"  Furniture : {meta.get('clip_furniture_type', '-')}")
    print(f"  Style     : {meta.get('clip_style', '-')}")
    print(f"  Material  : {meta.get('clip_material', '-')}")
    print(f"  Color     : {meta.get('clip_color', '-')}")
    print(f"  Room      : {meta.get('clip_room', '-')}")
    print(f"  Category  : {meta.get('clip_category', '-')}")
    cap = (meta.get("caption", "") or "")[:60]
    print(f"  Caption   : {cap}")
