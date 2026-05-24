import sqlite3

conn = sqlite3.connect("data/chroma_db/chroma.sqlite3")
cursor = conn.cursor()

# Koleksiyonlari listele
cursor.execute("SELECT id, name FROM collections")
collections = cursor.fetchall()
print("Koleksiyonlar:")
for c in collections:
    print(f"  {c[0]} -> {c[1]}")

# Her koleksiyon icin embedding sayisi
for coll_id, coll_name in collections:
    cursor.execute("SELECT COUNT(*) FROM embeddings WHERE segment_id IN (SELECT id FROM segments WHERE collection = ?)", (coll_id,))
    count = cursor.fetchone()[0]
    print(f"\n{coll_name}: {count} embedding kayit")

conn.close()
