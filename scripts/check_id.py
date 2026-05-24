import sqlite3
import json
conn = sqlite3.connect('data/chroma_db/chroma.sqlite3')
c = conn.cursor()
c.execute("SELECT id FROM embeddings LIMIT 10")
ids = [row[0] for row in c.fetchall()]

for doc_id in ids:
    c.execute("SELECT key, string_value FROM embedding_metadata WHERE id = ?", (doc_id,))
    meta = dict(c.fetchall())
    print(f"ID: {doc_id} -> image_path: {meta.get('image_path', 'YOK')} | msg_id: {meta.get('message_id', 'YOK')}")
