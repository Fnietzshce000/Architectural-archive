import sqlite3
conn = sqlite3.connect('data/chroma_db/chroma.sqlite3')
c = conn.cursor()
c.execute("SELECT COUNT(DISTINCT id) FROM embedding_metadata WHERE string_value = 'MegaSync'")
count = c.fetchone()[0]
print(f"Sisteme Yeni Giren Taze Model Sayisi: {count}")
