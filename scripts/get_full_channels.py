from indexer.chroma_store import ChromaStore
import json

def get_all_unique_channels():
    store = ChromaStore()
    total_count = store.get_count()
    chunk_size = 10000
    all_titles = set()
    
    print(f"--- Toplam {total_count} kayit taraniyor ---")
    
    for offset in range(0, total_count, chunk_size):
        # Sadece metadata'ları çek (embeddings gerekmez, bellek koruma)
        res = store.collection.get(
            include=['metadatas'], 
            limit=chunk_size, 
            offset=offset
        )
        if res and res['metadatas']:
            for m in res['metadatas']:
                if m and 'channel_title' in m:
                    all_titles.add(m['channel_title'])
        
        print(f"-> Ilerleme: %{min(100, int((offset + chunk_size) / total_count * 100))}")

    # Dosyaya yaz
    with open('full_channels.json', 'w', encoding='utf-8') as f:
        json.dump(sorted(list(all_titles)), f, ensure_ascii=False, indent=2)
    
    print(f"DONE! Toplam {len(all_titles)} farkli kanal bulundu.")

if __name__ == "__main__":
    get_all_unique_channels()
