from indexer.chroma_store import ChromaStore
import json

def find_channels():
    store = ChromaStore()
    # İlk 10.000 kaydı kontrol et
    res = store.collection.get(include=['metadatas'], limit=10000)
    titles = set()
    for m in res['metadatas']:
        if 'channel_title' in m:
            titles.add(m['channel_title'])
    
    with open('found_channels.json', 'w', encoding='utf-8') as f:
        json.dump(list(titles), f, ensure_ascii=False, indent=2)
    print(f"✅ {len(titles)} farklı kanal bulundu. 'found_channels.json' dosyasına yazıldı.")

if __name__ == "__main__":
    find_channels()
