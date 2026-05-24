import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indexer.chroma_store import ChromaStore

def check():
    store = ChromaStore()
    data = store.collection.get(where={"ai_enriched": True}, include=[])
    count = len(data["ids"])
    print(f"\nIslenen Toplam Model: {count}")

if __name__ == "__main__":
    check()
