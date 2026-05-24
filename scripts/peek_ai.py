import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from indexer.chroma_store import ChromaStore
import json

def peek():
    store = ChromaStore()
    results = store.collection.get(
        where={"ai_enriched": True},
        limit=5,
        include=["metadatas"]
    )
    
    print("\n" + "="*50)
    print("AI ANALIZ RONTGENI (SON 5 KAYIT)")
    print("="*50)
    
    if not results or not results["ids"]:
        print("Henuz islenmis kayit bulunamadi veya isleme devam ediyor...")
        return
        
    for i in range(len(results["ids"])):
        meta = results["metadatas"][i]
        print(f"ID: {results['ids'][i]}")
        print(f"Gorsel: {meta.get('image_path')}")
        print(f"AI Tanimi: {meta.get('ai_description')}")
        print("-" * 50)

if __name__ == "__main__":
    peek()
