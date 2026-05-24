from indexer.chroma_store import ChromaStore
from config import get_settings
import logging

logging.basicConfig(level=logging.ERROR)

def check():
    settings = get_settings()
    store = ChromaStore(settings.chroma_db_path, settings.get_collection_name())
    
    # Sadece bugün MegaSync ile inen son 20 kaydı çek
    res = store.collection.get(
        limit=20, 
        where={"source": "MegaSync"},
        include=['metadatas', 'documents']
    )
    
    print("\n--- MEGA SYNC (YENI MANTIK) QC RAPORU ---")
    for i in range(len(res['ids'])):
        meta = res['metadatas'][i]
        msg_id = res['ids'][i]
        # Mesajı temizle (Sadece ASCII)
        cap = meta.get('caption', '').encode('ascii', 'ignore').decode('ascii').strip()
        tags = meta.get('deep_tags', '').strip()
        
        # QC Mantığı
        if len(cap) >= 15 and not tags:
            qc_status = "DOGRU: Mesaj var, AI pas gecildi"
        elif len(cap) < 15 and tags:
            qc_status = "DOGRU: Mesaj yok, AI etiketledi"
        else:
            qc_status = "INCELENMELI: Hibrit/Diger"
            
        print(f"ID: {msg_id}")
        print(f"DURUM: {qc_status}")
        print(f"MESAJ: {cap[:50]}...")
        print(f"AI_TAGS: {tags if tags else '[YOK]'}")
        print("-" * 50)

if __name__ == "__main__":
    check()
