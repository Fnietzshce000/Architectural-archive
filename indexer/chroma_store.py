"""
ChromaDB Vektör Deposu — CLIP vektörlerini saklar ve kosinüs benzerliği ile arar.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    ChromaDB üzerinde vektör CRUD işlemleri.
    HNSW indeksi ile milisaniyelik arama sağlar.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        collection_name: Optional[str] = None,
    ):
        from config import get_settings
        settings = get_settings()
        
        self.db_path = str(db_path or settings.chroma_db_path)
        # 🛡️ GÜVENLİK: Eğer isim verilmediyse, model adına göre otomatik güvenli isim al
        self.collection_name = collection_name or settings.get_collection_name()

        # Kalıcı istemci
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)

        # Koleksiyon oluştur veya al
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",           # Kosinüs benzerliği
                "hnsw:construction_ef": 200,       # Graf kalitesi
                "hnsw:search_ef": 100,             # Arama hassasiyeti
                "hnsw:M": 16,                      # Komşu sayısı
            },
        )
        logger.info(
            f"📦 ChromaDB: koleksiyon '{self.collection_name}' — "
            f"{self.collection.count()} kayıt mevcut"
        )

    def add(
        self,
        doc_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
        document: str = "",
    ) -> bool:
        """
        Tek bir kayıt ekler.

        Args:
            doc_id: Benzersiz ID (ör: "123_456")
            embedding: CLIP vektörü (512 veya 768 boyutlu)
            metadata: Ek bilgiler (channel_id, deep_link, vb.)
            document: Aranabilir metin (caption + etiketler)
        """
        # ChromaDB metadata None/list kabul etmez, temizle
        clean_meta = {}
        for k, v in metadata.items():
            if v is None:
                clean_meta[k] = ""
            elif isinstance(v, list):
                clean_meta[k] = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)

        try:
            self.collection.upsert(
                ids=[doc_id],
                embeddings=[embedding.tolist()],
                metadatas=[clean_meta],
                documents=[document],
            )
            return True
        except Exception as e:
            logger.error(f"ChromaDB ekleme hatası ({doc_id}): {e}")
            return False

    def add_batch(
        self,
        doc_ids: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
        batch_size: int = 500,
    ):
        """Toplu ekleme — 500'lük batch'ler halinde."""
        total = len(doc_ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            batch_meta = []
            for meta in metadatas[start:end]:
                clean = {}
                for k, v in meta.items():
                    if v is None:
                        clean[k] = ""
                    elif isinstance(v, list):
                        clean[k] = json.dumps(v, ensure_ascii=False)
                    elif isinstance(v, (str, int, float, bool)):
                        clean[k] = v
                    else:
                        clean[k] = str(v)
                batch_meta.append(clean)

            try:
                self.collection.upsert(
                    ids=doc_ids[start:end],
                    embeddings=[e.tolist() for e in embeddings[start:end]],
                    metadatas=batch_meta,
                    documents=documents[start:end],
                )
                logger.info(f"  📝 Batch {start}-{end}/{total} ChromaDB'ye yazıldı")
            except Exception as e:
                logger.error(f"❌ ChromaDB batch hatası ({start}-{end}): {e}")
                # 🛑 GÜVENLİK: Hatayı yukarı fırlat ki sistem yanlışlıkla başarılı sanmasın
                raise RuntimeError(f"ChromaDB batch yazma başarısız: {e}") from e

    def query_by_vector(
        self,
        query_vector: np.ndarray,
        n_results: int = 20,
        where: Optional[Dict] = None,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """
        Vektör ile kosinüs benzerliği araması yapar.

        Returns:
            Sonuç listesi: [{"id", "score", "metadata", "document"}, ...]
        """
        query_params = {
            "query_embeddings": [query_vector.tolist()],
            "n_results": n_results,
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            query_params["where"] = where

        try:
            results = self.collection.query(**query_params)
        except Exception as e:
            logger.error(f"ChromaDB sorgu hatası: {e}")
            return []

        # Sonuçları düzenle
        formatted = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                # ChromaDB cosine distance döner (0 = aynı, 2 = zıt)
                # Score'a çevir (1 = aynı, 0 = alakasız)
                distance = results["distances"][0][i]
                score = 1.0 - (distance / 2.0)

                if score < min_score:
                    continue

                formatted.append({
                    "id": results["ids"][0][i],
                    "score": round(score, 4),
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "document": results["documents"][0][i] if results["documents"] else "",
                })

        return formatted

    def get_count(self) -> int:
        """Toplam kayıt sayısını döndürür."""
        return self.collection.count()

    def get_all_ids(self) -> List[str]:
        """Tüm kayıt ID'lerini döndürür."""
        result = self.collection.get(include=[])
        return result["ids"] if result else []

    def delete(self, doc_ids: List[str]):
        """Belirtilen kayıtları siler."""
        if doc_ids:
            self.collection.delete(ids=doc_ids)
            logger.info(f"🗑️ {len(doc_ids)} kayıt silindi.")

    def reset(self):
        """Koleksiyonu tamamen siler ve yeniden oluşturur."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 100,
                "hnsw:M": 16,
            },
        )
        logger.info("🔄 ChromaDB koleksiyonu sıfırlandı.")
