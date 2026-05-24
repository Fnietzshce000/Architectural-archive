"""
🔍 Archi Scene Parser — OWLv2 ile Pinterest/Referans Fotoğraflarını Parçalama

Bir oda fotoğrafı yükle → OWLv2 objeleri tespit eder → her objeyi kırpar →
SigLIP 2 ile encode eder → ChromaDB'den benzer 3D modelleri bulur.

Kullanım:
    from search.scene_parser import SceneParser
    parser = SceneParser()
    results = parser.parse_and_search("room_photo.jpg")
    # results = [
    #   {"label": "sofa", "confidence": 0.82, "bbox": [x1,y1,x2,y2],
    #    "crop_path": "/tmp/crop_0.jpg", "similar_models": [...]},
    #   ...
    # ]
"""
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings

logger = logging.getLogger("SceneParser")

# ── Mobilya/Obje Sorguları (OWLv2'ye gönderilecek) ──
FURNITURE_QUERIES = [
    "sofa",
    "armchair",
    "chair",
    "dining table",
    "coffee table",
    "side table",
    "desk",
    "console table",
    "bed",
    "wardrobe",
    "dresser",
    "nightstand",
    "bookshelf",
    "cabinet",
    "sideboard",
    "TV unit",
    "chandelier",
    "pendant light",
    "floor lamp",
    "table lamp",
    "wall sconce",
    "mirror",
    "rug",
    "carpet",
    "curtain",
    "plant",
    "vase",
    "painting",
    "wall art",
    "sculpture",
    "clock",
    "pillow",
    "bathtub",
    "sink",
    "fireplace",
]


class SceneParser:
    """Pinterest/referans fotoğraflarını parçalayıp obje bazlı 3D model arama."""

    def __init__(
        self,
        model_name: str = "google/owlv2-base-patch16-ensemble",
        confidence_threshold: float = 0.15,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._detector = None
        self._detector_processor = None
        self._clip_encoder = None

    def _load_detector(self):
        """OWLv2 modelini lazy load et (ilk kullanımda)."""
        if self._detector is not None:
            return

        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        logger.info(f"🦉 OWLv2 yükleniyor: {self.model_name}")
        self._detector_processor = Owlv2Processor.from_pretrained(self.model_name)
        self._detector = Owlv2ForObjectDetection.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        ).to(self.device).eval()
        logger.info(f"✅ OWLv2 yüklendi → {self.device} (FP16)")

    def detect_objects(
        self,
        image_path: str,
        queries: Optional[list[str]] = None,
        max_objects: int = 15,
    ) -> list[dict]:
        """
        Fotoğraftaki objeleri tespit et.

        Returns:
            [{"label": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]
        """
        self._load_detector()
        queries = queries or FURNITURE_QUERIES

        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        inputs = self._detector_processor(
            text=[queries],
            images=img,
            return_tensors="pt",
        )
        # FP16'ya çevir ve GPU'ya gönder
        inputs = {
            k: v.to(self.device).half() if v.dtype == torch.float32 else v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            outputs = self._detector(**inputs)

        # Post-process
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self._detector_processor.image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]

        detections = []
        for score, label_idx, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy(),
        ):
            x1, y1, x2, y2 = box.astype(int)
            # Çok küçük kutuları atla (gürültü)
            box_area = (x2 - x1) * (y2 - y1)
            img_area = w * h
            if box_area < img_area * 0.01:  # %1'den küçük kutular gürültü
                continue

            detections.append({
                "label": queries[label_idx],
                "confidence": float(score),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            })

        # Güvene göre sırala, max_objects kadar tut
        detections.sort(key=lambda d: -d["confidence"])
        detections = detections[:max_objects]

        # NMS benzeri: çok yakın kutuları birleştir
        detections = self._simple_nms(detections, iou_threshold=0.5)

        logger.info(f"🎯 {len(detections)} obje tespit edildi")
        return detections

    def _simple_nms(self, detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
        """Basit NMS — çok yakın ve aynı türdeki kutuları birleştir."""
        if len(detections) <= 1:
            return detections

        keep = []
        used = set()

        for i, det_i in enumerate(detections):
            if i in used:
                continue
            keep.append(det_i)
            for j, det_j in enumerate(detections):
                if j <= i or j in used:
                    continue
                if self._iou(det_i["bbox"], det_j["bbox"]) > iou_threshold:
                    used.add(j)

        return keep

    @staticmethod
    def _iou(box1: list, box2: list) -> float:
        """İki kutunun IoU (Intersection over Union) değerini hesapla."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def crop_objects(
        self,
        image_path: str,
        detections: list[dict],
        output_dir: Optional[str] = None,
        padding: float = 0.05,
    ) -> list[dict]:
        """Tespit edilen objeleri kırparak diske kaydeder."""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        if output_dir is None:
            output_dir = str(Path(PROJECT_ROOT) / "data" / "temp" / "scene_crops")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]

            # Padding ekle (kenardan biraz daha al)
            pad_w = int((x2 - x1) * padding)
            pad_h = int((y2 - y1) * padding)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)

            crop = img.crop((x1, y1, x2, y2))
            crop_path = str(Path(output_dir) / f"crop_{i}_{det['label'].replace(' ', '_')}.jpg")
            crop.save(crop_path, quality=90)
            det["crop_path"] = crop_path

        return detections

    def search_similar_models(
        self,
        detections: list[dict],
        n_results: int = 8,
    ) -> list[dict]:
        """Her kırpılmış obje için ChromaDB'den benzer 3D modelleri bul."""
        from indexer.clip_encoder import encode_image
        from indexer.chroma_store import ChromaStore

        settings = get_settings()
        store = ChromaStore(
            db_path=settings.chroma_db_path,
            collection_name=settings.get_collection_name(),
        )

        for det in detections:
            crop_path = det.get("crop_path")
            if not crop_path or not Path(crop_path).exists():
                det["similar_models"] = []
                continue

            # SigLIP 2 ile encode et
            embedding = encode_image(
                crop_path,
                model_name=settings.clip_model_name,
                pretrained=settings.clip_pretrained,
            )
            if embedding is None:
                det["similar_models"] = []
                continue

            # ChromaDB'den benzer modelleri bul
            results = store.query_by_vector(
                query_vector=embedding,
                n_results=n_results,
                min_score=0.15,
            )
            det["similar_models"] = results

        return detections

    def parse_and_search(
        self,
        image_path: str,
        queries: Optional[list[str]] = None,
        n_results_per_object: int = 8,
        max_objects: int = 12,
    ) -> list[dict]:
        """
        Tam pipeline: fotoğraf → obje tespiti → kırpma → arama.

        Args:
            image_path: Oda/sahne fotoğrafı yolu
            queries: Aranacak obje isimleri (None = varsayılan mobilya listesi)
            n_results_per_object: Her obje için kaç benzer model
            max_objects: Maksimum tespit edilecek obje sayısı

        Returns:
            [{"label", "confidence", "bbox", "crop_path", "similar_models"}, ...]
        """
        start = time.time()

        logger.info(f"🔍 Sahne analizi başlıyor: {Path(image_path).name}")

        # 1. Obje tespiti
        detections = self.detect_objects(image_path, queries, max_objects)
        if not detections:
            logger.info("❌ Hiçbir obje tespit edilemedi")
            return []

        # 2. Kırpma
        detections = self.crop_objects(image_path, detections)

        # 3. Benzer model arama
        detections = self.search_similar_models(detections, n_results_per_object)

        elapsed = time.time() - start
        total_models = sum(len(d.get("similar_models", [])) for d in detections)
        logger.info(
            f"✅ Sahne analizi tamamlandı: "
            f"{len(detections)} obje, {total_models} benzer model, "
            f"{elapsed:.1f}sn"
        )

        return detections

    def unload(self):
        """GPU belleğini serbest bırak."""
        if self._detector is not None:
            del self._detector
            del self._detector_processor
            self._detector = None
            self._detector_processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("🧹 OWLv2 bellekten kaldırıldı")
