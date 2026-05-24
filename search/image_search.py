import logging
import base64
import requests
from typing import Dict, List, Optional

import numpy as np

from indexer.clip_encoder import encode_image, encode_text
from indexer.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


def analyze_image_with_llm(image_path: str) -> str:
    """
    Yüklenen görseli LM Studio'daki (eğer destekliyorsa) Vision modeline gönderip,
    görselin tarzı, materyali ve renkleri hakkında anahtar kelimeler çıkarttırır.
    """
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            
        url = "http://localhost:1234/v1/chat/completions"
        system_prompt = (
            "Sen uzman bir iç mimarsın. Bu görsel bir 3D çalışma alanı (workspace) veya tasarım aşamasındaki bir mekandır. "
            "Görseli şu açılardan analiz et:\n"
            "1. TASARIM DİLİ: Mekanın tarzı nedir? (Modern, Loft, Klasik vb.)\n"
            "2. MATERYAL PALETİ: Hangi dokular hakim? (Beton, ahşap, metal, mermer vb.)\n"
            "3. EKSİK PARÇALAR: Bu sahneyi tamamlamak için hangi mobilya veya aksesuarlar eksik?\n"
            "Analizini kısa ve öz bir şekilde yap, sadece tasarımcıya yardımcı olacak kritik bilgileri ver."
        )
        from config import get_settings
        vision_model = get_settings().llm_vision_model
        
        payload = {
            "model": vision_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": "Extract detailed keywords from this image."}
                    ]
                }
            ],
            "temperature": 0.3,
            "max_tokens": 300
        }
        
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            analysis = resp.json()["choices"][0]["message"]["content"].strip()
            # Bazen tırnak içinde dönebiliyor, onları temizleyelim
            analysis = analysis.replace('"', '').replace('\n', ' ')
            if analysis:
                logger.info(f"✨ AI Görsel Analizi: {analysis}")
                return analysis
    except Exception as e:
        logger.debug(f"Görsel analizi başarısız (Model Vision desteklemiyor olabilir): {e}")
    return ""


def search_by_image(
    image_path: str,
    store: ChromaStore,
    n_results: int = 20,
    min_score: float = 0.15,
    where: Optional[Dict] = None,
    model_name: Optional[str] = None,
    pretrained: Optional[str] = None,
    use_ai_expansion: bool = True,
) -> tuple[List[Dict], str]:
    """
    Referans görsel ile benzer 3D model görselleri arar.
    """
    from config import get_settings
    settings = get_settings()
    
    model_name = model_name or settings.clip_model_name
    pretrained = pretrained or settings.clip_pretrained
    
    logger.info(f"🖼️ Görsel araması: {image_path}")

    # 1. Görseli CLIP vektörüne çevir
    image_vector = encode_image(image_path, model_name, pretrained)
    if image_vector is None:
        logger.error("Görsel vektörleştirilemedi.")
        return [], ""

    analysis_text = ""
    # 2. Opsiyonel: AI ile görseli analiz et ve vektörleri harmanla
    if use_ai_expansion:
        analysis_text = analyze_image_with_llm(image_path)
        if analysis_text:
            text_vector = encode_text(analysis_text, model_name, pretrained)
            if text_vector is not None:
                # Hibrit Arama: Görsel/Metin oranı config'den okunur
                blend = settings.image_text_blend_ratio
                image_vector = (blend * image_vector) + ((1.0 - blend) * text_vector)
                image_vector /= np.linalg.norm(image_vector)

    # 3. ChromaDB sorgusu
    results = store.query_by_vector(
        query_vector=image_vector,
        n_results=n_results,
        where=where,
        min_score=min_score,
    )

    logger.info(f"  📊 {len(results)} sonuç bulundu")
    return results, analysis_text
