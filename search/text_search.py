import logging
from typing import Dict, List, Optional
import requests
import numpy as np

from indexer.clip_encoder import encode_text
from indexer.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


def expand_query_with_llm(query: str) -> str:
    """
    Kullanıcının kısa arama terimini önce Gemini API (çok hızlı),
    başarısız olursa LM Studio'daki yerel LLM'i kullanarak
    çok daha geniş ve zengin bir CLIP arama metnine (Query Expansion) dönüştürür.
    """
    system_prompt = (
        "You expand search queries for a 3D model archive. "
        "CRITICAL RULE: You MUST include keywords in ALL THREE languages: English, Turkish, AND Russian (Cyrillic script). "
        "Every response MUST have at least 3 Russian words in Cyrillic and 3 Turkish words. This is mandatory.\n\n"
        "Format: comma-separated keywords only, nothing else.\n\n"
        "Example input: 'modern chair'\n"
        "Example output: modern chair, modern sandalye, современный стул, armchair, koltuk, кресло, seating, oturma, сиденье, wood, minimalist\n\n"
        "Example input: 'wall art'\n"
        "Example output: wall art, duvar tablosu, настенное искусство, painting, tablo, картина, poster, dekoratif pano, постер, canvas, tuval, холст\n\n"
        "Example input: 'masa'\n"
        "Example output: masa, table, стол, desk, çalışma masası, письменный стол, dining table, yemek masası, обеденный стол, wood, ahşap, дерево"
    )
    
    # 1. Gemini (Hibrid İlk Seçenek)
    import os
    from dotenv import load_dotenv
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={gemini_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": f"{system_prompt}\n\nQuery: '{query}'"}]}],
                "generationConfig": {"temperature": 0.4, "maxOutputTokens": 800}
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                expanded = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                expanded = expanded.replace('"', '').replace('\n', ' ')
                if expanded:
                    logger.info(f"✨ Gemini Genişletilmiş Sorgu: {expanded}")
                    return expanded
        except Exception as e:
            logger.warning(f"⚠️ Gemini Query Expansion başarısız ({e}). Yerel LLM'e geçiliyor...")

    # 2. Local LM Studio (Fallback)
    try:
        url = "http://localhost:1234/v1/chat/completions"
        from config import get_settings
        text_model = get_settings().llm_text_model
        
        payload = {
            "model": text_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: '{query}'"}
            ],
            "temperature": 0.4,
            "max_tokens": 800
        }
        
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            expanded = resp.json()["choices"][0]["message"]["content"].strip()
            # Remove <think> tags if any
            import re
            expanded = re.sub(r"<think>.*?</think>", "", expanded, flags=re.DOTALL).strip()
            
            if not expanded:
                return query
                
            expanded = expanded.replace('"', '').replace('\n', ' ')
            logger.info(f"✨ Local LLM Genişletilmiş Sorgu: {expanded}")
            return expanded
    except Exception as e:
        logger.debug(f"LLM ile sorgu genişletme başarısız oldu (Hata: {e}). Orijinal sorgu kullanılacak.")
        
    return query


def search_by_text(
    query: str,
    store: ChromaStore,
    n_results: int = 20,
    min_score: float = 0.15,
    where: Optional[Dict] = None,
    model_name: Optional[str] = None,
    pretrained: Optional[str] = None,
    use_ai_expansion: bool = True,
) -> tuple[List[Dict], str]:
    """
    Doğal dil sorgusu ile 3D model görselleri arar.
    """
    from config import get_settings
    settings = get_settings()
    
    model_name = model_name or settings.clip_model_name
    pretrained = pretrained or settings.clip_pretrained
    if not query.strip():
        return [], query

    logger.info(f"🔍 Orijinal Metin araması: '{query}'")

    # Yapay zeka ile kelimeleri genişlet ("armchair" -> "armchair, sofa, lounge chair...")
    search_text = query
    if use_ai_expansion:
        search_text = expand_query_with_llm(query)

    # Metni CLIP vektörüne çevir
    text_vector = encode_text(search_text, model_name, pretrained)
    if text_vector is None:
        logger.error("Metin vektörleştirilemedi.")
        return [], search_text

    # ChromaDB sorgusu (CLIP Vektörü ile)
    # Hibrit arama için n_results'ı biraz daha büyük alıp sonra eliyoruz/sıralıyoruz
    search_limit = n_results * 3
    results = store.query_by_vector(
        query_vector=text_vector,
        n_results=search_limit,
        where=where,
        min_score=min_score,
    )

    # ── HİBRİT RE-RANKING (SigLIP 2 Etiketleri + Aesthetic Predictor V2.5) ──
    from config import MetadataSchema
    query_keywords = query.lower().split()
    hybrid_results = []
    
    for res in results:
        meta = res.get("metadata", {})
        # 1. Orijinal CLIP Benzerliği (0-1 arası)
        clip_score = res.get("score", 0)
        
        # 2. Estetik Kalite Puanı — Aesthetic Predictor V2.5 (1-10 ölçeği)
        # 0-1 aralığına normalize et: (skor - 1) / 9
        raw_aesthetic = meta.get("aesthetic_score") or meta.get(MetadataSchema.AESTHETIC_SCORE)
        has_aesthetic = raw_aesthetic not in ("", None, 0, "0")
        try:
            aes_raw = float(raw_aesthetic) if has_aesthetic else 0
        except (ValueError, TypeError):
            aes_raw = 0
            has_aesthetic = False
        
        # V2.5 kalite katsayısı: 5.5+ = kaliteli, 7+ = elite, <3.5 = ceza
        if has_aesthetic and aes_raw > 0:
            aes_normalized = max(0.0, min(1.0, (aes_raw - 1.0) / 9.0))  # 1-10 → 0-1
            # Kalite çarpanı: elite render'lar ekstra boost alır
            if aes_raw >= 7.0:
                quality_mult = 1.15  # Elite: %15 boost
            elif aes_raw >= 5.5:
                quality_mult = 1.05  # İyi: %5 boost
            elif aes_raw < 3.5:
                quality_mult = 0.90  # Düşük: %10 ceza
            else:
                quality_mult = 1.0   # Ortalama: nötr
        else:
            aes_normalized = 0.5  # Bilinmeyen → nötr
            quality_mult = 1.0
            
        # 3. SigLIP 2 Etiket Eşleşme — Tüm clip_ alanlarını tara
        combined_tags = " ".join(
            str(meta.get(k, "")).lower()
            for k in [
                "clip_tags", "clip_style", "clip_material",
                "clip_furniture_type", "clip_color", "clip_room",
                "clip_category", MetadataSchema.AI_DESCRIPTION,
                MetadataSchema.DEEP_TAGS,
            ]
        )
        
        keyword_hits = sum(1 for kw in query_keywords if len(kw) > 2 and kw in combined_tags)
        keyword_boost = min(keyword_hits * 0.04, 0.20)  # Maks %20, kelime başı %4
        
        # ── ELITE SIRALAMA FORMÜLÜ V2 ──
        # Estetik skor varsa: Benzerlik %55 + Kalite %25 + Etiket %20 → kalite çarpanı
        # Estetik skor yoksa: Benzerlik %75 + Etiket %25 (adaletli mod)
        if has_aesthetic:
            base_score = (clip_score * 0.55) + (aes_normalized * 0.25) + (keyword_boost * 0.20)
            final_score = base_score * quality_mult
        else:
            final_score = (clip_score * 0.75) + (keyword_boost * 0.25)
        
        res["score"] = round(final_score, 4)
        hybrid_results.append(res)
    
    # Elite skorlara göre sırala
    hybrid_results.sort(key=lambda x: x["score"], reverse=True)
    final_results = hybrid_results[:n_results]

    logger.info(f"  📊 {len(final_results)} 'Elite' sonuç (V2.5 Kalite Filtresi)")
    return final_results, search_text
