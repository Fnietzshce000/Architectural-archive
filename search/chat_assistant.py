import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import get_settings

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"

OBJECT_TERMS = {
    "sofa": ["sofa", "kanepe", "couch", "sectional", "koltuk"],
    "armchair": ["armchair", "berjer", "accent chair", "lounge chair"],
    "chair": ["chair", "sandalye", "seat", "stool", "bar stool", "dining chair", "office chair"],
    "table": ["table", "masa", "sehpa", "desk", "coffee table", "dining table", "side table", "console"],
    "bed": ["bed", "yatak", "headboard", "nightstand", "karyola"],
    "lamp": ["lamp", "lighting", "isik", "lamba", "chandelier", "pendant", "sconce", "spotlight", "avize", "aplik"],
    "cabinet": ["cabinet", "dolap", "wardrobe", "shelf", "bookshelf", "dresser", "sideboard", "tv unit", "kitaplık"],
    "decor": ["decor", "dekor", "vase", "mirror", "art", "aksesuar", "sculpture", "painting", "clock", "pillow", "heykel"],
    "rug": ["rug", "carpet", "hali", "kilim"],
    "plant": ["plant", "bitki", "tree", "greenery", "saksı"],
    "sanitary": ["bathtub", "küvet", "sink", "lavabo", "toilet", "klozet", "shower", "duş", "faucet", "musluk"],
    "kitchen": ["kitchen cabinet", "mutfak dolabı", "kitchen island", "mutfak adası"],
    "architectural": ["door", "kapı", "window", "pencere", "staircase", "merdiven", "fireplace", "şömine", "column", "wall panel"],
    "curtain": ["curtain", "perde", "drape"],
}

STYLE_TERMS = {
    "modern": ["modern", "contemporary", "çağdaş", "cagdas"],
    "classic": ["classic", "klasik", "traditional", "geleneksel"],
    "luxury": ["luxury", "luks", "lüks", "premium"],
    "minimalist": ["minimal", "minimalist", "sade"],
    "scandinavian": ["scandinavian", "scandi", "nordic", "iskandinav"],
    "industrial": ["industrial", "loft", "endüstriyel"],
    "japandi": ["japandi", "japanese scandinavian"],
    "mid_century": ["mid-century", "mid century modern", "retro modern"],
    "baroque": ["baroque", "barok"],
    "neoclassic": ["neoclassic", "neoklasik", "neo classic"],
    "art_deco": ["art deco", "art deko"],
    "provence": ["provence", "provans", "french country"],
    "bohemian": ["bohemian", "boho", "bohem"],
    "rustic": ["rustic", "rustik", "country", "köy"],
    "retro": ["retro", "vintage", "antika"],
    "futuristic": ["futuristic", "futuristik", "sci-fi"],
}

MATERIAL_TERMS = {
    "wood": ["wood", "wooden", "ahsap", "ahşap"],
    "walnut": ["walnut", "ceviz"],
    "oak": ["oak", "meşe"],
    "metal": ["metal", "steel", "iron", "chrome", "çelik", "krom"],
    "brass": ["brass", "pirinç", "gold", "altın", "bronze", "bronz"],
    "marble": ["marble", "mermer", "travertine", "granite", "granit"],
    "leather": ["leather", "deri"],
    "fabric": ["fabric", "kumas", "kumaş", "linen", "keten"],
    "velvet": ["velvet", "kadife"],
    "boucle": ["boucle", "bukle"],
    "glass": ["glass", "cam"],
    "concrete": ["concrete", "beton"],
    "stone": ["stone", "taş", "brick", "tuğla"],
    "terrazzo": ["terrazzo"],
    "ceramic": ["ceramic", "seramik", "porcelain", "porselen"],
    "rattan": ["rattan", "bamboo", "bambu", "hasır"],
    "acrylic": ["acrylic", "plastic", "akrilik", "plastik"],
}

COLOR_TERMS = {
    "black": ["black", "siyah"],
    "white": ["white", "beyaz", "ivory", "cream", "krem"],
    "brown": ["brown", "kahve", "kahverengi"],
    "beige": ["beige", "bej"],
    "gray": ["gray", "grey", "gri", "anthracite", "antrasit"],
    "gold": ["gold", "altin", "altın", "brass", "pirinç"],
    "blue": ["blue", "mavi", "navy", "lacivert", "cyan"],
    "green": ["green", "yeşil", "yesil", "olive", "zeytin", "emerald", "zümrüt"],
    "red": ["red", "kırmızı", "kirmizi", "burgundy", "bordo"],
    "terracotta": ["terracotta", "terra cotta", "kiremit"],
    "yellow": ["yellow", "sarı", "sari", "mustard", "hardal"],
    "pink": ["pink", "pembe", "purple", "mor"],
    "orange": ["orange", "turuncu"],
    "silver": ["silver", "gümüş", "gumus"],
}

ROOM_TERMS = {
    "living_room": ["living room", "salon", "oturma"],
    "bedroom": ["bedroom", "yatak odasi", "yatak odası"],
    "kitchen": ["kitchen", "mutfak"],
    "dining_room": ["dining room", "yemek odası", "yemek odasi"],
    "bathroom": ["bathroom", "banyo"],
    "office": ["office", "ofis", "workspace", "calisma", "çalışma"],
    "hallway": ["hallway", "corridor", "koridor", "antre", "entryway", "giriş"],
    "kids_room": ["kids room", "çocuk odası", "cocuk odasi", "nursery"],
    "dressing_room": ["dressing room", "giyinme odası", "giyinme"],
    "outdoor": ["outdoor", "garden", "bahce", "bahçe", "terrace", "teras", "balcony", "balkon"],
    "commercial": ["cafe", "restaurant", "restoran", "hotel", "otel", "lobby", "lobi", "commercial"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_prompt(name: str, fallback: str) -> str:
    path = PROMPTS_DIR / name
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("Prompt okunamadi (%s): %s", path, exc)
    return fallback


def strip_think(content: str) -> str:
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()
    if "<think>" in content and "</think>" not in content:
        content = content[: content.find("<think>")].strip()
    return content


def _get_gemini_client():
    """Gemini genai client'ı lazy init et."""
    import os
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not found in .env")
    from google import genai
    return genai.Client(api_key=gemini_key)


def call_gemini_with_tools(system_prompt: str, user_message: str, chat_history: list, visual_context: str = "") -> Dict[str, Any]:
    """
    Gemini Function Calling ile akıllı intent detection.
    Model kendi karar verir: sohbet mi, arama mı, tavsiye mi.
    """
    from google import genai
    from google.genai import types

    client = _get_gemini_client()

    # ── Tool Tanımları ──
    search_tool = types.FunctionDeclaration(
        name="search_3d_models",
        description="Kullanıcı bir 3D model, mobilya, dekorasyon veya tasarım öğesi aramak istediğinde bu tool'u çağır. Örnek: 'modern koltuk bul', 'ahşap masa lazım', 'salon için avize'",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "search_queries": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), description="3-5 İngilizce arama sorgusu. CLIP/SigLIP modeli için kısa ve açıklayıcı."),
                "reply_tr": types.Schema(type=types.Type.STRING, description="Kullanıcıya Türkçe, samimi cevap. Arama yapacağını belirt."),
                "object_type": types.Schema(type=types.Type.STRING, description="Tespit edilen obje tipi: sofa/chair/table/bed/lamp/cabinet/decor/rug/plant/unknown"),
                "style": types.Schema(type=types.Type.STRING, description="Stil: modern/classic/minimalist/scandinavian/industrial/luxury/unknown"),
                "material": types.Schema(type=types.Type.STRING, description="Malzeme: wood/metal/marble/leather/velvet/glass/unknown"),
                "color_family": types.Schema(type=types.Type.STRING, description="Renk: white/black/brown/beige/gray/blue/green/gold/unknown"),
                "room": types.Schema(type=types.Type.STRING, description="Oda: living_room/bedroom/bathroom/kitchen/office/unknown"),
                "complementary_queries": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), description="0-3 tamamlayıcı ürün araması"),
            },
            required=["search_queries", "reply_tr"],
        ),
    )

    chat_tool = types.FunctionDeclaration(
        name="chat_reply",
        description="Kullanıcı selamlama, sohbet, teşekkür, veda, hal hatır sorma, veya genel konuşma yapıyorsa bu tool'u çağır. Arama gerektirmeyen her mesaj için bunu kullan. Örnek: 'merhaba', 'nasılsın', 'teşekkürler', 'ne yapabilirsin?'",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "reply_tr": types.Schema(type=types.Type.STRING, description="Samimi, sıcak ve doğal Türkçe cevap. Archi karakterinde ol — profesyonel ama arkadaşça bir iç mimar asistanı."),
            },
            required=["reply_tr"],
        ),
    )

    advice_tool = types.FunctionDeclaration(
        name="give_design_advice",
        description="Kullanıcı tasarım tavsiyesi, stil önerisi, renk kombinasyonu, mekan düzenleme veya dekorasyon ipucu istediğinde bu tool'u çağır. Arama yapmadan bilgi ver. Örnek: 'salon nasıl dekore edilir', 'hangi renkler uyar', 'scandinavian stil nedir'",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "reply_tr": types.Schema(type=types.Type.STRING, description="Detaylı, profesyonel Türkçe tasarım tavsiyesi. Örnekler ve pratik ipuçları içersin."),
                "suggested_search": types.Schema(type=types.Type.STRING, description="Opsiyonel: Eğer tavsiyeyi destekleyecek bir model araması öneriyorsan, tek İngilizce sorgu yaz."),
            },
            required=["reply_tr"],
        ),
    )

    tool = types.Tool(function_declarations=[search_tool, chat_tool, advice_tool])

    # ── Geçmiş sohbeti formatla ──
    history_text = compact_history(chat_history)

    full_system = (
        "Sen Archi, büyük bir 3D model arşivi üzerinde çalışan profesyonel iç mimari asistanısın.\n"
        "460.000'den fazla 3D model arşivin var.\n"
        "Türkçe, samimi ve doğal konuş. Kısa ve öz cevaplar ver.\n"
        "Kullanıcı sohbet ediyorsa sohbet et, arama istiyorsa arama yap.\n"
        "HER ZAMAN uygun tool'u çağır — asla düz metin döndürme.\n"
    )

    user_content = f"Geçmiş sohbet:\n{history_text}\n\n"
    if visual_context:
        user_content += f"Görsel analizi: {visual_context}\n\n"
    user_content += f"Kullanıcı mesajı: {user_message}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Content(role="user", parts=[types.Part(text=f"{full_system}\n\n{user_content}")]),
            ],
            config=types.GenerateContentConfig(
                tools=[tool],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                temperature=0.4,
            ),
        )

        # Function call'ı parse et
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    logger.info(f"🤖 Gemini tool: {fc.name} | args keys: {list(args.keys())}")
                    
                    if fc.name == "search_3d_models":
                        return {
                            "intent": "search",
                            "reply_tr": args.get("reply_tr", "Arşivde arıyorum..."),
                            "search_queries": list(args.get("search_queries", [])),
                            "complementary_queries": list(args.get("complementary_queries", [])),
                            "object_type": args.get("object_type", "unknown"),
                            "style": args.get("style", "unknown"),
                            "material": args.get("material", "unknown"),
                            "color_family": args.get("color_family", "unknown"),
                            "room": args.get("room", "unknown"),
                            "source": "gemini_tools",
                        }
                    elif fc.name == "chat_reply":
                        return {
                            "intent": "chat",
                            "reply_tr": args.get("reply_tr", "Merhaba!"),
                            "search_queries": [],
                            "complementary_queries": [],
                            "source": "gemini_tools",
                        }
                    elif fc.name == "give_design_advice":
                        suggested = args.get("suggested_search", "")
                        return {
                            "intent": "advice",
                            "reply_tr": args.get("reply_tr", ""),
                            "search_queries": [suggested] if suggested else [],
                            "complementary_queries": [],
                            "source": "gemini_tools",
                        }
            
            # Eğer function call yoksa text response'u al
            text = response.text or ""
            if text:
                return {"intent": "chat", "reply_tr": strip_think(text), "search_queries": [], "source": "gemini_text"}

    except Exception as e:
        logger.warning(f"⚠️ Gemini Function Calling hata: {e}")
    
    return None  # Fallback'e düş


def call_local_llm_with_tools(user_message: str, chat_history: list, visual_context: str = "") -> Optional[Dict[str, Any]]:
    """
    LM Studio (OpenAI-uyumlu) function calling ile intent detection.
    Gemini çalışmadığında devreye girer.
    """
    settings = get_settings()
    
    # OpenAI formatında tool tanımları
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_3d_models",
                "description": "Kullanici bir 3D model, mobilya, dekorasyon veya tasarim ogesi aramak istediginde bu tool'u cagir.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_queries": {"type": "array", "items": {"type": "string"}, "description": "3-5 Ingilizce arama sorgusu"},
                        "reply_tr": {"type": "string", "description": "Kullaniciya Turkce samimi cevap"},
                        "object_type": {"type": "string"},
                        "style": {"type": "string"},
                        "material": {"type": "string"},
                        "color_family": {"type": "string"},
                        "room": {"type": "string"},
                        "complementary_queries": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["search_queries", "reply_tr"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "chat_reply",
                "description": "Kullanici selamlama, sohbet, tesekkur, veda, hal hatir sorma veya genel konusma yapiyorsa bu tool'u cagir. Arama gerektirmeyen her mesaj icin bunu kullan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reply_tr": {"type": "string", "description": "Samimi, sicak ve dogal Turkce cevap"},
                    },
                    "required": ["reply_tr"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "give_design_advice",
                "description": "Kullanici tasarim tavsiyesi, stil onerisi, renk kombinasyonu veya dekorasyon ipucu istediginde bu tool'u cagir.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reply_tr": {"type": "string", "description": "Detayli, profesyonel Turkce tasarim tavsiyesi"},
                        "suggested_search": {"type": "string", "description": "Opsiyonel Ingilizce arama sorgusu"},
                    },
                    "required": ["reply_tr"],
                },
            },
        },
    ]

    system_msg = (
        "Sen Archi, buyuk bir 3D model arsivi uzerinde calisan profesyonel ic mimari asistanisin.\n"
        "460.000'den fazla 3D model arsivin var.\n"
        "Turkce, samimi ve dogal konus.\n"
        "Kullanici sohbet ediyorsa chat_reply, arama istiyorsa search_3d_models, tavsiye istiyorsa give_design_advice tool'unu cagir.\n"
        "HER ZAMAN uygun tool'u cagir."
    )

    history_text = compact_history(chat_history)
    user_content = f"Gecmis sohbet:\n{history_text}\n\n"
    if visual_context:
        user_content += f"Gorsel analizi: {visual_context}\n\n"
    user_content += f"Kullanici mesaji: {user_message}"

    payload = {
        "model": settings.llm_text_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.3,
        "max_tokens": 1000,
    }

    try:
        resp = requests.post("http://localhost:1234/v1/chat/completions", json=payload, timeout=60)
        if resp.status_code != 200:
            return None

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        # Function call var mı?
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            fn_name = tc.get("function", {}).get("name", "")
            fn_args_raw = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
            except json.JSONDecodeError:
                args = {}

            logger.info(f"🖥️ Local LLM tool: {fn_name}")

            if fn_name == "search_3d_models":
                return {
                    "intent": "search",
                    "reply_tr": args.get("reply_tr", "Arsivde ariyorum..."),
                    "search_queries": list(args.get("search_queries", [])),
                    "complementary_queries": list(args.get("complementary_queries", [])),
                    "object_type": args.get("object_type", "unknown"),
                    "style": args.get("style", "unknown"),
                    "material": args.get("material", "unknown"),
                    "color_family": args.get("color_family", "unknown"),
                    "room": args.get("room", "unknown"),
                    "source": "local_tools",
                }
            elif fn_name == "chat_reply":
                return {
                    "intent": "chat",
                    "reply_tr": args.get("reply_tr", "Merhaba!"),
                    "search_queries": [],
                    "complementary_queries": [],
                    "source": "local_tools",
                }
            elif fn_name == "give_design_advice":
                suggested = args.get("suggested_search", "")
                return {
                    "intent": "advice",
                    "reply_tr": args.get("reply_tr", ""),
                    "search_queries": [suggested] if suggested else [],
                    "complementary_queries": [],
                    "source": "local_tools",
                }

        # Function call yoksa, düz text cevap
        content = strip_think(message.get("content", ""))
        if content:
            return {"intent": "chat", "reply_tr": content, "search_queries": [], "source": "local_text"}

    except Exception as e:
        logger.warning(f"⚠️ Local LLM Function Calling hata: {e}")

    return None


def call_llm(messages: List[Dict[str, Any]], *, temperature: float = 0.4, max_tokens: int = 1200, timeout: int = 60) -> str:
    """Legacy LLM çağrısı — eski kod uyumluluğu için korundu."""
    # 1. Gemini REST
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={gemini_key}"
            system_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
            user_text = "\n".join(m["content"] for m in messages if m["role"] != "system")
            payload = {
                "contents": [{"parts": [{"text": f"{system_text}\n\n{user_text}"}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
            }
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
            if resp.status_code == 200:
                return strip_think(resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip())
    except Exception as e:
        logger.warning(f"Gemini REST fallback hata: {e}")

    # 2. Local LM Studio
    settings = get_settings()
    payload = {"model": settings.llm_text_model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    response = requests.post("http://localhost:1234/v1/chat/completions", json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"LM Studio HTTP {response.status_code}: {response.text[:300]}")
    return strip_think(response.json()["choices"][0]["message"]["content"].strip())


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def first_match(text: str, mapping: Dict[str, List[str]], default: str = "unknown") -> str:
    haystack = f" {text.lower()} "
    for label, terms in mapping.items():
        for term in terms:
            needle = f" {term.lower()} "
            if needle in haystack or term.lower() in haystack:
                return label
    return default


def fallback_plan(user_message: str, visual_context: str = "") -> Dict[str, Any]:
    text = f"{user_message} {visual_context}"
    object_type = first_match(text, OBJECT_TERMS)
    style = first_match(text, STYLE_TERMS)
    material = first_match(text, MATERIAL_TERMS)
    color = first_match(text, COLOR_TERMS)
    room = first_match(text, ROOM_TERMS)

    parts = [value for value in [style, color, material, object_type] if value and value != "unknown"]
    if room != "unknown":
        parts.append(room.replace("_", " "))
    base_query = " ".join(parts).strip() or user_message.strip()

    complementary = []
    s = style if style != "unknown" else "modern"
    m = material if material != "unknown" else "wood"
    if object_type in {"chair", "sofa", "armchair"}:
        complementary = [f"{s} coffee table", f"{s} floor lamp", f"{s} decorative pillow"]
    elif object_type == "table":
        complementary = [f"{s} dining chair", f"{m} pendant lamp", f"{s} vase"]
    elif object_type == "bed":
        complementary = [f"{s} nightstand", f"{s} bedroom table lamp", f"{s} dresser"]
    elif object_type == "lamp":
        complementary = [f"{s} side table", f"{s} decorative vase"]
    elif object_type == "cabinet":
        complementary = [f"{s} mirror", f"{s} decorative object"]
    elif object_type == "decor":
        complementary = [f"{s} console table", f"{s} wall art"]
    elif object_type == "rug":
        complementary = [f"{s} sofa", f"{s} coffee table"]
    elif object_type == "sanitary":
        complementary = [f"{s} bathroom mirror", f"{s} bathroom cabinet"]
    elif object_type == "architectural":
        complementary = [f"{s} wall sconce", f"{s} decorative panel"]

    return {
        "intent": "visual_reference" if visual_context else "search",
        "summary_tr": user_message[:180],
        "object_type": object_type,
        "style": style,
        "material": material,
        "color_family": color,
        "room": room,
        "search_queries": [base_query],
        "complementary_queries": complementary[:3],
        "reply_tr": "Tamam dostum, istegini parcalayip arsivde daha akilli bir arama planiyla tarayacagim.",
        "source": "fallback_rules",
    }


def normalize_plan(raw: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    plan = dict(fallback)
    for key in ["intent", "summary_tr", "object_type", "style", "material", "color_family", "room", "reply_tr"]:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            plan[key] = value.strip()
    for key in ["search_queries", "complementary_queries"]:
        value = raw.get(key)
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            if cleaned:
                plan[key] = cleaned[:5]
    plan["source"] = raw.get("source", "llm") if isinstance(raw.get("source"), str) else "llm"
    return plan


def _is_chat_message(text: str) -> Optional[str]:
    """Basit sohbet mesajlarını anında tanı — API'ye hiç gitme."""
    t = text.lower().strip().rstrip("!?.…")
    
    greetings = ["merhaba", "selam", "selamlar", "hey", "heyy", "hello", "hi", "meraba", "mrb", "slm", "sa", "selamun aleykum", "as", "gunaydin", "iyi gunler", "iyi aksamlar"]
    thanks = ["tesekkurler", "tesekkur", "sagol", "saol", "eyvallah", "thanks", "thank you", "mersi", "tsk"]
    byes = ["gorusuruz", "hosca kal", "bye", "bb", "iyi geceler", "gule gule"]
    how_are = ["nasilsin", "nasılsın", "ne haber", "naber", "nbr"]
    
    for g in greetings:
        if t == g or t.startswith(g + " "):
            return "Merhaba dostum! Ben Archi, senin kişisel 3D model asistanıyım. Bugün hangi projen için yardımcı olabilirim?"
    for th in thanks:
        if t == th or t.startswith(th + " "):
            return "Rica ederim dostum! Başka bir şey lazım olursa buradayım, her zaman."
    for b in byes:
        if t == b or t.startswith(b + " "):
            return "Görüşürüz dostum! İyi çalışmalar, projen harika olacak."
    for h in how_are:
        if t == h or t.startswith(h + " "):
            return "İyiyim dostum, teşekkür ederim! Hadi bugün ne arıyoruz? Modern bir koltuk mu, şık bir lamba mı?"
    return None


def build_search_plan(user_message: str, chat_history: list, visual_context: str = "") -> Dict[str, Any]:
    # ⚡ Anlık Sohbet Tespiti — API'ye gitmeden milisaniyede cevap ver
    instant_reply = _is_chat_message(user_message)
    if instant_reply and not visual_context:
        return {
            "intent": "chat",
            "summary_tr": user_message[:100],
            "object_type": "unknown", "style": "unknown", "material": "unknown",
            "color_family": "unknown", "room": "unknown",
            "search_queries": [], "complementary_queries": [],
            "reply_tr": instant_reply,
            "source": "instant_local",
        }
    
    # 🤖 Gemini Function Calling — Model kendi karar versin
    try:
        system_prompt = load_prompt("archi_system.md", "Sen Archi adinda Turkce konusan bir ic mimari asistanisin.")
        result = call_gemini_with_tools(system_prompt, user_message, chat_history, visual_context)
        if result:
            # Eksik alanları doldur
            result.setdefault("summary_tr", user_message[:100])
            result.setdefault("object_type", "unknown")
            result.setdefault("style", "unknown")
            result.setdefault("material", "unknown")
            result.setdefault("color_family", "unknown")
            result.setdefault("room", "unknown")
            result.setdefault("search_queries", [])
            result.setdefault("complementary_queries", [])
            logger.info(f"✅ Gemini Tools intent: {result.get('intent')} | source: {result.get('source')}")
            return result
    except Exception as exc:
        logger.warning(f"Gemini Tools fallback: {exc}")

    # 🖥️ Local LLM Function Calling — Gemini yoksa local model denesin
    try:
        result = call_local_llm_with_tools(user_message, chat_history, visual_context)
        if result:
            result.setdefault("summary_tr", user_message[:100])
            result.setdefault("object_type", "unknown")
            result.setdefault("style", "unknown")
            result.setdefault("material", "unknown")
            result.setdefault("color_family", "unknown")
            result.setdefault("room", "unknown")
            result.setdefault("search_queries", [])
            result.setdefault("complementary_queries", [])
            logger.info(f"✅ Local LLM Tools intent: {result.get('intent')} | source: {result.get('source')}")
            return result
    except Exception as exc:
        logger.warning(f"Local LLM Tools fallback: {exc}")

    # 📋 Son Fallback: Kural tabanlı plan
    fallback = fallback_plan(user_message, visual_context)
    try:
        planner_prompt = load_prompt("archi_search_planner.md", "Mesaji JSON arama planina cevir.")
        messages = [
            {"role": "system", "content": f"{load_prompt('archi_system.md', '')}\n\n{planner_prompt}"},
            {"role": "user", "content": f"GECMIS SOHBET:\n{compact_history(chat_history)}\n\nGORSEL BAGLAM:\n{visual_context or '-'}\n\nKULLANICI MESAJI:\n{user_message}"},
        ]
        content = call_llm(messages, temperature=0.25, max_tokens=1200, timeout=50)
        parsed = extract_json_object(content)
        if parsed:
            return normalize_plan(parsed, fallback)
    except Exception as exc:
        logger.warning("Archi planner fallback kullandi: %s", exc)
    return fallback


def compact_history(chat_history: list) -> str:
    rows = []
    for msg in chat_history[-6:]:
        role = msg.get("role", "user") if isinstance(msg, dict) else "user"
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        rows.append(f"{role}: {content[:300]}")
    return "\n".join(rows)


def parse_legacy_search(content: str) -> Tuple[str, str]:
    search_query = ""
    if "[SEARCH:" in content:
        start_idx = content.find("[SEARCH:") + 8
        end_idx = content.find("]", start_idx)
        if end_idx != -1:
            search_query = content[start_idx:end_idx].strip()
            content = content[: content.find("[SEARCH:")].strip()
    if not search_query and "[SEARCH:" in content:
        start_idx = content.find("[SEARCH:") + 8
        search_query = content[start_idx:].strip()
        content = content[: content.find("[SEARCH:")].strip()
    return content, search_query


def memory_path() -> Path:
    return get_settings().get_data_path() / "archi_memory.json"


def load_archi_memory() -> Dict[str, Any]:
    path = memory_path()
    if not path.exists():
        return {"version": 1, "updated_at": None, "preferences": {}, "history": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"version": 1, "preferences": {}, "history": []}
    except Exception:
        return {"version": 1, "preferences": {}, "history": []}


def save_archi_memory(memory: Dict[str, Any]) -> None:
    path = memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def summarize_memory_for_prompt(memory: Dict[str, Any]) -> str:
    prefs = memory.get("preferences", {}) if isinstance(memory.get("preferences"), dict) else {}
    if not prefs:
        return "Henuz belirgin tercih yok."
    parts = []
    for key in ["object_type", "style", "material", "color_family", "room"]:
        values = prefs.get(key, {})
        if isinstance(values, dict) and values:
            top = sorted(values.items(), key=lambda x: x[1], reverse=True)[:3]
            parts.append(f"{key}: " + ", ".join(f"{name}({count})" for name, count in top))
    return "\n".join(parts) if parts else "Henuz belirgin tercih yok."


def update_archi_memory(user_message: str, plan: Dict[str, Any], selected_results: Optional[List[Dict[str, Any]]] = None) -> None:
    memory = load_archi_memory()
    prefs = memory.setdefault("preferences", {})
    for key in ["object_type", "style", "material", "color_family", "room"]:
        value = str(plan.get(key) or "unknown")
        if value and value != "unknown":
            bucket = prefs.setdefault(key, {})
            bucket[value] = int(bucket.get(value, 0)) + 1
            # 🛡️ Bucket boyutunu sınırla: Sadece en sık kullanılan 20 tercihi tut
            if len(bucket) > 20:
                top_items = sorted(bucket.items(), key=lambda x: x[1], reverse=True)[:20]
                prefs[key] = dict(top_items)
    memory.setdefault("history", []).append({
        "timestamp": utc_now(),
        "message": user_message[:500],
        "plan": {k: plan.get(k) for k in ["object_type", "style", "material", "color_family", "room", "search_queries"]},
        "result_count": len(selected_results or []),
    })
    memory["history"] = memory.get("history", [])[-100:]
    memory["updated_at"] = utc_now()
    save_archi_memory(memory)


def result_match_reasons(result: Dict[str, Any], plan: Dict[str, Any]) -> List[str]:
    meta = result.get("metadata", {}) or {}
    reasons = []
    checks = [
        ("object_type", "kategori"),
        ("style", "stil"),
        ("material", "materyal"),
        ("color_family", "renk"),
        ("room", "mekan"),
    ]
    blob = " ".join(str(meta.get(key, "")) for key, _ in checks).lower()
    blob += " " + str(meta.get("deep_tags", "")).lower()
    blob += " " + str(meta.get("caption", "")).lower()
    for key, label in checks:
        value = str(plan.get(key) or "unknown").lower()
        if value and value != "unknown" and value.replace("_", " ") in blob.replace("_", " "):
            reasons.append(f"{label}: {value.replace('_', ' ')}")
    if result.get("score"):
        reasons.append(f"benzerlik skoru %{int(float(result.get('score', 0)) * 100)}")
    return reasons[:4]


def annotate_results(results: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    seen = set()
    annotated = []
    for result in results:
        doc_id = result.get("id") or (result.get("metadata") or {}).get("image_path")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        result = dict(result)
        result["archi_reasons"] = result_match_reasons(result, plan)
        annotated.append(result)
    return annotated


def explain_results(plan: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    if not results:
        return "Arsivde bu brief'e yakin sonuc bulamadim; daha genis bir stil veya kategoriyle tekrar deneyebiliriz."
    lines = ["Secimleri su mantikla one aldim:"]
    for idx, result in enumerate(results[:5], start=1):
        meta = result.get("metadata", {}) or {}
        title = meta.get("channel_title") or meta.get("object_type") or result.get("id", "sonuc")
        reasons = result.get("archi_reasons") or result_match_reasons(result, plan)
        lines.append(f"{idx}. {title}: " + (", ".join(reasons) if reasons else "genel gorsel benzerligi yuksek."))
    return "\n".join(lines)


def chat_with_designer(user_message: str, chat_history: list, image_path: str = None) -> tuple[str, str]:
    response = chat_with_designer_v2(user_message, chat_history, image_path=image_path)
    query = ""
    if response.get("search_queries"):
        query = response["search_queries"][0]
    return response.get("reply_text", ""), query


def chat_with_designer_v2(user_message: str, chat_history: list, image_path: str = None) -> Dict[str, Any]:
    from search.image_search import analyze_image_with_llm

    visual_context = ""
    if image_path:
        visual_context = analyze_image_with_llm(image_path)
        if visual_context:
            logger.info("Chat gorsel analizi: %s", visual_context)

    plan = build_search_plan(user_message, chat_history, visual_context)
    
    # Eğer intent "chat" ise (selamlama, sohbet, teşekkür vs.) arama yapma
    intent = plan.get("intent", "search")
    if intent == "chat":
        reply = plan.get("reply_tr") or "Merhaba dostum! Ben Archi, senin 3D model asistanıyım. Bugün sana nasıl yardımcı olabilirim?"
        return {
            "reply_text": reply,
            "search_queries": [],
            "plan": plan,
            "visual_context": visual_context,
        }
    
    search_queries = list(dict.fromkeys(plan.get("search_queries", []) + plan.get("complementary_queries", [])))[:6]
    reply = plan.get("reply_tr") or "Tamam dostum, arsivde daha akilli bir arama planiyla tarayacagim."
    if search_queries:
        reply += "\n\nArama stratejim: " + ", ".join(search_queries[:4])
    return {
        "reply_text": reply,
        "search_queries": search_queries,
        "plan": plan,
        "visual_context": visual_context,
    }
