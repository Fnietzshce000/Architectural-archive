"""Kürasyon pipeline bağlantı ve Qwen testi."""
import base64
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings

LM_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen/qwen3-vl-8b"

def test():
    settings = get_settings()
    images_dir = settings.get_images_path()

    # 1. Bağlantı testi
    print("🔌 LM Studio bağlantısı test ediliyor...")
    r = requests.get("http://localhost:1234/v1/models", timeout=5)
    r.raise_for_status()
    models = [m["id"] for m in r.json()["data"]]
    print("✅ Bağlantı OK. Yüklü modeller:")
    for m in models:
        print(f"   - {m}")

    # 2. Qwen görsel testi
    imgs = list(Path(images_dir).glob("*.jpg"))[:5]
    if not imgs:
        print("❌ Görsel bulunamadı!")
        return

    print(f"\n🧪 5 rastgele görsel ile Qwen testi yapılıyor...")
    for img_path in imgs:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Answer only YES or NO."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": "Is this a 3D model render or 3D visualization? YES or NO only."},
                    ],
                },
            ],
            "max_tokens": 5,
            "temperature": 0.0,
        }

        t0 = time.time()
        resp = requests.post(LM_URL, json=payload, timeout=30)
        elapsed = time.time() - t0
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"   [{elapsed:.1f}s] {img_path.name[:30]:30s} → {answer}")

    print("\n✅ Test tamamlandı! Pipeline çalışmaya hazır.")


if __name__ == "__main__":
    test()
