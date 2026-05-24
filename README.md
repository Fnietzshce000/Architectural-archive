<div align="center">

# 🏛️ Archi — AI-Powered 3D Model Search Engine

**527,000+ 3D model arasında yapay zeka ile arama yapan açık kaynak arama motoru.**

Telegram kanallarından toplanan devasa mimari model arşivini SigLIP 2 yapay zekasıyla indeksleyip,
doğal dilde veya görsel referansla arama yapmanızı sağlar.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-00A67E?style=for-the-badge)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 🎯 Ne İşe Yarar?

Mimarlar, iç mimarlar ve 3D sanatçılar için tasarlandı. Telegram'daki onlarca 3D model kanalından görselleri otomatik toplar, yapay zeka ile analiz eder ve aranabilir hale getirir.

**Örnek:** `"mermer ayaklı altın detaylı modern koltuk"` yazın → saniyeler içinde en uygun modeller karşınızda.

<div align="center">

| 🔍 Metin Araması | 🖼️ Görsel Araması | 💬 AI Asistan | 🤖 Telegram Bot |
|:-:|:-:|:-:|:-:|
| Doğal dilde yaz, AI anlasın | Pinterest'ten referans at | Sohbet ederek model bul | Cebinden ara, her yerden |

</div>

---

## ✨ Özellikler

### 🧠 Yapay Zeka Motoru
- **SigLIP 2 (ViT-SO400M-14)** — Google'ın en güçlü görsel-dil modeli ile vektör araması
- **Aesthetic Predictor V2.5** — Her görseli 1-10 arası kalite puanı ile skorlar
- **Zero-Shot Tagging** — Mobilya tipi, stil, malzeme, renk ve oda otomatik etiketleme
- **Gemini Query Expansion** — Arama sorgularını AI ile 3 dile (TR/EN/RU) genişletir
- **Hybrid Re-ranking** — Benzerlik + Estetik + Etiket eşleşmesi ile elit sıralama

### 🔍 Arama Modları
- **Metin Araması** — `"İskandinav tarzı ahşap masa"` gibi doğal dilde
- **Görsel Araması** — Referans fotoğraf yükle, benzerleri bul
- **Semantik Filtreler** — Mobilya tipi, stil, malzeme, renk, oda bazında filtreleme
- **Estetik Filtresi** — Sadece elit kalite (8+) modelleri göster

### 📡 Veri Toplama
- **59 Telegram kanalından** otomatik görsel indirme (Telethon)
- **Akıllı Resume** — Kaldığı yerden devam eden scraper
- **Deduplikasyon** — pHash ile kopya görselleri otomatik eleme
- **Live Listener** — Kanallara yeni atılan modelleri anlık yakalama

### 🤖 Telegram Bot
- Metin yaz → sonuçlar gelir
- Fotoğraf gönder → benzer modeller bulunur
- `/elite` komutu → sadece en kaliteli modeller
- Her yerden erişim (telefon, tablet, PC)

### 🖥️ Web Arayüzü (Streamlit)
- Premium koyu tema, glassmorphism tasarım
- Sahne analizi (OWLv2 ile obje tespiti)
- Arşiv sağlık paneli
- Konuşarak arama yapan AI asistan (Gemini + Local LLM)

---

## 🏗️ Mimari

```
┌─────────────────────────────────────────────────┐
│                 📱 Kullanıcı                     │
│         (Web UI / Telegram Bot / API)           │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│              🔍 Arama Katmanı                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │  Metin   │ │  Görsel  │ │  Chat Asistan    │ │
│  │  Arama   │ │  Arama   │ │  (Gemini + LLM)  │ │
│  └────┬─────┘ └────┬─────┘ └────────┬─────────┘ │
│       └─────────┬───┘               │            │
│           ┌─────▼──────┐            │            │
│           │  SigLIP 2  │◄───────────┘            │
│           │  Encoder   │                         │
│           └─────┬──────┘                         │
└─────────────────┼────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────┐
│              📦 Veri Katmanı                      │
│  ┌──────────────┐  ┌────────────────────────┐    │
│  │  ChromaDB    │  │  Dosya Sistemi         │    │
│  │  527K Vektör │  │  500K+ Görsel (150GB)  │    │
│  └──────────────┘  └────────────────────────┘    │
└──────────────────────────────────────────────────┘
                  ▲
┌─────────────────┼────────────────────────────────┐
│              📡 Veri Toplama                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Scraper  │ │  Live    │ │  Indexer         │ │
│  │ (59 ch)  │ │ Listener │ │  (SigLIP + Aes.) │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
└──────────────────────────────────────────────────┘
```

---

## 🚀 Kurulum

### Gereksinimler
- Python 3.11+
- NVIDIA GPU (CUDA destekli, en az 6GB VRAM)
- Telegram API kimlik bilgileri ([my.telegram.org](https://my.telegram.org))

### 1. Repoyu Klonla
```bash
git clone https://github.com/Fnietzshe002/Architectural-archive.git
cd Architectural-archive
```

### 2. Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### 3. Ortam Değişkenlerini Ayarla
```bash
cp .env.example .env
# .env dosyasını düzenle ve kendi API bilgilerini gir
```

`.env` dosyasında doldurulması gereken alanlar:
```env
# Telegram API (my.telegram.org'dan al)
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_PHONE=+905551234567

# Hedef Kanallar
TARGET_CHANNELS=@Free_3D_Collection,@model3dpro,...

# AI Model
CLIP_MODEL_NAME=hf-hub:timm/ViT-SO400M-14-SigLIP2

# ChromaDB
CHROMA_COLLECTION=models_3d_siglip2
CHROMA_DB_PATH=./data/chroma_db

# Opsiyonel: Gemini (sorgu genişletme için)
GEMINI_API_KEY=your_gemini_key

# Opsiyonel: Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token
```

### 4. Veri Topla (Scraper)
```bash
python scripts/run_scraper.py
```

### 5. İndeksle (AI ile Puanlama)
```bash
python scripts/run_indexer.py
```

### 6. Arayüzü Başlat
```bash
streamlit run ui/app.py
```

### 7. (Opsiyonel) Telegram Botu Başlat
```bash
python scripts/run_bot.py
```

---

## 📂 Proje Yapısı

```
├── bot/                    # Telegram Bot
│   └── telegram_bot.py     # Bot komutları ve handler'ları
├── config.py               # Merkezi konfigürasyon (pydantic)
├── indexer/                # Vektör veritabanı katmanı
│   ├── chroma_store.py     # ChromaDB CRUD operasyonları
│   └── clip_encoder.py     # SigLIP 2 / OpenCLIP encoder
├── preprocessing/          # Görsel ön işleme
│   ├── deduplicator.py     # pHash ile kopya tespiti
│   └── image_processor.py  # Görsel doğrulama ve tarama
├── prompts/                # AI asistan prompt'ları
├── scraper/                # Telegram veri toplama
│   ├── channel_scraper.py  # Kanal tarayıcı
│   ├── live_listener.py    # Canlı dinleyici
│   └── models.py           # Veri modelleri
├── scripts/                # Çalıştırılabilir scriptler
│   ├── run_scraper.py      # Scraper başlatıcı
│   ├── run_indexer.py      # İndeksleme pipeline
│   ├── run_bot.py          # Bot başlatıcı
│   └── ...                 # Yardımcı scriptler
├── search/                 # Arama motoru
│   ├── text_search.py      # Metin araması + re-ranking
│   ├── image_search.py     # Görsel benzerlik araması
│   ├── chat_assistant.py   # Konuşma tabanlı AI asistan
│   └── scene_parser.py     # Sahne analizi (OWLv2)
├── ui/                     # Streamlit web arayüzü
│   └── app.py              # Ana uygulama (1600+ satır)
├── requirements.txt        # Python bağımlılıkları
└── .env.example            # Ortam değişkenleri şablonu
```

---

## 🧪 Teknoloji Yığını

| Katman | Teknoloji | Açıklama |
|--------|-----------|----------|
| **Görsel AI** | SigLIP 2 (ViT-SO400M-14) | Google'ın SOTA görsel-dil modeli |
| **Kalite AI** | Aesthetic Predictor V2.5 | Görsel kalite puanlama (1-10) |
| **Vektör DB** | ChromaDB | 527K+ vektör depolama ve sorgu |
| **Scraper** | Telethon (MTProto) | Telegram kanal tarama |
| **Web UI** | Streamlit | Premium koyu tema arayüz |
| **Bot** | python-telegram-bot v20+ | Async Telegram Bot API |
| **LLM** | Gemini 2.5 Flash + Local LLM | Sorgu genişletme & sohbet |
| **Etiketleme** | Zero-Shot Classification | Mobilya, stil, malzeme otomatik tag |

---

## 📊 İstatistikler

| Metrik | Değer |
|--------|-------|
| İndeksli Model | 527,582 |
| Taranan Kanal | 59 |
| Toplam Görsel | 550,000+ |
| Arşiv Boyutu | ~150 GB |
| Arama Süresi | < 1 saniye |
| Desteklenen Diller | TR / EN / RU |

---

## 🤝 Katkıda Bulunma

Pull request'ler ve issue'lar açıktır! Projeyi fork'layıp kendi kanallarınızla genişletebilirsiniz.

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit atın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

---

## ⭐ Beğendiyseniz

Bu proje işinize yaradıysa bir ⭐ bırakmayı unutmayın!

<div align="center">

**Archi ile aradığınız 3D modeli saniyeler içinde bulun.**

*Mimarlar tarafından, mimarlar için yapıldı.* 🏛️

</div>
