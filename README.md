<div align="center">

# 🏛️ Archi — AI-Powered 3D Model Search Engine

**Search through 527,000+ architectural 3D models using state-of-the-art AI.**

An open-source search engine that scrapes Telegram channels for 3D model renders,
indexes them with SigLIP 2 vision AI, and lets you search using natural language or reference images.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-00A67E?style=for-the-badge)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 🎯 What Is This?

Built for architects, interior designers, and 3D artists. Archi automatically collects renders from dozens of Telegram 3D model channels, analyzes them with vision AI, and makes them instantly searchable.

**Example:** Type `"marble leg gold detail modern sofa"` → matching models appear in seconds.

<div align="center">

| 🔍 Text Search | 🖼️ Image Search | 💬 AI Assistant | 🤖 Telegram Bot |
|:-:|:-:|:-:|:-:|
| Type naturally, AI understands | Upload a Pinterest reference | Chat your way to models | Search from your pocket |

</div>

---

## ✨ Features

### 🧠 AI Engine
- **SigLIP 2 (ViT-SO400M-14)** — Google's state-of-the-art vision-language model for vector search
- **Aesthetic Predictor V2.5** — Scores every image on a 1-10 quality scale
- **Zero-Shot Tagging** — Auto-detects furniture type, style, material, color, and room
- **Gemini Query Expansion** — Expands search queries into 3 languages (EN/TR/RU)
- **Hybrid Re-ranking** — Elite ranking using similarity + aesthetics + tag matching

### 🔍 Search Modes
- **Text Search** — Natural language queries like `"Scandinavian wooden dining table"`
- **Image Search** — Upload a reference photo, find similar models
- **Semantic Filters** — Filter by furniture type, style, material, color, room
- **Aesthetic Filter** — Show only elite quality (8+) renders

### 📡 Data Pipeline
- **59 Telegram channels** — Automated scraping via Telethon (MTProto)
- **Smart Resume** — Picks up exactly where it left off
- **Deduplication** — Perceptual hashing (pHash) to eliminate duplicates
- **Live Listener** — Captures new models as they're posted in real-time

### 🤖 Telegram Bot
- Send text → get matching models
- Send/forward a photo → find similar models
- `/elite` command → only the highest quality results
- Access from anywhere (phone, tablet, desktop)

### 🖥️ Web UI (Streamlit)
- Premium dark theme with glassmorphism design
- Scene analysis (OWLv2 object detection)
- Archive health monitoring dashboard
- Conversational AI assistant (Gemini + Local LLM function calling)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                 📱 User Interface                │
│         (Web UI / Telegram Bot / API)           │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│              🔍 Search Layer                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │  Text    │ │  Image   │ │  Chat Assistant  │ │
│  │  Search  │ │  Search  │ │  (Gemini + LLM)  │ │
│  └────┬─────┘ └────┬─────┘ └────────┬─────────┘ │
│       └─────────┬───┘               │            │
│           ┌─────▼──────┐            │            │
│           │  SigLIP 2  │◄───────────┘            │
│           │  Encoder   │                         │
│           └─────┬──────┘                         │
└─────────────────┼────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────┐
│              📦 Data Layer                        │
│  ┌──────────────┐  ┌────────────────────────┐    │
│  │  ChromaDB    │  │  File System           │    │
│  │  527K vectors│  │  500K+ images (150GB)  │    │
│  └──────────────┘  └────────────────────────┘    │
└──────────────────────────────────────────────────┘
                  ▲
┌─────────────────┼────────────────────────────────┐
│              📡 Data Collection                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Scraper  │ │  Live    │ │  Indexer         │ │
│  │ (59 ch)  │ │ Listener │ │ (SigLIP + Aes.)  │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
└──────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- NVIDIA GPU (CUDA-enabled, 6GB+ VRAM recommended)
- Telegram API credentials ([my.telegram.org](https://my.telegram.org))

### 1. Clone the Repository
```bash
git clone https://github.com/Fnietzshce000/Architectural-archive.git
cd Architectural-archive
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your own API credentials
```

Key fields in `.env`:
```env
# Telegram API (from my.telegram.org)
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_PHONE=+905551234567

# Target Channels
TARGET_CHANNELS=@Free_3D_Collection,@model3dpro,...

# AI Model
CLIP_MODEL_NAME=hf-hub:timm/ViT-SO400M-14-SigLIP2

# ChromaDB
CHROMA_COLLECTION=models_3d_siglip2
CHROMA_DB_PATH=./data/chroma_db

# Optional: Gemini (for query expansion)
GEMINI_API_KEY=your_gemini_key

# Optional: Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token
```

### 4. Scrape Data
```bash
python scripts/run_scraper.py
```

### 5. Index with AI
```bash
python scripts/run_indexer.py
```

### 6. Launch Web UI
```bash
streamlit run ui/app.py
```

### 7. (Optional) Launch Telegram Bot
```bash
python scripts/run_bot.py
```

---

## 📂 Project Structure

```
├── bot/                    # Telegram Bot
│   └── telegram_bot.py     # Bot commands & handlers
├── config.py               # Central configuration (pydantic)
├── indexer/                # Vector database layer
│   ├── chroma_store.py     # ChromaDB CRUD operations
│   └── clip_encoder.py     # SigLIP 2 / OpenCLIP encoder
├── preprocessing/          # Image preprocessing
│   ├── deduplicator.py     # pHash duplicate detection
│   └── image_processor.py  # Image validation & scanning
├── prompts/                # AI assistant prompts
├── scraper/                # Telegram data collection
│   ├── channel_scraper.py  # Channel scraper
│   ├── live_listener.py    # Real-time listener
│   └── models.py           # Data models
├── scripts/                # Executable scripts
│   ├── run_scraper.py      # Scraper launcher
│   ├── run_indexer.py      # Indexing pipeline
│   ├── run_bot.py          # Bot launcher
│   └── ...                 # Utility scripts
├── search/                 # Search engine
│   ├── text_search.py      # Text search + hybrid re-ranking
│   ├── image_search.py     # Image similarity search
│   ├── chat_assistant.py   # Conversational AI assistant
│   └── scene_parser.py     # Scene analysis (OWLv2)
├── ui/                     # Streamlit web interface
│   └── app.py              # Main app (1600+ lines)
├── requirements.txt        # Python dependencies
└── .env.example            # Environment variables template
```

---

## 🧪 Tech Stack

| Layer | Technology | Description |
|-------|-----------|-------------|
| **Vision AI** | SigLIP 2 (ViT-SO400M-14) | Google's SOTA vision-language model |
| **Quality AI** | Aesthetic Predictor V2.5 | Visual quality scoring (1-10) |
| **Vector DB** | ChromaDB | 527K+ vector storage & retrieval |
| **Scraper** | Telethon (MTProto) | Telegram channel scraping |
| **Web UI** | Streamlit | Premium dark theme interface |
| **Bot** | python-telegram-bot v20+ | Async Telegram Bot API |
| **LLM** | Gemini 2.5 Flash + Local LLM | Query expansion & chat |
| **Tagging** | Zero-Shot Classification | Auto furniture/style/material tags |

---

## 📊 Stats

| Metric | Value |
|--------|-------|
| Indexed Models | 527,582 |
| Channels Scraped | 59 |
| Total Images | 550,000+ |
| Archive Size | ~150 GB |
| Search Latency | < 1 second |
| Supported Languages | EN / TR / RU |

---

## 🤝 Contributing

Pull requests and issues are welcome! Fork the project and extend it with your own channels.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## ⭐ Star This Repo

If you found this useful, please give it a ⭐ — it helps others discover the project!

<div align="center">

**Find the perfect 3D model in seconds with Archi.**

*Built by architects, for architects.* 🏛️

</div>
