"""
🤖 Archi Telegram Bot — Başlatıcı Script

Kullanım:
    python scripts/run_bot.py
"""
import sys
from pathlib import Path

# Proje kökünü Python path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bot.telegram_bot import run

if __name__ == "__main__":
    run()
