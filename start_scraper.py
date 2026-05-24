"""
🚀 Scraper Başlatıcı — Kaldığı yerden devam eder.
Çift tıklayın veya: python start_scraper.py
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, "scripts/run_scraper.py"])
