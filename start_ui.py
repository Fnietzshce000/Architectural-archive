"""
🚀 Tek Tıkla Streamlit Arayüzünü Başlat
Çift tıklayın veya: python start_ui.py
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"])
