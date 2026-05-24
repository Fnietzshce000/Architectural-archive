"""
🛑 Scraper Durdurucu
Çalışan Telegram scraper işlemini bulur ve sonlandırır.
"""
import os
import subprocess
import sys

def stop_scraper():
    print("🔍 Çalışan scraper işlemi aranıyor...")
    
    if sys.platform == "win32":
        # Windows'ta komut satırında 'run_scraper.py' geçen işlemleri bul ve sonlandır
        try:
            # Taskkill ile filtreleme yaparak sadece ilgili scripti hedefliyoruz
            cmd = 'taskkill /F /FI "COMMANDLINE eq python scripts/run_scraper.py" /T'
            # Alternatif olarak daha geniş bir arama
            subprocess.run(['powershell', '-Command', 'Get-Process python | Where-Object {$_.CommandLine -like "*run_scraper.py*"} | Stop-Process -Force'], capture_output=True)
            print("✅ Scraper durduruldu.")
        except Exception as e:
            print(f"❌ Bir hata oluştu: {e}")
    else:
        # Linux/Mac için pkill
        os.system("pkill -f scripts/run_scraper.py")
        print("✅ Scraper durduruldu.")

if __name__ == "__main__":
    stop_scraper()
    input("\nKapatmak için Enter'a basın...")
