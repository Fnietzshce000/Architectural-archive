import subprocess
import sys
import os
from pathlib import Path

# Calisma dizinini ayarla
base_dir = Path(__file__).parent.parent
os.chdir(base_dir)

print("\n" + "="*40)
print("ARCHI AI: TARAMA SISTEMI BASLATILIYOR")
print("="*40)

# smart_core.py'yi yeni bir konsol penceresinde baslat
try:
    subprocess.Popen(
        [sys.executable, "tarama/smart_core.py"], 
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    print("\n[OK] Tarama motoru basariyla ateslendi!")
    print("Izleme: Acilan yeni pencereden ilerlemeyi takip edebilirsin.")
    print("Durdurma: 'python tarama/stop.py' komutunu kullan.")
except Exception as e:
    print(f"[HATA] Baslatma hatasi: {e}")

print("="*40 + "\n")
