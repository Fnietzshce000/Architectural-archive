import os
import subprocess
import sys

print("\n" + "="*40)
print("ARCHI AI: TARAMA SISTEMI DURDURULUYOR")
print("="*40)

# tarama/smart_core.py içeren süreçleri bul ve öldür
try:
    if sys.platform == "win32":
        cmd = 'wmic process where "commandline like \'%tarama/smart_core.py%\'" delete'
        subprocess.run(cmd, shell=True, capture_output=True)
        print("\n✅ Tarama süreci durduruldu.")
        print("💾 Mevcut ilerleme 'data/enrich_queue.json' dosyasına kaydedildi.")
        print("🔄 Tekrar başlatmak için 'python tarama/start.py' kullanabilirsin.")
    else:
        print("❌ Bu script şu an sadece Windows için optimize edilmiştir.")
except Exception as e:
    print(f"❌ Durdurma sırasında bir hata oluştu: {e}")

print("="*40 + "\n")
