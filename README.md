# Telegram 3D Model Archive

Telegram 3D Model Archive, Telegram kanallarindan indirilen 3D model gorsellerini CLIP embeddingleri ile indeksleyen, ChromaDB uzerinde arayan ve Streamlit arayuzuyle yoneten yerel bir arsiv uygulamasidir.

Projenin ana hedefi sadece indirme yapmak degil; buyuk bir 3D model arsivini guvenilir, aranabilir, onarilabilir ve zamanla daha akilli hale getirmektir.

## Ne Yapar?

- Telegram kanallarini tarar ve gorselleri indirir.
- Gorselleri atomik indirme mantigiyle kaydeder; yarim/bozuk indirmeleri tamamlanmis saymaz.
- CLIP / OpenCLIP ile gorsel embedding uretir.
- ChromaDB koleksiyonuna metadata ve embedding yazar.
- Metin ile arama ve gorsel ile benzerlik aramasi sunar.
- Arsiv sagligi paneliyle DB, dosya, failed ve audit durumunu gosterir.
- Yeni kayitlarda 3D kalite metadata alanlari uretir: object_type, style, material, room, color_family, render_type.
- Failed kayitlari ve resume state dosyalarini tutarak kaldigi yerden devam eder.

## Proje Yapisi

```text
telegram/
  config.py                  Merkezi ayarlar ve metadata sozlesmesi
  indexer/                   ChromaDB ve CLIP indeksleme katmani
  scraper/                   Telegram client yonetimi
  search/                    Metin/gorsel arama ve tasarim asistani
  scripts/                   MegaSync, index, repair, curator ve yardimci scriptler
  ui/                        Streamlit arayuzu
  data/                      Yerel veri klasoru; git'e alinmaz
  docs/                      Arastirma ve gelistirme notlari
```

## Onemli Dosyalar

- `scripts/mega_sync.py`: Ana Telegram audit, indirme, encode ve Chroma upsert akisi.
- `ui/app.py`: Streamlit arama motoru, tasarim asistani ve arsiv sagligi paneli.
- `indexer/chroma_store.py`: ChromaDB yazma, arama ve silme katmani.
- `config.py`: `.env` ayarlari ve `MetadataSchema` alanlari.
- `data/sync_state.json`: MegaSync resume/audit state dosyasi.
- `data/mega_failed.json`: Basarisiz islerin tekrar denenebilir kaydi.
- `data/audit_report.json`: MegaSync durum raporu.

## Kurulum

1. Python ortamini hazirla.
2. Gereksinimleri yukle:

```powershell
pip install -r requirements.txt
```

3. `.env.example` dosyasini `.env` olarak kopyala ve Telegram bilgilerini doldur:

```text
TELEGRAM_API_ID=...
TELEGRAM_API_HASH=...
TELEGRAM_PHONE=...
TARGET_CHANNELS=@kanal1,@kanal2
DATA_DIR=./data
```

4. UI baslat:

```powershell
streamlit run ui/app.py
```

## MegaSync Kullanimi

Direkt calistirma varsayilan olarak tam audit davranisina ayarlidir:

```powershell
python scripts/mega_sync.py
```

Durum kontrolu:

```powershell
python scripts/mega_sync.py --status
```

Rapor yazdirma:

```powershell
python scripts/mega_sync.py --write-report
```

Sinirli test:

```powershell
python scripts/mega_sync.py --limit 10 --workers 4 --retry 2
```

Guvenli dry-run:

```powershell
python scripts/mega_sync.py --dry-run --limit 10
```

## UI Ozellikleri

- Metin ile semantik arama
- Gorsel ile benzer gorsel arama
- Sonuclar icin Telegram linki
- Benzerlerini bul aksiyonu
- Kategori, stil ve materyal filtreleri
- Arsiv Sagligi sekmesi
- Tasarim asistani sohbet alani



## Archi Intelligence

Archi tasarim asistani artik tek sorgu ureten basit chat akisi yerine structured search plan kullanir.

Eklenenler:

- Promptlar `prompts/` klasorune tasindi.
- Kullanici mesaji `object_type`, `style`, `material`, `color_family`, `room` ve coklu arama sorgularina ayrilir.
- Tek sorgu yerine ana sorgular + tamamlayici sorgular calistirilir.
- Sonuclar tekillestirilir ve metadata eslesmelerine gore kisa gerekcelerle aciklanir.
- Kullanici tercihleri `data/archi_memory.json` icinde kalici olarak tutulur.
- LM Studio kapaliysa kural tabanli fallback plan devreye girer.

## UI Pro Pack

Streamlit arayuzu artik daha fazla yonetim sayfasi icerir:

- `Arama Motoru`: Metin ve gorsel ile CLIP aramasi.
- `Arsiv Sagligi`: DB/dosya/failed/audit durumu ve orneklem tutarlilik kontrolu.
- `Kalite Laboratuvari`: CleanVision, imagededup ve Recognize Anything raporlarini inceler.
- `Gorev Merkezi`: MegaSync status/report komutlarini calistirir ve guvenli kalite raporu islerini baslatir.
- `Tasarim Asistani`: Sohbet tabanli tasarim ve arsiv arama asistanı.

Gorev Merkezi veri silmez veya tasimaz. CleanVision ve duplicate analizleri `data/reports/` altina rapor yazar; UI bu raporlari okur.

## Arsiv Sagligi

UI icindeki `Arsiv Sagligi` sekmesi su bilgileri okur:

- Chroma kayit sayisi
- Gorsel dosya sayisi
- Dosya/DB farki
- Failed kayit sayisi
- Karantina klasoru sayisi
- Audit kanal ilerlemesi
- Taranan foto sayisi
- Gecici `.part` dosyalari
- Son MegaSync log satiri
- Orneklem tabanli DB/dosya tutarlilik kontrolu

Bu ekran veri silmez, tasimaz veya DB yazmaz; sadece okuma ve raporlama yapar.

## Metadata Sozlesmesi

Yeni MegaSync kayitlari su temel alanlari yazar:

```text
channel_id
message_id
image_path
deep_link
channel_title
channel_username
timestamp
caption
deep_tags
source
model_name
schema_version
object_type
style
material
room
color_family
render_type
category_source
category_confidence
quality_tags
```

Bu alanlar arama, filtreleme, kalite kontrol ve ileride katalog gorunumu icin kullanilir.

## Git Guvenligi

Bu projede kod git'e alinabilir, veri alinmamalidir. Asagidaki dosya/klasorler git disinda kalmalidir:

```text
.env
data/
*.session
__pycache__/
*.pyc
.vscode/
```

Ozellikle su veriler git'e konmamalidir:

- Telegram session dosyalari
- `.env` API bilgileri
- ChromaDB klasoru
- indirilen gorseller
- failed/state JSON dosyalari
- buyuk metadata dump dosyalari


## Opsiyonel Kalite Araçları

Bu proje bazı harici GitHub/PyPI araçlarını ana akışı bozmadan opsiyonel rapor katmanı olarak kullanabilir:

```powershell
pip install -r requirements.optional.txt
```

Kalite audit raporu:

```powershell
python scripts/image_quality_audit.py --sample 5000
```

Duplicate aday raporu:

```powershell
python scripts/duplicate_candidates.py --method phash --sample 10000
```

Recognize Anything POC planı:

```powershell
python scripts/recognize_anything_poc.py --dry-run --sample 10
```

Bu scriptler varsayılan olarak dosya silmez, taşımaz ve ChromaDB yazmaz. Sadece `data/reports/` altında rapor üretir.

## Dis Entegrasyon Arastirmasi

GitHub uzerinden projeye eklenebilecek adaylar `docs/external_integrations.md` dosyasinda izlenir.

Ilk ciddi adaylar:

- `idealo/imagededup`: Guvenli duplicate/near-duplicate tespiti
- `cleanlab/cleanvision`: Blurry, dark, corrupt, duplicate gibi veri kalitesi denetimi (AGPL-3.0; opsiyonel tutulur)
- `voxel51/fiftyone`: Gorsel dataset explorer ve embedding/duplicate analiz laboratuvari
- `xinyu1205/recognize-anything`: Daha guclu otomatik tag uretimi
- `qdrant/qdrant` veya `lancedb/lancedb`: ChromaDB siniri buyurse alternatif vektor veritabani
- `rom1504/clip-retrieval`: Cok buyuk olcekli CLIP index/retrieval mimarisi fikirleri

## Gelistirme Yol Haritasi

1. Git repo temiz kurulumu ve ilk guvenli commit
2. README ve dokumantasyonun tamamlanmasi
3. Search benchmark sistemi
4. Metadata v3 enrichment scripti
5. Repair Center
6. Duplicate review paneli
7. Katalog/set gruplama sistemi
8. Feedback ile re-ranking
9. Alternatif vector backend denemesi

## Guvenlik Notu

Temizlik, silme, repair ve duplicate islemleri varsayilan olarak dry-run davranisinda tasarlanmalidir. Gercek silme/tasima sadece acik `--confirm` veya UI onayi ile yapilmalidir.



