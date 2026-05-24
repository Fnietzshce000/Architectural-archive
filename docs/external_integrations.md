# External Integration Research

Bu dosya Telegram 3D Model Archive projesine GitHub uzerinden eklenebilecek acik kaynak projeleri takip eder. Amac hazir kodu rastgele iceri almak degil; lisansi, bakim durumu, entegrasyon maliyeti ve projeye katacagi gercek degeri tartarak kontrollu denemeler yapmaktir.

Arastirma tarihi: 2026-04-21

## Secim Kriterleri

Bir repo ancak su sorulara iyi cevap veriyorsa entegre edilmeli:

- 300k+ gorselde pratik fayda sagliyor mu?
- Windows ortaminda calisma sansi yuksek mi?
- Lisansi projede kullanmaya uygun mu?
- Mevcut Chroma + Streamlit + MegaSync akisini bozmadan yan modul olarak denenebilir mi?
- Basarisiz olursa kolayca geri alinabilir mi?
- Veri silme/tasima gibi riskli islemlerde dry-run ile kullanilabilir mi?

## Kisa Liste

| Repo | Ne Ise Yarar | Uygunluk | Risk | Ilk Deneme |
| --- | --- | --- | --- | --- |
| https://github.com/idealo/imagededup | Exact/near duplicate bulma | Cok yuksek | Orta; 300k gorselde sure ve bellek | `scripts/duplicate_candidates.py` |
| https://github.com/cleanlab/cleanvision | Blur, dark, light, low-info, odd-size, duplicate kalite denetimi | Cok yuksek | AGPL-3.0 lisans ve buyuk dataset sure riski | `scripts/image_quality_audit.py` |
| https://github.com/voxel51/fiftyone | Gorsel dataset explorer, embedding, duplicate ve curation lab | Yuksek | Agir dependency ve ayri UI | Harici explorer export scripti |
| https://github.com/xinyu1205/recognize-anything | RAM/RAM++ ile daha guclu image tagging | Yuksek | Model/checkpoint buyuk, GPU isteyebilir | 100 gorsellik tag benchmark |
| https://github.com/qdrant/qdrant | Payload filtreli vector DB | Orta-yuksek | Migration maliyeti | Yan index POC |
| https://github.com/lancedb/lancedb | Multimodal tablo + vector/full-text/SQL | Orta-yuksek | Backend degisimi | Yan index POC |
| https://github.com/rom1504/clip-retrieval | Buyuk olcekli CLIP index/retrieval mimarisi | Orta | Mimari degisim buyuk | Sadece benchmark/fikir alinacak |
| https://github.com/visual-layer/fastdup | Cok hizli duplicate/outlier/kalite analizi | Teknik olarak yuksek | Lisans ticari/degistirme kisitli | Dogrudan entegre etme; sadece lisans uygun ise opsiyonel |

## Aday 1: idealo/imagededup

Kaynak: https://github.com/idealo/imagededup

Neden iyi:

- Exact ve near-duplicate gorseller icin hazir cozum.
- PHash, DHash, WHash, AHash ve CNN tabanli yontemleri destekliyor.
- Apache-2.0 lisansli.
- Windows ve Python 3.9+ destegi belirtilmis.

Bizde nasil kullanilir:

- `data/images` icin hash tabanli aday duplicate listesi uretir.
- Sonucu `data/reports/duplicates_candidates.json` dosyasina yazar.
- UI `Arsiv Sagligi` altina duplicate adaylari sayisini ekler.
- Silme yapmaz; sadece aday gosterir.

Ilk POC:

```text
scripts/duplicate_candidates.py --method phash --limit 10000 --dry-run
```

Karar: Ilk entegre edilecek en guvenli adaylardan biri.

## Aday 2: cleanlab/cleanvision

Kaynak: https://github.com/cleanlab/cleanvision

Neden iyi:

- Blurry, dark, light, low_information, odd_aspect_ratio, odd_size, exact_duplicates, near_duplicates gibi veri kalitesi problemlerini otomatik bulur.
- Lisans notu: GitHub sayfasinda AGPL-3.0 gorunuyor. Bu yuzden CleanVision ana dependency degil, opsiyonel analiz araci olarak tutulmali.
- UI Saglik Paneli icin dogrudan anlamli metrik uretir.

Bizde nasil kullanilir:

- `data/images` uzerinde kalite audit calisir.
- Sonuc `data/reports/image_quality_report.json` olarak saklanir.
- MegaSync bozuk gorsel karantinasi ile birlesebilir.
- UI'da `Blurry`, `Dark`, `Low Info`, `Odd Size` sayilari gosterilir.

Ilk POC:

```text
scripts/image_quality_audit.py --sample 5000 --dry-run
```

Karar: Arşiv Sagligi panelinin ikinci seviyesi icin cok uygun.

## Aday 3: voxel51/fiftyone

Kaynak: https://github.com/voxel51/fiftyone

Neden iyi:

- Gorsel datasetleri gezmek, embeddingleri incelemek, duplicate ve edge-case bulmak icin guclu bir harici laboratuvar.
- Apache-2.0 lisansli.
- Embedding exploration, data curation ve model evaluation ozellikleri var.

Bizde nasil kullanilir:

- Ana Streamlit UI yerine gecmez.
- `scripts/export_to_fiftyone.py` ile Chroma metadata + image_path alanlari FiftyOne datasetine aktarilir.
- Kullanici isterse FiftyOne App acip buyuk arsivi detayli inceler.

Ilk POC:

```text
scripts/export_to_fiftyone.py --limit 5000
```

Karar: Ana uygulamaya gommek yerine harici analiz modu olarak dusunulmeli.

## Aday 4: xinyu1205/recognize-anything

Kaynak: https://github.com/xinyu1205/recognize-anything

Neden iyi:

- RAM/RAM++ genel gorsel tagging konusunda CLIP zero-shot sozluk yaklasimindan daha guclu olabilir.
- Tag2Text hem tag hem caption uretimi saglayabilir.
- Apache-2.0 lisansli.

Bizde nasil kullanilir:

- Caption'da zaten guclu sinyal varsa calismasin.
- Caption zayifsa veya yoksa RAM++ devreye girsin.
- `object_type`, `room`, `material`, `style` alanlarini daha iyi doldurabilir.

Ilk POC:

```text
scripts/tagging_benchmark.py --sample 100 --backend ram_plus
```

Karar: Arama kalitesini artirmak icin iyi aday, ama model dosyasi/GPU ihtiyaci once test edilmeli.

## Aday 5: qdrant/qdrant

Kaynak: https://github.com/qdrant/qdrant

Neden iyi:

- Vector search + payload filtering konusunda guclu.
- Payload indexleri, hybrid search ve on-disk/quantization ozellikleri var.
- Apache-2.0 lisansli.
- Python client ile lokal path veya server modunda kullanilabilir.

Bizde nasil kullanilir:

- ChromaDB'yi hemen degistirmeyiz.
- Yan index olarak 10k-50k kayitta POC yapariz.
- Ozellikle metadata filtreli aramada hiz/kalite farki olcülür.

Ilk POC:

```text
scripts/vector_backend_benchmark.py --backend qdrant --sample 50000
```

Karar: Chroma yetersiz kalirsa guclu alternatif.

## Aday 6: lancedb/lancedb

Kaynak: https://github.com/lancedb/lancedb

Neden iyi:

- Vector search, full-text search, SQL ve multimodal metadata icin tek tablo yaklasimi sunuyor.
- Apache-2.0 lisansli.
- Gorsel yolu, metadata ve embeddingleri ayni tablo mantiginda saklamak uzun vadede guzel olabilir.

Bizde nasil kullanilir:

- `data/lancedb` altinda yan index kurulur.
- Chroma ile ayni sorgular calistirilip benchmark edilir.
- Full-text + vector hybrid arama kalite karsilastirmasi yapilir.

Ilk POC:

```text
scripts/vector_backend_benchmark.py --backend lancedb --sample 50000
```

Karar: Katalog/analytics tarafinda ilginc; Chroma yerine gecmeden once POC sart.

## Aday 7: rom1504/clip-retrieval

Kaynak: https://github.com/rom1504/clip-retrieval

Neden iyi:

- Cok buyuk CLIP retrieval sistemleri icin olgun fikirler sunuyor.
- FAISS, metadata parquet ve backend/frontend ayrimi gibi mimari dersler var.
- MIT lisansli.

Bizde nasil kullanilir:

- Dogrudan entegre etmek yerine mimari fikir alinir.
- Ozellikle cok buyuk embedding batch, FAISS index ve metadata performansi icin benchmark referansi olur.

Karar: Kisa vadede entegre etmeyelim; uzun vadeli arama mimarisi icin inceleyelim.

## Aday 8: visual-layer/fastdup

Kaynak: https://github.com/visual-layer/fastdup

Neden iyi:

- Duplicate, outlier, broken image, low-quality image ve similarity gallery konusunda cok guclu gorunuyor.
- Buyuk veri performansi iddiasi yuksek.

Risk:

- Lisans `Creative Commons Attribution-NonCommercial-NoDerivatives 4.0` olarak gorunuyor.
- Bu lisans kod entegrasyonu ve ticari kullanim acisindan daha kisitli.

Karar:

- Dogrudan projeye dependency olarak eklemeyelim.
- Sadece lisans uygunlugu netlesirse harici opsiyonel analiz araci olabilir.

## Onerilen Entegrasyon Sirasi

1. `imagededup` ile duplicate aday raporu
2. `cleanvision` ile kalite audit raporu
3. UI Arsiv Sagligi paneline bu raporlari baglama
4. `recognize-anything` icin 100 gorsellik tagging benchmark
5. `qdrant` veya `lancedb` yan index benchmark
6. FiftyOne export modu

## Eklenen POC Scriptleri`n`n- `scripts/image_quality_audit.py`: CleanVision ile kalite raporu uretir; varsayilan sample 5000, veri silmez.`n- `scripts/duplicate_candidates.py`: imagededup ile duplicate aday raporu uretir; varsayilan sample 10000, veri silmez.`n- `scripts/recognize_anything_poc.py`: Recognize Anything resmi inference scriptlerini kucuk orneklemde calistirir; Chroma/metadata yazmaz.`n- `requirements.optional.txt`: Bu araclar ana dependency olmadan opsiyonel kurulabilir.`n`n## Ilk Uygulanacak Paket

Paket adi: `archive-quality-lab`

Icerik:

- `scripts/duplicate_candidates.py`
- `scripts/image_quality_audit.py`
- `data/reports/duplicates_candidates.example.json`
- `data/reports/image_quality_report.example.json`
- UI Arsiv Sagligi panelinde kalite raporu bolumu
- Dry-run varsayilan davranis
- Gercek silme yok

## Notlar

- Her yeni dependency once `requirements.optional.txt` icine konmali.
- Ana `requirements.txt` hemen sisirilmemeli.
- POC basarili olursa dependency ana requirements'a alinabilir.
- Her rapor JSON olarak yazilmali, UI sadece raporu okumali.
- Veri silen hicbir entegrasyon otomatik calismamali.

