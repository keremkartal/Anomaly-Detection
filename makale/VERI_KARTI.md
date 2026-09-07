# Veri Kartı — STANet Sentetik Trafik Trajektori Veri Kümesi

**Hakem 1, Madde 2'ye yanıt:** *"You do not name the original vehicle trajectory
dataset, define what constitutes an anomaly, nor explain how ground-truth labels
were obtained."*

> ### ⚠️ Doğrudan cevap
> **Orijinal bir kaynak veri kümesi yoktur.** Trajektoriler, OSM'den türetilmiş
> Kocaeli yol ağı üzerinde fizik-bilgili bir simülasyonla **üretilmiştir**.
> Anomaliler simülasyon sırasında bilinen kurallarla **enjekte edilmiştir**;
> yer-gerçeği etiketleri bu enjeksiyon kayıtlarından gelir. Bu, makalenin
> ilk sürümünde açıkça belirtilmemişti.

---

## 1. Künye

| Alan | Değer |
|---|---|
| Ad | STANet Sentetik Trafik Trajektori Veri Kümesi |
| Sürüm | 1.0 (Şubat 2026) |
| Üretim yöntemi | Fizik-bilgili Markov zinciri simülasyonu |
| Uzamsal taban | OpenStreetMap — Kocaeli, Türkiye |
| Zamansal çözünürlük | 1 saniye |
| Lisans | *(belirlenmeli — ODbL önerilir, OSM türevi olduğu için)* |
| Erişim | *(Zenodo DOI alınacak)* |

## 2. İçerik

### 2.1 Trajektori tablosu (`lstm_data.csv`)

| Özellik | Değer |
|---|---|
| Satır | 293.168 |
| Trip (sürüş) | 90 |
| Tekil segment | 4.599 |
| Trip başına satır | ort. 3.257 · min 460 · maks 10.006 |
| Zaman adımı | 1 sn, trip içinde kesintisiz (Δt = 1) |

Sütunlar: `trip_id, segment_id, time, speed, accel, bearing, percent, traffic,
dwel_duration, dwel_flag, anomaly, maxspeed`

Güzergâhlar Kocaeli ilçeleri arasındadır (Gölcük–Gebze, Derince–Kandıra,
Başiskele–Çayırova, …); **90 tekil güzergâh, her biri bir trip**.

### 2.2 Yol ağı tablosu (`GNN_DATA.csv`)

| Özellik | Değer |
|---|---|
| Satır | 4.678 |
| **Tekil fiziksel yol (`osm_segment_id`)** | **628** |
| Yol tipleri | trunk 1.483 · primary 822 · secondary 721 · motorway 603 · *_link 1.049 |

> ⚠️ **Kritik yapısal not:** `segment_id` trip'e özeldir — hiçbir `segment_id` iki
> tripte geçmez. Aynı fiziksel yol (`osm_segment_id`) ortalama **7,4 kez** ayrı
> satır olarak tekrarlanır. Makalede bildirilen "4.678 düğüm, ortalama derece
> 11,47" bu tekrardan kaynaklanır; **birleştirilmiş gerçek ağ 628 düğüm ve
> ortalama derece 1,11'dir**.

### 2.3 İmputasyon payları

| Alan | OSM'den | Varsayılan | Yumuşatılmış |
|---|---|---|---|
| `lanes` | 2.515 (%53,8) | 2.163 (%46,2) | — |
| `max_speed` | 1.651 (%35,3) | 2.334 (%49,9) | 693 (%14,8) |

Yani hız limitlerinin **yaklaşık yarısı yazar tanımlı varsayılandır**. Bu, GNN
dalının taşıdığı `max_speed` bilgisinin güvenilirliğini doğrudan etkiler.

## 3. Üretim modeli

Aracın kinematik durumu 4 durumlu bir Markov zinciriyle evrilir
(sabit hız, hızlanma, yavaşlama, duruş). Geçiş olasılıkları yerel trafik
rejimine (serbest akış / geçiş / tıkalı) koşullanır; rejim, log-normal bir
trafik yoğunluk dağılımından örneklenir.

Hız–yoğunluk ilişkisi (Pipes takip modelinden uyarlanmış):

```
v̄_s = (v_max − 0,08·v_max) · (1 − τ/10)^4        τ ∈ [1, 9]
```

Normal sürüşte ivme, AASHTO geometrik tasarım kılavuzuna dayanarak
`[−3, +1,5] m/s²` zarfında sınırlanır. Hız ve yön kanallarına her adımda
GPS titremesini taklit eden gürültü enjekte edilir.

## 4. Anomali taksonomisi ve etiket üretimi

**Etiketler modelden bağımsız değildir** — enjeksiyon anında kaydedilir.
Aşağıdaki eşikler hem makale Tablo 6'dan hem de **veriden ampirik olarak geri
doğrulanmıştır**.

| # | Kategori | Alt tip | Üretim kuralı | n (satır) | Pay |
|---|---|---|---|---|---|
| 1 | Sürücü | Hız limiti ihlali | `v > 1,10·v_max` | 1.405 | %0,479 |
| 2 | Sürücü | Anormal duraklama | `>30 sn` hareketsiz | 11.610 | %3,960 |
| 3 | Sürücü | Anormal ivme | `a ∈ [−6,−3] ∪ [1,5, 2,5]` m/s² | 299 | %0,102 |
| 4 | Sürücü | Kaza | bileşik hız/ivme/yön bozulması + uzun duruş | 5.741 | %1,958 |
| 5 | Sürücü | Anormal yön değişimi | `\|Δθ\| > 10°` | 213 | %0,073 |
| 6 | Cihaz | GPS hız artefaktı | geçmişe göre örneklenen bozulma | 66 | %0,023 |
| 7 | Cihaz | GPS ivme artefaktı | geçmişe göre örneklenen bozulma | 76 | %0,026 |
| 8 | Cihaz | GPS yön artefaktı | geçmişe göre örneklenen bozulma | 94 | %0,032 |
| 9 | Cihaz | Sinyal donması | `≥2 sn` özdeş değerler | 606 | %0,207 |
| 0 | — | Normal | — | 273.058 | %93,140 |

**Toplam anomali oranı: %6,86** (satır düzeyi) · %7,07 (50 adımlık pencere düzeyi)

### 4.1 Ampirik doğrulama (veriden geri çıkarılan imzalar)

| Tip | Ölçülen imza |
|---|---|
| 1 | `speed/maxspeed` ortalama **1,13** |
| 2 | `speed ∈ [0,0 – 0,9]` |
| 3 | `accel ∈ [−5,98 – −3,24]` |
| 6 | `speed/maxspeed` ortalama **3,28** |
| 7 | `accel ∈ [−13,1 – +16,2]` |
| 9 | son ≥2 sn'de değerler sabit |

## 5. Bilinen sınırlar — ⚠️ dürüstçe bildirilmeli

### 5.1 Etiket döngüselliği (Hakem 2, Madde 3)

Modele verilen 4 öznitelik `[v, a, sin θ, cos θ]` ile etiket üretim kuralları
**aynı sinyaller üzerinde tanımlıdır**. Tip 1, 2, 3, 5 ve 9 doğrudan bu
kanalların eşik fonksiyonlarıdır.

Ölçülen sonuç: eşiklerden türetilmiş **sıfır parametreli 3 kural, test setinde
F1 = 0,8536** elde eder; 225.156 parametreli STANet ise 0,8678 (fark
istatistiksel olarak **anlamsız**, McNemar p = 0,332).

**Tip 1 istisnadır:** `v_max` modelin zamansal girdisinde yoktur, yalnızca graf
düğüm özniteliğinde bulunur. Uzamsal dalın meşru katkı gösterebileceği tek
alt tip budur.

### 5.2 Graf yapısı

- Segmentler tripler arasında paylaşılmaz → her test düğümü eğitimde görülmemiştir
- 546 kopya grubunun 538'i **birebir aynı öznitelik vektörüne** sahiptir
- Aynı düğüme farklı etiketli sekanslar düşer → yol kimliğinden elde edilebilecek
  **teorik en iyi doğruluk %93,45**

### 5.3 Bölme dengesizliği

| | Train | Val | Test |
|---|---|---|---|
| Trip | 63 | 13 | 14 |
| Sekans | 19.255 | 4.036 | 4.799 |
| Anomali oranı | %7,90 | %7,58 | **%3,27** |
| Tip 4 (kaza) payı | %37,0 | %0 | **%3,8** |

Tek sabit bölme temsili değildir; grup K-fold zorunludur.

### 5.4 Genel

- Gerçek sensör verisi, doğrulanmış olay kaydı veya saha ölçümü **içermez**
- Tek şehir (Kocaeli), tek ağ, tek simülasyon konfigürasyonu
- Hız limitlerinin ~%50'si imputasyon
- Hava durumu, görüş mesafesi, yol yüzeyi gibi çevresel değişkenler yoktur
- Gerçek dünya performansı hakkında **hiçbir çıkarım yapılamaz**

## 6. Uygun kullanım

**Uygundur:** kontrollü ortamda mimari/füzyon karşılaştırması, ablasyon
çalışmaları, metodoloji geliştirme.

**Uygun değildir:** operasyonel dağıtım iddiası, gerçek dünya performans
tahmini, şehirler arası transfer iddiası, güvenlik-kritik karar desteği.

## 7. Yeniden üretilebilirlik durumu

| Artefakt | Durum |
|---|---|
| Üretilmiş CSV'ler | ✅ Mevcut |
| Ön işleme + model + deney kodu | ✅ Mevcut (`stanet/`, `experiments/`) |
| **Simülasyon üretici kodu** | ❌ **BULUNAMADI — hocadan istenmeli** |
| Rastgelelik tohumları | ❌ Kayıt yok |

> **Aksiyon:** Üretici kod bulunamazsa, veri kümesi tanımı gereği yeniden
> üretilemez ve Hakem 1 Madde 2 tam olarak kapanmaz. Bu durumda §4'teki
> ampirik olarak doğrulanmış spesifikasyondan üretici yeniden yazılmalı ve
> **"yeniden üretilmiş"** olarak açıkça etiketlenmelidir.

## 8. Data Availability Statement (makaleye eklenecek)

> The trajectory data analysed in this study are synthetic and were generated by
> a physics-informed simulation over an OpenStreetMap-derived road network of
> Kocaeli, Türkiye. No real-world vehicle trajectory dataset was used. The
> generated dataset, the preprocessing and modelling code, and the complete
> experimental scripts are available at *[Zenodo DOI]*. Ground-truth anomaly
> labels were recorded at injection time according to the taxonomy in Table 6;
> because several anomaly subtypes are defined as threshold functions of the
> same kinematic channels supplied to the model, a zero-parameter rule-based
> classifier is reported alongside all learned models as a reference point.
