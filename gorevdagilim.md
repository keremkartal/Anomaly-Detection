# Sensors revizyonu — görev dağılımı ve çalışma sırası

**Makale:** sensors-4582518 · *How Much Does Fusion Design Matter? Partition, Seed, and Cluster-Aware Evaluation of Hybrid LSTM–GAT Traffic Anomaly Detectors*
**Durum:** 3 hakem, toplam 25 madde · **Tarih:** 25 Eylül 2026
**Hazırlayan:** Kerem — hakem dosyaları okundu, iddialar kodda **doğrulandı**, etkileri **ölçüldü**

---

## 0. Önce bir uyarı: elimizdeki yol haritasında hata var

`LSTM_GAT_Uc_Hakem_Revizyon_Yol_Haritasi.docx` dosyası **hakem numaralarını
kaydırmış**. İçeriği tek tek eşleştirdim:

| Yol haritasında | Gerçekte |
|---|---|
| "Birinci hakem" (H1.1–H1.4, 4 madde) | **Hakem 2** |
| "İkinci hakem" (H2.1–H2.8, 8 madde) | **Hakem 3** |
| "Üçüncü hakem" (H3.1–H3.13, 13 madde) | **Hakem 1** |

Doğrulama: Hakem 1'in dosyasında 13 madde var, Hakem 2'de 4, Hakem 3'te 8.
Yol haritasının tablolarındaki satır sayıları sırasıyla 4, 8, 13.

**Bu önemli, çünkü yanıt mektubu hakem hakem yazılır.** O dosyadaki kodlarla
yanıt yazarsak her maddeyi yanlış hakeme cevaplamış oluruz — hakemler kendi
yorumlarının cevabını göremez. Aşağıdaki dağılımda **gerçek** numaralar
kullanıldı.

Yol haritasının kendisi de şunu yazıyor: *"makale, kod ve ham sonuçlar bu plan
hazırlanırken denetlenmemiştir"*. Yani hangi maddenin gerçekten sorun olduğu,
hangisinin zaten çözülmüş olduğu orada belli değil. Ben kodu denetledim,
aşağısı ona dayanıyor.

---

## 1. Doğruladığım iki kritik bulgu

Hakemlerin 25 maddesinin çoğu açıklama ve tartışma. Ama **ikisi gerçek hata**
ve ikisi de benim yazdığım kodda.

### 1.1 🔴 Havuz analizinde eşik sızıntısı — **en kritik madde** (Hakem 3, madde 2)

`experiments/f5_cv.py` satır 132:

```python
th_pool = float(np.mean([m["threshold_val"] for m in run_metrics]))
pooled_m = compute_metrics(y, p, th_pool)
```

Her fold-seed koşusu kendi doğrulama setinde bir eşik seçiyor. Havuzlanmış
analizde ise **hepsinin ortalaması** alınıp bütün tahminlere uygulanıyor.
Fold 0'ın test tahminlerine uygulanan eşik, fold 1–4'ün doğrulama setlerinden
etkileniyor — o setler fold 0'ın **test trip'lerini** içeriyor. Bu sızıntıdır.

Eşikler hiç de birbirine yakın değil:

```
hybrid_gat   fold0 s42: 0.880   fold1 s43: 0.230   fold4 s42: 0.900
             min 0.230  maks 0.940  →  hepsine uygulanan: 0.791
```

**Ölçtüğüm etki:**

| Model | Havuz F1 (mevcut) | Havuz F1 (düzeltilmiş) | Fark |
|---|---|---|---|
| hybrid_gat | 0,8906 | 0,8891 | −0,0014 |
| stanet_gated | 0,8865 | 0,8849 | −0,0016 |
| hybrid_concat | 0,8772 | 0,8528 | **−0,0244** |
| hybrid_crossatt | 0,8913 | 0,8838 | −0,0074 |
| lstm_only | 0,8812 | 0,8901 | +0,0089 |

Tablo 13 için bu tolere edilebilir. **Asıl darbe Tablo 14'te:**

| Alt tip | Δ (mevcut) | Δ (düzeltilmiş) |
|---|---|---|
| Hız aşımı | −0,0843 | −0,0780 |
| **Uzun duraklama** | **+0,0335** (Holm p = 0,002) | **+0,0074** |
| Sinyal donması | −0,0367 | −0,0587 |

> ⚠️ **Makalenin tek anlamlı pozitif sonucu +0,0335'ti. Düzeltilince +0,0074'e
> düşüyor — 4,5 kat.** Bu neredeyse kesin olarak anlamlılığını yitirir.
>
> Yani elimizde "uzamsal dal uzun duraklamada işe yarıyor" diyebileceğimiz
> kanıt kalmıyor olabilir. Hakem 3 bunu zaten öngörmüş: *"Reassess dwell using
> corrected thresholds and multiple seeds."*

**İyi haber:** Bu düzeltme **yeniden eğitim gerektirmiyor.** 130 koşunun
tahminleri önbellekte duruyor; sadece eşik uygulaması ve sonraki istatistikler
yeniden hesaplanacak. Birkaç saatlik iş.

### 1.2 🔴 Graf seçim kriteri test etiketlerini kullanıyor (Hakem 3, madde 4)

`experiments/f2_graph.py`:

```python
def label_ambiguity(ds, graph):
    for sp in ("train", "val", "test"):     # <-- test de dahil
```

Makalede trip-instance grafı seçmemizin gerekçesi "etiket saflığı tavanı
%95,13 vs %93,45". O tavan **test etiketleriyle** hesaplanmış.

Bu, hocamızın bir önceki turda haklı olarak eleştirdiği *"performansa göre graf
seçimi"* sorununu çözmek için koyduğum gerekçeydi. Ama çözüm, sorunun daha
kötü bir versiyonunu getirmiş: model seçimini test etiketine dayandırmak.

**Çözüm iki seçenek:**
- (a) Tavanı **yalnızca eğitim verisiyle** hesapla, gerekçeyi ona dayandır
- (b) İki grafı da **önceden belirlenmiş deney faktörü** say, seçim yapma, her
  sonucu iki grafta da raporla

Bence (b) daha güvenli — zaten Tablo 15'te ikisi de var. Ama (b) bütün ana
tabloları iki grafta raporlamayı gerektirir, yani ek koşum.

### 1.3 Ayrıca: gönderilen sürümde kalan küçük hata

Tablo 11'in başlığı hâlâ **"three seeds"** diyor, tablo beş tohumlu. Bunu
gönderimden sonra fark edip depoda düzelttim (commit `b719b3b`), ama Sensors'a
giden PDF'te var. Revizyonda düzelecek.

---

## 2. 25 maddenin sahipleri

Üç sütun: **[V]** veri ekibi · **[K]** Kerem (modelleme/değerlendirme) ·
**[O]** ortak karar / hoca

### Hakem 1 — 13 madde

| # | Konu | Sahip | Not |
|---|---|---|---|
| 1.1 | Gerçek veri / ikinci ağ ya da dürüst sınırlılık | **V + O** | Veri ekibi ikinci ağ üretebilir mi? Üretemezse kapsam daraltılır |
| 1.2 | 14 trip — geniş GA'yı vurgula, trip bazlı dağılım | **K** | Kod hazır, sunum işi |
| 1.3 | İki graf için sistematik duyarlılık | **K** | 1.2 (§1.2) ile birlikte çözülür |
| 1.4 | Pencere düzeyi p değerlerini belirgin etiketle | **K** | Tablo düzeni |
| 1.5 | Anomaliler aynı kinematik değişkenlerden üretiliyor | **V + K** | Veri ekibi üretim kuralını yazar, ben tartışmayı yazarım |
| 1.6 | Markov geçiş olasılıkları nasıl seçildi | **V** | Bende bu bilgi yok |
| 1.7 | Pencere/stride duyarlılık analizi | **K** | Veri yeniden pencerelenir, model yeniden koşar |
| 1.8 | Dört füzyon için eğitim + çıkarım süresi + parametre | **K** | Çoğu ölçülmüş, derlenecek |
| 1.9 | Gate çökmesi tanıları (eğitim eğrileri, gate dağılımı) | **K** | Yeni analiz |
| 1.10 | Betimsel dil tutarlılığı (özet/bulgu/tartışma/sonuç) | **K** | Metin |
| 1.11 | RF 0,0014 farkla neden bu kadar yakın — tartış | **K** | Metin |
| 1.12 | İmputasyon duyarlılığı (şerit, hız limiti) | **V + K** | Veri ekibi alternatif imputasyon üretir, ben koşarım |
| 1.13 | Tablolarda düzey etiketleri (pencere/trip/seed/CV) | **K** | Tablo düzeni |

### Hakem 2 — 4 madde

| # | Konu | Sahip | Not |
|---|---|---|---|
| 2.1 | Farklı trafik akışı, gürültü, anomali yoğunluğu | **V** | Yeni veri üretimi gerekiyor |
| 2.2 | Graf test bilgisine erişiyor mu? Görülmemiş ağa genelleme | **K** | Transdüktif kurulumu belgeleyeceğim (§1.2 ile bağlantılı) |
| 2.3 | STGCN/DCRNN/GWN için arama uzayı, erken durdurma, init; her modele kendi eşiği | **K** | Kısmen yapılmış, belgelenecek |
| 2.4 | Figür yazıları küçük | **K** | Figür üretimi bende |

### Hakem 3 — 8 madde

| # | Konu | Sahip | Not |
|---|---|---|---|
| 3.1 | Tablo 1'de değerlendirme farkları; STANet yayın durumu | **O** | *STANet yayınlandı mı, hangi statüde? Hoca bilir* |
| 3.2 | **Fold–seed eşik düzeltmesi, Tablo 13–14 yeniden hesap** | **K** | 🔴 **En kritik. §1.1** |
| 3.3 | Tekrarlı gruplu bölümleme, ortak seed, hiyerarşik belirsizlik | **K** | 🔴 En pahalı koşum |
| 3.4 | **Graf seçiminde etiket kullanımı**; kenar üretimi; grafsız kontroller | **K** | 🔴 **§1.2** |
| 3.5 | Bağımsız üretilmiş veri kümeleri; imputasyon–etiket ilişkisi | **V** | 🔴 Veri ekibinin en büyük maddesi |
| 3.6 | Klasik modeller + 3-eşikli kural CV'ye; kodlayıcı/graf eşit koşul | **K** | Orta maliyet |
| 3.7 | Denklem 15 ↔ Tablo 5; gate tanıları; seed 45'i CV'de dene | **K** | |
| 3.8 | Dokuz alt tip: sayılar, precision/recall/FP; dwell yeniden | **K** | §1.1 ile birlikte |

**Özet:** 25 maddenin **18'i bende**, **4'ü veri ekibinde**, **3'ü ortak/hoca**.

---

## 3. Alt tip sorunu — veri ekibiyle konuşulacak

Hakem 3 "dokuz alt tipin tamamı için sayı verin" diyor. Veriye baktım,
gerçek durum şu:

| Tip | Eğitim penceresi | Eğitim trip | Test penceresi | Test trip |
|---|---|---|---|---|
| 1 hız aşımı | 102 | 18 | 13 | 5 |
| 2 uzun duraklama | 774 | 24 | 119 | 4 |
| 3 anormal ivme | 16 | 12 | 2 | 2 |
| 4 kaza | 563 | 6 | 6 | 1 |
| 5 anormal yön değişimi | 15 | 11 | 2 | 1 |
| 7 gps ivme artefaktı | 5 | 5 | 3 | 3 |
| 8 gps yön artefaktı | 6 | 5 | 0 | 0 |
| 9 sinyal donması | 41 | 15 | 12 | 5 |

Dokuz değil **sekiz** tip var (tip 6 hiç üretilmemiş), testte **yedi**.
Çoğu tipte test verisi 1–3 trip — bunlarla F1 raporlamak anlamsız.

**Veri ekibine soru:** Tip 6 neden yok? Tip 3, 5, 7, 8 bu kadar seyrek olacak
şekilde mi tasarlandı, yoksa üretim parametresi mi böyle çıktı? Hakem 3'e
"bu tipler betimsel" demek için bunu bilmemiz lazım.

---

## 4. Çalışma sırası — kim kimi bekliyor

```
  HAFTA 1              HAFTA 2-3            HAFTA 4          HAFTA 5
  ─────────            ─────────            ───────          ───────
  [K] Eşik düzeltmesi  [K] Tekrarlı         [K] Duyarlılık   [K] Metin
      ve yeniden           bölümleme            analizleri       + tablolar
      hesaplama            koşumu                                + figürler
         │                    │                   │                 │
         ▼                    │                   │                 │
  ┌──────────────┐            │                   │                 │
  │ KARAR NOKTASI│            │                   │                 │
  │ dwell bulgusu│            │                   │                 │
  │ duruyor mu?  │            │                   │                 │
  └──────┬───────┘            │                   │                 │
         │                    │                   │                 │
  [K] Graf seçim              │                   │                 │
      gerekçesi ──────────────┘                   │                 │
                                                  │                 │
  [V] Markov + imputasyon  [V] Alternatif ────────┘                 │
      dokümantasyonu            veri üretimi                        │
         │                         │                                │
         └─────────────────────────┴────────────────────────────────┘
                                                                    ▼
                                                            [O] Yanıt mektubu
```

### Bağımlılıklar — kritik olanlar

| Bekleyen | Beklenen | Neden |
|---|---|---|
| **Hiçbiri** | Eşik düzeltmesi (3.2) | Bunu **hemen** yapabilirim, kimseyi beklemiyorum. Diğer her şey bunun sonucuna bağlı |
| Metin yazımı | Eşik düzeltmesi | Dwell bulgusu düşerse özet, sonuç ve katkı listesi yeniden yazılır |
| Tekrarlı bölümleme (3.3) | Graf kararı (3.4) | Hangi graf(lar) üzerinde koşacağımı bilmem lazım |
| İmputasyon deneyi (1.12) | **Veri ekibi** | Alternatif imputasyonlu veri olmadan koşamam |
| Akış/gürültü deneyleri (2.1, 3.5) | **Veri ekibi** | Yeni veri kümeleri gerekiyor |
| Yanıt mektubu | Hepsi | En son |

> 🔑 **En önemli nokta:** Veri ekibinin işi benim ilk iki haftamı **bloke
> etmiyor**. Ben eşik düzeltmesi ve tekrarlı bölümlemeyle başlayabilirim,
> onlar paralel çalışır. Ama **4. haftadan sonra onları beklerim** — yeni veri
> gelmezse duyarlılık deneyleri yapılamaz.

---

## 5–6. Görev listeleri

> Bu iki bölümün ilk hâli kod denetiminden **önce** yazılmıştı ve iki maddeyi
> yanlış tarafa koymuştu (imputasyon duyarlılığı ve görülmemiş-ağ kontrolü
> aslında bende yapılabiliyor). Denetim sonrası kesinleşmiş hâlleri aşağıda:
>
> - **Benim tam programım:** §9
> - **Ekipten istenecekler:** §10

## 7. Grupta karara bağlanacaklar

| # | Karar | Neden şimdi |
|---|---|---|
| 1 | **STANet yayınlandı mı?** Hangi dergi, hangi statü? | Hakem 3, madde 1 doğrudan soruyor. Tablo 1'de ve atıflarda düzeltme gerekiyor. Bunu hoca bilir |
| 2 | **Graf: seçim mi, faktör mü?** | §1.2. Faktör yaparsak bütün ana tablolar iki grafta raporlanır → koşum iki katına çıkar. Seçim yaparsak gerekçe yalnızca eğitim verisine dayanmalı |
| 3 | **İkinci ağ / gerçek veri üretilebilir mi?** | Üç hakemin ortak maddesi. Cevap hayırsa dilin daraltılması gerekiyor |
| 4 | **Kaç veri varyantı üretilecek?** | Veri ekibinin ve benim yükümü belirliyor |
| 5 | **Dwell bulgusu düşerse ne yapıyoruz?** | Muhtemelen düşecek (§1.1). Özet ve katkı listesi yeniden yazılır. Buna hazırlıklı olalım |
| 6 | **Hedef takvim** | Sensors revizyon süresi ne kadar verdi? Ona göre kapsam belirleyelim |

---

## 8. Dürüst değerlendirme

**İyi taraf:** Üç hakem de makalenin şeffaflığını olumlu bulmuş. Hakem 1
*"unusually transparent about its limitations"* diyor, Hakem 3 katkıyı
*"potentially valuable empirical contribution"* olarak tanımlıyor. Kimse
"reddedilsin" demiyor — hepsi düzeltme istiyor. İngilizce iki hakemde temiz,
birinde geliştirilmeli.

**Zor taraf:** Hakem 3 kodu gerçekten okumuş gibi. Eşik sızıntısını ve graf
seçimindeki etiket kullanımını bulmuş — ikisi de gerçek hata, ikisi de bende.
Bu maddeler geçiştirilemez, düzeltilmesi gerekiyor.

**Beklentiyi baştan söyleyeyim:** Eşik düzeltmesi muhtemelen elimizdeki tek
anlamlı pozitif sonucu götürecek. O zaman makale şunu söyler hale gelir:
*"bu ölçüm tasarımıyla hiçbir mimari fark çözülemiyor, ve bunu kendi
sonuçlarımız üzerinden gösteriyoruz"*. Bu hâlâ yayınlanabilir bir metodoloji
katkısı — ama performans iddiası içermeyen bir makale.

**Kapsam uyarısı:** 25 maddenin hepsini eksiksiz karşılamak birkaç ay sürer.
Hakemler de bunu bekliyor olamaz. Bence şu ayrımı yapmalıyız:

- **Mutlaka yapılacak:** 3.2, 3.4, 3.3, 3.6, 3.8 (hatalar ve ölçüm zinciri)
- **Yapılabilirse:** 2.1, 3.5, 1.12, 1.7 (duyarlılık — veri ekibine bağlı)
- **Yazıyla karşılanacak:** 1.1, 1.5, 1.10, 1.11, 3.1 (tartışma ve kapsam)

Hangi maddenin hangi kutuya gireceğini grupta konuşalım.

---

## 9. Tam program — neyi kendimiz kapatıyoruz

Kodu ve veriyi denetledikten sonra kapsamı yeniden çizdim. **25 maddenin
20'sini ekipten hiçbir şey beklemeden kapatabiliyoruz.** İki madde başta
"veri ekibine bağlı" görünüyordu, denetleyince öyle olmadığı çıktı:

| Madde | Önce sandığım | Gerçek |
|---|---|---|
| 1.12 İmputasyon duyarlılığı | veri ekibi alternatif üretmeli | `GNN_DATA.csv` içinde **`lanes_source`** ve **`speed_source`** sütunları var; hangi değerin OSM'den hangisinin varsayılan olduğu etiketli (şerit: 2.515 osm / 2.163 varsayılan · hız: 1.651 osm / 2.334 varsayılan / 693 yumuşatılmış). **Alternatif imputasyonu kendim kurabilirim** |
| 2.2 Görülmemiş ağa genelleme | yeni ağ gerekiyor | 628 yol, trip başına 7,4 paylaşım. **İndüktif kontrol kurulabilir**: eğitim sırasında graf yalnızca eğitim trip'lerinin düğümlerini içerir, test düğümleri çıkarım anında eklenir |

### FAZ A · Ölçüm zinciri — yeniden eğitim yok

Hepsi önbellekteki 130 koşudan hesaplanır. **Birkaç gün.**

| # | İş | Hakem |
|---|---|---|
| A1 | 🔴 **Eşik düzeltmesi** — her fold–seed kendi doğrulama eşiğini kendi testine uygular; Tablo 13, 14 ve trip düzeyi testleri yeniden hesaplanır | 3.2 |
| A2 | 🔴 **Graf seçim gerekçesi** — etiket saflığı tavanı yalnızca eğitim verisiyle; ya da iki graf önceden belirlenmiş faktör | 3.4 |
| A3 | Trip bazlı performans dağılımı + etki boyutları (Cohen's d, fark dağılımı) | 1.2 |
| A4 | Dört füzyon için parametre + eğitim süresi + çıkarım süresi tablosu | 1.8 |
| A5 | **Sekiz** alt tipin tam raporu: pencere ve pozitif-trip sayıları, precision/recall/FP, seyrek olanlar betimsel | 3.8 |
| A6 | Pencere düzeyi istatistiklerin her tabloda belirgin etiketlenmesi; düzey sütunu (pencere/trip/seed/CV) | 1.4 · 1.13 |
| A7 | Denklem 15 ↔ Tablo 5 hizalaması | 3.7 |
| A8 | Transdüktif kurulumun belgelenmesi: graf ne zaman kuruluyor, test segmentleri mesaj geçişine dahil mi, veri kökeni tablosu | 2.2 · 3.4 |

> **A1 karar noktası:** Dwell bulgusu (+0,0335 → tahminen +0,0074) anlamlılığını
> yitirirse özet, katkı listesi ve sonuç bölümü yeniden yazılır. FAZ C'nin
> şekli buna bağlı.

### FAZ B · Yeni koşumlar — ekipten bağımsız

**~25–35 saat GPU**, haftalara yayılabilir.

| # | İş | Koşu | Hakem |
|---|---|---|---|
| B1 | 🔴 **Tekrarlı gruplu bölümleme** — 5 farklı trip bölümlemesi × 5 fold × 2 seed × 5 konfig; hiyerarşik belirsizlik (bölümleme + seed + trip) | ~250 | 3.3 |
| B2 | **Grafsız ve kenarsız kontroller** — aynı bağlamsal öznitelikler graf mesajlaşması olmadan; kenarları kaldırılmış GAT. *GAT'ın katkısı özniteliklerden mi, mesaj geçişinden mi?* | ~40 | 3.4 |
| B3 | Klasik modeller (RF/XGB/LGBM) + 3-eşikli kural **gruplu CV'ye** | CPU | 3.6 |
| B4 | **Seed 45'i CV'de** — çökme bölmeye mi seed'e mi bağlı | ~10 | 3.7 |
| B5 | **Her iki graf üzerinde tüm ana deneyler** — sistematik duyarlılık | ~130 | 1.3 · 3.4 |
| B6 | **İndüktif graf kontrolü** — eğitimde test düğümleri yok | ~20 | 2.2 |
| B7 | **Pencere/stride duyarlılığı** — en az 3 ayar (τ=25/50/100, adım 10/25/50) | ~60 | 1.7 |
| B8 | **İmputasyon duyarlılığı** — 3 varyant: (a) imputasyonlu değerleri maskele + gösterge öznitelik, (b) OSM kaynaklı satırların yol-tipi medyanı, (c) imputasyonlu öznitelikleri tamamen çıkar | ~60 | 1.12 |
| B9 | **Gürültü duyarlılığı** — girdilere kademeli GPS gürültüsü (çıkarım anında, eğitim gerekmez) | ucuz | 2.1 kısmi |
| B10 | **Anomali yaygınlığı duyarlılığı** — mevcut veriden alt örnekleme ile %3 / %8 / %15 | ~30 | 2.1 · 3.5 kısmi |
| B11 | **Baseline hiperparametre araması** — STGCN/DCRNN/GWN için arama uzayı, erken durdurma, başlangıç; her modele kendi doğrulama eşiği; yakınsama kontrolü | ~80 | 2.3 |
| B12 | **Gate çökme tanıları** — eğitim eğrileri, epoch bazında gate dağılımı, gradyan normu. *Koşu deposuna `history` kaydı eklenecek, F3 yeniden koşulacak* | ~20 | 1.9 · 3.7 |

### FAZ C · Metin, tablo, figür, yanıt mektubu

| # | İş | Hakem |
|---|---|---|
| C1 | Figür yazı boyutları ve kontrast; basılı ölçekte gözden geçirme | 2.4 |
| C2 | Tablo 1: önceki çalışmalar için veri türü, bölümleme, seed, bağımlılık birimi, belirsizlik raporu sütunları | 3.1 kısmi |
| C3 | Betimsel dil tutarlılığı (özet, bulgular, tartışma, sonuç) | 1.10 |
| C4 | RF farkı (0,0014) ve daha yüksek ROC-AUC üzerine pratik fayda/maliyet tartışması | 1.11 |
| C5 | Kinematik girdilerden türetilmiş etiketlerin yüksek F1'e etkisi | 1.5 |
| C6 | 14 trip sınırının her yorumda vurgulanması; anlamsızlık ≠ eşdeğerlik | 1.2 · 3.8 |
| C7 | Sentetik tek ağ sınırı; kapsamın daraltılması | 1.1 · 3.5 kısmi |
| C8 | **Yanıt mektubu — 25 maddeye tek tek**, yapılan değişiklik + yeni sayı + sayfa/tablo konumu | hepsi |
| C9 | Doğrulayıcıyı genişlet (eşik kaynağı, düzey etiketi kontrolü), depo + Zenodo yeni sürüm | — |

### Nereye kadar geleceğiz

| Durum | Madde | Oran |
|---|---|---|
| ✅ Tam kapatılan (biz) | 1.2 · 1.3 · 1.4 · 1.7 · 1.8 · 1.9 · 1.10 · 1.11 · 1.12 · 1.13 · 2.2 · 2.3 · 2.4 · 3.2 · 3.3 · 3.4 · 3.6 · 3.7 · 3.8 | **19/25** |
| 🟡 Kısmen (biz) + tam cevap ekipten | 1.5 · 2.1 · 3.5 | 3/25 |
| 🔵 Yalnızca ekip | 1.6 | 1/25 |
| 🟣 Yalnızca hoca | 3.1 (STANet statüsü) | 1/25 |
| ⚪ Grup kararı | 1.1 (ikinci ağ / gerçek veri) | 1/25 |

---

## 10. Ekipten istenecekler — kısa ve net liste

Denetim sonrası liste kısaldı. **Beş madde.**

### 10.1 Markov geçiş olasılıkları — *yalnızca sizde* 🔵

> *Hakem 1: "additional information about how transition probabilities are
> selected or calibrated would improve reproducibility"*

Geçiş matrisi nereden geldi: literatürden mi alındı, elle mi ayarlandı, bir
veriye mi kalibre edildi? Yeniden üretilebilir olacak kadar açık yazılmalı.
Bende bu bilgi yok.

### 10.2 Etiket üretim kuralı — *büyük kısmını ölçtüm* 🟡

> *Hakem 3: "Clarify whether identical imputed speed limits are used for label
> generation and model inputs."*

**Bu soruyu ölçerek cevapladım.** Veri kartındaki kuralı (`v > 1,10 · v_max`)
grafın düğüm özniteliğindeki `max_speed` ile elle uyguladım:

```
v > 1,10 · v_max   ->   TP 13   FP 128   FN 0   recall 1,000
tip-1 pozitiflerin v/v_max orani: min 1,105 · medyan 1,127 · maks 1,171
1,10'un uzerinde olan: 13/13
```

**Evet, aynı değer.** On üç tip-1 pozitifin on üçü de tam olarak o kuralı
sağlıyor. Etiket üretimi ile model girdisi aynı `max_speed`'i kullanıyor.

Üstelik o pozitiflerin **5'i `varsayilan` (imputasyonlu) limitli segmentlerde**
— yani döngüsellik gerçek ve üçte birinden fazlası yazar tanımlı bir
varsayılana dayanıyor. Bunu makalede açıkça yazacağım.

**Sizden hâlâ gereken tek şey:** Kuralda **ek bir koşul** olmalı. 128 pencere
limiti aşıyor ama tip-1 etiketi almamış — süre koşulu mu, süreklilik mi, yoksa
başka bir tipe mi atanmış? Tam kuralı yazarsanız tartışmayı kesinleştiririm.

Aynı soru diğer alt tipler için de geçerli: hangi anomali hangi değişkenden,
hangi ek koşulla üretiliyor? Tam liste iyi olur.

### 10.3 Alt tip tasarımı 🟡

Veriye baktım:

- **Tip 6 hiç üretilmemiş** (taksonomide var, veride yok)
- Tip 3 (anormal ivme): eğitimde 16 pencere, testte 2
- Tip 5 (anormal yön değişimi): eğitimde 15, testte 2
- Tip 7 (gps ivme artefaktı): eğitimde 5, testte 3
- Tip 8 (gps yön artefaktı): eğitimde 6, **testte 0**

Sorular: Tip 6 neden yok? Bu seyreklikler tasarım kararı mı, üretim parametresi
sonucu mu? Hakem 3'e "bu tipler betimsel raporlanmıştır" diyebilmek için
gerekçesini bilmemiz lazım.

### 10.4 Bağımsız veri varyantları — *en büyük iş* 🟡

> *Hakem 2: "how different traffic-flow distributions, noise levels, and
> anomaly-injection intensities may affect the reported conclusions"*
> *Hakem 3: "Vary contextual dependence, anomaly prevalence, noise, and
> imputation across independently generated datasets"*

Ben gürültü ve yaygınlığı mevcut veriden **türetebiliyorum** (B9, B10) ama o
bir perturbasyon analizi, bağımsız üretim değil. Tam cevap için üretici
gerekiyor.

**İstenen: 3–4 varyant**, her biri için:

| Eksen | Değişecek |
|---|---|
| Trafik akışı | Farklı yoğunluk/rejim dağılımı |
| Gürültü | GPS gürültü seviyesi |
| Anomali yaygınlığı | Mevcut %7,9 dışında en az iki seviye |
| Anomali şiddeti | Enjeksiyon yoğunluğu |
| Bağlamsal bağımlılık | Anomalinin yol tipine bağlılığı |

Format mevcutla aynı olmalı (`lstm_data.csv` + `GNN_DATA.csv` şeması), üretim
parametreleri belgelenmiş. Gelince aynı protokolde koşarım.

**Zaman sıkıntımız yok** — bu haftalar sürebilir, ben paralel çalışırım.

### 10.5 İkinci yol ağı / gerçek veri — *grup kararı* ⚪

Üç hakemin de ortak maddesi. Kocaeli dışında ikinci bir OSM ağı üretmek mümkün
mü? Mümkünse en güçlü cevap bu olur ve makalenin kabul şansını ciddi artırır.

Değilse: "genelleme" iddiası tamamen çıkar, sonuçlar üretici koşullarına
bağlanır, gelecek çalışmaya bırakılır. Hakem 1 bu seçeneğe açık kapı bırakmış
(*"or mentioned this into future work"*), Hakem 3 daha sert
(*"otherwise, retain generator-conditional architectural conclusions"*).

### 10.6 Hocaya: STANet yayın durumu 🟣

> *Hakem 3: "Clarify STANet's publication status and substantiate claims about
> common evaluation practices."*

STANet yayınlandı mı? Hangi dergi, hangi statü (yayında / kabul / red / hakem
sürecinde)? Tablo 1'de ve atıflarda buna göre düzeltme gerekiyor. Bu bilgi
yalnızca sizde.

---

## 11. Zaman çizelgesi

Zaman sıkıntımız yok, o yüzden sıkıştırmıyorum.

```
 Hafta 1-2   [Kerem] FAZ A tamamı          → KARAR NOKTASI: dwell duruyor mu?
 Hafta 1-3   [Ekip]  10.1, 10.2, 10.3      (paralel, beni bloke etmiyor)
 Hafta 3-6   [Kerem] FAZ B1-B6             (ana koşumlar, ~15 saat GPU)
 Hafta 4-8   [Ekip]  10.4 veri varyantları
 Hafta 6-8   [Kerem] FAZ B7-B12            (duyarlılık, ~15 saat GPU)
 Hafta 8-9   [Kerem] Yeni veriler üzerinde koşum
 Hafta 9-11  [Kerem] FAZ C metin + tablolar + figürler
 Hafta 11-12 [Ortak] Yanıt mektubu, son okuma, gönderim
```

**Tek gerçek bağımlılık:** 8. haftadan sonra veri varyantları gelmezse (10.4)
o deneyler yapılamaz ve 2.1/3.5 maddeleri yazıyla karşılanır.

---

## 12. Öneri — hemen başlıyorum

FAZ A'nın tamamı kimseyi beklemiyor ve yeniden eğitim gerektirmiyor. **A1'i
(eşik düzeltmesi) hemen yapıyorum**, çünkü sonucu diğer her şeyin çerçevesini
belirliyor.

Çıkan sayıyı gruba bildiririm; §7'deki kararları o sayıyı görerek
konuşabiliriz.
