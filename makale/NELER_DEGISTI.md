# Makalede neler değişti — sayfa sayfa

**Dosya:** `makale/STANet_Kartal_v2_final.pdf` (17 sayfa, IEEEtran iki sütun)
**Karşılaştırma tabanı:** `main (6).pdf` — Sensors şablonu, 27 sayfa, eleştirdiğiniz sürüm
**Protokol:** `0902cd97ec17` · Doğrulama: `python makale/_verify_numbers.py` → 5/5 geçiyor

> Bu dosya PDF'in **kendi** sayfa ve tablo numaralarına göre yazıldı. MDPI
> sürümünün numaralarına göre olan eşleme ayrı dosyada: `MDPI_ESLEME.md`.

---

## En hızlı yol: yeni olan 4 bölüme bakın

Tüm sayılar değişti ama **yapısal olarak yeni** olan dört bölüm var. Bunlar
maddelerinizin doğrudan karşılığı:

| Sayfa | Bölüm | Hangi maddenizi kapatıyor |
|---|---|---|
| **s. 8** | VI-A-4 *Protocol Stamping and Shared Runs* | 1 — aynı model farklı tablolarda farklı sonuç |
| **s. 8** | VI-B *Unit of Statistical Analysis* | 4 — istatistiksel birim |
| **s. 13** | VII-I *Statistical Reassessment at the Trip Level* + **Tablo 16** | 4 — sayısal karşılığı |
| **s. 14** | VII-J *Scope of the Contribution* | 7 — sentetik veri |

Sadece bunları okusanız bile revizyonun ne yaptığını görürsünüz. **s. 8** ve
**s. 13** en kritik ikisi.

Bir de **s. 10**'da *"Note on the previous version"* diye kısa bir paragraf var:
üç seed'den beş seed'e geçince neyin değiştiğini açıkça yazıyor.

---

## Tablo tablo

### Değişen tablolar

| Tablo | Sayfa | Neydi | Ne oldu |
|---|---|---|---|
| **9** | 9 | Hibrit 0,8836 ± 0,0033 | **0,8713 ± 0,0188.** Precision/recall/AUC de değişti. "Hibritin seed bandı ağaç toplulukları ile örtüşmüyor" cümlesi **çıkarıldı** — üstünlük +0,0014 |
| **10** | 9 | 4 sütun, RF satırında GA [+0,038, +0,116] ortalamayı içermiyordu | **5 sütun.** ΔF1(ortalama) ve ΔF1(medyan seed) ayrı sütunlar; GA ikincisini bracket'liyor. RF için GA artık **[−0,0132, +0,0303]**. Altına iki paragraf eklendi: neyin yanlış olduğu ve neden |
| **11** | 10 | 3 seed, DCRNN ve GWN'e karşı anlamlı | **5 seed.** Liderin std'si 0,0033 → 0,0188. GRU (p=0,119), Transformer (p=0,815), dilated conv (p=0,115) — **anlamlılık düştü.** STGCN + üç uzamsal operatöre karşı duruyor |
| **12** | 10 | weighted_sum 0,8484 · gated 0,6811 | **0,8713 · 0,6953.** Çöküş aynı seed'de tekrarlandı (0,0833 → 0,0715). Sıralama aynı |
| **13** | 11 | 3 model, 5 fold, **1 seed** | **5 model, 5 fold, 2 seed = 50 koşu.** Çökme sayacı eklendi. Alt tabloda gated'in diğerlerine karşı trip düzeyi + Holm karşılaştırması |
| **14** | 12 | 3 alt-tip, pencere düzeyi p | Aynı 3 alt-tip ama **trip düzeyi GA + Holm.** Uzun duraklama satırı işaret değiştirdi ve **anlamlı oldu** (+0,0335, Holm p=0,002) |
| **15** | 12 | 4 satır, füzyon adı yazmıyordu | **6 satır + füzyon sütunu.** Her iki füzyon her iki grafta. Yeni bulgu: gated birleştirilmiş grafta **çökmüyor** |
| **17** | 14 | Gecikme **628 düğümlü** grafta ölçülmüş | **4.678 düğümlü** grafta yeniden ölçüldü — doğruluk tablolarıyla aynı graf. Önbellek kazancı 3,5× → **4,5×** GPU |

### Yeni tablo

| Tablo | Sayfa | Ne |
|---|---|---|
| **16** | 13 | 🆕 On altı karşılaştırmanın **pencere ve trip düzeyi p değerleri yan yana**, Holm sütunu, GA genişleme oranı. Yedi karar değişikliği burada görünüyor |

### Dokunulmayan tablolar

1–8 (ilişkili çalışma, notasyon, LSTM/GAT/füzyon konfigürasyonu, yol
varsayılanları, anomali taksonomisi, bölme) — bunlar sonuç tablosu değil,
tanım tablosu. Sayıları değişmedi.

---

## Figür figür

| Şekil | Sayfa | Değişiklik |
|---|---|---|
| **1** | 5 | ⚠️ **"fixed threshold 0.50" yazıyordu** — oysa eşik validation'da seçiliyor. Baştan çizildi: dört füzyon, iki graf, trip düzeyi kanıt kutusu |
| **2** | 9 | ⚠️ **`F1_zemin.json`'dan üretiliyordu**, Tablo 9 ise `F1b_fair.json`'dan — aynı modeller iki yerde farklı değer gösteriyordu (**2. maddeniz**). Artık tek kaynak + hata çubukları |
| **3** | 10 | Beş seed'e güncellendi |
| **4** | 10 | 🔄 **Baştan tasarlandı: sabit bölme ve CV yan yana.** Makalenin tezi tek bakışta görünüyor — gated solda 4/4 ve çöküyor, sağda 1/4 ve çökmüyor |
| **5** | 11 | Yeni kapı kayma değerleri |
| **6** | 12 | Türkçe etiketler (`hiz_asimi` vb.) → İngilizce; Holm p değerleri eklendi |
| **7** | 13 | 2 model → 5 model |
| **8** | 14 | Doğru grafta yeniden ölçüldü |

---

## Geri çekilen beş iddia — nerede duruyordu, şimdi ne diyor

| # | Eski iddia | Nerede | Şimdi |
|---|---|---|---|
| 1 | LSTM–GAT, STGCN/DCRNN/Graph WaveNet'i anlamlı geçiyor | Özet, Giriş, Tablo 11, Sonuç | Yalnızca STGCN'in zamansal operatörü ve üç uzamsal operatör. **s. 10**'da açıkça yazılı |
| 2 | Uzamsal dal anlamlı katkı sağlıyor (p = 0,035) | Tablo 10, s. 11 | Geri çekildi. **s. 11** *"We withdraw that claim"* |
| 3 | Gated füzyon diğerlerinden kötü | Tablo 12, Sonuç | CV'de birinci. **s. 11**: fark çözülemiyor, iddia edilmiyor |
| 4 | Gated en iyi kalibre (ECE 0,0338) | Kalibrasyon bölümü | Sıralama tersine döndü. **s. 12**'de düzeltme olarak yazılı |
| 5 | Hibrit klasik ML'i geçiyor | Tablo 9 | +0,0014 — kendi std'sinin onda biri. **s. 9** |

Hepsi metinde **açıkça** geri çekiliyor, sessizce çıkarılmadı. Hakemin önceki
sürümle karşılaştırma yapması durumunda savunulabilir olması için böyle yaptım.

---

## Metinde başka ne değişti

| Bölüm | Sayfa | Değişiklik |
|---|---|---|
| **Başlık** | 1 | *"A Multi-Seed and Trip-Grouped Evaluation…"* → *"How Much Does Fusion Design Matter? **Partition, Seed, and Cluster Effects**…"* |
| **Özet** | 1 | Baştan yazıldı. Dört bulguyu sırayla veriyor. *"Cross-experiment inconsistencies limit comparability"* cümlesi **çıkarıldı** |
| **Anahtar kelimeler** | 1 | Clustered Data, Cluster Bootstrap, Permutation Test, Reproducibility eklendi |
| **Giriş — katkı listesi** | 2 | Kodlayıcı katkısı iki soruya bölündü (uzamsal çözülebiliyor / zamansal çözülemiyor); trip düzeyi istatistik eklendi |
| **Graf gerekçesi** | 6 | ⚠️ *"because it yields higher hybrid performance"* idi — **8. maddeniz**. Yerine eğitimden önce hesaplanan bir veri özelliği: etiket saflığı tavanı (%95,13 vs %93,45). Ters yöndeki ödünç de yazıldı |
| **Tekrarlanabilirlik** | 8 | "beş seed füzyon için, üç seed benchmark için" → **her yerde beş seed** |
| **Threats to Validity** | 14 | *Statistical power* maddesi trip sayısı üzerinden yeniden yazıldı |
| **Implications** | 15 | Üç parçaya bölündü: birim, tek bölme, hata çubukları |
| **Summary of Findings** | 15 | Sekiz madde, tamamen yeni |
| **Data and Code Availability** | 16 | Depo URL'si + iki DOI + iki mekanizmanın açıklaması |
| **Kaynakça** | 16–17 | ⚠️ **Önceki sürümde hiç basılmıyormuş** — `stanet.bib` yoktu, 66 atıf çözümsüzdü, PDF 13 sayfa çıkıyordu. Düzeltildi. `gcda_bearing` künyesi Crossref'ten doğrulandı |

---

## Değişmeyenler

Bunlara hiç dokunulmadı, karşılaştırırken vakit kaybetmeyin:

- Bölüm I–V (Giriş'in katkı listesi hariç), Problem Formülasyonu, Mimari
  tanımı, Veri Seti bölümünün graf gerekçesi dışındaki kısmı
- Tablo 1–8
- Denklemler
- Yazar listesi ve kurumlar

---

## Kontrol etmek isterseniz

```bash
python makale/_verify_numbers.py
```

```
A. PROTOKOL     7 sonuç dosyası, hepsi 0902cd97ec17   [OK]
B. PAYLASILAN   4 tablo, fark 0.00e+00                [OK]
C. SAYILAR      107 kontrol / 107 eşleşme             [OK]
D. FIGURLER     hepsi kaynağından yeni                [OK]
E. YER TUTUCU   yok                                   [OK]
SONUC: TUM KONTROLLER GECTI
```

**C kontrolü** makaledeki 107 sayıyı tek tek `results/*.json` dosyalarıyla
karşılaştırıyor. Yani PDF'te yazan hiçbir sayı elle yazılmış değil, hepsi
deney çıktısından geliyor ve bu otomatik doğrulanıyor.

**B kontrolü** de şunu kanıtlıyor: Tablo 9, 11, 12 ve 15'te geçen aynı
konfigürasyon dört yerde de **0,871338** — çünkü artık dördü de aynı diskteki
koşuyu okuyor.
