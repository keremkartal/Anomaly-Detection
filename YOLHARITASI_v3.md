# Yol Haritası v3 — Sensors revizyonu

**Hedef:** 25 maddeyi **ikinci tur revizyon almayacak** biçimde kapatmak.
**Kısıt:** Zaman yok. Kalite var.
**Protokol tabanı:** `0902cd97ec17` · 260 önbellekli koşu · doğrulayıcı 5/5 geçiyor

---

## 0. İlke — "bir daha revize almamak" ne demek

İkinci tur revizyon beş sebepten gelir. Plan bu beşini engelleyecek şekilde
kuruldu:

| # | İkinci turu tetikleyen | Bizim önlemimiz |
|---|---|---|
| 1 | Hakem **deney** istemiş, biz **yazıyla** geçiştirmişiz | Her maddenin **cevap biçimi** önceden belirlendi (§2 sütun "biçim"). Yazıyla kapanacaklar, hakemin kendi verdiği izinle kapanıyor |
| 2 | Yeni sayılar makalenin **başka yerindeki** iddiayla çelişiyor | Her fazın sonunda `_verify_numbers.py` + yeni **E3 iddia denetimi** koşacak |
| 3 | Düzeltme **yeni bir sorun** yaratmış | Bu tam olarak bize oldu: graf gerekçesini düzeltirken etiket sızıntısı ekledim. Artık her düzeltme için "bu neyi bozabilir" kontrolü var (§5) |
| 4 | İddia, yeni veriden **fazlasını** söylüyor | Anlamsızlık ≠ eşdeğerlik. **TOST** ile eşdeğerliği ölçeceğiz ya da iddia etmeyeceğiz (§3, B13) |
| 5 | Yanıt mektubu **nereye baktığını** söylemiyor | Her maddeye sayfa/tablo/şekil konumu + yeni sayı. `_verify_letter.py` bunu denetleyecek (§4, C9) |

**En önemli kural:** Bir maddeyi kapatmadan önce şunu soracağım —
*"hakem bunu okuyup tekrar açabilir mi?"* Açabiliyorsa kapatılmamıştır.

---

## 1. Faz haritası

```
 FAZ 0  Hazırlık ve emniyet ağı          (1 gün, koşum yok)
    │
 FAZ A  Ölçüm zinciri düzeltmeleri       (3-4 gün, yeniden eğitim yok)
    │
    ├── 🚦 KARAR KAPISI 1 ─ dwell bulgusu + graf kararı ─ GRUP
    │
 FAZ B  Deneyler                          (3-5 hafta, ~30 saat GPU)
    │      B0 pilot → B1-B6 ana → B7-B13 duyarlılık
    │
    ├── 🚦 KARAR KAPISI 2 ─ hangi iddialar ayakta ─ GRUP
    │
 FAZ C  Metin, tablo, figür                (1-2 hafta)
    │
 FAZ D  Düşman gözüyle okuma + yanıt mektubu (1 hafta)
    │
    └── 🚦 KARAR KAPISI 3 ─ gönderim onayı ─ GRUP + HOCA
```

---

## 2. FAZ 0 · Hazırlık ve emniyet ağı

Koşum yok. Bu faz **ikinci turu engelleyen altyapıyı** kuruyor.

| # | İş | Çıktı | Kabul ölçütü |
|---|---|---|---|
| 0.1 | **Madde izleme dosyası** — 25 maddenin her biri için: hakem, metin, cevap biçimi (deney/analiz/yazı), sorumlu, durum, kanıt konumu | `HAKEM_TAKIP.md` | 25 satır, hepsi dolu |
| 0.2 | **Koşu deposuna `history` kaydı** — eğitim eğrileri ve epoch bazında gate dağılımı saklansın | `stanet/runstore.py` | Yeni koşular `history` taşıyor |
| 0.3 | **E3 iddia denetimi** — `.tex` içindeki her "significant/anlamlı/outperforms" iddiası bir JSON değerine bağlanmalı | `makale/_verify_numbers.py` | Bağlanmamış iddia = hata |
| 0.4 | **Deney kayıt defteri** — her koşum bloğu için tarih, protokol, koşu sayısı, çıktı dosyası | `DENEY_DEFTERI.md` | Her FAZ B adımı kayıtlı |

> **0.3 neden kritik:** Şu anki doğrulayıcı *sayıların* doğruluğunu kontrol
> ediyor, *iddiaların* doğruluğunu değil. Tablo 11'in "three seeds" hatası bu
> yüzden kaçmıştı. Yeni sayılar geldiğinde metinde kalan eski iddiaları bu
> yakalayacak.

---

## 3. FAZ A · Ölçüm zinciri — yeniden eğitim yok

Hepsi önbellekteki 260 koşudan hesaplanır. **3-4 gün.**

| # | İş | Hakem | Biçim | Kabul ölçütü |
|---|---|---|---|---|
| A1 | 🔴 **Eşik düzeltmesi** — her fold–seed kendi doğrulama eşiğini kendi testine uygular | 3.2 | deney | Tablo 13, 14 ve trip testleri yeniden; `pooled_threshold` alanı kalkar, yerine `per_run_thresholds` |
| A2 | 🔴 **Graf gerekçesi** — etiket tavanı yalnızca eğitim verisiyle; karar §4'te | 3.4 | analiz | `label_ambiguity(split="train")`; eski değer arşive |
| A3 | Trip bazlı performans dağılımı + etki boyutları (Cohen's d) | 1.2 | analiz | Her karşılaştırmada trip dağılımı figürü |
| A4 | Dört füzyon: parametre + eğitim süresi + çıkarım süresi | 1.8 | analiz | Tek tablo, önbellekten |
| A5 | **Sekiz** alt tipin tam raporu: pencere + pozitif-trip sayısı, precision/recall/FP | 3.8 | analiz | Seyrek tipler "betimsel" etiketli |
| A6 | Her tabloya **düzey sütunu** (pencere / trip / seed / CV) | 1.4 · 1.13 | yazı | Hiçbir sayı düzeysiz kalmaz |
| A7 | Denklem 15 ↔ Tablo 5 hizalaması | 3.7 | yazı | Aynı füzyon tanımı |
| A8 | **Veri kökeni tablosu** — hangi bilgi hangi aşamada, test erişimi var mı | 2.2 · 3.4 | yazı | Graf, düğüm öznitelikleri, etiketler, eşik için ayrı satır |
| A9 | **Hız limiti döngüselliği** — ölçülen bulgu metne | 3.5 kısmi | yazı | 13/13, imputasyon payı |

### 🚦 KARAR KAPISI 1 — gruba gider

A1 ve A2 bitince **koşum yapmadan önce** iki karar:

**Karar 1.1 — Dwell bulgusu ayakta mı?**
Ön ölçüm: +0,0335 → **+0,0074**. Anlamlılığını yitirirse:
- Özet, katkı listesi, sonuç bölümü yeniden yazılır
- Makale "hiçbir mimari fark çözülemiyor" konumuna geçer
- **Bu FAZ C'nin şeklini belirliyor, o yüzden önce karar**

**Karar 1.2 — Graf: seçim mi, faktör mü?**

| Seçenek | Sonuç | Koşum maliyeti |
|---|---|---|
| (a) Seçim, eğitim verisiyle gerekçelendir | Tek grafta ana tablolar | ~15 saat GPU |
| (b) İki grafı da önceden belirlenmiş faktör say | Tüm ana tablolar iki grafta | ~30 saat GPU |

**Önerim (b).** Zaman kısıtımız yok ve hakem 1 zaten *"systematic sensitivity
analysis ... across both constructions"* istiyor. (a) yaparsak hakem "neden
sistematik değil" diye tekrar sorabilir — yani (a) ikinci tur riski taşıyor.

---

## 4. FAZ B · Deneyler

### B0 · Pilot — büyük koşumdan önce tasarımı doğrula

**15 saat GPU harcamadan önce yarım saatlik pilot.** Amaç: istatistiksel
tasarımın doğru olduğunu küçük ölçekte görmek.

| # | İş | Kabul ölçütü |
|---|---|---|
| B0.1 | 2 tekrar × 5 fold × 1 seed × 1 konfig ile boru hattını koş | Varyans bileşenleri hesaplanıyor, çökme yok |
| B0.2 | **Nadeau–Bengio düzeltmesini** uygula ve doğrula | Düzeltilmiş GA, düzeltilmemişten geniş |
| B0.3 | Tasarımı yaz: kaç tekrar, kaç seed, hangi belirsizlik modeli | `DENEY_DEFTERI.md` |

> **Neden Nadeau–Bengio:** Hakem 3 *"uncertainty estimation that accounts for
> seed variability and **overlapping training sets**"* diyor. Tekrarlı CV'de
> eğitim kümeleri örtüşür, standart t-testi bunu yok sayar ve GA'yı dar verir.
> Nadeau–Bengio düzeltmesi tam bu durum için. İsmini yanıt mektubunda açıkça
> yazacağız — hakemin ne istediğini anladığımızı gösterir.

### B1–B6 · Ana koşumlar

| # | İş | Hakem | Koşu | Kabul ölçütü |
|---|---|---|---|---|
| B1 | 🔴 **Tekrarlı gruplu bölümleme** — 5 bölümleme × 5 fold × 2 seed × 5 konfig | 3.3 | ~250 | Varyans bileşenleri (bölümleme/fold/seed) ayrıştırılmış; NB düzeltmeli GA |
| B2 | **Grafsız + kenarsız kontroller** — aynı öznitelikler mesaj geçişi olmadan; kenarları silinmiş GAT | 3.4 | ~40 | GAT'ın katkısı öznitelik mi mesaj mı, sayıyla ayrılmış |
| B3 | Klasik modeller + 3-eşikli kural **gruplu CV'ye** | 3.6 | CPU | Tablo 13'e RF/XGB/LGBM/kural satırları |
| B4 | **Seed 45'i CV'de** | 3.7 | ~10 | Çökme bölmeye mi seed'e mi bağlı, cevaplı |
| B5 | **Her iki grafta tüm ana deneyler** (Karar 1.2 (b) ise) | 1.3 · 3.4 | ~250 | Her ana tablo iki grafta |
| B6 | **İndüktif graf kontrolü** — eğitimde test düğümleri yok | 2.2 | ~20 | Transdüktif ↔ indüktif farkı ölçülmüş |

### B7–B13 · Duyarlılık

| # | İş | Hakem | Koşu | Kabul ölçütü |
|---|---|---|---|---|
| B7 | **Pencere/stride** — τ ∈ {25, 50, 100} × adım ∈ {10, 25, 50} | 1.7 | ~60 | Örtüşme oranı ↔ sonuç ilişkisi tablo |
| B8 | **İmputasyon** — 3 varyant: maskele+gösterge · OSM medyanı · özniteliği çıkar | 1.12 | ~60 | Uzamsal dal bulgusu imputasyona duyarlı mı |
| B9 | **Gürültü** — girdilere kademeli GPS gürültüsü (çıkarım anında) | 2.1 kısmi | ucuz | Bozulma eğrisi |
| B10 | **Anomali yaygınlığı** — alt örneklemeyle %3 / %8 / %15 | 2.1 · 3.5 kısmi | ~30 | Yaygınlık ↔ F1 ilişkisi |
| B11 | **Baseline hiperparametre araması** — STGCN/DCRNN/GWN için arama uzayı, erken durdurma, init, yakınsama | 2.3 | ~80 | Her modelin araması belgeli; her biri kendi eşiğini kullanıyor |
| B12 | **Gate çökme tanıları** — eğitim eğrisi, epoch bazında gate dağılımı, gradyan normu | 1.9 · 3.7 | ~20 | Çökme mekanizması eğriyle gösterilmiş |
| B13 | **TOST eşdeğerlik testi** — "fark yok" yerine "fark şu sınırın içinde" | 3.8 | ucuz | Eşdeğerlik sınırı önceden belirlenmiş ve gerekçeli |

> **B13 neden var:** Hakem 3 *"Nonsignificance establishes neither absence of
> benefit nor equivalence; equivalence claims require appropriate testing"*
> diyor. Bizim ana bulgumuz "hiçbir füzyon ayırt edilemiyor" — bu bir
> eşdeğerlik iması. TOST yapmazsak hakem haklı olarak tekrar açar.
> **Bu madde doğrudan ikinci tur riskini kapatıyor.**

---

## 5. Her düzeltme için yan etki kontrolü

Graf gerekçesi hatası bize şunu öğretti: **bir maddeyi düzeltirken başkasını
bozabiliyoruz.** Her düzeltmeden sonra şu üçlü kontrol:

| Kontrol | Soru |
|---|---|
| **Sızıntı** | Bu düzeltme test verisinden bir bilgi kullanıyor mu? |
| **Tutarlılık** | Bu sayı makalenin başka yerinde geçiyor mu, orada da güncellendi mi? |
| **Aşırı iddia** | Bu sonuç, verinin desteklediğinden fazlasını söylüyor mu? |

`_verify_numbers.py` ilk ikisini otomatik yakalıyor (B ve E3 kontrolleri).
Üçüncüsü elle — FAZ D'de düşman okumasında.

---

## 6. FAZ C · Metin, tablo, figür

| # | İş | Hakem |
|---|---|---|
| C1 | Figür yazı boyutları ve kontrast, basılı ölçekte gözden geçirme | 2.4 |
| C2 | Tablo 1: önceki çalışmalar için **değerlendirme protokolü sütunları** — veri türü, bölümleme, seed sayısı, bağımlılık birimi, belirsizlik raporu | 3.1 kısmi |
| C3 | Betimsel dil tutarlılığı: özet, bulgular, tartışma, sonuç | 1.10 |
| C4 | RF farkı (0,0014) ve yüksek ROC-AUC üzerine fayda/maliyet tartışması | 1.11 |
| C5 | Kinematik türetilmiş etiketlerin yüksek F1'e etkisi + hız limiti döngüselliği | 1.5 · 3.5 |
| C6 | 14 trip sınırı her yorumda; **anlamsızlık ≠ eşdeğerlik** (B13'e atıf) | 1.2 · 3.8 |
| C7 | Sentetik tek ağ sınırı; kapsamın üretici koşullarına bağlanması | 1.1 · 3.5 |
| C8 | Yeni bölüm: **değerlendirme protokolü** — tekrarlı bölümleme, NB düzeltmesi, TOST | 3.3 · 3.8 |

---

## 7. FAZ D · Düşman gözüyle okuma — ikinci turu burada engelliyoruz

Yanıt mektubundan **önce** yapılacak. En önemli faz.

| # | İş | Kabul ölçütü |
|---|---|---|
| D1 | **Hakem rolü oynama** — 25 maddenin her biri için "bunu okuyup tekrar açar mıyım?" | Açılabilir madde kalmaz ya da gerekçesi yazılır |
| D2 | **Yeni itiraz avı** — revizyonun *kendisinin* yarattığı itirazlar | Liste + her birine önceden cevap |
| D3 | `_verify_numbers.py` + `_verify_pdf.py` tam koşum | 5/5 + iddia denetimi geçer |
| D4 | **Yanıt mektubu** — 25 maddeye tek tek: ne yapıldı, yeni sayı, konum | Her madde sayfa/tablo/şekil referanslı |
| D5 | **`_verify_letter.py`** — mektuptaki her sayı sonuç dosyalarıyla, her konum PDF'le eşleşiyor mu | Otomatik geçer |
| D6 | Depo + Zenodo yeni sürüm (v3.0.0), DOI makaleye | Doğrulayıcı 5/5 |

> **D5 yeni:** Yanıt mektubunda yanlış sayı veya yanlış sayfa referansı vermek
> ikinci turun en kolay sebebi. Mektubu da makale gibi otomatik denetleyeceğiz.

---

## 8. Ekipten gelecekler ve bağımlılık

| İstek | Kim | Ne zaman gerekli | Gelmezse |
|---|---|---|---|
| Markov geçiş olasılıkları | Ekip | FAZ C (C7) | Yazıda "belgelenemedi" demek zorunda kalırız — **kötü görünür** |
| Etiket kuralının ek koşulu | Ekip | FAZ C (C5) | Kısmi cevap veririz, ölçtüğüm 13/13 bulgusu yeter ama tam olmaz |
| Tip 6 neden yok | Ekip | FAZ A (A5) | "Veride yok" deriz, gerekçesiz |
| **3-4 veri varyantı** | Ekip | FAZ B sonu | 2.1 ve 3.5 yazıyla kapanır — **ikinci tur riski buradadır** |
| İkinci yol ağı | Grup | FAZ B sonu | Genelleme iddiası tamamen çıkar |
| STANet statüsü | Hoca | FAZ C (C2) | Tablo 1 eksik kalır |

> **Tek gerçek darboğaz veri varyantları.** Diğer dördü dakikalık iş ama
> gelmezlerse yanıt mektubunda "bu bilgiye erişemedik" yazmak zorunda kalırız,
> ki bu kötü bir izlenim bırakır.

---

## 9. Takip tablosu

Her faz bitiminde güncellenecek.

| Faz | Durum | Başlangıç | Bitiş | Çıktı |
|---|---|---|---|---|
| FAZ 0 | ⏳ | — | — | `HAKEM_TAKIP.md`, E3 kontrolü |
| FAZ A | ⏳ | — | — | Düzeltilmiş Tablo 13/14, veri kökeni tablosu |
| 🚦 Kapı 1 | ⏳ | — | — | Grup kararı: dwell + graf |
| FAZ B0 | ⏳ | — | — | Tasarım doğrulaması |
| FAZ B1-B6 | ⏳ | — | — | Ana sonuçlar |
| FAZ B7-B13 | ⏳ | — | — | Duyarlılık + TOST |
| 🚦 Kapı 2 | ⏳ | — | — | Hangi iddialar ayakta |
| FAZ C | ⏳ | — | — | Metin, tablo, figür |
| FAZ D | ⏳ | — | — | Düşman okuma + yanıt mektubu |
| 🚦 Kapı 3 | ⏳ | — | — | Gönderim onayı |

---

## 10. İlk adım

**FAZ 0.1 + 0.3** ile başlıyorum — madde izleme dosyası ve iddia denetimi.
Koşum yok, yarım gün. Sebebi: bunlar olmadan FAZ A'da ne kapattığımızı
takip edemeyiz ve eski iddialar metinde kalır.

Hemen ardından **A1 (eşik düzeltmesi)**.
