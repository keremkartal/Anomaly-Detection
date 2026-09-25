# Hakem maddeleri — tek kaynak takip tablosu

**25 madde.** Her maddenin cevap biçimi **önceden** sabitlendi; bu, "hakem
deney istedi biz yazıyla geçiştirdik" hatasını engelliyor.

**Durum kodları:** ⏳ bekliyor · 🔄 sürüyor · ✅ kapandı · 🚫 bloke

**Biçim:** `D` deney (yeni koşum) · `A` analiz (mevcut veriden) · `Y` yazı

> **Kapatma kuralı:** Bir madde ancak *"hakem bunu okuyup tekrar açamaz"*
> dediğimde ✅ olur. Kanıt konumu boşsa kapanmış sayılmaz.

---

## Hakem 1 — 13 madde

| # | Madde (özet) | Biçim | Faz | Sahip | Durum | Kanıt konumu |
|---|---|---|---|---|---|---|
| 1.1 | Gerçek veri / ikinci ağ ya da gelecek çalışma | Y | C7 | Kerem + grup | ⏳ | |
| 1.2 | 14 trip vurgusu, etki boyutu, trip bazlı dağılım | A | A3 | Kerem | ⏳ | |
| 1.3 | İki graf için sistematik duyarlılık | D | B5 | Kerem | ⏳ | |
| 1.4 | Pencere düzeyi p değerlerini belirgin etiketle | Y | A6 | Kerem | ⏳ | |
| 1.5 | Kinematik türetilmiş etiketlerin etkisi | Y | C5 | Kerem | ⏳ | |
| 1.6 | Markov geçiş olasılıkları | Y | C7 | **Ekip** | 🚫 | |
| 1.7 | Pencere/stride duyarlılığı | D | B7 | Kerem | ⏳ | |
| 1.8 | Dört füzyon: parametre + eğitim + çıkarım süresi | A | A4 | Kerem | ⏳ | |
| 1.9 | Gate çökme tanıları (eğri, dağılım) | D | B12 | Kerem | ⏳ | |
| 1.10 | Betimsel dil tutarlılığı | Y | C3 | Kerem | ⏳ | |
| 1.11 | RF 0,0014 farkı — neden karmaşık model | Y | C4 | Kerem | ⏳ | |
| 1.12 | İmputasyon duyarlılığı | D | B8 | Kerem | ⏳ | |
| 1.13 | Tablolarda düzey etiketleri | Y | A6 | Kerem | ⏳ | |

## Hakem 2 — 4 madde

| # | Madde (özet) | Biçim | Faz | Sahip | Durum | Kanıt konumu |
|---|---|---|---|---|---|---|
| 2.1 | Akış/gürültü/anomali şiddeti tartışması | D+Y | B9·B10·C7 | Kerem (+ekip güçlendirir) | ⏳ | |
| 2.2 | Graf test bilgisine erişiyor mu; görülmemiş ağ | D+Y | A8·B6 | Kerem | ⏳ | |
| 2.3 | Baseline arama uzayı, durdurma, init, kendi eşiği | D | B11 | Kerem | ⏳ | |
| 2.4 | Figür yazı boyutu | Y | C1 | Kerem | ⏳ | |

## Hakem 3 — 8 madde

| # | Madde (özet) | Biçim | Faz | Sahip | Durum | Kanıt konumu |
|---|---|---|---|---|---|---|
| 3.1 | Tablo 1 değerlendirme protokolleri; STANet statüsü | Y | C2 | Kerem + **hoca** | 🚫 | |
| 3.2 | 🔴 **Fold–seed eşik düzeltmesi** | D | A1 | Kerem | 🔄 | |
| 3.3 | 🔴 Tekrarlı bölümleme, NB düzeltmesi | D | B1 | Kerem | ⏳ | |
| 3.4 | 🔴 Graf seçimi, kenar kuralı, grafsız kontroller | D+Y | A2·A8·B2 | Kerem | ⏳ | |
| 3.5 | Bağımsız veri kümeleri; imputasyon–etiket | D | B8·B10 | Kerem + **ekip** | 🚫 kısmi | |
| 3.6 | Klasik modeller + kural CV'ye | D | B3 | Kerem | ⏳ | |
| 3.7 | Eq 15 ↔ Tablo 5; gate tanıları; seed 45 CV | D+Y | A7·B4·B12 | Kerem | ⏳ | |
| 3.8 | Alt tipler; precision/recall/FP; eşdeğerlik testi | A+D | A5·B13 | Kerem | ⏳ | |

---

## Bloke maddeler — ne bekliyoruz

| # | Kimden | Ne | Gelmezse ne yazacağız |
|---|---|---|---|
| 1.6 | Ekip | Markov geçiş olasılıklarının seçimi/kalibrasyonu | *"Üretim parametreleri bu çalışmada belgelenemedi"* — **kötü görünür** |
| 3.1 | Hoca | STANet yayın durumu | Tablo 1'de satır eksik kalır |
| 3.5 | Ekip | 3–4 bağımsız veri varyantı | Perturbasyon analizleriyle (B9, B10) kısmi cevap; **ikinci tur riski burada** |

Ek olarak ekipten istenen ama bloke etmeyen:
- Etiket kuralının **ek koşulu** (128 pencere limiti aşıyor ama tip-1 değil)
- Tip 6'nın neden hiç üretilmediği

---

## İlerleme

| | Sayı |
|---|---|
| ✅ Kapandı | 0 / 25 |
| 🔄 Sürüyor | 1 |
| ⏳ Bekliyor | 21 |
| 🚫 Bloke | 3 |
