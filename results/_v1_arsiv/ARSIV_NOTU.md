# v1 sonuç arşivi — protokol karışımının kaydı

Bu klasör, makalenin **MDPI'ye gönderilen sürümündeki** ham sonuç dosyalarını
saklar. Amaç, v2 protokolüyle yeniden koşulan sonuçların neyi değiştirdiğini
denetlenebilir kılmaktır. **Bu dosyalar makalede artık kullanılmıyor.**

## Neden yeniden koşuldu

`stanet/utils.py` içindeki `set_seed`, 2 Ağustos 00:10'da değiştirildi:
`cudnn.deterministic` varsayılan olarak `True` iken `False` yapıldı
(gerekçe: dilated Conv1d geri yayılımında 627 ms/batch → 14 ms/batch,
43 kat yavaşlama; ayrıca `GATConv`'un `scatter_add`'i zaten deterministik
olmadığı için ayar vaat ettiği garantiyi sağlamıyordu).

Sonuç: deneyler **iki farklı protokolde** koşulmuş oldu.

| Dosya | Zaman | cuDNN det. | Seed |
|---|---|---|---|
| `F2_graph.json` | 01 Ağu 21:41 | **True** | 3 |
| `F3_fusion.json` | 01 Ağu 22:24 | **True** | 5 |
| `F4_benchmark.json` | 02 Ağu 01:20 | False | 3 |
| `F1b_fair.json` | 02 Ağu 01:24 | False | 3 (derin) / 5 (klasik) |
| `F5_cv.json` | 02 Ağu 01:44 | False | — |
| `F6_efficiency.json` | 02 Ağu 01:45 | False | — |

Bu yüzden `lstm + gat + weighted_sum` konfigürasyonu iki tabloda farklı
göründü — hakemin işaretlediği ana kusur:

- Tablo 9 / 11 (F1b, F4 `T:lstm`): **0,8836 ± 0,0033**
- Tablo 12 (F3 `weighted_sum`): **0,8484 ± 0,0138**

Ayrıca Tablo 15 (`F2 original/gated`, **0,8513 ± 0,0176**) yalnızca "Hybrid"
diye etiketlenmişti; oysa o satır `gated` füzyonuydu ve ayrıca farklı bir
model sınıfı (`models.FusionHybrid`, `hybrid.GenericHybrid` değil)
kullanıyordu — aynı seed'de farklı başlangıç ağırlıkları çekiliyordu.

## v1 F3 sonuçları (JSON kaybı — burada kayıt altına alınmıştır)

`F3_fusion.json` v1 dosyası, v2 duman testi sırasında üzerine yazıldı.
Değerleri `BULGULAR.md` §F3 ve `makale/main_revised.tex` Tablo 12'de
korunmaktadır; ham tahmin matrisi `F3_predictions.npz` olarak burada durur.

| Sıra | Füzyon | F1 (ort ± std) | ROC-AUC | Füzyon par. | Çökme |
|---|---|---|---|---|---|
| 1 | weighted_sum | 0,8484 ± 0,0138 | 0,9443 ± 0,0449 | 1 | 0/5 |
| 2 | cross_attention | 0,8438 ± 0,0099 | 0,9162 ± 0,0220 | 16.640 | 0/5 |
| 3 | concat | 0,8212 ± 0,0408 | 0,9093 ± 0,0557 | 0 | 0/5 |
| 4 | gated | 0,6811 ± 0,3366 | 0,8539 ± 0,1548 | 4.161 | 1/5 |

Seed bazında F1:

| Füzyon | s42 | s43 | s44 | s45 | s46 |
|---|---|---|---|---|---|
| weighted_sum | 0,8591 | 0,8339 | 0,8658 | 0,8464 | 0,8369 |
| cross_attention | 0,8502 | 0,8525 | 0,8429 | 0,8462 | 0,8274 |
| concat | 0,8310 | 0,8369 | 0,8491 | 0,8399 | 0,7491 |
| gated | 0,7623 | 0,8425 | 0,8481 | **0,0833** | 0,8693 |

v1 F2 (`original/*` = trip-instance graf, `merged/*` = osm-birleştirilmiş):

| Koşu | F1 (3 seed) |
|---|---|
| original/gat_only | 0,0756 |
| original/gated | 0,8513 |
| merged/gat_only | 0,0739 |
| merged/gated | 0,8141 |

## v2'de ne değişti

1. **Protokol damgası** (`stanet.utils.protocol_stamp`) — her sonuç JSON'una
   12 haneli özet yazılır. F1b, F4'ün damgası kendisininkiyle uyuşmuyorsa
   `AssertionError` verir; sessiz karışım artık mümkün değil.
2. **Paylaşılan koşu deposu** (`stanet.runstore`) — `lstm+gat+weighted_sum`
   bir kez eğitilir; F3 (`weighted_sum`), F4 (`T:lstm`), F1b (`hibrit`) ve
   F2 (`trip_instance`) aynı diskteki kaydı okur. Dört tablo bit-aynı sayı
   gösterir.
3. **Tek seed kümesi** — her yerde 42–46 (v1'de F2/F4 3, F3/F1b 5 idi).
4. **Tek model sınıfı** — F2 artık `GenericHybrid` kullanıyor.
5. **Tablo 15 füzyon sütunu** — her iki füzyon her iki grafta koşuluyor.
6. **Tablo 10 düzeltmesi** — `mean_delta` (çok-seed ortalama farkı) ile
   `bootstrap_diff` (medyan seed farkı) artık ayrı sütunlar; JSON ayrıca
   `ci_contains_median_delta` bayrağını taşır.

v2 protokol özeti: `0902cd97ec17`
