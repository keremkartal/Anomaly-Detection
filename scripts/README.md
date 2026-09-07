# Koşu betikleri

v2 revizyonunda kullanılan iki sürücü. Sıra önemli: `run_v2.sh` paylaşılan
koşuları oluşturur, `run_v2b.sh` onları kullanır.

| Betik | İçerik | Süre |
|---|---|---|
| `run_v2.sh` | F3 → F4 → F2 → F1b (80 yeni eğitim) | ~3 sa 15 dk |
| `run_v2b.sh` | F5 → F7 → F6 → F9 (50 yeni eğitim + figürler) | ~1 sa 45 dk |

```bash
bash scripts/run_v2.sh  > logs/v2_faz2.log 2>&1
bash scripts/run_v2b.sh > logs/v2_faz4.log 2>&1
python makale/_verify_numbers.py
```

Koşular `results/_runs/<protokol_hash>/` altında önbelleğe alındığı için
yeniden çalıştırmak eğitimi tekrarlamaz; yalnızca eksik olanlar eğitilir.
Protokolde herhangi bir alan değişirse özet değişir, önbellek geçersizleşir
ve ilgili koşular yeniden eğitilir.
