"""
Hocaya gidecek teslim paketini kurar ve zip'ler.

ICERIK
    tablolar_csv/   39 tablo CSV + koken listesi
    figurler/       makalede kullanilan 10 figur
    sonuclar_json/  43 sonuc dosyasi (her sayinin kaynagi)
    *.docx          revizyon raporu, makale, yanit mektubu
    OKUBENI.txt     klasor haritasi + yeniden uretme komutlari

NE KONMUYOR
    results/_runs/  1.280 onbelleklenmis kosu kaydi -- boyut nedeniyle.
                    Tamami GitHub'da; OKUBENI.txt adresi veriyor.

KULLANIM
    python makale/_build_teslim_zip.py
"""
import datetime as dt
import pathlib
import shutil
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
OUT = ROOT / "teslim"
REPO = "https://github.com/keremkartal/Anomaly-Detection"

OKUBENI = """sensors-4582518 — REVIZYON TESLIM PAKETI
{tarih}

Bu paket, revizyonda uretilen butun tablo, figur ve sonuc dosyalarini
icerir. Amac: makaledeki her sayinin nereden geldigini tek tek
izlenebilir kilmak.

ONCE BUNU OKUYUN
----------------
  REVIZYON_RAPORU.docx
      Ne yapildi, veriler nereden geldi, makalenin neresi nasil
      guncellendi, geri alinan iddialar, acik kalan konular.

KLASORLER
---------
  tablolar_csv/
      Makaledeki 39 tablonun CSV hali. Her dosyanin ilk satirinda tablo
      numarasi ve basligi var.

      >> 00_TABLO_LISTESI.csv <<
      Her tablo icin: numara, etiket, baslik, CSV adi, satir sayisi, VE
      hangi sonuc dosyasindan beslendigi (kapsama yuzdesiyle). Bazi
      tablolar birden fazla deneyden beslenir, liste bunu gosterir.

  figurler/
      Makalede kullanilan 10 figur (PNG). Hepsi okudugu JSON'dan sonra
      uretildi; bunu `_verify_numbers.py` D kontrolu denetliyor.

  sonuclar_json/
      43 sonuc dosyasi. Makaledeki 204 sayinin tamami bunlardan birinde
      geciyor. Her dosya bir PROTOKOL DAMGASI tasir: tohum listesi, epoch
      butcesi, patience, batch, ogrenme orani, scheduler, dropout,
      checkpoint olcutu, esik kurali, cuDNN bayraklari, pencere, adim ve
      kutuphane surumleri uzerinden hash.

BELGELER
--------
  main_revised_MDPI.docx
      Makale, MDPI duzeninde Word. Bolumler ondalik numarali (1, 2.1,
      4.2.1), 41 tablo ve 10 figur numarali, appendix'ler Appendix A-C.

  Response_to_Reviewers_sensors-4582518_DOLDURULMUS.docx
      Yanit mektubu, sizin sablonunuzda. 26 maddenin hepsi cevapli.
      EKLENEN HER SEY MAVI; siyah kalan kisimlar sablonun kendisi.

ZIP'TE OLMAYAN
--------------
  results/_runs/ altindaki 1.280 onbelleklenmis egitim kosusu (boyut).
  Tamami depoda:
      {repo}

HER SEYI YENIDEN URETMEK
------------------------
  python makale/_verify_numbers.py        14 kontrol (204/204 sayi)
  python makale/_verify_letter.py         mektup (26/26 madde)
  python makale/_tables_to_csv.py         tablo CSV'leri + koken listesi
  python makale/_to_mdpi_word.py          makale -> MDPI Word
  python makale/_fill_letter_template.py  sablonu doldur
"""


def main():
    OUT.mkdir(exist_ok=True)
    stats = {}

    # --- figurler
    figdir = OUT / "figurler"
    figdir.mkdir(exist_ok=True)
    src = ROOT / "results" / "figures"
    n = 0
    for p in sorted(src.glob("*.png")):
        shutil.copy2(p, figdir / p.name)
        n += 1
    stats["figur"] = n

    # --- sonuc json
    jdir = OUT / "sonuclar_json"
    jdir.mkdir(exist_ok=True)
    n = 0
    for p in sorted((ROOT / "results").glob("*.json")):
        shutil.copy2(p, jdir / p.name)
        n += 1
    stats["sonuc"] = n

    # --- belgeler
    n = 0
    for name in ("main_revised_MDPI.docx",
                 "Response_to_Reviewers_sensors-4582518_DOLDURULMUS_v4.docx"):
        p = ROOT / name
        if p.exists():
            tgt = name.replace("_v4", "")
            shutil.copy2(p, OUT / tgt)
            n += 1
    stats["belge"] = n

    # --- okubeni
    (OUT / "OKUBENI.txt").write_text(
        OKUBENI.format(tarih=dt.date.today().strftime("%d.%m.%Y"), repo=REPO),
        encoding="utf-8")

    csvdir = OUT / "tablolar_csv"
    stats["csv"] = len(list(csvdir.glob("*.csv"))) if csvdir.exists() else 0

    # --- zip
    zpath = ROOT / ("sensors-4582518_revizyon_%s.zip"
                    % dt.date.today().strftime("%Y%m%d"))
    skip = {".md"}
    total = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(OUT.rglob("*")):
            if not p.is_file() or p.suffix.lower() in skip:
                continue
            z.write(p, p.relative_to(OUT))
            total += 1

    mb = zpath.stat().st_size / 1048576
    print("zip: %s  (%.1f MB, %d dosya)\n" % (zpath.name, mb, total))
    for k, v in (("tablo CSV", stats["csv"]), ("figur", stats["figur"]),
                 ("sonuc JSON", stats["sonuc"]), ("belge (docx)", stats["belge"])):
        print("  %-14s %d" % (k, v))
    if stats["belge"] < 2:
        print("\n  UYARI: belgelerden biri bulunamadi")


if __name__ == "__main__":
    main()
