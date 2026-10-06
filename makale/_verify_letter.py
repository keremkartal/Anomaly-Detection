"""
Yanit mektubu dogrulayicisi.

NEDEN
-----
Ikinci tur revizyonun en ucuz sebebi yanit mektubunda YANLIS SAYI ya da
YANLIS KONUM vermektir. Hakem "Tablo 13'e bakin" der, bakar, orada o sayiyi
bulamaz; guven biter ve butun mektup sorgulanir. Makaleyi otomatik
denetliyoruz, mektubu da denetlemeliyiz.

NE KONTROL EDILIYOR
-------------------
  1. KAPSAM      — 25 maddenin hepsi mektupta var mi
  2. SAYI        — mektuptaki her sayi bir sonuc dosyasinda geciyor mu
  3. KONUM       — "Table 13", "Figure 4", "Section V-B", "page 12" gibi her
                   gonderme derlenmis makalede gercekten var mi
  4. DURUM       — her madde ne yapildigini soyluyor mu (bos madde kalmasin)

KONUM DOGRULAMASI NASIL
-----------------------
`main_revised.aux` her etiket icin numarayi, sayfayi ve turu tutar
(`\\newlabel{tab:cv}{{13}{15}{...}{table.13}{}}`). Mektuptaki "Table 13"
ifadesi, aux'ta 13 numarali bir tablo varsa gecerlidir. Boylece PDF'i
ayristirmaya gerek kalmaz ve kontrol derlemeyle her zaman senkron kalir.

KULLANIM
  python makale/_verify_letter.py [mektup.md]
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

AUX = HERE / "main_revised.aux"
RESULTS = ROOT / "results"
DEFAULT_LETTER = HERE.parent / "HAKEM_YANIT.md"

OK, FAIL = "  [OK]  ", "  [HATA]"
failures = []

# Madde listesi yerel takip dosyasindan okunur. Listeyi buraya gomulu
# tutmuyoruz: degerlendirme surecinin yapisi (kac hakem, kac madde) yayina
# kadar gizlidir ve bu betik herkese acik depoda duruyor. Takip dosyasi
# yoksa kapsam kontrolu atlanir, digerleri calisir.
TRACKER = ROOT / "HAKEM_TAKIP.md"


def load_items():
    if not TRACKER.exists():
        return []
    txt = TRACKER.read_text(encoding="utf-8", errors="replace")
    return sorted({m.group(1) for m in
                   re.finditer(r"^\|\s*(\d+\.\d+)\s*\|", txt, re.M)},
                  key=lambda s: tuple(int(x) for x in s.split(".")))


ITEMS = load_items()


# ------------------------------------------------------------------ kaynaklar
def load_labels():
    """aux -> {etiket: (numara, sayfa, tur)} ve {tur: {numara}}"""
    if not AUX.exists():
        return {}, {}
    txt = AUX.read_text(encoding="utf-8", errors="replace")
    labels, by_kind = {}, {}
    # Basliklar suslu parantez icerebildigi icin butun girdiyi tek bir duzenli
    # ifadeyle yakalamak kirilgan: o yuzden once ad, numara ve sayfa aliniyor,
    # sonra AYNI girdinin govdesinde tur adi araniyor.
    KINDS = ("subsubsection", "subsection", "section", "table", "figure",
             "equation")
    # Bolum numaralari `{\mbox  {IV-D}}` ya da `{\mbox  {V-A}2}` biciminde
    # sarmalanmis geliyor; tablo/figur numaralari duz. Iki bicimi de aliyoruz.
    num_pat = (r"(?:\{\{\\mbox\s*\{([^{}]*)\}([^{}]*)\}"     # {{\mbox {V-A}2}
               r"|\{\{([^{}]*)\})")                          # {{13}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}" + num_pat + r"\{(\d+)\}", txt):
        name = m.group(1)
        num = ((m.group(2) or "") + (m.group(3) or "")) or (m.group(4) or "")
        page = m.group(5)
        if not num:
            continue
        # Girdinin sonu SATIR sonudur. Bir sonraki \newlabel'a kadar okumak
        # yanlis: aradaki \@writefile satirlari da "{5}{table.3}" gibi
        # capalar icerir ve tur yanlis okunur.
        eol = txt.find("\n", m.end())
        body = txt[m.end(): eol if eol > 0 else len(txt)]
        kind = next((k for k in KINDS if (k + ".") in body), None)
        if kind is None:
            continue
        if kind.endswith("section"):
            kind = "section"
        labels[name] = (num, int(page), kind)
        by_kind.setdefault(kind, set()).add(num)

    # Etiketsiz bolumler de gondermeye konu olabilir. Icindekiler kayitlari
    # (`\contentsline`) etiketten bagimsiz olarak HER bolumun numarasini
    # tutar, o yuzden onlari da topluyoruz; ayrica "V-B" varsa "V" de
    # gecerli bir gondermedir.
    for m in re.finditer(r"\\contentsline\s*\{(?:sub)*section\}"
                         r"\{\\numberline\s*\{([^{}]*)\}", txt):
        num = re.sub(r"\\mbox\s*", "", m.group(1)).strip()
        if num:
            by_kind.setdefault("section", set()).add(num)
    for num in list(by_kind.get("section", ())):
        head = num.split("-")[0]
        if head:
            by_kind["section"].add(head)
    return labels, by_kind


def collect_numbers():
    """Sonuc dosyalarindaki her sayiyi 4 ve 2 ondalikli metin olarak topla."""
    found = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            return
        elif isinstance(o, (int, float)):
            f = float(o)
            found.add("%.4f" % f)
            found.add("%.3f" % f)
            found.add("%.2f" % f)
            found.add("%.1f" % f)
            if abs(f - round(f)) < 1e-9:
                found.add("%d" % round(f))
                found.add("{:,}".format(int(round(f))))
            # yuzde olarak da anilabilir
            found.add("%.1f" % (f * 100))
            found.add("%.2f" % (f * 100))

    for p in sorted(RESULTS.glob("*.json")):
        try:
            walk(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return found


# ------------------------------------------------------------------ kontroller
def check_coverage(letter):
    print("1. KAPSAM")
    if not ITEMS:
        print("    yerel takip dosyasi yok, kapsam kontrolu atlandi\n")
        return
    missing = [i for i in ITEMS if not re.search(r"(?<![\d.])" + re.escape(i)
                                                 + r"(?![\d])", letter)]
    print(f"    mektupta gecen madde: {len(ITEMS) - len(missing)}/{len(ITEMS)}")
    if missing:
        print(f"{FAIL} eksik madde: {', '.join(missing)}")
        failures.append(f"{len(missing)} madde mektupta yok")
    else:
        print(f"{OK} {len(ITEMS)} maddenin hepsi mektupta")
    print()


def check_locations(letter, labels, by_kind):
    print("2. KONUM GONDERMELERI (aux ile)")
    kinds = {"table": "table", "tablo": "table", "figure": "figure",
             "sekil": "figure", "şekil": "figure", "equation": "equation",
             "eq": "equation", "denklem": "equation"}
    pat = re.compile(r"\b(Table|Tablo|Figure|Sekil|Şekil|Equation|Eq\.?|Denklem)"
                     r"[~\s]*(\d+)", re.IGNORECASE)
    # Bolum gondermeleri IEEEtran'da roman: "Section V-B", "Bolum VII-C"
    sec_pat = re.compile(r"\b(?:Section|Sect\.?|Bolum|Bölüm)[~\s]*"
                         r"([IVXL]+(?:-[A-Z]\d*)?)", re.IGNORECASE)
    bad, good = [], 0
    # Mektup numara yerine etiket tasiyor: {tab:cv}. Etiketin aux'ta
    # var olmasi, numaranin dogru nesneyi gosterdigini sorgulamayi
    # gereksiz kilar -- etiket nesnenin kendisi.
    for m in re.finditer(r"\{((?:tab|fig|eq):[a-z_0-9]+)\}", letter):
        lbl = m.group(1)
        if lbl in labels:
            good += 1
        else:
            bad.append((f"{{{lbl}}}", f"etiket {lbl} aux'ta yok"))
    for m in pat.finditer(letter):
        word, num = m.group(1).lower().rstrip("."), m.group(2)
        kind = kinds.get(word)
        if kind is None:
            continue
        if num in by_kind.get(kind, set()):
            good += 1
        else:
            bad.append((m.group(0), kind, num))
    for m in sec_pat.finditer(letter):
        num = m.group(1).upper()
        if num in by_kind.get("section", set()):
            good += 1
        else:
            bad.append((m.group(0), "section", num))
    # sayfa gondermeleri
    pages = {p for (_, p, _) in labels.values()}
    maxpage = max(pages) if pages else 0
    for m in re.finditer(r"\b(?:page|sayfa)[~\s]*(\d+)", letter, re.IGNORECASE):
        if maxpage and int(m.group(1)) > maxpage:
            bad.append((m.group(0), "page", m.group(1)))
        else:
            good += 1
    print(f"    dogrulanan gonderme: {good}")
    if bad:
        print(f"{FAIL} {len(bad)} gonderme makalede yok:")
        for txt, kind, num in bad:
            print(f"      \"{txt}\"  ({kind} {num} bulunamadi)")
        failures.append(f"{len(bad)} konum gondermesi gecersiz")
    else:
        print(f"{OK} butun gondermeler makalede karsiligini buluyor")
    print()


def check_numbers(letter, known):
    print("3. SAYILAR (sonuc dosyalarindan geliyor mu)")
    # madde numaralarini ve bolum numaralarini sayi sanmayalim
    cleaned = re.sub(r"(?<![\d.])[123]\.\d{1,2}(?![\d])", " ", letter)
    cleaned = re.sub(r"\b(?:Table|Tablo|Figure|Sekil|Şekil|Equation|Eq\.?|"
                     r"Denklem|page|sayfa|Section|Bolum|Bölüm)[~\s]*[\dIVXA-Z-]+",
                     " ", cleaned, flags=re.IGNORECASE)
    nums = re.findall(r"(?<![\w.])(\d+\.\d{2,4})(?![\w])", cleaned)
    miss = [n for n in set(nums) if n not in known]
    print(f"    mektupta sayi: {len(set(nums))}   eslesen: {len(set(nums)) - len(miss)}")
    if miss:
        print(f"{FAIL} {len(miss)} sayi hicbir sonuc dosyasinda yok:")
        for n in sorted(miss):
            ctx = ""
            m = re.search(r".{0,60}" + re.escape(n) + r".{0,60}", letter)
            if m:
                ctx = m.group(0).replace("\n", " ").strip()
            print(f"      {n}   ...{ctx}...")
        failures.append(f"{len(miss)} sayi sonuc dosyalariyla eslesmiyor")
    else:
        print(f"{OK} her sayi bir sonuc dosyasindan geliyor")
    print()


def check_answers(letter):
    """Her maddenin altinda en az bir konum gondermesi olmali."""
    print("4. HER MADDE BIR YERE ISARET EDIYOR MU")
    blocks, cur, buf = {}, None, []
    for line in letter.split("\n"):
        m = re.match(r"\s*#{1,6}\s*(?:Madde\s*)?([123]\.\d{1,2})\b", line)
        if m:
            if cur:
                blocks[cur] = "\n".join(buf)
            cur, buf = m.group(1), []
        elif cur:
            buf.append(line)
    if cur:
        blocks[cur] = "\n".join(buf)
    loc = re.compile(r"\b(Table|Tablo|Figure|Sekil|Şekil|Section|Bolum|Bölüm|"
                     r"Equation|Denklem|page|sayfa|Appendix)\b"
                     r"|\{(?:tab|fig|eq):[a-z_0-9]+\}", re.IGNORECASE)
    empty = [k for k, v in blocks.items() if not loc.search(v)]
    print(f"    baslikli madde blogu: {len(blocks)}")
    if not blocks:
        print("    (mektup henuz madde basliklari icermiyor, atlandi)")
    elif empty:
        print(f"{FAIL} {len(empty)} madde konum vermiyor: {', '.join(sorted(empty))}")
        failures.append(f"{len(empty)} madde kanit konumu vermiyor")
    else:
        print(f"{OK} her madde makalede bir yere isaret ediyor")
    print()


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LETTER
    print("=" * 78)
    print(f"YANIT MEKTUBU DOGRULAMASI — {path.name}")
    print("=" * 78 + "\n")
    if not path.exists():
        print(f"mektup bulunamadi: {path}")
        print("(FAZ D'de yazilacak; dogrulayici hazir)")
        return 0
    letter = path.read_text(encoding="utf-8", errors="replace")
    labels, by_kind = load_labels()
    if not labels:
        print("UYARI: main_revised.aux yok, once makaleyi derleyin\n")
    print(f"aux'tan okunan etiket: {len(labels)}   "
          f"turler: {', '.join(sorted(by_kind))}\n")

    check_coverage(letter)
    check_locations(letter, labels, by_kind)
    check_numbers(letter, collect_numbers())
    check_answers(letter)

    print("=" * 78)
    if failures:
        print(f"SONUC: {len(failures)} KONTROL BASARISIZ")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SONUC: TUM KONTROLLER GECTI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
