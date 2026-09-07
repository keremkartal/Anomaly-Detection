"""Orijinal PDF'ten cikarilan kaynakcayi atif anahtarlariyla eslestirir
ve revize makale icin yeniden numaralandirir."""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
here = Path(__file__).parent

# ---------------------------------------------------------------- 1) kaynaklari ayristir
raw = (here / "_extracted" / "references_raw.txt").read_text(encoding="utf-8")
raw = raw[raw.find("[1]"):]
parts = re.split(r"\n?\[(\d+)\]\s*", raw)
entries = {}
for i in range(1, len(parts) - 1, 2):
    num = int(parts[i])
    body = re.sub(r"\s+", " ", parts[i + 1]).strip()
    entries[num] = body
print(f"ayristirilan kaynak: {len(entries)} (1..{max(entries)})")

# ---------------------------------------------------------------- 2) orijinal atif sirasi
orig = (here / "main.tex").read_text(encoding="utf-8")
body_start = orig.find(r"\begin{document}")
orig_body = orig[body_start:]

CITE = re.compile(r"\\(?:no)?cite\{([^}]+)\}")
order, seen = [], set()
for m in CITE.finditer(orig_body):
    for k in m.group(1).split(","):
        k = k.strip()
        if k and k not in seen:
            seen.add(k)
            order.append(k)
print(f"orijinal atif sirasi: {len(order)} anahtar")

key2num = {k: i + 1 for i, k in enumerate(order)}
key2ref = {k: entries.get(key2num[k], "") for k in order}
missing = [k for k, v in key2ref.items() if not v]
print(f"kaynak metni bulunamayan: {missing or 'yok'}")

# dogrulama: birkac anahtar-kaynak eslesmesi mantikli mi?
print("\n--- eslesme kontrolu (ilk 5) ---")
for k in order[:5]:
    print(f"  [{key2num[k]:2d}] {k:32s} {key2ref[k][:60]}...")

# ---------------------------------------------------------------- 3) revize atif sirasi
rev = (here / "main_revised.tex").read_text(encoding="utf-8")
rev_body = rev[rev.find(r"\begin{document}"):]
new_order, seen2 = [], set()
for m in CITE.finditer(rev_body):
    for k in m.group(1).split(","):
        k = k.strip()
        if k and k not in seen2:
            seen2.add(k)
            new_order.append(k)
print(f"\nrevize atif sirasi: {len(new_order)} anahtar")

GCDA = ("“Digital twin-assisted graph contrastive domain adaptation "
        "for small-sample bearing fault diagnosis,” 2025. "
        "[Authors, venue, volume, pages and DOI must be verified from the "
        "publisher record before submission.]")

bib = []
for i, k in enumerate(new_order, 1):
    text = key2ref.get(k) or (GCDA if k == "gcda_bearing" else None)
    if text is None:
        print(f"  UYARI: {k} icin kaynak metni yok")
        text = f"[{k} — kaynak metni bulunamadi]"
    bib.append({"n": i, "key": k, "text": text})

out = {"new_order": new_order, "bibliography": bib,
       "dropped": [k for k in order if k not in seen2]}
(here / "_extracted" / "bibliography.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\ncikarilan kaynaklar: {out['dropped']}")
print(f"revize kaynakca: {len(bib)} girdi -> _extracted/bibliography.json")
