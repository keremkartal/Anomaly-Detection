"""Revize .tex icin capraz referans, atif ve figur kontrolu."""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
here = Path(__file__).parent
new = (here / "main_revised.tex").read_text(encoding="utf-8")
old = (here / "main.tex").read_text(encoding="utf-8")

LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\ref\{([^}]+)\}")
CITE = re.compile(r"\\(?:no)?cite\{([^}]+)\}")
FIG = re.compile(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}")


def keys(text):
    return {k.strip() for grp in CITE.findall(text) for k in grp.split(",")}


labels, refs = set(LABEL.findall(new)), set(REF.findall(new))
print("tanimsiz \\ref :", sorted(refs - labels) or "YOK - hepsi tanimli")
print("kullanilmayan label:", sorted(labels - refs) or "yok")

nk, ok = keys(new), keys(old)
print()
print(f"atif anahtari: eski {len(ok)} -> yeni {len(nk)}")
print("  yeni eklenen :", sorted(nk - ok) or "yok")
print("  cikarilan    :", sorted(ok - nk) or "yok")

print()
print("figurler:")
for f in sorted(set(FIG.findall(new))):
    print(("  OK    " if (here / f).exists() else "  EKSIK ") + f)

print()
print(f"satir: eski {len(old.splitlines())} -> yeni {len(new.splitlines())}")
# kaba denge kontrolu
for env in ("table", "table*", "figure", "figure*", "equation", "align", "itemize", "tabular", "tabularx"):
    b = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", new))
    e = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", new))
    if b != e:
        print(f"  DENGESIZ {env}: begin={b} end={e}")
print("ortam denge kontrolu tamam")
print("suslu parantez dengesi:", new.count("{") - new.count("}"), "(0 olmali)")
