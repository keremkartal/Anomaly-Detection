"""
main_revised.tex -> MDPI Sensors duzeninde Word (.docx).

NICIN
-----
Hoca MDPI sablonunu istedi. Gonderilen MDPI surumunun (`main (6).pdf`,
27 sayfa) LaTeX kaynagi elimizde YOK -- yalnizca PDF var; bunu
`makale/MDPI_ESLEME.md` de soyluyor ve dogru yolun revize kaynaktan
yeniden uretmek oldugunu yaziyor. Bu script onu yapar.

UC SORUN, UC COZUM
------------------
1. `\\resizebox{\\columnwidth}{!}{\\begin{tabular}...}` -- pandoc bu sarmali
   okumuyor ve 21 tablonun tamamini dusuruyor. Sarmal cikarilir, ciplak
   tabular birakilir.

2. `\\ref{...}` cozulmuyor; pandoc "Table [tab:cv]" yaziyor. Butun
   gondermeler ONCEDEN cozulur. Tablo/figur numarasi aux'tan; BOLUM
   numarasi ise MDPI'nin ondalik duzenine gore (1, 2.1, 2.1.1) yeniden
   hesaplanir -- IEEEtran'in roman numaralari (V-B) Word surumunde
   anlamsiz olurdu.

3. MDPI bolum basliklarini metinde numarali yazar. Pandoc numara uretmez,
   bu yuzden basliklara ondalik numara eklenir.

KULLANIM
    python makale/_to_mdpi_word.py [--out YOL]
"""
import argparse
import pathlib
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
TEX = HERE / "main_revised.tex"
AUX = HERE / "main_revised.aux"
DEFAULT_OUT = ROOT / "main_revised_MDPI.docx"

# ---------------------------------------------------------------- numaralar

def float_numbers():
    """Tablo/figur/denklem etiketi -> numara (aux'tan)."""
    if not AUX.exists():
        return {}
    aux = AUX.read_text(encoding="utf-8", errors="ignore")
    pat = (r"newlabel\{((?:tab|fig|eq):[a-z_0-9]+)\}\{\{(?:"
           + "\\\\" + r"mbox\s*\{)?([0-9]+)")
    return {m.group(1): m.group(2) for m in re.finditer(pat, aux)}


def section_numbers(tex):
    """Bolum etiketi -> MDPI ondalik numarasi, belge sirasina gore."""
    out, c = {}, [0, 0, 0]
    pat = re.compile(r"\\(section|subsection|subsubsection)\*?\{"
                     r"|\\label\{(sec:[a-z_0-9]+)\}")
    cur = None
    for m in pat.finditer(tex):
        if m.group(1):
            lvl = {"section": 0, "subsection": 1, "subsubsection": 2}[m.group(1)]
            c[lvl] += 1
            for k in range(lvl + 1, 3):
                c[k] = 0
            cur = ".".join(str(x) for x in c[:lvl + 1] if True)[:]
            cur = ".".join(str(c[k]) for k in range(lvl + 1))
        elif m.group(2) and cur:
            out[m.group(2)] = cur
    return out


# ---------------------------------------------------------------- temizlik

def strip_resizebox(tex):
    """\\resizebox{..}{..}{ ... } sarmalini kaldirir, icerigi birakir."""
    out, i, n = [], 0, 0
    while True:
        m = re.compile(r"\\resizebox\{[^}]*\}\{[^}]*\}\{%?\s*").search(tex, i)
        if not m:
            out.append(tex[i:])
            break
        out.append(tex[i:m.start()])
        j, depth = m.end(), 1
        while j < len(tex) and depth:
            if tex[j] == "{":
                depth += 1
            elif tex[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(tex[m.end():j])
        i = j + 1
        n += 1
    return "".join(out), n


def resolve_refs(tex, floats, secs):
    """Butun \\ref ve \\eqref cagrilarini duz metne cevirir."""
    miss = set()

    def rep(m):
        lbl = m.group(1)
        if lbl.startswith("sec:"):
            v = secs.get(lbl)
            if v:
                return v
        else:
            v = floats.get(lbl)
            if v:
                return v
        miss.add(lbl)
        return "??"

    tex = re.sub(r"\\eqref\{([^}]+)\}", lambda m: "(" + rep(m) + ")", tex)
    tex = re.sub(r"\\ref\{([^}]+)\}", rep, tex)
    return tex, sorted(miss)


def bibliography():
    """bibliography.tex -> (anahtar->numara, [girdi metinleri])

    Kaynakca elle yazilmis bir thebibliography ortami ve .bib dosyasi yok.
    Pandoc bu durumda \\cite cagrilarini SESSIZCE dusuruyor (36 atifin
    tamami kayboldu) ve 35 girdiyi tek paragrafa cokertiyor. O yuzden
    numaralandirmayi burada yapiyoruz: atif numarasi, girdinin
    bibliography.tex icindeki sirasidir.
    """
    p = HERE / "bibliography.tex"
    if not p.exists():
        return {}, []
    raw = p.read_text(encoding="utf-8")
    items = re.findall(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem\{|\\end\{thebibliography\})",
                       raw, flags=re.S)
    keys, texts = {}, []
    for i, (k, body) in enumerate(items, 1):
        keys[k] = str(i)
        texts.append(" ".join(body.split()))
    return keys, texts


def resolve_cites(tex, keys):
    """\\cite{a,b} -> [1, 7]. Cozulemeyen anahtari rapor eder."""
    miss = set()

    def rep(m):
        out = []
        for k in (x.strip() for x in m.group(1).split(",")):
            n = keys.get(k)
            if n:
                out.append(n)
            else:
                miss.add(k)
        return "[" + ", ".join(out) + "]" if out else ""

    tex = re.sub(r"\\cite\{([^}]+)\}", rep, tex)
    return tex, sorted(miss)


def preprocess(tex, floats, secs):
    notes = {}

    # IEEEtran'a ozgu olanlar
    for pat in (r"\\IEEEpeerreviewmaketitle", r"\\IEEEtablecaptionsepspace",
                r"\\IEEEfigurecaptionsepspace", r"\\IEEEtablestring\{[^}]*\}"):
        tex = re.sub(pat, "", tex)
    tex = re.sub(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}",
                 r"\n\n\\textbf{Keywords:} \1\n\n", tex, flags=re.S)

    tex, notes["resizebox"] = strip_resizebox(tex)
    tex, notes["unresolved"] = resolve_refs(tex, floats, secs)

    # Atiflar ve kaynakca: pandoc elle yazilmis thebibliography'yi
    # beceremiyor, o yuzden ikisini de kendimiz uretiyoruz.
    keys, entries = bibliography()
    tex, notes["cites"] = resolve_cites(tex, keys)
    notes["n_cites"] = len(keys)
    notes["n_entries"] = len(entries)
    refs = "\n\n".join("%d. %s" % (i, t) for i, t in enumerate(entries, 1))
    tex = re.sub(r"\\input\{bibliography\.tex\}",
                 "\\\\section*{References}\n\n" + refs.replace("\\", "\\\\"),
                 tex, count=1)

    # \bm -> \mathbf (pandoc \bm bilmiyor)
    tex = tex.replace("\\bm{", "\\mathbf{")
    # tablo/figur etiketlerini artik gerek yok
    tex = re.sub(r"\\label\{[^}]*\}", "", tex)
    # IEEEtran tablo numaralandirmasi
    tex = re.sub(r"\\renewcommand\{\\thetable\}[^\n]*\n", "", tex)
    tex = re.sub(r"\\makeatletter.*?\\makeatother", "", tex, flags=re.S)
    return tex, notes


# ---------------------------------------------------------------- baslik no

def number_headings(docx_path, tex):
    """Word basliklarina MDPI ondalik numarasi ekler.

    `\\appendices`ten sonraki ust duzey bolumler MDPI'da "Appendix A",
    "Appendix B" diye adlandirilir; 10/11/12 olarak numaralanmalari yanlis
    olurdu.
    """
    from docx import Document

    app_at = tex.find("\\appendices")
    seq = []
    c = [0, 0, 0]
    app_i = 0
    for m in re.finditer(r"\\(section|subsection|subsubsection)\*?\{", tex):
        lvl = {"section": 0, "subsection": 1, "subsubsection": 2}[m.group(1)]
        in_app = app_at >= 0 and m.start() > app_at
        if in_app and lvl == 0:
            seq.append((0, "Appendix " + chr(ord("A") + app_i)))
            app_i += 1
            c[1] = c[2] = 0
            continue
        c[lvl] += 1
        for k in range(lvl + 1, 3):
            c[k] = 0
        seq.append((lvl, ".".join(str(c[k]) for k in range(lvl + 1))))

    d = Document(str(docx_path))
    heads = [p for p in d.paragraphs
             if p.style.name in ("Heading 1", "Heading 2", "Heading 3")]
    n = 0
    for p, (lvl, num) in zip(heads, seq):
        want = "Heading %d" % (lvl + 1)
        if p.style.name != want or not p.runs:
            continue
        if re.match(r"^(?:\d+(\.\d+)*\.?|Appendix [A-Z]\.?)\s", p.text):
            continue
        p.runs[0].text = "%s. %s" % (num, p.runs[0].text)
        n += 1
    d.save(str(docx_path))
    return n, len(heads), len(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()

    tex = TEX.read_text(encoding="utf-8")
    floats = float_numbers()
    secs = section_numbers(tex)
    body, notes = preprocess(tex, floats, secs)

    tmp = HERE / "_mdpi_tmp.tex"
    tmp.write_text(body, encoding="utf-8")
    try:
        # cwd=HERE sart: \input{bibliography.tex} goreli yoldur ve pandoc
        # onu calisma dizinine gore cozer.
        r = subprocess.run(
            ["pandoc", tmp.name, "-o", str(pathlib.Path(a.out).resolve()),
             "--from", "latex", "--to", "docx",
             "--resource-path", str(HERE)],
            capture_output=True, text=True, cwd=str(HERE))
        if r.returncode:
            sys.exit("pandoc hatasi:\n" + r.stderr[:1500])
        warn = [l for l in r.stderr.splitlines() if l.strip()][:5]
    finally:
        tmp.unlink(missing_ok=True)

    nnum, nheads, nseq = number_headings(a.out, tex)

    from docx import Document
    d = Document(a.out)
    print(f"yazildi: {a.out}\n")
    print(f"  bolum etiketi cozuldu   : {len(secs)}")
    print(f"  tablo/figur etiketi     : {len(floats)}")
    print(f"  resizebox sarmali acildi: {notes['resizebox']}")
    print(f"  tablo (docx)            : {len(d.tables)}  /  tex'te 36")
    print(f"  paragraf                : {len(d.paragraphs)}")
    print(f"  numaralanan baslik      : {nnum} / {nheads} (tex: {nseq})")
    print(f"  kaynakca girdisi        : {notes['n_entries']}")
    print(f"  cozulen atif anahtari   : {notes['n_cites']}")
    if notes["cites"]:
        print(f"  COZULEMEYEN ATIF        : {notes['cites']}")
    if notes["unresolved"]:
        print(f"\n  COZULEMEYEN GONDERME    : {len(notes['unresolved'])}")
        for l in notes["unresolved"][:10]:
            print(f"      {l}")
    if warn:
        print("\n  pandoc uyarisi:")
        for w in warn:
            print("     ", w[:110])


if __name__ == "__main__":
    main()
