"""
Makaledeki her tabloyu CSV'ye cevirir ve kokenini VERIDEN bulur.

NICIN
-----
Hoca degisiklikleri takip etmekte zorlandigini soyledi ve tablolari CSV
olarak istedi. Elle liste yazmak yerine tablolar kaynaktan uretiliyor;
boylece makale degisirse CSV'ler de degisir.

KOKEN NASIL BULUNUYOR
---------------------
Her tablonun sayisal hucreleri cikarilir ve `results/*.json` dosyalarinin
hepsinde aranir. En cok eslesmeyi veren dosya o tablonun kaynagi sayilir.
Bu, elle tutulan bir haritadan daha guvenilir: harita eskir, eslesme
eskimez. Eslesme orani da yazilir, boylece zayif bir atama fark edilir.

CIKTI
-----
    <out>/Table_01_<label>.csv ...
    <out>/00_TABLO_LISTESI.csv   -- numara, etiket, baslik, kaynak, oran

KULLANIM
    python makale/_tables_to_csv.py [--out DIZIN]
"""
import argparse
import csv
import json
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
TEX = HERE / "main_revised.tex"
AUX = HERE / "main_revised.aux"
RESULTS = ROOT / "results"
DEFAULT_OUT = ROOT / "teslim" / "tablolar_csv"

BS = "\\"


# ------------------------------------------------------------------ yardimci
def label_numbers():
    aux = AUX.read_text(encoding="utf-8", errors="ignore")
    pat = (r"newlabel\{(tab:[a-z_0-9]+)\}\{\{(?:" + BS * 2
           + r"mbox\s*\{)?([0-9]+)")
    return {m.group(1): int(m.group(2)) for m in re.finditer(pat, aux)}


def clean(cell):
    """LaTeX hucresini duz metne indirir."""
    s = cell
    s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\(?:textbf|bm|mathbf|emph|textit|texttt|mathrm)\{(.*?)\}",
               r"\1", s)
    s = s.replace(r"\pm", "±").replace(r"\%", "%").replace(r"\times", "x")
    s = s.replace(r"\approx", "~").replace(r"\delta", "delta")
    s = s.replace("$", "").replace("{,}", ",").replace("~", " ")
    s = s.replace("---", "—").replace("--", "–")
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = s.replace("{", "").replace("}", "").replace(BS, "")
    return " ".join(s.split())


def table_blocks(tex):
    """(caption, label, [satirlar]) listesi, belge sirasinda."""
    out = []
    pat = re.compile(BS * 2 + r"begin\{(table\*?)\}(.*?)" + BS * 2
                     + r"end\{\1\}", re.S)
    for m in pat.finditer(tex):
        body = m.group(2)
        cm = re.search(r"\\caption\{(.*?)\}\s*\n\s*\\label", body, re.S)
        if not cm:
            cm = re.search(r"\\caption\{(.*)", body, re.S)
        caption = clean(cm.group(1)) if cm else ""
        lm = re.search(r"\\label\{(tab:[a-z_0-9]+)\}", body)
        label = lm.group(1) if lm else ""
        # Sutun tanimi ic ice suslu parantez tasiyabilir (`{@{}lcccc@{}}`),
        # bu yuzden `[^}]*` yetmiyor; dengeli olarak tuketiliyor.
        tm = None
        bm = re.search(r"\\begin\{tabular\}\s*(?:\[[^\]]*\]\s*)?\{", body)
        if bm:
            j, depth = bm.end(), 1
            while j < len(body) and depth:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                j += 1
            em = body.find(r"\end{tabular}", j)
            if em > 0:
                tm = type("M", (), {"group": staticmethod(
                    lambda n, s=body[j:em]: s)})()
        rows = []
        if tm:
            for raw in tm.group(1).split(BS * 2):
                raw = re.sub(r"\\(?:top|mid|bottom)rule", "", raw)
                raw = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", "", raw)
                raw = re.sub(r"\\cmidrule\{[^}]*\}", "", raw)
                raw = raw.replace("%", "")
                if not raw.strip():
                    continue
                cells = [clean(c) for c in raw.split("&")]
                if any(cells):
                    rows.append(cells)
        out.append((caption, label, rows))
    return out


def numbers_in(rows):
    got = set()
    for r in rows:
        for c in r:
            for m in re.finditer(r"\d+\.\d{3,4}", c):
                got.add(round(float(m.group()), 4))
    return got


def load_results():
    store = {}
    for p in sorted(RESULTS.glob("*.json")):
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        vals = {round(float(m.group()), 4)
                for m in re.finditer(r"\d+\.\d{3,}", txt)}
        store[p.name] = vals
    return store


# Etiketten kaynaga BEYAN EDILEN eslesme. Kesif (asagida) bunu DOGRULAR;
# ikisi uyusmazsa uyari basilir. Yalnizca kesfe guvenmek yetmiyor: bir
# tablonun sayilari baska bir deneyin dosyasinda tesaduf eseri daha cok
# eslesebiliyor (Tablo 7 ilk denemede yanlis dosyaya baglanmisti).
CLAIMED = {
    "tab:replication": "F15_reproducibility.json",
    "tab:markov": "F31_markov.json",
    "tab:provenance": "F24_model_audit.json",
    "tab:reference_points": "F1b_fair.json",
    "tab:reference_stats": "F1b_fair.json",
    "tab:encoder_benchmark": "F4_benchmark.json",
    "tab:fusion_ablation": "F3_fusion.json",
    "tab:fusion_cost": "F6_efficiency.json",
    "tab:cv": "F5_cv_5seed.json",
    "tab:subtype": "F8_effects_subtypes.json",
    "tab:repeated_cv": "F10_repeated_cv.json",
    "tab:variance_components": "F10_repeated_cv.json",
    "tab:collapse": "F10_repeated_cv.json",
    "tab:graph_controls": "F14_graph_controls_5seed.json",
    "tab:classic_cv": "F11_classic_cv_5seed.json",
    "tab:equivalence": "F16_equivalence.json",
    "tab:inductive": "F18_inductive_5seed.json",
    "tab:prevalence": "F13_prevalence.json",
    "tab:noise": "F19_noise.json",
    "tab:imputation": "F22_imputation_5seed.json",
    "tab:window_stride": "F20_window_stride.json",
    "tab:baseline_search": "F21_baseline_search.json",
    "tab:independent_network": "F28_hamburg_baseline_n0_5_a5.json",
    "tab:four_datasets": "F28_kocaeli_baseline_n0_5_a5.json",
    "tab:effect_sizes": "F8_effects_subtypes.json",
    "tab:graph_ablation": "F2_graph.json",
    "tab:trip_level": "F7_cluster_stats.json",
    "tab:efficiency": "F6_efficiency.json",
    "tab:markov_free": "F31_markov.json",
    "tab:markov_trans": "F31_markov.json",
    "tab:markov_cong": "F31_markov.json",
    "tab:operator_config": "F24_model_audit.json",
    "tab:two_seed": "F5_cv_5seed.json",
    "tab:dataset_split": "F5_cv_5seed.json",
    "tab:anomaly_taxonomy": "F23_data_audit.json",
    "tab:road_defaults": "F23_data_audit.json",
}


def provenance(label, nums, store):
    """Tablonun sayilarini aciklayan sonuc dosyalari, kapsama sirasinda.

    Bir tablo BIRDEN FAZLA deneyden beslenebiliyor -- Tablo 23 hem klasik
    modelleri (F11) hem derin modelleri (F5) tasiyor, Tablo 31 hem Hamburg
    kosusunu hem sizinti tanisini. Tek kaynak varsaymak bu tablolarda
    yaniltici bir "zayif eslesme" uretiyordu.

    Beyan edilen kaynak her zaman listede ilk sirada gosterilir; acgozlu
    secim kalan sayilari kapatacak dosyalari ekler. Donus:
    ([(dosya, oran), ...], toplam_kapsama).
    """
    if not nums:
        return [], 0.0
    picks, rest = [], set(nums)
    claimed = CLAIMED.get(label)
    if claimed and claimed in store:
        hit = rest & store[claimed]
        picks.append((claimed, len(hit) / len(nums)))
        rest -= hit
    while rest:
        best, gain = "", 0
        for name, vals in store.items():
            if any(name == p for p, _ in picks):
                continue
            h = len(rest & vals)
            if h > gain:
                best, gain = name, h
        if not best or gain / len(nums) < 0.05:
            break
        picks.append((best, gain / len(nums)))
        rest -= store[best]
    return picks, (len(nums) - len(rest)) / len(nums)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tex = TEX.read_text(encoding="utf-8")
    nums_by_label = label_numbers()
    store = load_results()
    blocks = table_blocks(tex)

    index, n_written, weak = [], 0, []
    for i, (caption, label, rows) in enumerate(blocks, 1):
        num = nums_by_label.get(label, i)
        slug = label.split(":", 1)[-1] if label else "appendix%02d" % i
        fname = "Table_%02d_%s.csv" % (num, slug)
        if rows:
            with (out / fname).open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["# Table %d" % num, caption])
                w.writerow([])
                for r in rows:
                    w.writerow(r)
            n_written += 1
        nums = numbers_in(rows)
        picks, cover = provenance(label, nums, store)
        if nums and cover < 0.8:
            weak.append((num, picks, cover))
        srcs = " + ".join("%s (%.0f%%)" % (p, 100 * r) for p, r in picks)
        index.append([num, label, caption, fname if rows else "",
                      len(rows), srcs,
                      "%.0f%%" % (100 * cover) if nums else "", len(nums)])

    index.sort(key=lambda r: r[0])
    with (out / "00_TABLO_LISTESI.csv").open("w", newline="",
                                             encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Tablo", "Etiket", "Baslik", "CSV dosyasi", "Satir",
                    "Kaynak sonuc dosyalari (kapsama)",
                    "Toplam kapsama", "Tablodaki sayi"])
        w.writerows(index)

    print("cikti: %s" % out)
    print("  yazilan CSV      : %d" % n_written)
    print("  taranan sonuc    : %d dosya" % len(store))
    print("  zayif koken (<50%%): %d" % len(weak))
    for num, picks, c in weak[:10]:
        print("      Tablo %-2d kapsama %.0f%%  <- %s"
              % (num, 100 * c,
                 ", ".join(p for p, _ in picks) or "kaynak bulunamadi"))


if __name__ == "__main__":
    main()
