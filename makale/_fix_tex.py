"""Heredoc kaynakli bozuk '~\\nef{...}' kaliplarini ve olusan tekrarlari temizler."""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
p = Path(__file__).parent / "main_revised.tex"
s = p.read_text(encoding="utf-8")
BS = chr(92)

# 1) "Section~\n ef{x}" -> "Section~\ref{x}"
broken = re.compile(r"~\s*\n\s*ef\{([^}]+)\}")
n_broken = len(broken.findall(s))
s = broken.sub(lambda m: "~" + BS + "ref{" + m.group(1) + "}", s)

# 2) Ayni cumlenin iki kez uretilmis hallerini tekile indir
dupes = [
    ("Table~" + BS + "ref{tab:encoder_benchmark} and Fig.~" + BS + "ref{fig:encoder_benchmark} report the outcome. "
     "Table~" + BS + "ref{tab:encoder_benchmark} and Fig.~" + BS + "ref{fig:encoder_benchmark} report the outcome. ",
     "Table~" + BS + "ref{tab:encoder_benchmark} and Fig.~" + BS + "ref{fig:encoder_benchmark} report the outcome. "),
    ("Table~" + BS + "ref{tab:fusion_ablation} and Fig.~" + BS + "ref{fig:fusion_ablation} summarize the outcome. "
     "Table~" + BS + "ref{tab:fusion_ablation} and Fig.~" + BS + "ref{fig:fusion_ablation} summarize the outcome.",
     "Table~" + BS + "ref{tab:fusion_ablation} and Fig.~" + BS + "ref{fig:fusion_ablation} summarize the outcome."),
    ("Table~" + BS + "ref{tab:subtype} and Fig.~" + BS + "ref{fig:subtype} report the result. "
     "Table~" + BS + "ref{tab:subtype} and Fig.~" + BS + "ref{fig:subtype} report the result.",
     "Table~" + BS + "ref{tab:subtype} and Fig.~" + BS + "ref{fig:subtype} report the result."),
    ("The limitations bounding these conclusions are collected in Section~" + BS + "ref{sec:threats}. "
     "The limitations bounding these conclusions are collected in Section~" + BS + "ref{sec:threats}. ",
     "The limitations bounding these conclusions are collected in Section~" + BS + "ref{sec:threats}. "),
    ("Two evaluation regimes are used (see also Section~" + BS + "ref{sec:dataset_split} for the trip-grouping rationale):",
     "Two evaluation regimes are used:"),
]
n_dupes = 0
for a, b in dupes:
    if a in s:
        s = s.replace(a, b)
        n_dupes += 1

# 3) bosluk temizligi
s = re.sub(r"[ \t]+\n", "\n", s)
s = re.sub(r"\n{3,}", "\n\n", s)

p.write_text(s, encoding="utf-8")
print(f"bozuk ref duzeltildi: {n_broken}")
print(f"tekrar temizlendi   : {n_dupes}/{len(dupes)}")
