"""
FAZ B kuyrugu — F10 bitince kalan deneyleri SIRAYLA kosar.

NEDEN BOYLE
-----------
Deneyler GPU'yu paylasamaz: kart 8 GB ve tek bir egitim 7 GB'i buluyor.
Ikisini ayni anda baslatmak bellek hatasi demek. Bu yuzden kuyruk
seri calisir ve her adim bir oncekinin bitmesini bekler.

Her adim kendi onbellegine yazdigi icin kuyruk ISTENILEN YERDEN yeniden
baslatilabilir: tamamlanmis kosular yeniden egitilmez.

KULLANIM
  python experiments/run_queue.py              # F10'u bekler, sonra kosar
  python experiments/run_queue.py --now        # beklemeden basla
  python experiments/run_queue.py --only f14   # tek adim
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stanet import config as C
from stanet.utils import init_console

init_console()
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

WAIT_FOR = C.RESULTS_DIR / "F10_repeated_cv.json"

# (etiket, betik, beklenen cikti) — sira onemli: once KARAR verici olanlar,
# sonra duyarlilik. Boylece kuyruk yarida kalirsa elimizde en cok ise
# yarayan sonuclar olur.
QUEUE = [
    ("f14", "f14_graph_controls.py",  "F14_graph_controls.json"),   # 3.4
    ("f18", "f18_inductive.py",       "F18_inductive.json"),        # 2.2
    ("f17", "f17_gate_collapse.py",   "F17_gate_collapse.json"),    # 1.9 · 3.7
    ("f22", "f22_imputation.py",      "F22_imputation.json"),       # 1.12
    ("f21", "f21_baseline_search.py", "F21_baseline_search.json"),  # 2.3
    ("f20", "f20_window_stride.py",   "F20_window_stride.json"),    # 1.7
]


def stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def wait_for_f10(poll=120):
    if WAIT_FOR.exists():
        print("%s  F10 zaten bitmis, kuyruk basliyor" % stamp(), flush=True)
        return
    print("%s  F10 bekleniyor (%s)" % (stamp(), WAIT_FOR.name), flush=True)
    while not WAIT_FOR.exists():
        time.sleep(poll)
    print("%s  F10 bitti, kuyruk basliyor" % stamp(), flush=True)


def run(tag, script, out_name):
    out = C.RESULTS_DIR / out_name
    if out.exists():
        print("%s  [%s] atlandi, cikti zaten var: %s" % (stamp(), tag, out_name),
              flush=True)
        return True
    log = LOG_DIR / ("queue_%s.log" % tag)
    print("%s  [%s] basliyor -> %s" % (stamp(), tag, log.name), flush=True)
    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.call([sys.executable, "-u",
                              str(ROOT / "experiments" / script)],
                             stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    dt = (time.time() - t0) / 60
    ok = (rc == 0 and out.exists())
    print("%s  [%s] %s  (%.1f dk, cikis %d)"
          % (stamp(), tag, "BITTI" if ok else "BASARISIZ", dt, rc), flush=True)
    if not ok:
        print("       son satirlar:", flush=True)
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        for line in tail:
            print("       | " + line, flush=True)
    return ok


def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    if "--now" not in sys.argv:
        wait_for_f10()

    done, failed = [], []
    for tag, script, out_name in QUEUE:
        if only and tag != only:
            continue
        # Bir adimin basarisiz olmasi digerlerini durdurmaz: bagimsizlar ve
        # elimizde ne varsa onunla devam etmek, hicbir sey olmamasindan iyi.
        (done if run(tag, script, out_name) else failed).append(tag)

    print("\n%s  KUYRUK BITTI" % stamp())
    print("  tamamlanan: %s" % (", ".join(done) or "yok"))
    if failed:
        print("  BASARISIZ  : %s  (logs/queue_<etiket>.log)" % ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
