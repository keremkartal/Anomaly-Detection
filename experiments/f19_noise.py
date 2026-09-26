"""
F19 — Cikarim aninda sensor gurultusune dayaniklilik (degerlendirme: gercek
sensor kosullari).

Veri sentetiktir ve uretim sirasinda hiz ve yon kanallarina gurultu
enjekte edilmistir; ama modelin EGITILDIGI gurultu duzeyinden farkli bir
duzeyde nasil davrandigi olculmemisti. Burada egitilmis modeller hic
degistirilmeden, yalnizca girdiler bozularak yeniden degerlendirilir.

GURULTU MODELI — tahmin degil, olculmus donusum
-----------------------------------------------
Veri hattinin birimleri dogrulandi: hiz km/h, ornekleme araligi 1 s ve
ivme ile hiz arasinda tam bir bagimlilik var —

    a_t = (v_t - v_{t-1}) / 3.6      [m/s^2]

Bu oran test kumesinde %5-%95 dilimlerinin tamaminda 3,6000 cikti, yani
ivme turetilmis bir kanaldir. Dolayisiyla hiz kanalina gurultu eklerken
ivmeyi de tutarli bicimde YENIDEN HESAPLIYORUZ. Iki kanala bagimsiz gurultu
eklemek fiziksel olarak tutarsiz olurdu ve modele gercekte olmayan bir
celiski gosterirdi.

Pencerenin ilk adiminin ivmesi pencere disindaki bir orneğe bagli oldugu
icin degistirilmez; bu, 50 adimin 1'ini etkiler.

Yon kanali icin aci dogrudan dondurulur ve sin/cos yeniden hesaplanir.

ESIK POLITIKASI
---------------
Esik, kosunun KENDI temiz dogrulama kumesinden secilmis degerdir ve gurultu
arttikca DEGISTIRILMEZ. Sahada gurultu duzeyi bilinmez, dolayisiyla yeniden
kalibrasyon varsayimi gercekci olmaz. Karsilastirma icin gurultulu dogrulama
uzerinde yeniden secilen esik de raporlanir.

Bu betik egitim YAPMAZ; onbellekteki agirliklari (`*.pt`) okur. Agirligi
olmayan kosular atlanir ve sayisi raporlanir.

KULLANIM
  python experiments/f19_noise.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from stanet import config as C
from stanet.data import build_cv_folds
from stanet.evaluate import compute_metrics, pick_threshold, predict
from stanet.graph import build_graph, map_segments
from stanet.runstore import _build, _dirs, make_spec, spec_key
from stanet.train import make_loaders
from stanet.utils import init_console, get_device, load_json, protocol_stamp, save_json

init_console()
OUT = C.RESULTS_DIR / "F19_noise.json"

KMH_PER_MS2 = 3.6          # a = dv/3.6, dogrulandi
SPEED_SIGMAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]        # km/h
BEARING_SIGMAS = [0.0, 1.0, 2.0, 5.0, 10.0]            # derece
NOISE_SEED = 20260926
RHO = 0.90                     # AR(1) ilinti katsayisi (gercekci GPS)
SEEDS = [42, 43]
REPEATS = [0, 1, 2, 3, 4]      # agirligi olan her bolumleme kullanilir
MAX_RUNS_PER_CONFIG = 10       # yeter; None yaparsaniz hepsi
MERGE = False
CONFIGS = {
    "hybrid_gat":      dict(temporal="lstm", spatial="gat",  fusion="weighted_sum"),
    "stanet_gated":    dict(temporal="lstm", spatial="gat",  fusion="gated"),
    "lstm_only":       dict(temporal="lstm", spatial="none", fusion="weighted_sum"),
}


def verify_units(X, scalers, tol=1e-3):
    """a_t = (v_t - v_{t-1}) / 3.6 bagintisini dogrular; oran doner."""
    from stanet.data import inverse_kinematics
    s, a = inverse_kinematics(X, scalers)
    dv, aa = np.diff(s, axis=1), a[:, 1:]
    m = np.abs(aa) > 1e-6
    if not m.any():
        return None
    r = dv[m] / aa[m]
    lo, hi = np.percentile(r, [5, 95])
    return {"ratio_median": float(np.median(r)), "p5": float(lo), "p95": float(hi),
            "consistent": bool(abs(lo - KMH_PER_MS2) < tol
                               and abs(hi - KMH_PER_MS2) < tol)}


def perturb(X, scalers, sigma_speed=0.0, sigma_bearing_deg=0.0,
            rho=0.0, seed=NOISE_SEED):
    """Normalize pencereleri ham birimlere cevirir, bozar, geri normalize eder."""
    from stanet.data import inverse_kinematics
    rng = np.random.default_rng(seed)
    s, a = inverse_kinematics(X, scalers)
    Xn = X.copy()

    if sigma_speed > 0:
        if rho > 0:
            # AR(1) ile ZAMANDA ILINTILI gurultu. Gercek GPS hiz hatasi
            # adimdan adima bagimsiz degildir; beyaz gurultu, ivme birinci
            # fark oldugu icin en kotu durumu temsil eder. Iki modeli de
            # raporlamak "model kirilgan" ile "1 Hz'de beyaz gurultunun
            # farkini almak kirilgan" ayrimini yapar.
            e = rng.normal(0.0, sigma_speed * np.sqrt(1 - rho ** 2), s.shape)
            noise = np.empty_like(e)
            noise[:, 0] = rng.normal(0.0, sigma_speed, s.shape[0])
            for t in range(1, s.shape[1]):
                noise[:, t] = rho * noise[:, t - 1] + e[:, t]
        else:
            noise = rng.normal(0.0, sigma_speed, s.shape)
        s2 = s + noise
        s2 = np.clip(s2, 0.0, None)                 # hiz negatif olamaz
        a2 = a.copy()
        a2[:, 1:] = np.diff(s2, axis=1) / KMH_PER_MS2
        Xn[..., 0] = scalers["speed"].transform(
            s2.reshape(-1, 1)).reshape(s2.shape)
        Xn[..., 1] = scalers["accel"].transform(
            a2.reshape(-1, 1)).reshape(a2.shape)

    if sigma_bearing_deg > 0:
        th = np.arctan2(X[..., 2], X[..., 3])
        th2 = th + np.radians(rng.normal(0.0, sigma_bearing_deg, th.shape))
        Xn[..., 2] = np.sin(th2)
        Xn[..., 3] = np.cos(th2)
    return Xn


def evaluate(model, ds, Xn, node_idx, graph, device, threshold):
    """Bozulmus girdilerle test metrikleri; esik disaridan gelir."""
    import copy
    stub = copy.copy(ds)
    stub.X_test = Xn
    _, _, te = make_loaders(stub, node_idx)
    nx_, ei_ = graph.to_tensors(device)
    yt, pt, _ = predict(model, te, nx_, ei_, device)
    return compute_metrics(yt, pt, threshold), yt, pt


def main():
    device = get_device()
    stamp = protocol_stamp()
    graph = build_graph(merge_by_osm=MERGE)
    folds = build_cv_folds(n_folds=C.N_FOLDS, legacy=False)
    d, _ = _dirs()
    print("protokol %s | cihaz %s\n" % (stamp["hash"], device), flush=True)

    results = {"protocol": stamp,
               "noise_model": {
                   "speed_unit": "km/h", "accel_unit": "m/s^2",
                   "sampling_interval_s": 1,
                   "relation": "a_t = (v_t - v_{t-1}) / 3.6",
                   "speed_sigmas_kmh": SPEED_SIGMAS,
                   "bearing_sigmas_deg": BEARING_SIGMAS,
                   "noise_seed": NOISE_SEED, "ar1_rho": RHO,
                   "why_two_speed_models": ("beyaz gurultu ivme kanalini "
                       "yok eder cunku ivme birinci farktir; AR(1) gercek "
                       "GPS hatasina daha yakindir ve fark alma onu buyutmez"),
                   "threshold_policy": "kosunun kendi TEMIZ dogrulama esigi, sabit",
                   "first_step_accel": "pencere disina bagli oldugu icin degismez"},
               "unit_check": {}, "models": {}, "skipped": []}

    # birim dogrulamasi (bir fold yeter ama hepsini kontrol ediyoruz)
    for f in folds:
        v = verify_units(f.X_test, f.scalers)
        results["unit_check"]["fold%d" % f.info["fold"]] = v
    bad = [k for k, v in results["unit_check"].items() if not (v and v["consistent"])]
    print("birim dogrulamasi: %d/%d fold'da a=(dv)/3.6 tam tutuyor%s"
          % (len(folds) - len(bad), len(folds),
             ("  UYARI: " + ", ".join(bad)) if bad else ""), flush=True)

    # (tekrar, fold) -> Dataset ve dugum indeksleri; istenince kurulur
    fold_cache = {0: {f.info["fold"]: f for f in folds}}
    idx_cache = {}

    def get_fold(rep, k):
        if rep not in fold_cache:
            fold_cache[rep] = {g.info["fold"]: g for g in
                               build_cv_folds(n_folds=C.N_FOLDS, legacy=False,
                                              repeat=rep)}
        f = fold_cache[rep][k]
        if (rep, k) not in idx_cache:
            idx_cache[(rep, k)] = {sp: map_segments(getattr(f, "meta_" + sp), graph)[0]
                                   for sp in ("train", "val", "test")}
        return f, idx_cache[(rep, k)]

    def available(cfg):
        """Bu konfigurasyona ait, AGIRLIGI olan kosular: (rep, fold, seed, spec)."""
        out = []
        for rep in REPEATS:
            for k in range(C.N_FOLDS):
                for seed in SEEDS:
                    kw = {} if rep == 0 else {"rep": rep}
                    spec = make_spec(merge_by_osm=MERGE, fold=k, cv=C.N_FOLDS,
                                     **kw, **cfg)
                    key = spec_key(spec, seed)
                    if (d / (key + ".pt")).exists() and (d / (key + ".json")).exists():
                        out.append((rep, k, seed, spec, key))
                    else:
                        results["skipped"].append(key)
        return out

    for name, cfg in CONFIGS.items():
        rows = []
        runs_here = available(cfg)
        if MAX_RUNS_PER_CONFIG:
            runs_here = runs_here[:MAX_RUNS_PER_CONFIG]
        print("  %-13s agirligi olan kosu: %d" % (name, len(runs_here)), flush=True)
        for rep, k, seed, spec, key in runs_here:
                f, node_idx = get_fold(rep, k)
                jf = d / (key + ".json")
                wf = d / (key + ".pt")
                meta = load_json(jf)["metrics"]
                th = float(meta["threshold_val"])
                model = _build(spec, graph).to(device)
                model.load_state_dict(torch.load(wf, map_location=device))
                model.eval()

                row = {"repeat": rep, "fold": k, "seed": seed, "threshold": th,
                       "f1_cached": float(meta["f1"]), "speed": {}, "speed_ar1": {},
                       "bearing": {}}
                for sg in SPEED_SIGMAS:
                    Xn = f.X_test if sg == 0 else perturb(f.X_test, f.scalers,
                                                          sigma_speed=sg)
                    m, _, _ = evaluate(model, f, Xn, node_idx, graph, device, th)
                    row["speed"]["%.1f" % sg] = {"f1": m["f1"],
                                                 "precision": m["precision"],
                                                 "recall": m["recall"],
                                                 "roc_auc": m.get("roc_auc")}
                for sg in SPEED_SIGMAS:
                    Xn = f.X_test if sg == 0 else perturb(
                        f.X_test, f.scalers, sigma_speed=sg, rho=RHO)
                    m, _, _ = evaluate(model, f, Xn, node_idx, graph, device, th)
                    row["speed_ar1"]["%.1f" % sg] = {"f1": m["f1"],
                                                     "roc_auc": m.get("roc_auc")}
                for bg in BEARING_SIGMAS:
                    Xn = f.X_test if bg == 0 else perturb(f.X_test, f.scalers,
                                                          sigma_bearing_deg=bg)
                    m, _, _ = evaluate(model, f, Xn, node_idx, graph, device, th)
                    row["bearing"]["%.1f" % bg] = {"f1": m["f1"],
                                                   "roc_auc": m.get("roc_auc")}
                rows.append(row)
                print("  %-13s r%d f%d s%d: temiz F1=%.4f -> 10 km/h gurultude %.4f"
                      % (name, rep, k, seed, row["speed"]["0.0"]["f1"],
                         row["speed"]["10.0"]["f1"]), flush=True)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if not rows:
            continue
        summ = {"speed": {}, "speed_ar1": {}, "bearing": {}}
        for ch, sigmas in (("speed", SPEED_SIGMAS), ("speed_ar1", SPEED_SIGMAS),
                           ("bearing", BEARING_SIGMAS)):
            for sg in sigmas:
                key = "%.1f" % sg
                v = np.array([r[ch][key]["f1"] for r in rows])
                summ[ch][key] = {"f1_mean": float(v.mean()),
                                 "f1_sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                 "n": int(len(v))}
        results["models"][name] = {"runs": rows, "summary": summ}

    # ------------------------------------------------------------------ ozet
    for ch, sigmas, unit in (("speed", SPEED_SIGMAS, "km/h, beyaz"),
                             ("speed_ar1", SPEED_SIGMAS, "km/h, AR(1) rho=%.2f" % RHO),
                             ("bearing", BEARING_SIGMAS, "derece")):
        print("\n" + "=" * 92)
        print("%s KANALI GURULTUSU (%s) — F1 ort+-sd, esik sabit" % (ch.upper(), unit))
        print("=" * 92)
        print("  %-16s" % "model" + "".join("%15s" % ("s=%.1f" % s) for s in sigmas))
        for name, d2 in results["models"].items():
            line = "  %-16s" % name
            for s in sigmas:
                v = d2["summary"][ch]["%.1f" % s]
                line += "%15s" % ("%.3f±%.3f" % (v["f1_mean"], v["f1_sd"]))
            print(line)
        # bozulma orani
        for name, d2 in results["models"].items():
            base = d2["summary"][ch]["0.0"]["f1_mean"]
            last = d2["summary"][ch]["%.1f" % sigmas[-1]]["f1_mean"]
            print("  %-16s en yuksek gurultude goreli kayip: %.1f%%"
                  % (name, 100 * (base - last) / max(base, 1e-9)))

    if results["skipped"]:
        print("\n  agirligi olmadigi icin atlanan kosu: %d" % len(results["skipped"]))
        print("  (bu kosular F10 tamamlandiktan sonra agirlik tasiyacak)")
    save_json(results, OUT)
    print("\nkaydedildi: %s" % OUT)


if __name__ == "__main__":
    main()
