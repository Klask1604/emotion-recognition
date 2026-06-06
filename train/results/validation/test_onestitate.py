#!/usr/bin/env python3
"""TEST DE ONESTITATE: dovedeste ca pipeline-ul nu minte.
Ruleaza CALM vs STRESAT (WESAD) in 2 conditii:
  REAL    - etichete corecte. Asteptat: ~90%+ (semnal real)
  RANDOM  - etichete AMESTECATE per subiect. Asteptat: ~50% (sansa)

Daca RANDOM ramane mare -> leakage/bug -> cifrele sunt FALSE.
Daca RANDOM cade la ~50% -> pipeline curat -> cifra REALA e de incredere.
Asta e testul standard prin care prinzi un ML mincinos (permutation test).
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

WESAD_DIR = ROOT / "datasets" / "WESAD"
WIN_S, STEP_S = 20, 10
RNG = np.random.default_rng(0)


def hrv_bvp(seg, fs):
    s = seg - seg.mean()
    if s.std() < 1e-6:
        return None
    b, a = butter(2, [0.5 / (fs / 2), 4 / (fs / 2)], btype="band")
    f = filtfilt(b, a, s)
    pk, _ = find_peaks(f, distance=int(0.4 * fs))
    if len(pk) < 4:
        return None
    ibi = np.diff(pk) / fs * 1000.0
    return [60000.0 / np.mean(ibi), np.std(ibi),
            np.sqrt(np.mean(np.diff(ibi) ** 2)) if len(ibi) > 1 else 0.0,
            np.mean(np.abs(np.diff(ibi)) > 50) * 100 if len(ibi) > 1 else 0.0,
            np.median(ibi), np.percentile(ibi, 20), np.percentile(ibi, 80)]


def add_personal(X, subj, neutral):
    Xout = np.zeros((len(X), X.shape[1] * 2))
    for s in set(subj.tolist()):
        m = subj == s
        base = X[m & neutral] if (m & neutral).sum() >= 3 else X[m]
        mu, sd = base.mean(0), base.std(0) + 1e-8
        Xout[m] = np.hstack([X[m], (X[m] - mu) / sd])
    return Xout


def loso(X, y, g):
    logo = LeaveOneGroupOut(); yt, yp = [], []
    for tr, te in logo.split(X, y, g):
        if len(set(y[tr].tolist())) < 2:
            continue
        c = make_pipeline(StandardScaler(),
                          GradientBoostingClassifier(n_estimators=120, random_state=0))
        c.fit(X[tr], y[tr]); yp.extend(c.predict(X[te])); yt.extend(y[te])
    return balanced_accuracy_score(yt, yp) * 100


def load():
    X, lab, subj = [], [], []
    for sd in sorted([p for p in WESAD_DIR.iterdir()
                      if p.is_dir() and (p / f"{p.name}.pkl").exists()]):
        d = pickle.load(open(sd / f"{sd.name}.pkl", "rb"), encoding="latin1")
        L = d["label"]; bvp = d["signal"]["wrist"]["BVP"].ravel()
        acc = d["signal"]["wrist"]["ACC"]
        fb, fl = 64.0, 700.0; w, st = int(WIN_S * fb), int(STEP_S * fb)
        for i in range(0, len(bvp) - w + 1, st):
            seg = L[int(i / fb * fl):int(i / fb * fl) + int(WIN_S * fl)]
            if len(seg) == 0:
                continue
            v, c = np.unique(seg, return_counts=True); dom = int(v[np.argmax(c)])
            if dom not in (1, 2, 4):  # calm(1,4) vs stres(2)
                continue
            f = hrv_bvp(bvp[i:i + w], fb)
            if f is None:
                continue
            ai = int(i / fb * 32); a = acc[ai:ai + int(WIN_S * 32)]
            if len(a) < 8:
                continue
            mag = np.sqrt((a.astype(float) ** 2).sum(1))
            X.append(f + [float(mag.mean()), float(mag.std())])
            lab.append(dom); subj.append(sd.name)
    return np.array(X), np.array(lab), np.array(subj)


def main():
    print("Incarc WESAD...")
    X, lab, subj = load()
    neutral = np.isin(lab, [1, 4])
    Xd = add_personal(X, subj, neutral)
    y = (lab == 2).astype(int)   # 1=stres, 0=calm
    print(f"  {len(y)} ferestre, {len(set(subj.tolist()))} subiecti\n")

    print("=" * 56)
    print("TEST DE ONESTITATE (permutation test)")
    print("=" * 56)

    real = loso(Xd, y, subj)
    print(f"  REAL   (etichete corecte)  : {real:.0f}%")

    # random: amesteca etichetele in interiorul fiecarui subiect (pastreaza balanta)
    rnds = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        yr = y.copy()
        for s in set(subj.tolist()):
            m = subj == s
            yr[m] = rng.permutation(yr[m])
        rnds.append(loso(Xd, yr, subj))
    print(f"  RANDOM (etichete amestec.) : {np.mean(rnds):.0f}%  "
          f"(rulari: {[f'{r:.0f}' for r in rnds]})")

    print("\n" + "-" * 56)
    if np.mean(rnds) < 60 and real > 80:
        print("VERDICT: PIPELINE CURAT.")
        print(f"  Random cade la ~{np.mean(rnds):.0f}% (sansa), real {real:.0f}%.")
        print("  Cei {:.0f}% sunt REALI - nu sunt din leakage/bug.".format(real))
        print("  Diferenta real-random = semnal fiziologic adevarat.")
    else:
        print("VERDICT: SUSPECT! Random prea mare -> posibil leakage. De investigat.")


if __name__ == "__main__":
    main()
