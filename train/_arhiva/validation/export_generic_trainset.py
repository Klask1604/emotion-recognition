#!/usr/bin/env python3
"""One-time export of the generic EMOGNITION training matrix for the personal
retrain loop.

The personal trainer runs INSIDE the engine (Docker), where datasets/ is not
mounted. It needs the generic training set to combine with the user's own labels
(the proven recipe: generic + personal beats generic alone, 65% -> 89% at 5/state).
Instead of mounting 16 GB of EMOGNITION, we extract the 9-feature matrix ONCE here
and ship it next to the model as a small .npz (tens of KB).

Output: models/stari_3_generic.npz with X (n,9), y (n,), subj (n,) in the SAME
9-feature space the live state_classifier produces:
  [rmssd, mean_hr, sdnn, mean_ibi, pnn50, acc_rms, acc_std, gyro_rms, gyro_std].

This is the exact loader from curba_etichete_personal.py (the test that proved the
loop works), so train=serving parity holds.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

EMOG = ROOT / "datasets" / "EMOGNITION"
OUT = ROOT / "models" / "stari_3_generic.npz"
WIN_S, STEP_S = 20, 10
# EMOGNITION stimulus -> 3 states (CALM=0, DISCONFORT=1, PLACUT=2), same as the test.
ST = {
    "BASELINE": 0, "NEUTRAL": 0,
    "ANGER": 1, "FEAR": 1, "DISGUST": 1, "SADNESS": 1,
    "AMUSEMENT": 2, "ENTHUSIASM": 2, "AWE": 2, "LIKING": 2,
}


def hrv(ibi):
    ibi = ibi[(ibi > 300) & (ibi < 1500)]
    if len(ibi) < 5:
        return None
    d = np.diff(ibi)
    return [
        np.sqrt(np.mean(d ** 2)),       # rmssd
        60000.0 / np.mean(ibi),         # mean_hr
        np.std(ibi),                    # sdnn
        np.mean(ibi),                   # mean_ibi
        np.mean(np.abs(d) > 50) * 100,  # pnn50
    ]


def load():
    X, lab, subj = [], [], []
    for sd in sorted([p for p in EMOG.iterdir() if p.is_dir() and p.name.isdigit()],
                     key=lambda p: int(p.name)):
        for e, st in ST.items():
            bf = sd / f"{sd.name}_{e}_STIMULUS_SAMSUNG_WATCH.json"
            if not bf.exists():
                continue
            dd = json.load(open(bf))
            pp = dd.get("PPInterval") or []
            if len(pp) < WIN_S:
                continue
            t = (pd.to_datetime([r[0] for r in pp], format="%Y-%m-%dT%H:%M:%S:%f")
                 .astype("int64") // 1_000_000).to_numpy()
            v = np.array([float(r[1]) for r in pp])
            acc = dd.get("acc") or []
            gyr = dd.get("gyr") or []
            accm = (np.array([np.linalg.norm([float(x) for x in a[1:4]]) for a in acc])
                    if acc else np.zeros(1))
            gyrm = (np.array([np.linalg.norm([float(x) for x in g[1:4]]) for g in gyr])
                    if gyr else np.zeros(1))
            cur = t[0]
            while cur + WIN_S * 1000 <= t[-1]:
                m = (t >= cur) & (t < cur + WIN_S * 1000)
                h = hrv(v[m])
                if h:
                    X.append(h + [float(accm.mean()), float(accm.std()),
                                  float(gyrm.mean()), float(gyrm.std())])
                    lab.append(st)
                    subj.append(int(sd.name))
                cur += STEP_S * 1000
    return np.array(X), np.array(lab), np.array(subj)


def main():
    print("Extracting generic EMOGNITION matrix (9 features)...", flush=True)
    X, lab, subj = load()
    if len(X) == 0:
        print("ERROR: no windows extracted. Is datasets/EMOGNITION present?")
        sys.exit(1)
    n_states = len(set(lab.tolist()))
    print(f"  {len(X)} windows, {len(set(subj.tolist()))} subjects, {n_states} states")
    for s in sorted(set(lab.tolist())):
        print(f"    state {s}: {int((lab == s).sum())} windows")
    # Key names match what personal_trainer.maybe_retrain() reads: X, y, subj.
    np.savez(OUT, X=X, y=lab, subj=subj)
    print(f"Saved {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
