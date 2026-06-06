#!/usr/bin/env python3
"""DOVADA: metoda EMOGNITION/Nandini (split aleatoriu ferestre) vs onest (LOSO).
ACELASI model, ACELEASI date, valenta binara. Singura diferenta = cum imparti train/test.

  A) SPLIT ALEATORIU FERESTRE (ca Nandini): ferestre din acelasi clip in train SI test
  B) LOSO subiect-nou (onest):              subiectul de test NU e in train

Daca A da 90%+ si B da 50% -> diferenta E leakage-ul lor. Cifra lor de 97% e A.
Pe TINE live = cazul B (esti subiect nou). De-aia metoda lor NU functioneaza pe tine.
"""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.metrics import balanced_accuracy_score

EMOG_DIR = ROOT / "datasets" / "EMOGNITION"
EMOTIONS = ["AMUSEMENT","ANGER","AWE","BASELINE","DISGUST","ENTHUSIASM",
            "FEAR","LIKING","NEUTRAL","SADNESS","SURPRISE"]
WIN_S, STEP_S = 20, 10
HV, LV = 6, 4


def hrv(ibi):
    ibi = ibi[(ibi > 300) & (ibi < 1500)]
    if len(ibi) < 5: return None
    return np.array([60000.0/np.mean(ibi), np.std(ibi), np.sqrt(np.mean(np.diff(ibi)**2)),
                     np.mean(np.abs(np.diff(ibi))>50)*100, np.median(ibi),
                     np.percentile(ibi,20), np.percentile(ibi,80)])


def load():
    X, y, subj = [], [], []
    for sd in sorted([p for p in EMOG_DIR.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name)):
        qf = sd/f"{sd.name}_QUESTIONNAIRES.json"
        if not qf.exists(): continue
        q = json.load(open(qf)).get("questionnaires", [])
        lab = {it["movie"]: int(it["sam"]["VALENCE"]) for it in q
               if it.get("movie") and (it.get("sam") or {}).get("VALENCE") is not None}
        for e in EMOTIONS:
            if e not in lab: continue
            v = lab[e]; yv = 1 if v>=HV else (0 if v<=LV else -1)
            if yv < 0: continue
            sf = sd/f"{sd.name}_{e}_STIMULUS_SAMSUNG_WATCH.json"
            if not sf.exists(): continue
            pp = json.load(open(sf)).get("PPInterval") or []
            if len(pp) < WIN_S: continue
            t = (pd.to_datetime([r[0] for r in pp], format="%Y-%m-%dT%H:%M:%S:%f").astype("int64")//1_000_000).to_numpy()
            vv = np.array([float(r[1]) for r in pp])
            cur = t[0]
            while cur + WIN_S*1000 <= t[-1]:
                m = (t>=cur)&(t<cur+WIN_S*1000)
                f = hrv(vv[m])
                if f is not None:
                    X.append(f); y.append(yv); subj.append(int(sd.name))
                cur += STEP_S*1000
    return np.array(X), np.array(y), np.array(subj)


def main():
    print("Incarc EMOGNITION valenta binara...")
    X, y, subj = load()
    print(f"  {len(y)} ferestre, {len(set(subj.tolist()))} subiecti\n")

    def clf():
        return make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=120, random_state=0))

    print("="*56)
    print("ACELASI MODEL, ACELEASI DATE - doar SPLIT-ul difera")
    print("="*56)

    # A) split aleatoriu ferestre (ca Nandini) - mediat pe 5 rulari
    accs = []
    for seed in range(5):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        c = clf(); c.fit(Xtr, ytr)
        accs.append(balanced_accuracy_score(yte, c.predict(Xte))*100)
    a = np.mean(accs)
    print(f"  A) SPLIT ALEATORIU FERESTRE (Nandini): {a:.0f}%  <- cifra 'frumoasa'")

    # B) LOSO onest
    logo = LeaveOneGroupOut(); yt, yp = [], []
    for tr, te in logo.split(X, y, subj):
        if len(set(y[tr].tolist())) < 2: continue
        c = clf(); c.fit(X[tr], y[tr]); yp.extend(c.predict(X[te])); yt.extend(y[te])
    b = balanced_accuracy_score(yt, yp)*100
    print(f"  B) LOSO subiect-nou (ONEST)          : {b:.0f}%  <- realitatea pe TINE")

    print("\n" + "-"*56)
    print(f"  LEAKAGE-ul lor = A - B = {a-b:.0f} puncte procentuale")
    print(f"  Cifra lor 'frumoasa' ({a:.0f}%) vine din ferestre din acelasi clip")
    print(f"  in train+test. Pe TINE (subiect nou) modelul lor da {b:.0f}% = sansa.")
    print("  De-aia metoda EMOGNITION NU functioneaza in realitate. Nu e magie, e bug.")


if __name__ == "__main__":
    main()
