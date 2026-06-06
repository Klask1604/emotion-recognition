#!/usr/bin/env python3
"""Antreneaza modelul 3-stari pe WESAD folosind FIX cele 5 features ale engine-ului
live (rmssd, mean_hr, sdnn, mean_ibi, pnn50) + baseline personal. Paritate train=serving.
Verific cifra cu 5 features (poate difera de 66% care avea 7+motion), apoi salvez modelul.
Stari: NEUTRU(baseline=1)/STRES(2)/RELAXARE(meditatie=4). Amuse(3) exclus.
"""
from __future__ import annotations

import pickle, sys, warnings, json
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import joblib

WESAD_DIR = ROOT / "datasets" / "WESAD"
OUT = ROOT / "models"
WIN_S, STEP_S = 20, 10
NAME = {0: "NEUTRU", 1: "STRES", 2: "RELAXARE"}
# 5 features HRV ale engine-ului + 2 motion (acc_rms, acc_std) = ce are engine-ul live
FEAT_NAMES = ["rmssd", "mean_hr", "sdnn", "mean_ibi", "pnn50", "acc_rms", "acc_std"]


def features_7(seg, fs, accseg):
    """5 HRV (engine) + 2 motion (acc rms/std). Motion = ce publica engine live."""
    s = seg - seg.mean()
    if s.std() < 1e-6: return None
    b, a = butter(2, [0.5/(fs/2), 4/(fs/2)], btype="band")
    f = filtfilt(b, a, s); pk, _ = find_peaks(f, distance=int(0.4*fs))
    if len(pk) < 5: return None
    ibi = np.diff(pk)/fs*1000.0
    ibi = ibi[(ibi > 300) & (ibi < 1500)]
    if len(ibi) < 4: return None
    diffs = np.diff(ibi)
    rmssd = float(np.sqrt(np.mean(diffs**2))) if len(diffs) else 0.0
    mean_ibi = float(np.mean(ibi))
    mean_hr = 60000.0/mean_ibi if mean_ibi > 0 else 0.0
    sdnn = float(np.std(ibi))
    pnn50 = float(100.0*np.sum(np.abs(diffs) > 50)/len(diffs)) if len(diffs) else 0.0
    mag = np.sqrt((accseg.astype(float)**2).sum(1))
    return [rmssd, mean_hr, sdnn, mean_ibi, pnn50, float(mag.mean()), float(mag.std())]


def personal(X, subj, neu):
    out = np.zeros((len(X), X.shape[1]*2))
    for s in set(subj.tolist()):
        m = subj == s
        base = X[m & neu] if (m & neu).sum() >= 3 else X[m]
        mu, sd = base.mean(0), base.std(0)+1e-8
        out[m] = np.hstack([X[m], (X[m]-mu)/sd])
    return out


def main():
    print("Antrenez 3-stari pe WESAD cu cele 5 features LIVE ale engine-ului...")
    X, lab, subj = [], [], []
    for sd in sorted([p for p in WESAD_DIR.iterdir() if p.is_dir() and (p/f"{p.name}.pkl").exists()]):
        d = pickle.load(open(sd/f"{sd.name}.pkl","rb"), encoding="latin1")
        L = d["label"]; bvp = d["signal"]["wrist"]["BVP"].ravel()
        acc = d["signal"]["wrist"]["ACC"]
        fb, fl = 64.0, 700.0; w, st = int(WIN_S*fb), int(STEP_S*fb)
        for i in range(0, len(bvp)-w+1, st):
            seg = L[int(i/fb*fl):int(i/fb*fl)+int(WIN_S*fl)]
            if len(seg)==0: continue
            v,c = np.unique(seg, return_counts=True); dom = int(v[np.argmax(c)])
            if dom not in (1,2,4): continue
            ai = int(i/fb*32); accseg = acc[ai:ai+int(WIN_S*32)]
            if len(accseg) < 8: continue
            f = features_7(bvp[i:i+w], fb, accseg)
            if f: X.append(f); lab.append(dom); subj.append(sd.name)
    X, lab, subj = np.array(X), np.array(lab), np.array(subj)
    neu = np.isin(lab, [1,4]); Xd = personal(X, subj, neu)
    y = np.array([{1:0, 2:1, 4:2}[l] for l in lab])
    print(f"  {len(y)} ferestre, {len(set(subj.tolist()))} subiecti, {Xd.shape[1]} features (7 abs + 7 dev)\n")

    # LOSO pt cifra onesta cu 5 features
    logo = LeaveOneGroupOut(); yt, yp = [], []
    for tr, te in logo.split(Xd, y, subj):
        if len(set(y[tr].tolist())) < 3: continue
        c = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=150, random_state=0))
        c.fit(Xd[tr], y[tr]); yp.extend(c.predict(Xd[te])); yt.extend(y[te])
    yt, yp = np.array(yt), np.array(yp)
    acc = balanced_accuracy_score(yt, yp)*100
    cm = confusion_matrix(yt, yp, labels=[0,1,2]); cmn = cm/cm.sum(1,keepdims=True)*100
    print("="*52)
    print(f"3 STARI cu 5 HRV + motion | LOSO balanced={acc:.0f}% (sansa 33%)")
    print("="*52)
    for i in range(3):
        print(f"  {NAME[i]:<9}: recall {cmn[i][i]:.0f}%")

    # modelul FINAL antrenat pe TOT (pt serving)
    final = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=150, random_state=0))
    final.fit(Xd, y)
    OUT.mkdir(exist_ok=True)
    bundle = {
        "model": final,
        "feature_names": FEAT_NAMES,       # cele 5 abs ale engine-ului
        "uses_personal_deviation": True,   # serving: concat [abs | (abs-mu)/sd] din baseline personal
        "classes": {0: "NEUTRU", 1: "STRES", 2: "RELAXARE"},
        "loso_balanced_acc": round(acc, 1),
        "loso_recall": {NAME[i]: round(cmn[i][i], 1) for i in range(3)},
        "trained_on": "WESAD wrist BVP->IBI, 5 features live + personal deviation",
        "window_s": WIN_S,
    }
    p = OUT / "stari_3.joblib"
    joblib.dump(bundle, p)
    print(f"\nSalvat model: {p}")
    print(f"  features: {FEAT_NAMES} (+ deviatia personala a fiecaruia)")
    print(f"  serving: engine da cele 5, aplica baseline personal (rest_baseline), concat, model.predict")


if __name__ == "__main__":
    main()
