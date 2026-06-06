#!/usr/bin/env python3
"""Verifica paritate: features REALE vs APROXIMARE (mean+/-0.84sdnn) vs FARA percentile.
Daca aproximarea ~ real -> ok de pus live. Daca difera -> re-antrenez fara percentile.
EMOGNITION Galaxy, 3 stari disconfort, LOSO, RF rapid.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score

EMOG_DIR = ROOT / "datasets" / "EMOGNITION"
WIN_S, STEP_S = 20, 10
EMOG_STATE = {"BASELINE":0,"NEUTRAL":0,"ANGER":1,"FEAR":1,"DISGUST":1,"SADNESS":1,
              "AMUSEMENT":2,"ENTHUSIASM":2,"AWE":2,"LIKING":2}


def base_hrv(ibi):
    """rmssd, mean_hr, sdnn, pnn50, mean_ibi - ce ARE engine-ul sigur."""
    ibi = ibi[(ibi>300)&(ibi<1500)]
    if len(ibi)<5: return None
    d=np.diff(ibi)
    return {
        "rmssd": np.sqrt(np.mean(d**2)), "mean_hr": 60000.0/np.mean(ibi),
        "sdnn": np.std(ibi), "pnn50": np.mean(np.abs(d)>50)*100,
        "mean_ibi": np.mean(ibi),
        "median_real": np.median(ibi), "p20_real": np.percentile(ibi,20), "p80_real": np.percentile(ibi,80),
    }


def rf(): return make_pipeline(StandardScaler(), RandomForestClassifier(150,random_state=0,n_jobs=-1,class_weight="balanced"))


def load():
    rows, lab, subj = [], [], []
    for sd in sorted([p for p in EMOG_DIR.iterdir() if p.is_dir() and p.name.isdigit()],key=lambda p:int(p.name)):
        for e,st in EMOG_STATE.items():
            bf=sd/f"{sd.name}_{e}_STIMULUS_SAMSUNG_WATCH.json"
            if not bf.exists(): continue
            dd=json.load(open(bf)); pp=dd.get("PPInterval") or []
            if len(pp)<WIN_S: continue
            t=(pd.to_datetime([r[0] for r in pp],format="%Y-%m-%dT%H:%M:%S:%f").astype("int64")//1_000_000).to_numpy()
            v=np.array([float(r[1]) for r in pp]); acc=dd.get("acc") or []; gyr=dd.get("gyr") or []
            accm=np.array([np.linalg.norm([float(x) for x in a[1:4]]) for a in acc]) if acc else np.zeros(1)
            gyrm=np.array([np.linalg.norm([float(x) for x in g[1:4]]) for g in gyr]) if gyr else np.zeros(1)
            cur=t[0]
            while cur+WIN_S*1000<=t[-1]:
                m=(t>=cur)&(t<cur+WIN_S*1000); h=base_hrv(v[m])
                if h:
                    h["acc_rms"]=float(accm.mean()); h["acc_std"]=float(accm.std())
                    h["gyro_rms"]=float(gyrm.mean()); h["gyro_std"]=float(gyrm.std())
                    rows.append(h); lab.append(st); subj.append(int(sd.name))
                cur+=STEP_S*1000
    return rows, np.array(lab), np.array(subj)


def build(rows, mode):
    """mode: 'real' / 'approx' / 'none'."""
    X=[]
    for h in rows:
        base=[h["rmssd"],h["mean_hr"],h["sdnn"],h["pnn50"],h["mean_ibi"]]
        mot=[h["acc_rms"],h["acc_std"],h["gyro_rms"],h["gyro_std"]]
        if mode=="real":
            pct=[h["median_real"],h["p20_real"],h["p80_real"]]
        elif mode=="approx":
            pct=[h["mean_ibi"], h["mean_ibi"]-0.84*h["sdnn"], h["mean_ibi"]+0.84*h["sdnn"]]
        else:
            pct=[]
        X.append(base+pct+mot)
    return np.array(X)


def dev(X,subj,lab):
    out=np.zeros((len(X),X.shape[1]*2))
    for s in set(subj.tolist()):
        m=subj==s; neu=m&(lab==0)
        base=X[neu] if neu.sum()>=3 else X[m]
        mu,sd=base.mean(0),base.std(0)+1e-8
        out[m]=np.hstack([X[m],(X[m]-mu)/sd])
    return out


def loso(X,lab,subj):
    logo=LeaveOneGroupOut(); yt,yp=[],[]
    for tr,te in logo.split(X,lab,subj):
        if len(set(lab[tr].tolist()))<3: continue
        c=rf(); c.fit(X[tr],lab[tr]); yp.extend(c.predict(X[te])); yt.extend(lab[te])
    return balanced_accuracy_score(np.array(yt),np.array(yp))*100


def main():
    print("Incarc EMOGNITION...", flush=True)
    rows,lab,subj=load()
    print(f"  {len(rows)} ferestre\n", flush=True)
    print("="*50)
    print("PARITATE: features REALE vs APROXIMARE vs FARA percentile")
    print("="*50)
    for mode,name in [("real","REALE (median/p20/p80 din IBI)"),
                      ("approx","APROXIMARE (mean+/-0.84sdnn) = ce da engine-ul"),
                      ("none","FARA percentile (doar ce e sigur)")]:
        X=build(rows,mode); Xd=dev(X,subj,lab); acc=loso(Xd,lab,subj)
        print(f"  {name:<42}: {acc:.0f}%", flush=True)
    print("\nDaca APROXIMARE ~ REALE -> aproximarea e OK live.")
    print("Daca FARA percentile ~ REALE -> scoatem percentilele, paritate garantata, zero risc.")


if __name__=="__main__": main()
