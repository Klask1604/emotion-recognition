#!/usr/bin/env python3
"""VALIDARE robustete model 3 stari pe EMOGNITION Galaxy:
  1) PERMUTATION TEST pe 3 clase (etichete amestecate -> trebuie ~33% = sansa)
  2) BINAR PER PERECHE: CALM-vs-DISCONFORT, CALM-vs-PLACUT, DISCONFORT-vs-PLACUT
     (arata unde modelul SEPARA clar vs unde se topeste)
  3) variabilitate intre subiecti (cat de stabil e 65%)
Tot LOSO personal, RF.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score

EMOG=Path(ROOT/"datasets"/"EMOGNITION"); WIN_S,STEP_S=20,10
ST={"BASELINE":0,"NEUTRAL":0,"ANGER":1,"FEAR":1,"DISGUST":1,"SADNESS":1,"AMUSEMENT":2,"ENTHUSIASM":2,"AWE":2,"LIKING":2}
NAME={0:"CALM",1:"DISCONFORT",2:"PLACUT"}

def hrv(ibi):
    ibi=ibi[(ibi>300)&(ibi<1500)]
    if len(ibi)<5: return None
    d=np.diff(ibi)
    return [np.sqrt(np.mean(d**2)),60000.0/np.mean(ibi),np.std(ibi),np.mean(ibi),np.mean(np.abs(d)>50)*100]

def load():
    X,lab,subj=[],[],[]
    for sd in sorted([p for p in EMOG.iterdir() if p.is_dir() and p.name.isdigit()],key=lambda p:int(p.name)):
        for e,st in ST.items():
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
                m=(t>=cur)&(t<cur+WIN_S*1000); h=hrv(v[m])
                if h: X.append(h+[float(accm.mean()),float(accm.std()),float(gyrm.mean()),float(gyrm.std())]); lab.append(st); subj.append(int(sd.name))
                cur+=STEP_S*1000
    return np.array(X),np.array(lab),np.array(subj)

def dev(X,subj,lab):
    out=np.zeros((len(X),X.shape[1]*2))
    for s in set(subj.tolist()):
        m=subj==s; neu=m&(lab==0); base=X[neu] if neu.sum()>=3 else X[m]
        mu,sd=base.mean(0),base.std(0)+1e-8; out[m]=np.hstack([X[m],(X[m]-mu)/sd])
    return out

def rf(): return make_pipeline(StandardScaler(),RandomForestClassifier(150,random_state=0,n_jobs=-1,class_weight="balanced"))

def loso(X,lab,subj, per_subj=False):
    logo=LeaveOneGroupOut(); yt,yp=[],[]; subj_accs=[]
    for tr,te in logo.split(X,lab,subj):
        if len(set(lab[tr].tolist()))<2: continue
        c=rf(); c.fit(X[tr],lab[tr]); pred=c.predict(X[te])
        yp.extend(pred); yt.extend(lab[te])
        if per_subj and len(set(lab[te].tolist()))>=1:
            try: subj_accs.append(balanced_accuracy_score(lab[te],pred)*100)
            except: pass
    acc=balanced_accuracy_score(np.array(yt),np.array(yp))*100
    return (acc, subj_accs) if per_subj else acc

def main():
    print("Incarc EMOGNITION...", flush=True)
    X,lab,subj=load(); Xd=dev(X,subj,lab)
    print(f"  {len(lab)} ferestre, {len(set(subj.tolist()))} subj\n", flush=True)

    print("="*56); print("1) PERMUTATION TEST 3 clase (random -> trebuie ~33%)"); print("="*56)
    real,sa=loso(Xd,lab,subj,per_subj=True)
    print(f"  REAL (etichete corecte) : {real:.0f}%", flush=True)
    rnds=[]
    for seed in range(3):
        rng=np.random.default_rng(seed); yr=lab.copy()
        for s in set(subj.tolist()):
            m=subj==s; yr[m]=rng.permutation(yr[m])
        rnds.append(loso(Xd,yr,subj))
    print(f"  RANDOM (amestecate)     : {np.mean(rnds):.0f}%  {[f'{r:.0f}' for r in rnds]}", flush=True)
    print(f"  => {'CURAT (real>>random)' if real-np.mean(rnds)>15 else 'SUSPECT'}\n", flush=True)

    print("="*56); print("2) BINAR PER PERECHE (unde separa clar vs se topeste)"); print("="*56)
    for a,b in [(0,1),(0,2),(1,2)]:
        m=np.isin(lab,[a,b]); yb=(lab[m]==b).astype(int)
        acc=loso(Xd[m],yb,subj[m])
        print(f"  {NAME[a]:10} vs {NAME[b]:10}: {acc:.0f}%", flush=True)

    print("\n"+"="*56); print("3) STABILITATE intre subiecti"); print("="*56)
    if sa:
        print(f"  media={np.mean(sa):.0f}% std={np.std(sa):.0f}% min={np.min(sa):.0f}% max={np.max(sa):.0f}%")
        print(f"  => {'stabil' if np.std(sa)<15 else 'variaza mult intre oameni'}")

if __name__=="__main__": main()
