#!/usr/bin/env python3
"""Antreneaza al 2-lea model: MORFOLOGIE PPG -> 3 stari, pt valenta (axa 2).
Features morfologice invariante la fs (verificat: 64Hz~100Hz <2% diff): rise_ms, width_ms,
amp, area (+ deviatie personala). Pe WESAD (64Hz, forma buna). Ruleaza pe ceas 100Hz live.

Complementar modelului HRV: HRV da CALM-vs-activat (97%), morfologia da disconfort-vs-placut
(unde HRV cade la 52%). Salveaza models/morfologie_3.joblib.
"""
from __future__ import annotations
import pickle, sys, warnings
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import joblib

WESAD=ROOT/"datasets"/"WESAD"; OUT=ROOT/"models"; WIN_S,STEP_S=20,10
NAME={0:"CALM",1:"DISCONFORT",2:"PLACUT"}
# features in unitati REALE (ms, amplitudine) = invariante la fs
FEAT=["amp_mean","amp_std","rise_ms_mean","rise_ms_std","width_ms_mean","width_ms_std","area_mean","area_std"]


def morph(sig, fs):
    s=sig-np.mean(sig)
    if np.std(s)<1e-6: return None
    b,a=butter(2,[0.5/(fs/2),min(8,fs/2-0.1)/(fs/2)],btype="band"); f=filtfilt(b,a,s)
    pk,_=find_peaks(f,distance=int(0.4*fs))
    if len(pk)<4: return None
    amps,rises,widths,areas=[],[],[],[]
    for i in range(1,len(pk)):
        seg=f[pk[i-1]:pk[i]]
        if len(seg)<3: continue
        amp=f[pk[i]]-seg.min(); amps.append(amp)
        mn=np.argmin(seg); rises.append((len(seg)-mn)/fs*1000)
        half=seg.min()+amp*0.5; widths.append(np.sum(seg>half)/fs*1000)
        areas.append(np.trapezoid(seg-seg.min())/fs)
    if len(amps)<3: return None
    def st(x): return [np.mean(x),np.std(x)]
    return st(amps)+st(rises)+st(widths)+st(areas)


def dev(X,subj,lab):
    out=np.zeros((len(X),X.shape[1]*2))
    for s in set(subj.tolist()):
        m=subj==s; neu=m&np.isin(lab,[0]); base=X[neu] if neu.sum()>=3 else X[m]
        mu,sd=base.mean(0),base.std(0)+1e-8; out[m]=np.hstack([X[m],(X[m]-mu)/sd])
    return out


def rf(): return make_pipeline(StandardScaler(),RandomForestClassifier(200,random_state=0,n_jobs=-1,class_weight="balanced"))


def main():
    print("Antrenez model MORFOLOGIE pe WESAD 64Hz...", flush=True)
    X,lab,subj=[],[],[]
    for sd in sorted([p for p in WESAD.iterdir() if p.is_dir() and (p/f"{p.name}.pkl").exists()]):
        d=pickle.load(open(sd/f"{sd.name}.pkl","rb"),encoding="latin1")
        L=d["label"]; bvp=d["signal"]["wrist"]["BVP"].ravel()
        fb,fl=64.0,700.0; w,st=int(WIN_S*fb),int(STEP_S*fb)
        for i in range(0,len(bvp)-w+1,st):
            seg=L[int(i/fb*fl):int(i/fb*fl)+int(WIN_S*fl)]
            if len(seg)==0: continue
            v,c=np.unique(seg,return_counts=True); dom=int(v[np.argmax(c)])
            if dom not in (1,2,3,4): continue
            st_lab={1:0,4:0,2:1,3:2}[dom]  # calm / disconfort(stres) / placut(amuse)
            mf=morph(bvp[i:i+w],fb)
            if mf: X.append(mf); lab.append(st_lab); subj.append(sd.name)
    X,lab,subj=np.array(X),np.array(lab),np.array(subj)
    Xd=dev(X,subj,lab)
    print(f"  {len(lab)} ferestre, {len(set(subj.tolist()))} subj (CALM={np.sum(lab==0)} DISC={np.sum(lab==1)} PLAC={np.sum(lab==2)})\n", flush=True)

    logo=LeaveOneGroupOut(); yt,yp=[],[]
    for tr,te in logo.split(Xd,lab,subj):
        if len(set(lab[tr].tolist()))<3: continue
        c=rf(); c.fit(Xd[tr],lab[tr]); yp.extend(c.predict(Xd[te])); yt.extend(lab[te])
    yt,yp=np.array(yt),np.array(yp)
    acc=balanced_accuracy_score(yt,yp)*100
    cm=confusion_matrix(yt,yp,labels=[0,1,2]); rec=[cm[i][i]/cm[i].sum()*100 for i in range(3)]
    print(f"MORFOLOGIE 3 stari LOSO: {acc:.0f}%", flush=True)
    for i in range(3): print(f"  {NAME[i]}: {rec[i]:.0f}%", flush=True)
    # binar disconfort-vs-placut (unde HRV cade la 52)
    m=np.isin(lab,[1,2]);
    yt2,yp2=[],[]
    for tr,te in logo.split(Xd,lab,subj):
        trm=tr[np.isin(lab[tr],[1,2])]; tem=te[np.isin(lab[te],[1,2])]
        if len(set(lab[trm].tolist()))<2 or len(tem)==0: continue
        c=rf(); c.fit(Xd[trm],(lab[trm]==1).astype(int))
        yp2.extend(c.predict(Xd[tem])); yt2.extend((lab[tem]==1).astype(int))
    if yt2: print(f"\n  DISCONFORT vs PLACUT (morfologie): {balanced_accuracy_score(yt2,yp2)*100:.0f}%  (HRV dadea 52%)", flush=True)

    fm=rf(); fm.fit(Xd,lab)
    OUT.mkdir(exist_ok=True)
    joblib.dump({"model":fm,"feature_names":FEAT,"uses_personal_deviation":True,
                 "classes":{0:"CALM",1:"DISCONFORT",2:"PLACUT"},"loso_balanced_acc":round(acc,1),
                 "loso_recall":{NAME[i]:round(rec[i],1) for i in range(3)},
                 "trained_on":"WESAD 64Hz morfologie PPG (rise_ms/width_ms/amp/area, invariant fs)",
                 "fs_invariant":True,"window_s":WIN_S}, OUT/"morfologie_3.joblib")
    print(f"\nSalvat models/morfologie_3.joblib ({acc:.0f}% LOSO)", flush=True)


if __name__=="__main__": main()
