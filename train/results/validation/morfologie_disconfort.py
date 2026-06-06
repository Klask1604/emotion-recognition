#!/usr/bin/env python3
"""FORMA UNDEI PPG (features morfologice explicite, nu CNN) pe DISCONFORT vs PLACUT.
Gaura dovedita: disconfort-vs-placut = 52% (sansa) din HRV. Vasoconstrictia la disconfort
schimba FORMA undei -> poate morfologia sparge acolo unde HRV nu.

Features morfologice per bataie, mediate pe fereastra:
  amplitudine, panta urcare (rise time), latime la 50%, arie sub unda, raport sistolic/diastolic.
Compar WESAD 64Hz (forma BUNA, dar E4) vs EMOGNITION 20Hz (Galaxy real, grosier).
Daca WESAD da semnal si EMOG nu -> e rezolutia (100Hz ar ajuta). LOSO personal, RF.
"""
from __future__ import annotations
import json, pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score

WESAD=ROOT/"datasets"/"WESAD"; EMOG=ROOT/"datasets"/"EMOGNITION"
WIN_S, STEP_S = 20, 10


def morph_features(sig, fs):
    """Features de FORMA per fereastra: morfologia pulsului mediata pe batai."""
    s = sig - np.mean(sig)
    if np.std(s) < 1e-6: return None
    b,a = butter(2,[0.5/(fs/2),min(8,fs/2-0.1)/(fs/2)],btype="band")
    f = filtfilt(b,a,s)
    pk,_ = find_peaks(f, distance=int(0.4*fs))
    if len(pk) < 4: return None
    amps, rises, widths, areas = [], [], [], []
    for i in range(1,len(pk)):
        seg = f[pk[i-1]:pk[i]]
        if len(seg) < 3: continue
        amp = f[pk[i]] - seg.min()
        amps.append(amp)
        # rise time: de la minim la varf
        mn = np.argmin(seg)
        rises.append((len(seg)-mn)/fs*1000)  # ms de la minim la urmatorul varf
        # latime la 50% amplitudine
        half = seg.min() + amp*0.5
        above = np.sum(seg > half)
        widths.append(above/fs*1000)
        # arie sub unda (pulsatilitate)
        areas.append(np.trapz(seg - seg.min())/fs)
    if len(amps) < 3: return None
    def st(x): return [np.mean(x), np.std(x)]
    return st(amps)+st(rises)+st(widths)+st(areas)


def dev(X,subj,base_mask):
    out=np.zeros((len(X),X.shape[1]*2))
    for s in set(subj.tolist()):
        m=subj==s; bm=m&base_mask; base=X[bm] if bm.sum()>=3 else X[m]
        mu,sd=base.mean(0),base.std(0)+1e-8; out[m]=np.hstack([X[m],(X[m]-mu)/sd])
    return out


def rf(): return make_pipeline(StandardScaler(),RandomForestClassifier(150,random_state=0,n_jobs=-1,class_weight="balanced"))


def loso(X,y,g):
    logo=LeaveOneGroupOut(); yt,yp=[],[]
    for tr,te in logo.split(X,y,g):
        if len(set(y[tr].tolist()))<2: continue
        c=rf(); c.fit(X[tr],y[tr]); yp.extend(c.predict(X[te])); yt.extend(y[te])
    return balanced_accuracy_score(np.array(yt),np.array(yp))*100 if yt else float("nan")


def wesad():
    # disconfort=stres(2), placut=amuse(3); baseline pt deviatie
    X,lab,subj=[],[],[]
    for sd in sorted([p for p in WESAD.iterdir() if p.is_dir() and (p/f"{p.name}.pkl").exists()]):
        d=pickle.load(open(sd/f"{sd.name}.pkl","rb"),encoding="latin1")
        L=d["label"]; bvp=d["signal"]["wrist"]["BVP"].ravel()
        fb,fl=64.0,700.0; w,st=int(WIN_S*fb),int(STEP_S*fb)
        for i in range(0,len(bvp)-w+1,st):
            seg=L[int(i/fb*fl):int(i/fb*fl)+int(WIN_S*fl)]
            if len(seg)==0: continue
            v,c=np.unique(seg,return_counts=True); dom=int(v[np.argmax(c)])
            if dom not in (1,2,3,4): continue  # 1,4=baseline pt deviatie
            mf=morph_features(bvp[i:i+w],fb)
            if mf: X.append(mf); lab.append(dom); subj.append(sd.name)
    X,lab,subj=np.array(X),np.array(lab),np.array(subj)
    base=np.isin(lab,[1,4])
    Xd=dev(X,subj,base)
    m=np.isin(lab,[2,3])  # disconfort vs placut
    y=(lab[m]==2).astype(int)
    return loso(Xd[m],y,subj[m]), m.sum()


def emognition():
    DISC={"ANGER","FEAR","DISGUST","SADNESS"}; PLAC={"AMUSEMENT","ENTHUSIASM","AWE","LIKING"}; NEU={"BASELINE","NEUTRAL"}
    X,lab,subj=[],[],[]
    for sd in sorted([p for p in EMOG.iterdir() if p.is_dir() and p.name.isdigit()],key=lambda p:int(p.name)):
        for e in DISC|PLAC|NEU:
            bf=sd/f"{sd.name}_{e}_STIMULUS_SAMSUNG_WATCH.json"
            if not bf.exists(): continue
            dd=json.load(open(bf)); raw=dd.get("BVPProcessed") or []
            if len(raw)<WIN_S: continue
            t=(pd.to_datetime([r[0] for r in raw],format="%Y-%m-%dT%H:%M:%S:%f").astype("int64")//1_000_000).to_numpy()
            v=np.array([float(r[1]) for r in raw])
            fs=round(1000/np.median(np.diff(t))) if len(t)>2 else 20
            cur=t[0]
            while cur+WIN_S*1000<=t[-1]:
                msk=(t>=cur)&(t<cur+WIN_S*1000)
                if msk.sum()>=WIN_S:
                    mf=morph_features(v[msk],fs)
                    if mf:
                        st=0 if e in NEU else (1 if e in DISC else 2)
                        X.append(mf); lab.append(st); subj.append(int(sd.name))
                cur+=STEP_S*1000
    X,lab,subj=np.array(X),np.array(lab),np.array(subj)
    base=lab==0
    Xd=dev(X,subj,base)
    m=np.isin(lab,[1,2])
    y=(lab[m]==1).astype(int)
    return loso(Xd[m],y,subj[m]), m.sum()


def main():
    print("FORMA UNDEI (morfologie explicita) pe DISCONFORT vs PLACUT", flush=True)
    print("(HRV singur a dat 52% pe perechea asta = sansa)\n", flush=True)
    print("Incarc WESAD 64Hz (forma buna)...", flush=True)
    w,nw=wesad()
    print(f"  WESAD 64Hz  : {w:.0f}%  ({nw} ferestre)\n", flush=True)
    print("Incarc EMOGNITION 20Hz (Galaxy real, grosier)...", flush=True)
    e,ne=emognition()
    print(f"  EMOG 20Hz   : {e:.0f}%  ({ne} ferestre)\n", flush=True)
    print("-"*56)
    print("CITIRE:")
    print(f"  WESAD 64Hz = {w:.0f}%, EMOG 20Hz = {e:.0f}% (sansa 50%)")
    if w>58 and e<=55:
        print("  => forma ajuta la rezolutie BUNA (64Hz) dar nu la 20Hz.")
        print("     100Hz-ul tau AR ajuta! Morfologia e reala, cere rezolutie.")
    elif w<=55 and e<=55:
        print("  => forma NU ajuta nici la 64Hz. Morfologia chiar nu separa valenta.")
        print("     PPG 100Hz = doar viz, poti opri pt baterie.")
    else:
        print("  => rezultat mixt, de interpretat.")


if __name__=="__main__": main()
