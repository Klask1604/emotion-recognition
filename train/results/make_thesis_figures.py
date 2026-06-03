#!/usr/bin/env python3
"""Regenereaza TOATE figurile de rezultate ale lucrarii, din datele brute, intr-un
singur loc consistent.

Reguli (cerute de autor):
  - acelasi clasificator peste tot: LogisticRegression (ce ruleaza in prezent),
    StandardScaler in fata, class_weight='balanced'. Fara cifre hardcodate.
  - acuratete = balanced accuracy sub LOSO (leave-one-subject-out), pragul sansei
    50% pe doua clase.
  - toate seturile cu adnotarile necesare.
  - iesirea merge in demo_documentatie/figuri/ (langa .tex), nu in train/figs.

Ruleaza din radacina proiectului:
  ./venv/Scripts/python.exe train/results/make_thesis_figures.py
"""

from __future__ import annotations

import glob
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = Path(r"C:/Users/doltu/Desktop/demo_documentatie/figuri")
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25})

# Toate cifrele se aadauga aici pe masura ce sunt calculate, ca sa le pot
# tipari la final si sa le aliniez cu .tex.
NUM: dict[str, float] = {}


# --------------------------------------------------------------------- model
def _model():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def _loso(X, y, s) -> float:
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.base import clone
    rows = np.all(np.isfinite(X), axis=1)
    X, y, s = X[rows], y[rows], s[rows]
    accs = []
    for tr, te in LeaveOneGroupOut().split(X, y, s):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = make_pipeline(StandardScaler(), clone(_model()))
        m.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], m.predict(X[te])))
    return float(np.mean(accs)) * 100 if accs else 0.0


def _random_split(X, y) -> float:
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = np.all(np.isfinite(X), axis=1)
    X, y = X[rows], y[rows]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pipe = make_pipeline(StandardScaler(), _model())
    return cross_val_score(pipe, X, y, cv=skf,
                           scoring="balanced_accuracy").mean() * 100


def _personal(X, y, s) -> float:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.base import clone
    rows = np.all(np.isfinite(X), axis=1)
    X, y, s = X[rows], y[rows], s[rows]
    accs = []
    for u in np.unique(s):
        m = s == u
        Xu, yu = X[m], y[m]
        if len(np.unique(yu)) < 2 or len(yu) < 10:
            continue
        nsplit = min(5, int(np.bincount(yu).min()))
        if nsplit < 2:
            continue
        skf = StratifiedKFold(n_splits=nsplit, shuffle=True, random_state=0)
        for tr, te in skf.split(Xu, yu):
            if len(np.unique(yu[tr])) < 2:
                continue
            c = make_pipeline(StandardScaler(), clone(_model()))
            c.fit(Xu[tr], yu[tr])
            accs.append(balanced_accuracy_score(yu[te], c.predict(Xu[te])))
    return float(np.mean(accs)) * 100 if accs else float("nan")


# --------------------------------------------------------------------- loaders
def _valence(name):
    """(X, y, subjects) pentru valenta binara negativ(0)/pozitiv(1)."""
    if name == "WESAD":
        d = np.load(ROOT / "data" / "wesad_features.npz", allow_pickle=True)
        return d["X"], d["y"].astype(int), d["subjects"]
    if name == "CASE":
        d = np.load(ROOT / "data" / "case_stratified.npz", allow_pickle=True)
        v = d["valence"]; sel = (v <= 3) | (v >= 7)
        return d["Xb"][sel], (v[sel] >= 7).astype(int), d["subjects"][sel]
    if name == "EEVR":
        d = np.load(ROOT / "data" / "eevr_stratified.npz", allow_pickle=True)
        q = d["quad"]
        return d["Xb"], np.isin(q, ["HVHA", "HVLA"]).astype(int), d["subjects"]
    if name == "EmoWear":
        d = np.load(ROOT / "data" / "emowear_features.npz", allow_pickle=True)
        v = d["valence"]; sel = (v <= 3) | (v >= 7)
        if sel.sum() < 20:
            y = (v >= np.median(v)).astype(int)
            return d["X"], y, d["subjects"]
        return d["X"][sel], (v[sel] >= 7).astype(int), d["subjects"][sel]
    if name == "DEAP":
        d = np.load(ROOT / "data" / "deap_valence_fd.npz", allow_pickle=True)
        return d["X"], d["y"].astype(int), d["subjects"]
    raise KeyError(name)


# ============================================================ FIG 1: split gap
def fig_split_inflation():
    """Split aleatoriu vs LOSO pe valenta, pentru cele 3 seturi cu adnotari clare."""
    sets = ["WESAD", "CASE", "EEVR"]
    rand, loso = [], []
    for n in sets:
        X, y, s = _valence(n)
        rand.append(_random_split(X, y))
        loso.append(_loso(X, y, s))
    for n, r, l in zip(sets, rand, loso):
        NUM[f"split_{n}_rand"] = r
        NUM[f"split_{n}_loso"] = l

    x = np.arange(len(sets)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, rand, w, color="#d98880",
                label="Split aleatoriu (subiectul se scurge)")
    b2 = ax.bar(x + w / 2, loso, w, color="#5dade2",
                label="LOSO (cinstit, cross-subiect)")
    ax.axhline(50, ls="--", c="gray", lw=1)
    ax.text(len(sets) - 0.55, 51, "sansa (50%)", color="gray", fontsize=9)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{b.get_height():.0f}", ha="center", fontsize=10)
    ax.set_ylabel("Acuratete echilibrata (%)"); ax.set_ylim(0, 100)
    ax.set_xticks(x); ax.set_xticklabels(sets)
    ax.set_title("Cat umfla split-ul aleatoriu acuratetea de valenta\n"
                 "(acelasi LogReg, aceleasi date — difera doar protocolul)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "split_inflation.png", dpi=150)
    plt.close(fig)


# ====================================================== FIG 2: graded valence
def fig_graded_valence():
    sets = ["WESAD", "CASE", "EEVR", "EmoWear", "DEAP"]
    notes = {"WESAD": "contrast extrem", "CASE": "valenta clara",
             "EEVR": "realitate virtuala", "EmoWear": "inducere blanda",
             "DEAP": "self-report retrospectiv"}
    vals = []
    for n in sets:
        X, y, s = _valence(n)
        a = _loso(X, y, s)
        vals.append(a); NUM[f"graded_{n}"] = a

    colours = ["#27ae60" if v >= 70 else "#f39c12" if v >= 57 else "#95a5a6"
               for v in vals]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sets))
    b = ax.bar(x, vals, color=colours, width=0.6)
    ax.axhline(50, ls="--", c="#c0392b", lw=1.2)
    ax.text(len(sets) - 0.6, 51, "sansa (50%)", color="#c0392b", fontsize=9)
    for bar, v, n in zip(b, vals, sets):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.0f}%",
                ha="center", fontsize=11, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 4, notes[n], ha="center",
                fontsize=8, color="white", rotation=90, va="bottom")
    ax.set_ylabel("Valenta — acuratete echilibrata (LOSO) %"); ax.set_ylim(0, 100)
    ax.set_xticks(x); ax.set_xticklabels(sets)
    ax.set_title("Detectabilitatea valentei creste cu intensitatea afectiva\n"
                 "(acelasi pipeline, acelasi LOSO — difera doar stimulul)")
    fig.tight_layout(); fig.savefig(OUT / "graded_valence.png", dpi=150)
    plt.close(fig)


# ================================================ FIG 3: personal vs LOSO
def fig_personal_vs_loso():
    sets = ["WESAD", "CASE", "EEVR", "EmoWear"]
    loso, pers = [], []
    for n in sets:
        X, y, s = _valence(n)
        loso.append(_loso(X, y, s)); pers.append(_personal(X, y, s))
    for n, l, p in zip(sets, loso, pers):
        NUM[f"pers_{n}_loso"] = l; NUM[f"pers_{n}_personal"] = p

    x = np.arange(len(sets)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w / 2, loso, w, color="#95a5a6",
                label="Model GENERAL (LOSO — testat pe strain)")
    b2 = ax.bar(x + w / 2, pers, w, color="#27ae60",
                label="Model PERSONAL (calibrat pe utilizator)")
    ax.axhline(50, ls="--", c="#c0392b", lw=1)
    ax.text(len(sets) - 0.6, 51, "sansa (50%)", color="#c0392b", fontsize=8)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{b.get_height():.0f}", ha="center", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("Valenta — acuratete echilibrata %"); ax.set_ylim(0, 100)
    ax.set_xticks(x); ax.set_xticklabels(sets)
    ax.set_title("Valenta din PPG: slaba pe STRAINI, recuperata PERSONAL\n"
                 "(acelasi LogReg — difera doar daca s-a calibrat pe utilizator)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "personal_vs_loso.png", dpi=150)
    plt.close(fig)


# =============================================== FIG 4: WESAD confusion matrix
def fig_wesad_confusion():
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.base import clone
    X, y, s = _valence("WESAD")
    rows = np.all(np.isfinite(X), axis=1); X, y, s = X[rows], y[rows], s[rows]
    yt, yp = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, s):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = make_pipeline(StandardScaler(), clone(_model())); m.fit(X[tr], y[tr])
        yt.extend(y[te]); yp.extend(m.predict(X[te]))
    yt, yp = np.array(yt), np.array(yp)
    bal = balanced_accuracy_score(yt, yp) * 100
    NUM["wesad_conf_bal"] = bal
    cm = confusion_matrix(yt, yp).astype(float)
    cmn = cm / cm.sum(1, keepdims=True) * 100
    NUM["wesad_neg_recall"] = cmn[0, 0]; NUM["wesad_pos_recall"] = cmn[1, 1]

    labels = ["Stres\n(negativ)", "Amuzament\n(pozitiv)"]
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cmn[i,j]:.0f}%\n(n={int(cm[i,j])})", ha="center",
                    va="center", fontsize=12,
                    color="white" if cmn[i, j] > 55 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Prezis"); ax.set_ylabel("Real")
    ax.set_title(f"Confuzie valenta WESAD (LOSO, echilibrat {bal:.0f}%)\n"
                 "succesul vine din diferenta de activare dintre conditii")
    ax.grid(False)
    fig.colorbar(im, fraction=0.046, pad=0.04, label="% pe rand")
    fig.tight_layout(); fig.savefig(OUT / "wesad_confusion.png", dpi=150)
    plt.close(fig)


# =============================================== FIG 5: SHAP proxy (Cohen's d)
def fig_wesad_feature_attrib():
    """Separarea bruta per trasatura (Cohen's d) pe WESAD valenta: arata ca HR
    medie + morfologia pulsului domina (proxy de activare), nu trasaturi de valenta."""
    d = np.load(ROOT / "data" / "wesad_features.npz", allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)
    # nume trasaturi: ordinea canonica a extractorului comun (sursa unica de adevar).
    from affectus.research.valence.features import VALENCE_FEATURE_NAMES
    names = list(VALENCE_FEATURE_NAMES)[:X.shape[1]]
    pos, neg = X[y == 1], X[y == 0]
    d_vals = []
    for j in range(X.shape[1]):
        a, b = pos[:, j], neg[:, j]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        sp = np.sqrt(((len(a)-1)*a.std()**2 + (len(b)-1)*b.std()**2) /
                     max(len(a)+len(b)-2, 1))
        d_vals.append((a.mean() - b.mean()) / sp if sp > 0 else 0.0)
    d_vals = np.array(d_vals)
    order = np.argsort(np.abs(d_vals))[::-1][:12]
    top_names = [names[i] for i in order]
    top_d = d_vals[order]
    NUM["cohen_top1"] = abs(top_d[0]); NUM["cohen_top1_name"] = top_names[0]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colours = ["#2980b9" if v > 0 else "#c0392b" for v in top_d]
    ax.barh(range(len(top_d)), top_d, color=colours)
    ax.set_yticks(range(len(top_d))); ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.axvline(0, c="black", lw=0.8)
    ax.set_xlabel("Cohen's d  (+ = mai mare la amuzament, − = mai mare la stres)")
    ax.set_title("Ce separa de fapt clasele de valenta pe WESAD\n"
                 "domina latimea pulsului si HR medie — trasaturi de ACTIVARE")
    fig.tight_layout(); fig.savefig(OUT / "wesad_feature_attrib.png", dpi=150)
    plt.close(fig)


# ============================================ FIG 6: polarity 3-class confusion
def fig_polarity_confusion():
    """Detector 3 clase negativ/neutru/pozitiv pe BVP de incheietura (WESAD,
    cele 3 conditii afective). LOSO, LogReg."""
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    from sklearn.base import clone
    d = np.load(ROOT / "data" / "states_wesad.npz", allow_pickle=True)
    X, lab, s = d["X"], d["y"], d["subjects"]
    # WESAD labels: 1 baseline(neutru), 2 stress(negativ), 3 amusement(pozitiv)
    keep = np.isin(lab, [1, 2, 3])
    X, lab, s = X[keep], lab[keep], s[keep]
    y = np.select([lab == 2, lab == 1, lab == 3], [0, 1, 2])  # neg/neu/poz
    rows = np.all(np.isfinite(X), axis=1); X, y, s = X[rows], y[rows], s[rows]
    yt, yp = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, s):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(), clone(_model())); m.fit(X[tr], y[tr])
        yt.extend(y[te]); yp.extend(m.predict(X[te]))
    yt, yp = np.array(yt), np.array(yp)
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2]).astype(float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    NUM["pol_neg"] = cmn[0, 0]; NUM["pol_neu"] = cmn[1, 1]; NUM["pol_pos"] = cmn[2, 2]
    NUM["pol_neg_pos_conf"] = cmn[0, 2]; NUM["pol_pos_neg_conf"] = cmn[2, 0]

    labels = ["negativ", "neutru", "pozitiv"]
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(cmn, cmap="Purples", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i,j]:.0f}%", ha="center", va="center",
                    fontsize=12, color="white" if cmn[i, j] > 55 else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Prezis"); ax.set_ylabel("Real")
    ax.set_title("Detector de polaritate (3 clase, LOSO, BVP incheietura)\n"
                 "prinde negativul, rateaza pozitivul; confuzia poli opusi mica")
    ax.grid(False)
    fig.colorbar(im, fraction=0.046, pad=0.04, label="% pe rand")
    fig.tight_layout(); fig.savefig(OUT / "polarity_confusion.png", dpi=150)
    plt.close(fig)


# =============================================== FIG 7: arousal WESAD conditions
def fig_arousal_wesad():
    """Indicele de stres (Baevsky) calculat din ECG WESAD, mediat pe conditie.
    Trebuie sa creasca cu activarea: meditatie < baseline < amuzament < stres."""
    from scipy.signal import butter, filtfilt, find_peaks
    from scipy.stats import spearmanr
    from affectus.dsp.hrv.metrics import compute_baevsky_indices
    FS, WIN = 700, 30
    COND = {1: "baseline", 2: "stres", 3: "amuzament", 4: "meditatie"}
    RANK = {4: 0, 1: 1, 3: 2, 2: 3}

    def ibi_from_ecg(seg):
        s = seg - seg.mean()
        nyq = FS / 2
        b, a = butter(2, [5 / nyq, min(15, nyq*0.9)/nyq], btype="band")
        sq = filtfilt(b, a, s) ** 2
        pk, _ = find_peaks(sq, height=sq.mean() + 0.5*sq.std(),
                           distance=int(0.4*FS))
        return np.diff(pk)/FS*1000.0 if len(pk) >= 4 else np.array([])

    by = {c: [] for c in COND}
    for path in sorted(glob.glob(str(ROOT / "datasets/WESAD/S*/S*.pkl"))):
        d = pickle.load(open(path, "rb"), encoding="latin1")
        ecg = np.array(d["signal"]["chest"]["ECG"]).flatten()
        lbl = np.array(d["label"]); wb = WIN * FS
        for i in range(len(ecg)//wb):
            seg = ecg[i*wb:(i+1)*wb]; ls = lbl[i*wb:(i+1)*wb]
            v, c = np.unique(ls, return_counts=True); dom = int(v[np.argmax(c)])
            if dom not in COND:
                continue
            ibi = ibi_from_ecg(seg)
            ibi = ibi[(ibi > 300) & (ibi < 2000)]
            if len(ibi) < 4:
                continue
            _, kub = compute_baevsky_indices(ibi)
            by[dom].append(float(kub))

    ordered = sorted(COND, key=lambda k: RANK[k])
    means = [np.mean(by[c]) for c in ordered]
    ns = [len(by[c]) for c in ordered]
    rows = [(RANK[c], v) for c in ordered for v in by[c]]
    rho, _ = spearmanr([r for r, _ in rows], [v for _, v in rows])
    NUM["arousal_rho"] = rho
    for c, m in zip(ordered, means):
        NUM[f"arousal_SI_{COND[c]}"] = m

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ordered))
    cols = ["#5dade2", "#85c1e9", "#f5b041", "#e74c3c"]
    b = ax.bar(x, means, color=cols, width=0.6)
    for bar, m, n in zip(b, means, ns):
        ax.text(bar.get_x()+bar.get_width()/2, m+0.3, f"{m:.1f}\n(n={n})",
                ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{COND[c]}\n(rang {RANK[c]})" for c in ordered])
    ax.set_ylabel("Indice de stres (Baevsky) — mediu pe fereastra de 30 s")
    ax.set_title("Indicele nostru de stres creste cu activarea conditiei (WESAD)\n"
                 f"Spearman rho (rang activare vs indice) = {rho:+.2f}")
    fig.tight_layout(); fig.savefig(OUT / "arousal_wesad.png", dpi=150)
    plt.close(fig)


# =============================================== FIG 8: respiration RSA vs PPG
def fig_respiration():
    """RSA-din-IBI vs amplitudine-PPG vs respiratie reala (banda toracica WESAD).
    Arata empiric care sursa, daca vreuna, e fiabila pe incheietura."""
    from scipy.signal import butter, filtfilt, find_peaks, welch
    from affectus.io.messages import InterbeatIntervalEntry
    from affectus.research.respiration.rsa import estimate_respiration_rsa
    from affectus.research.respiration.ppg import estimate_respiration_ppg

    FS_BVP, FS_RESP = 64, 700      # WESAD wrist BVP, chest Resp
    WIN = 60                        # respiratia are nevoie de ~zeci de secunde

    def true_resp_rate(seg):
        """Rata de respiratie din banda toracica via spectru Welch (0.1–0.5 Hz)."""
        f, p = welch(seg - seg.mean(), fs=FS_RESP, nperseg=min(len(seg), 4096))
        band = (f >= 0.1) & (f <= 0.5)
        if not band.any():
            return None
        return float(f[band][np.argmax(p[band])] * 60.0)

    def bvp_to_ibi_entries(seg):
        nyq = FS_BVP/2
        b, a = butter(2, [0.5/nyq, 4/nyq], btype="band")
        f = filtfilt(b, a, seg - seg.mean())
        pk, _ = find_peaks(f, distance=int(0.4*FS_BVP))
        if len(pk) < 5:
            return None
        t_ms = (pk/FS_BVP*1000.0).astype(int)
        ibi = np.diff(t_ms)
        return [InterbeatIntervalEntry(interval_ms=int(m), timestamp_ms=int(t))
                for m, t in zip(ibi, t_ms[1:])]

    rsa_err, ppg_err = [], []
    rsa_conf, ppg_conf = [], []
    pts = []  # (true, rsa, ppg)
    for path in sorted(glob.glob(str(ROOT / "datasets/WESAD/S*/S*.pkl"))):
        d = pickle.load(open(path, "rb"), encoding="latin1")
        bvp = np.array(d["signal"]["wrist"]["BVP"]).flatten()
        resp = np.array(d["signal"]["chest"]["Resp"]).flatten()
        lbl = np.array(d["label"])
        wb_bvp, wb_resp = WIN*FS_BVP, WIN*FS_RESP
        wb_lbl = WIN*FS_RESP  # label is at chest 700 Hz
        n = min(len(bvp)//wb_bvp, len(resp)//wb_resp)
        for i in range(n):
            ls = lbl[i*wb_lbl:(i+1)*wb_lbl]
            v, c = np.unique(ls, return_counts=True)
            if int(v[np.argmax(c)]) not in (1, 2, 3, 4):
                continue   # doar conditii cunoscute (corp asezat)
            tr = true_resp_rate(resp[i*wb_resp:(i+1)*wb_resp])
            if tr is None or not (6 <= tr <= 30):
                continue
            seg_bvp = bvp[i*wb_bvp:(i+1)*wb_bvp]
            ts_ms = (np.arange(len(seg_bvp))/FS_BVP*1000.0).astype(int).tolist()
            entries = bvp_to_ibi_entries(seg_bvp)
            r = estimate_respiration_rsa(entries) if entries else None
            p = estimate_respiration_ppg(seg_bvp.astype(int).tolist(), ts_ms)
            r_bpm = r.breaths_per_min if r and r.confidence > 0 else None
            p_bpm = p.breaths_per_min if p and p.confidence > 0 else None
            if r:
                rsa_conf.append(r.confidence)
            if p:
                ppg_conf.append(p.confidence)
            if r_bpm:
                rsa_err.append(abs(r_bpm - tr))
            if p_bpm:
                ppg_err.append(abs(p_bpm - tr))
            pts.append((tr, r_bpm, p_bpm))

    NUM["resp_rsa_mae"] = float(np.mean(rsa_err)) if rsa_err else float("nan")
    NUM["resp_ppg_mae"] = float(np.mean(ppg_err)) if ppg_err else float("nan")
    NUM["resp_rsa_conf"] = float(np.mean(rsa_conf)) if rsa_conf else 0.0
    NUM["resp_ppg_conf"] = float(np.mean(ppg_conf)) if ppg_conf else 0.0
    NUM["resp_rsa_usable"] = len(rsa_err)
    NUM["resp_ppg_usable"] = len(ppg_err)
    NUM["resp_total_windows"] = len(pts)

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    # panou stanga: estimat vs real, ambele surse. Jitter mic pe axa reala,
    # fiindca detectia de varf pe banda toracica cade pe o grila grosiera; fara
    # jitter punctele se suprapun in cateva benzi verticale si ascund densitatea.
    ax = axes[0]
    tr_r = [(t, r) for t, r, _ in pts if r]
    tr_p = [(t, p) for t, _, p in pts if p]
    if tr_r:
        jt = [t + rng.normal(0, 0.35) for t, _ in tr_r]
        ax.scatter(jt, [r for _, r in tr_r], s=14, alpha=0.45,
                   c="#2980b9", label=f"RSA (n={len(tr_r)})")
    if tr_p:
        jt = [t + rng.normal(0, 0.35) for t, _ in tr_p]
        ax.scatter(jt, [p for _, p in tr_p], s=14, alpha=0.45,
                   c="#e67e22", label=f"Amplitudine PPG (n={len(tr_p)})")
    lim = [5, 30]
    ax.plot(lim, lim, "k--", lw=1, label="acord perfect")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Respiratie reala — banda toracica (br/min)")
    ax.set_ylabel("Respiratie estimata din incheietura (br/min)")
    ax.set_title("Estimare vs adevar (WESAD, ferestre de 60 s)\n"
                 "norii verticali = estimarea nu urmeaza adevarul")
    ax.legend(fontsize=8, loc="upper left")

    # panou dreapta: MAE + cat de des produc o estimare increzatoare
    ax = axes[1]
    labels = ["RSA\n(din IBI)", "Amplitudine\nPPG"]
    maes = [NUM["resp_rsa_mae"], NUM["resp_ppg_mae"]]
    b = ax.bar(labels, maes, color=["#2980b9", "#e67e22"], width=0.5)
    for bar, m, ku, kc in zip(b, maes,
                              ["resp_rsa_usable", "resp_ppg_usable"],
                              ["resp_rsa_conf", "resp_ppg_conf"]):
        ax.text(bar.get_x()+bar.get_width()/2, m+0.12,
                f"{m:.1f} br/min\n{NUM[ku]} ferestre cu incredere\n"
                f"incredere medie {NUM[kc]:.2f}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, max(maes) + 2.2)
    ax.set_ylabel("Eroare absoluta medie fata de banda toracica (br/min)")
    ax.set_title(f"Cat de departe e fiecare sursa", pad=14)
    ax.text(0.5, -0.16, f"din {len(pts)} ferestre cu adevar valid",
            transform=ax.transAxes, ha="center", fontsize=9, color="gray")
    fig.tight_layout(); fig.savefig(OUT / "respiration_compare.png", dpi=150)
    plt.close(fig)


def main():
    print("Generez figurile de rezultate in:", OUT)
    for fn in (fig_split_inflation, fig_graded_valence, fig_personal_vs_loso,
               fig_wesad_confusion, fig_wesad_feature_attrib,
               fig_polarity_confusion, fig_arousal_wesad, fig_respiration):
        print(f"  -> {fn.__name__} ...", flush=True)
        fn()
    print("\n==== CIFRELE REALE (pentru .tex) ====")
    for k in sorted(NUM):
        v = NUM[k]
        print(f"  {k:28s} = {v}" if isinstance(v, str) else f"  {k:28s} = {v:.2f}")


if __name__ == "__main__":
    main()
