#!/usr/bin/env python3
"""GalaxyPPG diagnostic — NOT a valence experiment.

GalaxyPPG is stress-only (TSST/SSST, no amusement) so it cannot teach valence.
Its value is that it records the SAME Galaxy Watch PPG as the product, alongside
Empatica E4 BVP and a Polar H10 chest ECG (gold R-R). It answers two questions
the live failure raised:

  Part A — IBI quality:  how close is Galaxy-Watch-PPG-derived IBI to the Polar
           gold R-R, vs E4 BVP IBI? (Can the watch sustain HRV at all?)
  Part B — domain shift:  how different is Galaxy PPG from E4 BVP in the 33-feature
           space (Cohen's d + a domain-classifier AUC), WITH and WITHOUT the
           median+MAD normalize_ppg_window? This quantifies *why* a WESAD(E4)
           model collapses on the watch.

Rest segments only (baseline + rest-*), where the signal is cleanest and the two
devices are directly comparable.

Units: Galaxy timestamp = ms, ppg DC-coupled (~2e6), ~25 Hz, status 0=ok.
       E4 BVP timestamp = MICROseconds, AC-coupled (~0), 64 Hz.
       Polar IBI = phoneTimestamp ms + duration ms (R-R gold).

Run: ./venv/Scripts/python.exe train/galaxy_quality.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectus.legacy.valence_features import (  # noqa: E402
    extract_valence_feature_vector,
    normalize_ppg_window,
)

DATA = ROOT / "datasets/GalaxyPPG/Dataset"
REST_SESSIONS = {"baseline", "rest-1", "rest-2", "rest-3", "rest-4", "rest-5"}
FS_GALAXY = 25
FS_E4 = 64
WIN_S = 20


def _rest_windows(events: pd.DataFrame) -> list[tuple[int, int]]:
    """[(enter_ms, exit_ms)] for rest/baseline sessions from Event.csv."""
    spans, open_t = [], {}
    for _, r in events.iterrows():
        s, st, ts = r["session"], r["status"], int(r["timestamp"])
        if s not in REST_SESSIONS:
            continue
        if st == "ENTER":
            open_t[s] = ts
        elif st == "EXIT" and s in open_t:
            spans.append((open_t.pop(s), ts))
    return spans


def _peaks_ibi(sig: np.ndarray, fs: int) -> np.ndarray:
    """Band-pass 0.5-4 Hz + peak detect -> IBI in ms."""
    if len(sig) < 3 * fs:
        return np.array([])
    s = sig - sig.mean()
    b, a = butter(2, [0.5 / (fs / 2), 4 / (fs / 2)], btype="band")
    f = filtfilt(b, a, s)
    pk, _ = find_peaks(f, distance=int(0.4 * fs))
    if len(pk) < 2:
        return np.array([])
    return np.diff(pk) / fs * 1000.0


def _load_subject(pdir: Path):
    """Return (galaxy_ppg_df, e4_bvp_df, polar_ibi_df, rest_spans) or None."""
    gw = pdir / "GalaxyWatch" / "PPG.csv"
    e4 = pdir / "E4" / "BVP.csv"
    pol = pdir / "PolarH10" / "IBI.csv"
    ev = pdir / "Event.csv"
    if not (gw.exists() and e4.exists() and pol.exists() and ev.exists()):
        return None
    g = pd.read_csv(gw)
    g = g[g["status"] == 0][["timestamp", "ppg"]].astype({"timestamp": np.int64, "ppg": float})
    e = pd.read_csv(e4)
    e["timestamp"] = (e["timestamp"] / 1000).astype(np.int64)  # us -> ms
    # E4 IBI = Empatica's own R-R (research-grade reference, aligned to the phone
    # clock like Galaxy/Event). Polar H10 is the true gold but its phoneTimestamp
    # sits on a DIFFERENT clock origin (~9 h offset), so we recover it per-subject
    # by aligning the recording spans, and ALSO report E4 IBI as a clock-safe ref.
    eibi = pd.read_csv(pdir / "E4" / "IBI.csv")
    eibi["ts"] = (eibi["timestamp"] / 1000).astype(np.int64)   # us -> ms
    eibi["ibi"] = eibi["duration"] / 1000.0                     # us -> ms
    p = pd.read_csv(pol).rename(columns={"phoneTimestamp": "ts", "duration": "ibi"})
    spans = _rest_windows(pd.read_csv(ev))
    # estimate Polar clock offset: align its first beat to the session start
    sess_start = min((lo for lo, _ in spans), default=None)
    if sess_start is not None and len(p):
        p["ts"] = p["ts"] - (int(p["ts"].iloc[0]) - sess_start)
    return g, e, eibi, p, spans


def _slice(df, col_ts, lo, hi):
    return df[(df[col_ts] >= lo) & (df[col_ts] <= hi)]


def part_a_ibi_quality(subjects):
    """Galaxy-PPG IBI and E4-BVP IBI vs E4's own R-R (clock-safe reference)."""
    print("\n=== Part A — IBI quality (rest only) ===")
    print("  reference = E4 IBI (Empatica R-R, aligned clock); Polar via span-offset")
    rows = []
    for sid, (g, e, eibi, p, spans) in subjects.items():
        for lo, hi in spans:
            gw = _slice(g, "timestamp", lo, hi)["ppg"].to_numpy()
            e4 = _slice(e, "timestamp", lo, hi).iloc[:, 0].to_numpy()
            ref = _slice(eibi, "ts", lo, hi)["ibi"].to_numpy()
            if len(ref) < 5:
                continue
            gw_ibi = _peaks_ibi(gw, FS_GALAXY)
            e4_ibi = _peaks_ibi(e4, FS_E4)
            ref_med = float(np.median(ref))
            for tag, ibi in (("Galaxy", gw_ibi), ("E4-BVP", e4_ibi)):
                if len(ibi) < 3:
                    continue
                med = float(np.median(ibi))
                rows.append(dict(sid=sid, dev=tag,
                                 ibi_err=abs(med - ref_med),
                                 hr_err=abs(60000 / med - 60000 / ref_med)))
    df = pd.DataFrame(rows)
    if df.empty:
        print("  (no comparable rest windows found)")
        return
    for dev in ("Galaxy", "E4-BVP"):
        s = df[df.dev == dev]
        if s.empty:
            continue
        print(f"  {dev:7s}: median |IBI err| {s.ibi_err.median():5.1f} ms | "
              f"median |HR err| {s.hr_err.median():4.1f} bpm | "
              f"n={len(s)} windows over {s.sid.nunique()} subjects")


def _window_feats(sig, fs, normalize):
    """33-feature vectors over consecutive 20 s windows of one device's signal."""
    w = WIN_S * fs
    out = []
    for i in range(0, len(sig) - w + 1, w):
        seg = sig[i:i + w]
        g = [int(round(x)) for x in seg]
        if normalize:
            g = [int(round(x)) for x in normalize_ppg_window(g)]
        ts = [int(j * 1000 / fs) for j in range(len(seg))]
        v = extract_valence_feature_vector(g, ts)
        if v is not None and all(np.isfinite(v)):
            out.append(v)
    return out


def part_b_domain_shift(subjects, normalize: bool):
    """Can a classifier tell Galaxy PPG from E4 BVP? High AUC = large domain shift."""
    Xg, Xe = [], []
    for sid, (g, e, eibi, p, spans) in subjects.items():
        for lo, hi in spans:
            gw = _slice(g, "timestamp", lo, hi)["ppg"].to_numpy()
            e4 = _slice(e, "timestamp", lo, hi).iloc[:, 0].to_numpy()
            Xg += _window_feats(gw, FS_GALAXY, normalize)
            Xe += _window_feats(e4, FS_E4, normalize)
    Xg, Xe = np.asarray(Xg, float), np.asarray(Xe, float)
    if len(Xg) < 10 or len(Xe) < 10:
        print("  (insufficient windows for domain test)")
        return
    X = np.vstack([Xg, Xe])
    y = np.r_[np.ones(len(Xg)), np.zeros(len(Xe))]  # 1=Galaxy, 0=E4
    clf = make_pipeline(StandardScaler(),
                        SVC(kernel="rbf", probability=True, class_weight="balanced"))
    auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()
    # mean |Cohen's d| across features (g vs e)
    mu_g, mu_e = Xg.mean(0), Xe.mean(0)
    sd = np.sqrt((Xg.var(0) + Xe.var(0)) / 2) + 1e-9
    d = np.abs((mu_g - mu_e) / sd)
    tag = "WITH normalize" if normalize else "raw (no norm)"
    print(f"  [{tag:16s}] domain AUC={auc:.3f}  mean|Cohen d|={d.mean():.2f}  "
          f"max|d|={d.max():.1f}  (n_galaxy={len(Xg)}, n_e4={len(Xe)})")


def main():
    print("Loading GalaxyPPG subjects (P02-P24, guarding missing devices)...")
    subjects = {}
    for pdir in sorted(DATA.glob("P*")):
        loaded = _load_subject(pdir)
        if loaded is not None:
            subjects[pdir.name] = loaded
    print(f"  {len(subjects)} subjects with full device set")

    part_a_ibi_quality(subjects)

    print("\n=== Part B — domain shift Galaxy PPG vs E4 BVP (rest) ===")
    print("  (AUC ~0.5 = devices indistinguishable; ~1.0 = total shift)")
    part_b_domain_shift(subjects, normalize=False)
    part_b_domain_shift(subjects, normalize=True)
    print("\nDone. Diagnostic for the WESAD(E4)->Watch transfer failure.")


if __name__ == "__main__":
    main()
