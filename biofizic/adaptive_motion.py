#!/usr/bin/env python3
"""
Baseline adaptiv acc_rms — GMM nesupervizat (hibrid cu praguri Tukey relative).

  - Componentă 1 (quiet): GMM pe eșantioane liniștite → μ, σ personal
  - Componentă 2 (sesiune): GMM rest/active → intersecție ca prag auxiliar
  - Fallback: MAD robust → Tukey outer fence (warmup)

Ref: Reynolds et al. 2009 (HAR wearables); Tukey fences (outlier detection).
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

log = logging.getLogger("adaptive_motion")

QUIET_LEARN_MAX_MPS2 = 1.0
QUIET_GMM_MIN = 30
MAD_MIN_SAMPLES = 12
SESSION_GMM_MIN = 60
REFIT_EVERY = 20
MAX_QUIET_BUFFER = 600
MAX_SESSION_BUFFER = 400
TUKEY_K_FALLBACK = 2.5
SCALE_FLOOR_MPS2 = 0.08
GMM_REG_COVAR = 1e-3


@dataclass
class MotionBaseline:
    quiet_mean: float | None
    quiet_scale: float | None
    source: str
    gmm_ready: bool
    active_threshold: float | None
    n_quiet: int
    n_session: int

    def as_quiet_pair(self) -> tuple[float | None, float | None]:
        return self.quiet_mean, self.quiet_scale


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _mad(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _tukey_scale(values: np.ndarray) -> float:
    if values.size < 4:
        return max(SCALE_FLOOR_MPS2, float(np.std(values)))
    q75, q25 = np.percentile(values, [75, 25])
    return max(SCALE_FLOOR_MPS2, float(q75 - q25) * 0.7413)


def _gmm_intersection_1d(
    mu1: float,
    s1: float,
    w1: float,
    mu2: float,
    s2: float,
    w2: float,
) -> float:
    """Punct unde w1·N(μ1,σ1) = w2·N(μ2,σ2) între cele două medii."""
    lo, hi = (mu1, mu2) if mu1 <= mu2 else (mu2, mu1)
    if hi - lo < 1e-6:
        return float((mu1 + mu2) / 2.0)
    xs = np.linspace(lo, hi, 800)
    d = w1 * norm.pdf(xs, mu1, max(s1, 1e-4)) - w2 * norm.pdf(xs, mu2, max(s2, 1e-4))
    signs = np.sign(d)
    crossings = np.where(np.diff(signs) != 0)[0]
    if crossings.size == 0:
        return float((mu1 * w1 + mu2 * w2) / max(w1 + w2, 1e-9))
    i = int(crossings[0])
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = d[i], d[i + 1]
    if abs(y1 - y0) < 1e-12:
        return float(x0)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def _fit_quiet_gmm(values: np.ndarray) -> tuple[float, float] | None:
    if values.size < QUIET_GMM_MIN:
        return None
    x = values.reshape(-1, 1)
    try:
        gmm = GaussianMixture(
            n_components=1,
            reg_covar=GMM_REG_COVAR,
            random_state=42,
            max_iter=200,
        )
        gmm.fit(x)
        mu = float(gmm.means_[0, 0])
        var = float(gmm.covariances_[0, 0, 0])
        sigma = max(SCALE_FLOOR_MPS2, float(np.sqrt(max(var, 1e-8))))
        if not np.isfinite(mu) or not np.isfinite(sigma):
            return None
        return mu, sigma
    except Exception as e:
        log.debug("quiet GMM fit failed: %s", e)
        return None


def _fit_session_intersection(values: np.ndarray) -> float | None:
    if values.size < SESSION_GMM_MIN:
        return None
    x = values.reshape(-1, 1)
    try:
        gmm = GaussianMixture(
            n_components=2,
            reg_covar=GMM_REG_COVAR,
            random_state=42,
            max_iter=200,
        )
        gmm.fit(x)
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.reshape(-1))
        weights = gmm.weights_
        rest_i = int(np.argmin(means))
        act_i = 1 - rest_i
        mu_r, mu_a = float(means[rest_i]), float(means[act_i])
        s_r = max(SCALE_FLOOR_MPS2, float(stds[rest_i]))
        s_a = max(SCALE_FLOOR_MPS2, float(stds[act_i]))
        w_r, w_a = float(weights[rest_i]), float(weights[act_i])
        if mu_a <= mu_r + SCALE_FLOOR_MPS2:
            return None
        thr = _gmm_intersection_1d(mu_r, s_r, w_r, mu_a, s_a, w_a)
        if not np.isfinite(thr) or thr <= 0:
            return None
        return float(thr)
    except Exception as e:
        log.debug("session GMM fit failed: %s", e)
        return None


def _baseline_from_quiet(values: np.ndarray) -> MotionBaseline:
    n_q = int(values.size)
    n_s = n_q
    gmm_pair = _fit_quiet_gmm(values)
    if gmm_pair is not None:
        mu, sigma = gmm_pair
        return MotionBaseline(
            quiet_mean=mu,
            quiet_scale=sigma,
            source="gmm_quiet",
            gmm_ready=True,
            active_threshold=None,
            n_quiet=n_q,
            n_session=n_s,
        )

    med = float(np.median(values))
    if n_q >= MAD_MIN_SAMPLES:
        scale = max(SCALE_FLOOR_MPS2, _mad(values) * 1.4826)
        return MotionBaseline(
            quiet_mean=med,
            quiet_scale=scale,
            source="mad_robust",
            gmm_ready=False,
            active_threshold=None,
            n_quiet=n_q,
            n_session=n_s,
        )

    scale = _tukey_scale(values)
    mean = med if n_q >= 2 else float(values.mean())
    return MotionBaseline(
        quiet_mean=float(mean),
        quiet_scale=max(SCALE_FLOOR_MPS2, TUKEY_K_FALLBACK * scale * 0.4 + scale),
        source="tukey_warmup",
        gmm_ready=False,
        active_threshold=None,
        n_quiet=n_q,
        n_session=n_s,
    )


def motion_z(acc_rms: float, baseline: MotionBaseline) -> float:
    if acc_rms <= 0 or baseline.quiet_mean is None or baseline.quiet_scale is None:
        return 0.0
    z = (acc_rms - baseline.quiet_mean) / max(baseline.quiet_scale, SCALE_FLOOR_MPS2)
    return clip(z, -3.0, 3.0)


class AdaptiveMotionBaseline:
    """Învățare baseline personal din acc_rms; persistă în JSON."""

    def __init__(self, path: Path | None = None, *, persist: bool = True) -> None:
        self.path = path
        self.persist = persist
        self._quiet: deque[float] = deque(maxlen=MAX_QUIET_BUFFER)
        self._session: deque[float] = deque(maxlen=MAX_SESSION_BUFFER)
        self._samples_since_refit = 0
        self._cached: MotionBaseline | None = None
        self._load()

    def observe(self, acc_rms: float, *, allow_quiet: bool) -> None:
        if acc_rms > 0:
            self._session.append(acc_rms)
            self._samples_since_refit += 1
        if allow_quiet and 0 < acc_rms <= QUIET_LEARN_MAX_MPS2:
            self._quiet.append(acc_rms)
        if self._samples_since_refit >= REFIT_EVERY:
            self.refit()

    def refit(self) -> MotionBaseline:
        quiet_arr = np.asarray(self._quiet, dtype=float)
        session_arr = np.asarray(self._session, dtype=float)

        if quiet_arr.size >= MAD_MIN_SAMPLES:
            bl = _baseline_from_quiet(quiet_arr)
        else:
            bl = MotionBaseline(
                quiet_mean=None,
                quiet_scale=None,
                source="none",
                gmm_ready=False,
                active_threshold=None,
                n_quiet=int(quiet_arr.size),
                n_session=int(session_arr.size),
            )

        bl.n_session = int(session_arr.size)
        active_thr = _fit_session_intersection(session_arr)
        if active_thr is not None:
            bl = MotionBaseline(
                quiet_mean=bl.quiet_mean,
                quiet_scale=bl.quiet_scale,
                source=bl.source,
                gmm_ready=bl.gmm_ready,
                active_threshold=active_thr,
                n_quiet=bl.n_quiet,
                n_session=bl.n_session,
            )

        self._cached = bl
        self._samples_since_refit = 0
        self._save(bl)
        if bl.gmm_ready and bl.active_threshold is not None:
            log.info(
                "GMM baseline μ=%.3f σ=%.3f active_thr=%.3f n_q=%d n_s=%d",
                bl.quiet_mean or 0,
                bl.quiet_scale or 0,
                bl.active_threshold,
                bl.n_quiet,
                bl.n_session,
            )
        return bl

    def current(self) -> MotionBaseline:
        if self._cached is None:
            return self.refit()
        return self._cached

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._cached = MotionBaseline(
                quiet_mean=data.get("quiet_mean"),
                quiet_scale=data.get("quiet_scale"),
                source=str(data.get("source", "none")),
                gmm_ready=bool(data.get("gmm_ready", False)),
                active_threshold=data.get("active_threshold"),
                n_quiet=int(data.get("n_quiet", 0)),
                n_session=int(data.get("n_session", 0)),
            )
            for v in data.get("quiet_buffer", [])[-MAX_QUIET_BUFFER:]:
                if 0 < float(v) <= QUIET_LEARN_MAX_MPS2:
                    self._quiet.append(float(v))
            for v in data.get("session_buffer", [])[-MAX_SESSION_BUFFER:]:
                if float(v) > 0:
                    self._session.append(float(v))
        except Exception as e:
            log.warning("motion model load failed: %s", e)

    def _save(self, bl: MotionBaseline) -> None:
        if self.path is None or not self.persist:
            return
        payload = {
            **asdict(bl),
            "updated_at": int(time.time()),
            "quiet_buffer": list(self._quiet)[-120:],
            "session_buffer": list(self._session)[-120:],
        }
        try:
            text = json.dumps(payload, indent=2)
            # Pe bind mount Windows, replace() pe .tmp eșuează (EBUSY) — scriere directă.
            self.path.write_text(text, encoding="utf-8")
        except Exception as e:
            log.warning("motion model save failed: %s", e)
