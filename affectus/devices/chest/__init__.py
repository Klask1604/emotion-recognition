"""
CHEST device family (ECG chest strap) — TEMPLATE, not implemented.

A chest strap carries clean ECG R-peaks (so IBI is high-quality and barely
motion-coupled) and optionally a respiration belt. When implemented it will:
  - register a 'chest_ecg' profile (artifact_rate_max ~0.05, motion factor ~0.5)
  - provide an adapter mapping its wire payload -> SensorFrame
  - expose modules in modules/ (ecg_raw, resp) each declaring its Capability

Today it registers nothing and exposes no modules, so a chest device degrades to
the shared HRV/HR path. The structure exists so adding it later touches only this
folder, never the shared algorithms.
"""

from __future__ import annotations

# from affectus.devices.chest import profile as _profile  # registers 'chest_ecg'

PROFILE = "chest_ecg"
MODULES: list = []  # TODO: [ecg_raw, resp] when implemented

__all__ = ["PROFILE", "MODULES"]
