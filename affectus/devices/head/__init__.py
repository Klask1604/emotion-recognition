"""
HEAD device family (head-worn EEG) — TEMPLATE, not implemented.

A head-worn device carries EEG (and possibly PPG at the temple). When implemented
it will register a 'head_eeg' profile, an adapter, and EEG feature-modules. Today
it registers nothing and exposes no modules.
"""

from __future__ import annotations

# from affectus.devices.head import profile as _profile  # registers 'head_eeg'

PROFILE = "head_eeg"
MODULES: list = []  # TODO: [eeg] when implemented

__all__ = ["PROFILE", "MODULES"]
