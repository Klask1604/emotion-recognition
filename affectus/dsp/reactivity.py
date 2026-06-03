"""
One-time subject reactivity / expressiveness profile.

Affective reactivity varies between people: a "low-responder" produces small
absolute valence swings, a very expressive subject large (and noisier) ones. This
is a property of the SUBJECT, not of the moment, so it is asked once and persisted
(unlike arousal/valence self-reports, which are per-calibration). It scales the
valence dead-band: low → narrower neutral zone (small real excursions still
register), high → wider (transients don't flip the verdict). See
config.REACTIVITY_DEADBAND_SCALE.

Persisted in its own file (data/reactivity_profile.json), mirroring the other
per-subject stores. path=None keeps it purely in-memory (tests).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from affectus.config import REACTIVITY_DEADBAND_SCALE, reactivity_profile_path

log = logging.getLogger("reactivity")

_VALID = ("low", "normal", "high")
_DEFAULT = "normal"


class ReactivityProfile:
    def __init__(self, path: Path | None = None, *, persist: bool = False) -> None:
        self._path = path or (reactivity_profile_path() if persist else None)
        self._level = _DEFAULT
        if self._path is not None:
            self._load()

    @property
    def level(self) -> str:
        return self._level

    @property
    def deadband_scale(self) -> float:
        """Multiplier applied to the base valence dead-band for this subject."""
        return REACTIVITY_DEADBAND_SCALE.get(self._level, 1.0)

    def set(self, level: str) -> None:
        """Set the reactivity level (low/normal/high). Unknown values are ignored
        so a malformed watch payload can't corrupt the profile."""
        lvl = str(level).strip().lower()
        if lvl not in _VALID:
            log.warning("ignoring unknown reactivity level: %r", level)
            return
        self._level = lvl
        log.info("reactivity profile set to %s (deadband_scale=%.2f)",
                 lvl, self.deadband_scale)
        self._save()

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"reactivity": self._level}, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            lvl = str(data.get("reactivity", _DEFAULT)).lower()
            if lvl in _VALID:
                self._level = lvl
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load reactivity profile: %s", exc)
