"""
Parallel research / legacy engines facade.

`LegacyEngines.run()` executes only the toggled-on engines and returns their
outputs for the compute service to publish on `biofizic/legacy/*`. Nothing here
ever touches the production `PhysiologyDecision`. Sub-engine modules and their
heavy dependencies (scipy / scikit-learn) are imported lazily, so with all
toggles off this package costs nothing and imports cleanly without them.
"""

from __future__ import annotations

from dataclasses import dataclass

from biofizic.legacy import toggles


@dataclass(frozen=True)
class LegacyOutputs:
    ppg: dict | None = None       # detected peaks, PPA, reconstructed IBI
    wesad: dict | None = None     # {"p_stress": float}
    valence: dict | None = None   # {"valence": float, "ppa_z": float}
    respiration: dict | None = None  # {"rsa_bpm", "ppg_bpm", confidences, ...}

    def is_empty(self) -> bool:
        return (
            self.ppg is None
            and self.wesad is None
            and self.valence is None
            and self.respiration is None
        )


class LegacyEngines:
    """Holds the enabled sub-engines (lazy-constructed)."""

    def __init__(self) -> None:
        self._ppg = None
        self._wesad = None
        self._valence = None
        self._respiration = None
        if toggles.ENABLE_RESPIRATION_COMPARE:
            from biofizic.legacy.respiration_compare import RespirationCompareEngine

            self._respiration = RespirationCompareEngine()

        if toggles.ENABLE_RAW_PPG or toggles.ENABLE_PPG_PEAKS or toggles.ENABLE_VALENCE:
            from biofizic.legacy.raw_ppg import RawPpgEngine

            self._ppg = RawPpgEngine()
        if toggles.ENABLE_WESAD:
            from biofizic.legacy.wesad import WesadEngine

            try:
                self._wesad = WesadEngine()
            except FileNotFoundError as exc:
                # Toggle on but model not trained yet: skip gracefully (don't
                # crash the service) until models/wesad_rf.joblib exists.
                import logging

                logging.getLogger("legacy").warning("WESAD disabled: %s", exc)
                self._wesad = None
        if toggles.ENABLE_VALENCE:
            from biofizic.legacy.valence import ValenceEngine

            self._valence = ValenceEngine()

    @property
    def active(self) -> bool:
        return any((self._ppg, self._wesad, self._valence, self._respiration))

    def run(self, *, batch, result, baseline) -> LegacyOutputs:
        """Run the enabled engines for one epoch. `batch` is the parsed
        AcquisitionBatchMessage, `result` the production MultiWindowResult,
        `baseline` the RestBaselineStore (for personal z-scores)."""
        ppg_out = None
        ppa_z = 0.0
        if self._ppg is not None:
            ppg_out = self._ppg.process(batch)
            ppa_z = self._ppg.ppa_z

        decision = result.decision if result is not None else None
        primary = decision.multi_window.window_30_seconds if (decision and decision.multi_window) else None

        wesad_out = None
        if self._wesad is not None and primary is not None:
            wesad_out = self._wesad.predict(primary)

        valence_out = None
        if self._valence is not None and decision is not None:
            rmssd_z = baseline.rmssd_z_score(decision.rmssd_ms) if baseline.is_ready else 0.0
            valence_out = self._valence.compute(rmssd_z=rmssd_z, ppa_z=ppa_z)

        respiration_out = None
        if self._respiration is not None:
            respiration_out = self._respiration.compute(batch)

        return LegacyOutputs(
            ppg=ppg_out, wesad=wesad_out, valence=valence_out,
            respiration=respiration_out,
        )
