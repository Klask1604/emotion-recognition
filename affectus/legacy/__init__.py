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

from affectus.legacy import toggles


@dataclass(frozen=True)
class LegacyOutputs:
    ppg: dict | None = None       # detected peaks, PPA, reconstructed IBI
    wesad: dict | None = None     # {"p_stress": float}
    valence: dict | None = None   # {"valence": float, "ppa_z": float}
    respiration: dict | None = None  # {"rsa_bpm", "ppg_bpm", confidences, ...}
    valence_fd: dict | None = None   # PPG frequency-domain features (9)
    valence_wesad: dict | None = None  # {"p_positive", "valence_z", confidence}
    valence_eevr: dict | None = None   # EEVR-trained valence (same dict shape)
    valence_case: dict | None = None   # CASE-trained valence (same dict shape)

    def is_empty(self) -> bool:
        return (
            self.ppg is None
            and self.wesad is None
            and self.valence is None
            and self.respiration is None
            and self.valence_fd is None
            and self.valence_wesad is None
            and self.valence_eevr is None
            and self.valence_case is None
        )


class LegacyEngines:
    """Holds the enabled sub-engines (lazy-constructed)."""

    def __init__(self) -> None:
        self._ppg = None
        self._wesad = None
        self._valence = None
        self._respiration = None
        self._valence_fd = None
        self._valence_wesad = None
        self._valence_eevr = None
        self._valence_case = None
        if toggles.ENABLE_RESPIRATION_COMPARE:
            from affectus.legacy.respiration_compare import RespirationCompareEngine

            self._respiration = RespirationCompareEngine()
        if toggles.ENABLE_VALENCE_FD:
            from affectus.legacy.valence_fd_engine import ValenceFdEngine

            self._valence_fd = ValenceFdEngine()
        # WESAD / EEVR / CASE valence run on the SAME engine class; each just
        # loads a different bundle. A None model_path uses the WESAD default.
        def _load_valence(model_path):
            from affectus.legacy.valence_wesad_engine import ValenceWesadEngine

            try:
                return ValenceWesadEngine(model_path=model_path)
            except FileNotFoundError as exc:
                # Toggle on but bundle not trained yet: skip gracefully (don't
                # crash the service) until the .joblib exists.
                import logging

                logging.getLogger("legacy").warning("valence model disabled: %s", exc)
                return None

        from pathlib import Path as _Path
        _models = _Path(__file__).resolve().parents[2] / "models"
        if toggles.ENABLE_VALENCE_WESAD:
            self._valence_wesad = _load_valence(None)
        if toggles.ENABLE_VALENCE_EEVR:
            self._valence_eevr = _load_valence(_models / "valence_eevr.joblib")
        if toggles.ENABLE_VALENCE_CASE:
            self._valence_case = _load_valence(_models / "valence_case.joblib")

        if toggles.ENABLE_RAW_PPG or toggles.ENABLE_PPG_PEAKS or toggles.ENABLE_VALENCE:
            from affectus.legacy.raw_ppg import RawPpgEngine

            self._ppg = RawPpgEngine()
        if toggles.ENABLE_WESAD:
            from affectus.legacy.wesad import WesadEngine

            try:
                self._wesad = WesadEngine()
            except FileNotFoundError as exc:
                # Toggle on but model not trained yet: skip gracefully (don't
                # crash the service) until models/wesad_rf.joblib exists.
                import logging

                logging.getLogger("legacy").warning("WESAD disabled: %s", exc)
                self._wesad = None
        if toggles.ENABLE_VALENCE:
            from affectus.legacy.valence import ValenceEngine

            self._valence = ValenceEngine()

    @property
    def active(self) -> bool:
        return any((self._ppg, self._wesad, self._valence, self._respiration,
                    self._valence_fd, self._valence_wesad,
                    self._valence_eevr, self._valence_case))

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

        valence_fd_out = None
        if self._valence_fd is not None:
            valence_fd_out = self._valence_fd.compute(batch)

        valence_wesad_out = None
        if self._valence_wesad is not None:
            valence_wesad_out = self._valence_wesad.compute(batch)

        valence_eevr_out = None
        if self._valence_eevr is not None:
            valence_eevr_out = self._valence_eevr.compute(batch)

        valence_case_out = None
        if self._valence_case is not None:
            valence_case_out = self._valence_case.compute(batch)

        return LegacyOutputs(
            ppg=ppg_out, wesad=wesad_out, valence=valence_out,
            respiration=respiration_out, valence_fd=valence_fd_out,
            valence_wesad=valence_wesad_out,
            valence_eevr=valence_eevr_out, valence_case=valence_case_out,
        )
