"""
Parallel research / legacy engines facade.

`ResearchEngines.run()` executes only the toggled-on engines and returns their
outputs for the compute service to publish on `biofizic/legacy/*`. Nothing here
ever touches the production `PhysiologyDecision`. Sub-engine modules and their
heavy dependencies (scipy / scikit-learn) are imported lazily, so with all
toggles off this package costs nothing and imports cleanly without them.
"""

from __future__ import annotations

from dataclasses import dataclass

from affectus.research import toggles


@dataclass(frozen=True)
class ResearchOutputs:
    ppg: dict | None = None       # detected peaks, PPA, reconstructed IBI
    respiration: dict | None = None  # {"rsa_bpm", "ppg_bpm", confidences, ...}
    valence_fd: dict | None = None   # PPG frequency-domain features (9)
    valence_wesad: dict | None = None  # {"p_positive", "valence_z", confidence}
    valence_eevr: dict | None = None   # EEVR-trained valence (same dict shape)
    valence_case: dict | None = None   # CASE-trained valence (same dict shape)

    def is_empty(self) -> bool:
        return (
            self.ppg is None
            and self.respiration is None
            and self.valence_fd is None
            and self.valence_wesad is None
            and self.valence_eevr is None
            and self.valence_case is None
        )


class ResearchEngines:
    """Holds the enabled sub-engines (lazy-constructed)."""

    def __init__(self) -> None:
        self._ppg = None
        self._respiration = None
        self._valence_fd = None
        self._valence_wesad = None
        self._valence_eevr = None
        self._valence_case = None
        # Per-subject valence calibration (baseline + smoother + verdict stabiliser)
        # for each enabled valence model. Lives here, not in the production
        # pipeline, so the arousal engine holds zero research state and never
        # imports research/. One dict keyed by model name; adding a model is a
        # single entry in ENABLED_VALENCE_MODELS, no engine edits.
        from affectus.research.valence.tracks import build_valence_tracks

        self.valence_tracks = build_valence_tracks()
        if toggles.ENABLE_RESPIRATION_COMPARE:
            from affectus.research.respiration.compare import RespirationCompareEngine

            self._respiration = RespirationCompareEngine()
        if toggles.ENABLE_VALENCE_FD:
            from affectus.research.valence.fd_engine import ValenceFdEngine

            self._valence_fd = ValenceFdEngine()
        # WESAD / EEVR / CASE valence run on the SAME engine class; each just
        # loads a different bundle. A None model_path uses the WESAD default.
        def _load_valence(model_path):
            from affectus.research.valence.wesad_engine import ValenceWesadEngine

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
        if toggles.ENABLE_RAW_PPG or toggles.ENABLE_PPG_PEAKS:
            from affectus.research.ppg.raw_ppg import RawPpgEngine

            self._ppg = RawPpgEngine()

    @property
    def active(self) -> bool:
        return any((self._ppg, self._respiration,
                    self._valence_fd, self._valence_wesad,
                    self._valence_eevr, self._valence_case))

    def reset_valence_tracks(self, reported_valence: float | None = None) -> None:
        """Re-anchor every valence model's personal calibration on a recalibration.
        Called by the compute service alongside the production baseline reset; the
        arousal pipeline no longer knows these exist."""
        for track in self.valence_tracks.values():
            track.reset(reported_valence)

    def run(self, *, batch, result, baseline) -> ResearchOutputs:
        """Run the enabled engines for one epoch. `batch` is the parsed
        AcquisitionBatchMessage, `result` the production MultiWindowResult,
        `baseline` the RestBaselineStore (for personal z-scores)."""
        # The waveform engines all share one shape: engine.compute(batch) -> dict.
        # Run each enabled one by its output name. The two odd ones out (raw PPG
        # uses .process(), WESAD predicts from the HRV window not the batch) stay
        # explicit below.
        out: dict = {
            name: getattr(self, attr).compute(batch)
            for name, attr in (
                ("respiration", "_respiration"),
                ("valence_fd", "_valence_fd"),
                ("valence_wesad", "_valence_wesad"),
                ("valence_eevr", "_valence_eevr"),
                ("valence_case", "_valence_case"),
            )
            if getattr(self, attr) is not None
        }
        if self._ppg is not None:
            out["ppg"] = self._ppg.process(batch)

        return ResearchOutputs(**out)
