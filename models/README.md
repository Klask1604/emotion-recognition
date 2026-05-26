# Models

No trained model artifacts are needed. The pipeline is fully analytic: HRV
math, a robust per-user log-space baseline, an online signal-quality model
(`biofizic/decision/signal_quality.py`), and a CUSUM detector.

The earlier WISDM Random Forest HAR classifier was retired because its
train/serve feature spaces did not match the Galaxy Watch 7's 1 Hz aggregated
motion statistics (see `docs/THESIS_LIMITATIONS.md`). `scikit-learn` and
`joblib` are no longer dependencies, and this directory no longer ships any
`.joblib` files.
