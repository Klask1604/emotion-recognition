# ML models (local only)

`.joblib` files are not committed. Train locally:

```bash
python train/train_wisdm_har.py    # -> motion_har_wisdm.joblib
python wesad/train_wesad_epoch.py  # -> emotion WESAD (optional, gate failed)
```

Required for production HAR:

- `motion_har_wisdm.joblib`

Optional population stats (gitignored if regenerated):

- `population_stats.json`
