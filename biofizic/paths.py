"""Căi runtime — Docker (/data) vs development local (data/, models/)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    docker = Path("/data")
    if docker.is_dir():
        return docker
    local = ROOT / "data"
    local.mkdir(parents=True, exist_ok=True)
    return local


def models_dir() -> Path:
    docker_models = Path("/app/models")
    if docker_models.is_dir():
        return docker_models
    local = ROOT / "models"
    local.mkdir(parents=True, exist_ok=True)
    return local


def user_profile_path() -> Path:
    return data_dir() / "user_profile.json"


def user_baseline_path() -> Path:
    return data_dir() / "user_baseline.json"


def rest_baseline_path() -> Path:
    return data_dir() / "rest_baseline.json"


def motion_model_path() -> Path:
    return data_dir() / "motion_model.json"


def model_v3_path() -> Path:
    v4 = models_dir() / "model_v4.joblib"
    if v4.exists():
        return v4
    return models_dir() / "model_v3.joblib"


def model_v4_path() -> Path:
    return models_dir() / "model_v4.joblib"


def population_stats_path() -> Path:
    return models_dir() / "population_stats.json"
