"""Accès en lecture aux artefacts produits par le pipeline (sans dépendance Streamlit)."""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from . import models as MD
from .config import ART, EDA_DIR, MODELS_DIR


def artifacts_ready() -> bool:
    return (ART / "meta.json").exists() and (ART / "scores.parquet").exists()


def clear_caches() -> None:
    for fn in (meta, results, scores, sample, shap_importance, cat_maps):
        fn.cache_clear()


@lru_cache(maxsize=1)
def meta() -> dict:
    return json.loads((ART / "meta.json").read_text())


@lru_cache(maxsize=1)
def results() -> dict:
    return json.loads((ART / "results.json").read_text())


@lru_cache(maxsize=1)
def scores() -> pd.DataFrame:
    return pd.read_parquet(ART / "scores.parquet")


@lru_cache(maxsize=1)
def sample() -> pd.DataFrame:
    return pd.read_parquet(ART / "explain_sample.parquet")


@lru_cache(maxsize=1)
def shap_importance() -> list:
    return json.loads((ART / "shap_importance.json").read_text())


@lru_cache(maxsize=1)
def cat_maps() -> dict:
    return json.loads((ART / "cat_maps.json").read_text())


def eda(name: str) -> pd.DataFrame:
    return pd.read_csv(EDA_DIR / f"{name}.csv")


def eda_overview() -> dict:
    return json.loads((EDA_DIR / "overview.json").read_text())


@lru_cache(maxsize=8)
def load_model(key: str):
    if key == "xgboost":
        return MD.XGBScorer.load(MODELS_DIR / "xgboost.json")
    return joblib.load(MODELS_DIR / f"{key}.joblib")
