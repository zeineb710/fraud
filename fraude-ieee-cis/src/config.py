"""Configuration centrale du projet."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
ART = ROOT / "artifacts"
MODELS_DIR = ART / "models"
EDA_DIR = ART / "eda"

COMPETITION = "ieee-fraud-detection"
RAW_FILES = ["train_transaction.csv", "train_identity.csv"]

SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2        # le reste (20 %) = test
BUDGET_FRAC = 0.02                      # budget d'alertes k = 2 % des transactions du jeu
N_V_COLS = 40                           # nombre de variables Vesta V* conservées


def split_points(n: int) -> tuple[int, int]:
    """Indices de fin du train et de la validation (découpage chronologique 60/20/20)."""
    return int(TRAIN_FRAC * n), int((TRAIN_FRAC + VAL_FRAC) * n)


def max_rows() -> int:
    """0 = toutes les lignes. Sinon, tranche chronologique (utile pour limiter la mémoire)."""
    return int(os.getenv("MAX_ROWS", "0") or 0)


def default_cost_per_alert() -> float:
    """Coût d'analyse d'une alerte (en $, hypothèse modifiable dans l'application)."""
    return float(os.getenv("COST_PER_ALERT", "5") or 5)
