"""Données de travail de l'application : CSV bruts s'ils existent, sinon échantillon versionné dans le dépôt.

Construire l'échantillon (une fois, là où se trouvent les CSV Kaggle) :
    python -m src.data --rows 40000
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import features as F
from .config import DATA_RAW, ROOT

SAMPLE = ROOT / "data" / "sample" / "transactions_sample.parquet"


def has_raw() -> bool:
    return (DATA_RAW / "train_transaction.csv").exists() and (DATA_RAW / "train_identity.csv").exists()


def n_available() -> int:
    """Nombre de lignes utilisables (0 = aucune donnée)."""
    if has_raw():
        return 150000
    if SAMPLE.exists():
        import pyarrow.parquet as pq
        return pq.ParquetFile(SAMPLE).metadata.num_rows
    return 0


def source_label() -> str:
    return "CSV bruts locaux" if has_raw() else "échantillon du dépôt (data/sample)" if SAMPLE.exists() else "aucune"


def load(n_rows: int) -> pd.DataFrame:
    """Tranche chronologique des n premières transactions (utile : la vélocité a besoin de l'historique)."""
    df = F.load_raw(n_rows) if has_raw() else pd.read_parquet(SAMPLE).head(n_rows)
    return df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)


def build_sample(df: pd.DataFrame, n_v: int = 100) -> pd.DataFrame:
    """Garde les colonnes utiles ; les V les moins manquantes sont choisies sur les 60 % premiers (train)."""
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    v = F.select_v_cols(df.iloc[: int(0.6 * len(df))], n_v)
    keep = ["TransactionID", "isFraud", "TransactionDT"] + F.feature_list(df.columns, v)
    return df[list(dict.fromkeys(keep))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Construit data/sample/transactions_sample.parquet à partir des CSV Kaggle")
    ap.add_argument("--rows", type=int, default=40000)
    ap.add_argument("--n-v", type=int, default=100)
    a = ap.parse_args()
    out = build_sample(F.load_raw(a.rows), a.n_v)
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SAMPLE, compression="zstd", index=False)
    print(f"{len(out):,} lignes, {out.shape[1]} colonnes -> {SAMPLE} ({SAMPLE.stat().st_size / 1e6:.1f} Mo)")