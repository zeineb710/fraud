"""Tables d'exploration (Module 1) et de variables comportementales (Module 2)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import features as F
from .config import EDA_DIR

BEHAV_BINS = {
    "tx_count_1h": ([-1, 0, 1, 3, np.inf], ["0", "1", "2-3", "4+"]),
    "tx_count_24h": ([-1, 0, 2, 6, 15, np.inf], ["0", "1-2", "3-6", "7-15", "16+"]),
    "amt_vs_mean_30d": ([-np.inf, 0.5, 0.9, 1.1, 2, 5, np.inf],
                        ["<0.5x", "0.5-0.9x", "~1x", "1.1-2x", "2-5x", ">5x"]),
    "time_since_prev": ([-1, 60, 600, 3600, 86400, np.inf],
                        ["<1 min", "1-10 min", "10 min-1 h", "1-24 h", ">24 h"]),
}
CAT_EDA = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "M4"]


def _rate(g: pd.core.groupby.SeriesGroupBy) -> pd.DataFrame:
    out = g.agg(["size", "mean"]).reset_index()
    out.columns = [out.columns[0], "n", "fraud_rate"]
    out["fraud_rate"] = out["fraud_rate"] * 100
    return out


def build_eda(df: pd.DataFrame, n_tr: int, n_va: int) -> dict:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    y = df["isFraud"]
    amt = df["TransactionAmt"]
    sec = df["TransactionDT"]

    overview = {
        "n_rows": int(len(df)), "n_fraud": int(y.sum()), "fraud_rate": float(y.mean() * 100),
        "days": float((sec.max() - sec.min()) / 86400),
        "mean_amt_legit": float(amt[y == 0].mean()), "mean_amt_fraud": float(amt[y == 1].mean()),
        "median_amt_legit": float(amt[y == 0].median()), "median_amt_fraud": float(amt[y == 1].median()),
    }
    (EDA_DIR / "overview.json").write_text(json.dumps(overview, indent=2))

    week = _rate(y.groupby((sec // (7 * 86400)).astype(int)))
    week.columns = ["week", "n", "fraud_rate"]
    week.to_csv(EDA_DIR / "week.csv", index=False)

    hour = _rate(y.groupby(((sec // 3600) % 24).astype(int)))
    hour.columns = ["hour", "n", "fraud_rate"]
    hour.to_csv(EDA_DIR / "hour.csv", index=False)

    edges = np.linspace(-1, 4.3, 28)
    la = np.log10(amt.clip(lower=0.1))
    rows = []
    for cls, name in ((0, "Honnête"), (1, "Fraude")):
        h, _ = np.histogram(la[y == cls], bins=edges)
        for i, v in enumerate(h):
            rows.append({"classe": name, "log10_montant": (edges[i] + edges[i + 1]) / 2,
                         "part": 100 * v / max(h.sum(), 1)})
    pd.DataFrame(rows).to_csv(EDA_DIR / "amount.csv", index=False)

    rows = []
    for var in CAT_EDA:
        if var not in df.columns:
            continue
        s = df[var].astype(object).where(df[var].notna(), "(manquant)")
        t = _rate(y.groupby(s.values))
        t.columns = ["categorie", "n", "fraud_rate"]
        t = t.sort_values("n", ascending=False).head(12)
        t.insert(0, "variable", var)
        rows.append(t)
    pd.concat(rows).to_csv(EDA_DIR / "cat.csv", index=False)

    rows = []
    for var, (bins, labels) in BEHAV_BINS.items():
        if var not in df.columns:
            continue
        b = pd.cut(df[var], bins=bins, labels=labels).astype(object)
        b = b.where(b.notna(), "pas d'historique")
        t = _rate(y.groupby(b.values))
        t.columns = ["groupe", "n", "fraud_rate"]
        order = {l: i for i, l in enumerate(labels + ["pas d'historique"])}
        t["ordre"] = t["groupe"].map(order)
        t.insert(0, "variable", var)
        rows.append(t.sort_values("ordre"))
    pd.concat(rows).to_csv(EDA_DIR / "behav.csv", index=False)

    raw_cols = [c for c in df.columns if c not in F.ENGINEERED]
    miss = (df[raw_cols].isna().mean() * 100).sort_values(ascending=False).head(30)
    miss.rename_axis("colonne").reset_index(name="manquant_pct").to_csv(EDA_DIR / "missing.csv", index=False)

    bounds = [(0, n_tr, "train"), (n_tr, n_va, "validation"), (n_va, len(df), "test")]
    rows = []
    for a, b, name in bounds:
        part = df.iloc[a:b]
        rows.append({"jeu": name, "lignes": b - a, "fraudes": int(part["isFraud"].sum()),
                     "taux_fraude": float(part["isFraud"].mean() * 100),
                     "jour_debut": float(part["TransactionDT"].min() / 86400),
                     "jour_fin": float(part["TransactionDT"].max() / 86400)})
    pd.DataFrame(rows).to_csv(EDA_DIR / "split.csv", index=False)
    return overview
