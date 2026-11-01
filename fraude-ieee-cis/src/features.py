"""Chargement, variables temporelles/comportementales (sans fuite) et encodage."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import DATA_RAW, N_V_COLS

# ----------------------------------------------------------------------------
# Listes de colonnes
# ----------------------------------------------------------------------------
C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_COLS = [f"M{i}" for i in range(1, 10)]
ID_NUM = [f"id_{i:02d}" for i in range(1, 12)]
ID_CAT = ["id_12", "id_15", "id_16", "id_28", "id_29", "id_30", "id_31", "id_33",
          "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo"]
BASE_NUM = ["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2",
            "dist1", "dist2"] + C_COLS + D_COLS
CAT_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"] + M_COLS + ID_CAT

TIME_FEATS = ["hour", "h_sin", "h_cos", "dow"]
EXTRA_FEATS = ["amt_log", "amt_cents", "email_match"]
BEHAV_FEATS = ["tx_count_1h", "tx_count_24h", "amt_vs_mean_30d", "time_since_prev"]
ENGINEERED = TIME_FEATS + EXTRA_FEATS + BEHAV_FEATS

FEATURE_LABELS = {
    "TransactionAmt": "Montant de la transaction",
    "tx_count_1h": "Nb d'achats de la même identité sur 1 h",
    "tx_count_24h": "Nb d'achats de la même identité sur 24 h",
    "amt_vs_mean_30d": "Montant / moyenne des 30 derniers jours",
    "time_since_prev": "Temps depuis l'achat précédent (s)",
    "hour": "Heure de la journée", "h_sin": "Heure (sinus)", "h_cos": "Heure (cosinus)",
    "dow": "Jour de la semaine", "amt_log": "Montant (log)", "amt_cents": "Partie décimale du montant",
    "email_match": "Emails acheteur/destinataire identiques",
    "ProductCD": "Type de produit", "card1": "Carte (card1)", "card2": "Carte (card2)",
    "card3": "Carte (card3)", "card4": "Réseau de la carte", "card5": "Carte (card5)",
    "card6": "Débit / crédit", "addr1": "Adresse (addr1)", "addr2": "Pays (addr2)",
    "dist1": "Distance 1", "dist2": "Distance 2",
    "P_emaildomain": "Domaine email acheteur", "R_emaildomain": "Domaine email destinataire",
    "DeviceType": "Type d'appareil", "DeviceInfo": "Modèle d'appareil",
}


def label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if re.fullmatch(r"[CDMV]\d+", feature):
        return f"Variable anonymisée {feature}"
    if feature.startswith("id_"):
        return f"Identité numérique {feature}"
    return feature


# ----------------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------------
def load_raw(max_rows: int = 0) -> pd.DataFrame:
    """Charge et fusionne les deux CSV (types allégés pour limiter la mémoire)."""
    tpath, ipath = DATA_RAW / "train_transaction.csv", DATA_RAW / "train_identity.csv"
    header = pd.read_csv(tpath, nrows=0).columns
    dtypes = {c: "float32" for c in header if re.fullmatch(r"[VCD]\d+", c)}
    tx = pd.read_csv(tpath, dtype=dtypes, nrows=max_rows or None)

    idt = pd.read_csv(ipath)
    idt.columns = [c.replace("-", "_") for c in idt.columns]
    keep = ["TransactionID"] + [c for c in ID_NUM + ID_CAT if c in idt.columns]
    idt = idt[keep]
    for c in ID_NUM:
        if c in idt.columns:
            idt[c] = idt[c].astype("float32")
    df = tx.merge(idt, on="TransactionID", how="left")
    del tx, idt
    return df


# ----------------------------------------------------------------------------
# Variables temporelles et dérivées
# ----------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """TransactionDT est un décalage en secondes (pas une date réelle)."""
    sec = df["TransactionDT"].to_numpy()
    hour = ((sec // 3600) % 24).astype("float32")
    df["hour"] = hour
    df["h_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")   # codage cyclique
    df["h_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    df["dow"] = ((sec // 86400) % 7).astype("float32")
    amt = df["TransactionAmt"].astype("float64")
    df["amt_log"] = np.log1p(amt).astype("float32")
    df["amt_cents"] = (amt % 1).astype("float32")
    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        both = df["P_emaildomain"].notna() & df["R_emaildomain"].notna()
        same = (df["P_emaildomain"] == df["R_emaildomain"]).to_numpy()
        df["email_match"] = np.where(both, same, np.nan).astype("float32")
    else:
        df["email_match"] = np.float32("nan")
    return df


# ----------------------------------------------------------------------------
# Identité approximative + variables de vélocité (PASSÉ UNIQUEMENT)
# ----------------------------------------------------------------------------
def build_uid(df: pd.DataFrame) -> np.ndarray:
    """Identité reconstituée (approximation, pas de vraie colonne client dans IEEE-CIS).

    uid = card1 | card2 | card3 | card5 | addr1 | jour de première transaction estimé (jour - D1).
    """
    start_day = np.floor(df["TransactionDT"] / 86400) - df["D1"]
    parts = [df[c].astype(str) for c in ["card1", "card2", "card3", "card5", "addr1"] if c in df.columns]
    key = parts[0]
    for p in parts[1:]:
        key = key + "_" + p
    key = key + "_" + start_day.astype(str)
    return pd.factorize(key)[0].astype(np.int64)


_BIG = 10**9  # > TransactionDT maximum (~1,6e7 s)


def add_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule vélocité, écart au profil et délai depuis l'achat précédent.

    Pour chaque transaction on ne regarde que les transactions STRICTEMENT ANTÉRIEURES
    de la même identité : aucune information du futur n'est utilisée.
    Le df doit être trié par TransactionDT (stable).
    """
    uid = build_uid(df)
    t = df["TransactionDT"].to_numpy().astype(np.int64)
    amt = df["TransactionAmt"].to_numpy().astype(np.float64)

    order = np.lexsort((t, uid))                    # tri par identité puis par temps
    u, ts, a = uid[order], t[order], amt[order]
    key = u * _BIG + ts
    n = len(key)
    pos = np.arange(n)

    def lower(window):
        return np.searchsorted(key, u * _BIG + (ts - window), side="left")

    cnt_1h = pos - lower(3600)
    cnt_24h = pos - lower(86400)

    lo30 = lower(30 * 86400)
    cnt30 = pos - lo30
    cs = np.r_[0.0, np.cumsum(a)]
    mean_prev = np.where(cnt30 > 0, (cs[pos] - cs[lo30]) / np.maximum(cnt30, 1), np.nan)
    ratio = np.where(mean_prev > 0, a / np.where(mean_prev > 0, mean_prev, 1.0), np.nan)

    prev_t = np.r_[np.nan, ts[:-1].astype(float)]
    same = np.r_[False, u[1:] == u[:-1]]
    since_prev = np.where(same, ts - prev_t, np.nan)

    def unsort(x):
        out = np.empty(n, dtype=np.float64)
        out[order] = x
        return out.astype("float32")

    df["tx_count_1h"] = unsort(cnt_1h)
    df["tx_count_24h"] = unsort(cnt_24h)
    df["amt_vs_mean_30d"] = unsort(ratio)
    df["time_since_prev"] = unsort(since_prev)
    return df


# ----------------------------------------------------------------------------
# Sélection / encodage (statistiques apprises sur le TRAIN uniquement)
# ----------------------------------------------------------------------------
def select_v_cols(train: pd.DataFrame, n_keep: int = N_V_COLS) -> list[str]:
    v_cols = [c for c in train.columns if re.fullmatch(r"V\d+", c)]
    if not v_cols:
        return []
    miss = train[v_cols].isna().mean().sort_values(kind="stable")
    return list(miss.index[:n_keep])


def fit_cat_maps(train: pd.DataFrame, cat_cols: list[str]) -> dict[str, dict[str, int]]:
    maps = {}
    for c in cat_cols:
        vals = sorted(train[c].dropna().astype(str).unique())
        maps[c] = {v: i for i, v in enumerate(vals)}
    return maps


def feature_list(columns, v_cols: list[str]) -> list[str]:
    wanted = (BASE_NUM + ID_NUM + CAT_COLS + TIME_FEATS + EXTRA_FEATS + BEHAV_FEATS + list(v_cols))
    seen, out = set(), []
    for c in wanted:
        if c in columns and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def encode(df: pd.DataFrame, feats: list[str], maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Matrice float32 : catégories -> codes ordinaux appris sur le train (inconnu -> NaN)."""
    out = {}
    for f in feats:
        if f in maps:
            out[f] = df[f].astype(str).map(maps[f]).astype("float32")
        else:
            out[f] = pd.to_numeric(df[f], errors="coerce").astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_derived_from_row(row: dict) -> dict:
    """Pour l'API : calcule les variables dérivées simples si leurs sources sont fournies."""
    r = dict(row)
    if "TransactionDT" in r and r["TransactionDT"] is not None:
        sec = float(r["TransactionDT"])
        hour = (sec // 3600) % 24
        r.setdefault("hour", hour)
        r.setdefault("h_sin", float(np.sin(2 * np.pi * hour / 24)))
        r.setdefault("h_cos", float(np.cos(2 * np.pi * hour / 24)))
        r.setdefault("dow", (sec // 86400) % 7)
    if "TransactionAmt" in r and r["TransactionAmt"] is not None:
        amt = float(r["TransactionAmt"])
        r.setdefault("amt_log", float(np.log1p(amt)))
        r.setdefault("amt_cents", amt % 1)
    p, q = r.get("P_emaildomain"), r.get("R_emaildomain")
    if p is not None and q is not None:
        r.setdefault("email_match", float(p == q))
    return r
