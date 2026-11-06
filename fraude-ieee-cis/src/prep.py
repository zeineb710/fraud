"""Préparation configurable des données (appliquée dans l'application, sans fuite : tout est appris sur le train)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from . import features as F

# Groupes de variables proposés à l'utilisateur (nom affiché -> colonnes)
GROUPS = {
    "Montant et produit": ["TransactionAmt", "ProductCD"],
    "Carte": ["card1", "card2", "card3", "card4", "card5", "card6"],
    "Adresse et distance": ["addr1", "addr2", "dist1", "dist2"],
    "Emails": ["P_emaildomain", "R_emaildomain"],
    "Compteurs C (anonymisés)": F.C_COLS,
    "Délais D (anonymisés)": F.D_COLS,
    "Correspondances M": F.M_COLS,
    "Identité numérique (id_01 à id_11)": F.ID_NUM,
    "Navigateur et appareil": F.ID_CAT,
    "Heure et jour": F.TIME_FEATS,
    "Montant et emails dérivés": F.EXTRA_FEATS,
    "Comportement (vélocité, passé seul)": F.BEHAV_FEATS,
}
NUM_IMPUTE = ["Médiane", "Moyenne", "Zéro", "Laisser manquant"]
CAT_IMPUTE = ["Catégorie « inconnu »", "Valeur la plus fréquente"]
ENCODINGS = ["Ordinal (codes entiers)", "One-hot", "Fréquence"]
SCALINGS = ["Aucune", "Standardisation", "Min-Max"]


@dataclass
class PrepConfig:
    groups: list[str] = field(default_factory=lambda: list(GROUPS))
    n_v: int = 40
    n_rows: int = 40000
    val_frac: float = 0.2
    test_frac: float = 0.2
    num_impute: str = "Médiane"
    cat_impute: str = CAT_IMPUTE[0]
    encoding: str = ENCODINGS[0]
    rare: float = 0.01
    scaling: str = "Aucune"
    seed: int = 42


@dataclass
class Prepared:
    X: dict          # "train" / "validation" / "test" -> DataFrame float32
    y: dict
    amount: dict
    tid: dict
    dt: dict
    features: list
    cfg: PrepConfig
    report: dict


def prepare(raw: pd.DataFrame, cfg: PrepConfig) -> Prepared:
    df = raw.sort_values("TransactionDT", kind="stable").head(cfg.n_rows).reset_index(drop=True)
    df = F.add_time_features(df)
    if "Comportement (vélocité, passé seul)" in cfg.groups and "D1" in df.columns:
        df = F.add_behavior_features(df)
    n = len(df)
    n_te, n_va = int(cfg.test_frac * n), int(cfg.val_frac * n)
    n_tr = n - n_va - n_te
    bounds = {"train": (0, n_tr), "validation": (n_tr, n_tr + n_va), "test": (n_tr + n_va, n)}
    part = {k: df.iloc[a:b] for k, (a, b) in bounds.items()}
    tr = part["train"]

    feats = [c for g in cfg.groups for c in GROUPS[g] if c in df.columns]
    v_cols = F.select_v_cols(tr, cfg.n_v) if cfg.n_v > 0 else []   # choisies sur le train
    feats += [c for c in v_cols if c not in feats]
    cat = [c for c in feats if c in F.CAT_COLS]
    num = [c for c in feats if c not in cat]

    # 1. catégories : valeurs manquantes puis encodage (statistiques apprises sur le train)
    fill = {c: ("inconnu" if cfg.cat_impute == CAT_IMPUTE[0] else
                (tr[c].mode(dropna=True).iloc[0] if tr[c].notna().any() else "inconnu")) for c in cat}

    def cat_frame(d):
        return pd.DataFrame({c: d[c].where(d[c].notna(), fill[c]).astype(str) for c in cat}, index=d.index)

    onehot_cols = []
    if cat and cfg.encoding == ENCODINGS[2]:
        freq = {c: cat_frame(tr)[c].value_counts(normalize=True) for c in cat}
        enc = lambda d: pd.DataFrame({f"{c}_freq": cat_frame(d)[c].map(freq[c]).fillna(0.0) for c in cat}, index=d.index)
    elif cat and cfg.encoding == ENCODINGS[1]:
        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=cfg.rare,
                            sparse_output=False, dtype=np.float32).fit(cat_frame(tr))
        names = list(ohe.get_feature_names_out())
        enc = lambda d: pd.DataFrame(ohe.transform(cat_frame(d)), columns=names, index=d.index)
        onehot_cols = names
    elif cat:
        maps = F.fit_cat_maps(cat_frame(tr), cat)
        enc = lambda d: F.encode(cat_frame(d), cat, maps).add_suffix("_code")
    else:
        enc = lambda d: pd.DataFrame(index=d.index)

    def build(d):
        nums = pd.DataFrame({c: pd.to_numeric(d[c], errors="coerce").astype("float32") for c in num}, index=d.index)
        return pd.concat([nums, enc(d)], axis=1).astype("float32")

    X = {k: build(v) for k, v in part.items()}
    cols = list(X["train"].columns)
    missing_before = float(X["train"].isna().mean().mean() * 100)

    # 2. valeurs manquantes numériques (apprises sur le train)
    impute_cols = [c for c in cols if c not in onehot_cols]
    strat = {"Médiane": "median", "Moyenne": "mean", "Zéro": "constant"}.get(cfg.num_impute)
    if strat and impute_cols:
        imp = SimpleImputer(strategy=strat, fill_value=0.0, keep_empty_features=True).fit(X["train"][impute_cols])
        for k in X:
            X[k][impute_cols] = imp.transform(X[k][impute_cols]).astype("float32")

    # 3. normalisation (apprise sur le train ; jamais sur les one-hot)
    scale_cols = [c for c in cols if c not in onehot_cols]
    if cfg.scaling != SCALINGS[0] and scale_cols:
        sc = (StandardScaler() if cfg.scaling == SCALINGS[1] else MinMaxScaler()).fit(X["train"][scale_cols])
        for k in X:
            X[k][scale_cols] = np.clip(sc.transform(X[k][scale_cols]), -10, 10).astype("float32")

    y = {k: v["isFraud"].to_numpy().astype(int) for k, v in part.items()}
    report = {
        "n_features": len(cols), "n_raw_features": len(feats), "n_cat": len(cat),
        "missing_before": missing_before, "missing_after": float(X["train"].isna().mean().mean() * 100),
        "splits": pd.DataFrame([{"Jeu": k, "Lignes": len(y[k]), "Fraudes": int(y[k].sum()),
                                 "Taux de fraude (%)": round(100 * y[k].mean(), 2),
                                 "Jour début": round(float(part[k]["TransactionDT"].min() / 86400), 1),
                                 "Jour fin": round(float(part[k]["TransactionDT"].max() / 86400), 1)} for k in part]),
    }
    return Prepared(X=X, y=y, amount={k: v["TransactionAmt"].to_numpy().astype(float) for k, v in part.items()},
                    tid={k: v["TransactionID"].to_numpy() for k, v in part.items()},
                    dt={k: v["TransactionDT"].to_numpy() for k, v in part.items()},
                    features=cols, cfg=cfg, report=report)