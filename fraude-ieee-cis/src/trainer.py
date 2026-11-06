"""Entraînement des modèles choisis dans l'application et scores au même format que artifacts/scores.parquet."""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd

from . import metrics as M
from . import models as MD
from .prep import Prepared

UNSUP_SIZES = {"lof": 5000, "ocsvm": 5000, "autoencoder": 20000}   # lignes « normales » utilisées pour apprendre


def train(p: Prepared, keys: list[str], strategy_name: str, params: dict, ca: float, log=print) -> dict:
    ytr = p.y["train"]
    if min(int(p.y[s].sum()) for s in ("train", "validation", "test")) < 5:
        raise ValueError("Moins de 5 fraudes dans un des jeux : augmentez le nombre de lignes.")
    strategy = next(s for s in MD.STRATEGIES if s["name"] == strategy_name)
    seed, Xtr = p.cfg.seed, p.X["train"]
    rng = np.random.default_rng(seed)
    normal = np.where(ytr == 0)[0]
    sc = {s: {} for s in ("validation", "test")}
    fit_s = {}
    for key in keys:
        t0 = time.time()
        if MD.is_supervised(key):
            mdl, beta, _ = MD.fit_supervised(key, Xtr, ytr, strategy, seed=seed, params=params.get(key))
        else:
            beta, mdl = 1.0, MD.build_unsupervised(key, Xtr.shape[1], seed=seed, params=params.get(key))
            if key == "iforest":
                mdl.fit(Xtr)
            else:
                mdl.fit(Xtr.iloc[rng.choice(normal, min(UNSUP_SIZES[key], len(normal)), replace=False)])
        for split in sc:
            s = MD.raw_score(key, mdl, p.X[split])
            sc[split][key] = MD.calibrate(s, beta) if MD.is_supervised(key) else s
        fit_s[key] = round(time.time() - t0, 1)
        log(f"{MD.label(key)} entraîné en {fit_s[key]} s")

    frames = []
    for split in sc:
        d = pd.DataFrame({"split": split, "TransactionID": p.tid[split], "y": p.y[split],
                          "amount": p.amount[split], "dt": p.dt[split]})
        for key, v in sc[split].items():
            d[f"score_{key}"] = v.astype("float32")
        frames.append(d)
    scores = pd.concat(frames, ignore_index=True)

    pr = {k: M.pr_auc(p.y["validation"], sc["validation"][k]) for k in keys}
    n_all = sum(len(p.y[s]) for s in p.y)
    meta = {"source": "entraînement dans l'application", "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "n_rows": n_all, "fraud_rate": 100 * sum(int(p.y[s].sum()) for s in p.y) / n_all,
            "cost_per_alert": ca, "strategy": strategy_name, "fit_seconds": fit_s}
    return {"scores": scores, "meta": meta, "res": {"best_model": max(pr, key=pr.get)}, "keys": keys,
            "n_features": Xtr.shape[1]}


def summary_at_threshold(run: dict, ca: float, tau: float = 0.5) -> pd.DataFrame:
    """Jeu de test : métriques sans seuil (PR-AUC, ROC-AUC) et au seuil tau pour les modèles à probabilités."""
    d = run["scores"][run["scores"]["split"] == "test"]
    y, amt, rows = d["y"].to_numpy(), d["amount"].to_numpy(), []
    for key in run["keys"]:
        s = d[f"score_{key}"].to_numpy()
        r = {"Modèle": MD.label(key), "Famille": MD.MODELS[key][1], "PR-AUC": M.pr_auc(y, s), "ROC-AUC": M.roc_auc(y, s)}
        if MD.is_supervised(key):
            m = M.metrics_at_threshold(y, s, amt, tau, ca)
            r.update({"Alertes": m["n_alerts"], "Précision": m["precision"], "Rappel": m["recall"],
                      "F2": m["f2"], "MCC": m["mcc"], "Économie ($)": m["saving"]})
        rows.append(r)
    return pd.DataFrame(rows)