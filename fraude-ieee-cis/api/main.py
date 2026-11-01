"""API FastAPI de scoring de fraude (modèle XGBoost + codes motifs SHAP).

Lancer :  uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs   :  http://localhost:8000/docs  (Swagger)  |  /redoc
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src import features as F
from src import models as MD
from src import store

DESCRIPTION = """
Service de **détection de fraude explicable** (projet Data Mining, jeu IEEE-CIS).

* `POST /score` : score de risque d'une transaction + **codes motifs** (variables qui ont le plus
  poussé le score vers la fraude, valeurs de Shapley exactes).
* `POST /score/batch` : même chose pour un lot (max 1000).
* `GET /transactions/sample` et `GET /transactions/{id}` : transactions de démonstration (jeu de test).
* `GET /model/info` : modèle servi, seuil, variables attendues.

**Entrées** : un dictionnaire `features`. Les variables absentes sont traitées comme manquantes
(XGBoost gère les NaN). Les variables catégorielles peuvent être données en texte (ex. `"ProductCD": "W"`).
Les variables de vélocité (`tx_count_1h`, `tx_count_24h`, `amt_vs_mean_30d`, `time_since_prev`) doivent
être calculées côté appelant à partir de l'historique de l'identité, **passé uniquement**.
"""

app = FastAPI(title="API Détection de fraude explicable", version="1.0.0", description=DESCRIPTION)


# ----------------------------------------------------------------------------
# Modèles de données
# ----------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Variables de la transaction (clé = nom de variable).")
    threshold: float | None = Field(None, description="Seuil d'alerte ; par défaut le seuil optimal en coût.")
    top_reasons: int = Field(5, ge=1, le=10, description="Nombre de codes motifs renvoyés.")

    model_config = {"json_schema_extra": {"example": {
        "features": {"TransactionAmt": 311.5, "ProductCD": "C", "card4": "visa",
                     "P_emaildomain": "anonymous.com", "TransactionDT": 7200000,
                     "tx_count_1h": 4, "tx_count_24h": 6, "amt_vs_mean_30d": 7.2, "time_since_prev": 95},
        "top_reasons": 5}}}


class BatchRequest(BaseModel):
    items: list[dict[str, Any]] = Field(..., max_length=1000)
    threshold: float | None = None


class Reason(BaseModel):
    feature: str
    label: str
    value: Any
    contribution: float = Field(..., description="Contribution (log-odds) ; > 0 = pousse vers la fraude.")


class ScoreResponse(BaseModel):
    model: str
    score: float = Field(..., description="Probabilité estimée de fraude (calibrée).")
    threshold: float
    alert: bool
    risk_level: str
    reasons: list[Reason]
    ignored_features: list[str]


# ----------------------------------------------------------------------------
# Chargement paresseux des artefacts
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _state() -> dict:
    if not store.artifacts_ready():
        raise HTTPException(503, "Artefacts absents : lancez `python -m src.pipeline` d'abord.")
    meta = store.meta()
    smp = store.sample().set_index("TransactionID")
    return {"meta": meta, "model": store.load_model("xgboost"), "feats": meta["features"],
            "maps": store.cat_maps(), "beta": meta["beta_xgboost"], "sample": smp}


def _matrix(items: list[dict]) -> tuple[pd.DataFrame, list[list[str]], list[dict]]:
    st = _state()
    feats, maps = st["feats"], st["maps"]
    rows, ignored, originals = [], [], []
    for item in items:
        item = F.add_derived_from_row(item)
        row, ign = {}, []
        for k, v in item.items():
            if k not in feats and k != "TransactionDT":
                ign.append(k)
        for f in feats:
            v = item.get(f)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row[f] = np.nan
            elif f in maps:
                row[f] = float(maps[f].get(str(v), np.nan)) if isinstance(v, str) else float(v)
            else:
                try:
                    row[f] = float(v)
                except (TypeError, ValueError):
                    row[f] = np.nan
        rows.append(row)
        ignored.append(ign)
        originals.append(item)
    return pd.DataFrame(rows, columns=feats).astype("float32"), ignored, originals


def _risk(score: float, tau: float) -> str:
    return "élevé" if score >= tau else ("moyen" if score >= 0.5 * tau else "faible")


def _score_frame(X: pd.DataFrame, ignored, originals, threshold, top) -> list[ScoreResponse]:
    st = _state()
    tau = float(st["meta"]["threshold_xgboost"] if threshold is None else threshold)
    raw = st["model"].predict_proba(X)[:, 1]
    prob = MD.calibrate(raw, st["beta"])
    contrib = st["model"].contribs(X)[:, :-1]
    out = []
    for i in range(len(X)):
        order = np.argsort(-contrib[i])[:top]
        reasons = [Reason(feature=st["feats"][j], label=F.label(st["feats"][j]),
                          value=originals[i].get(st["feats"][j], None if np.isnan(X.iloc[i, j]) else float(X.iloc[i, j])),
                          contribution=float(contrib[i, j]))
                   for j in order if contrib[i, j] > 0]
        out.append(ScoreResponse(model="xgboost", score=float(prob[i]), threshold=tau,
                                 alert=bool(prob[i] >= tau), risk_level=_risk(float(prob[i]), tau),
                                 reasons=reasons, ignored_features=ignored[i]))
    return out


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/health", tags=["service"], summary="État du service")
def health():
    return {"status": "ok", "artifacts": store.artifacts_ready()}


@app.get("/model/info", tags=["service"], summary="Informations sur le modèle servi")
def model_info():
    st = _state()
    m, res = st["meta"], store.results()["models"]["xgboost"]
    return {"model": "xgboost", "strategy": m["selected_strategy"], "threshold": m["threshold_xgboost"],
            "cost_per_alert": m["cost_per_alert"], "n_features": len(st["feats"]), "features": st["feats"],
            "categorical_features": m["cat_cols"], "data_source": m["source"], "built_at": m["built_at"],
            "metrics_test": res["test"], "metrics_validation": res["validation"]}


@app.post("/score", response_model=ScoreResponse, tags=["scoring"], summary="Scorer une transaction")
def score(req: ScoreRequest):
    X, ign, orig = _matrix([req.features])
    return _score_frame(X, ign, orig, req.threshold, req.top_reasons)[0]


@app.post("/score/batch", tags=["scoring"], summary="Scorer un lot de transactions")
def score_batch(req: BatchRequest):
    X, ign, orig = _matrix(req.items)
    res = _score_frame(X, ign, orig, req.threshold, 3)
    return {"count": len(res), "alerts": int(sum(r.alert for r in res)), "results": res}


@app.get("/transactions/sample", tags=["démo"], summary="Transactions de démonstration (jeu de test)")
def sample(n: int = Query(10, ge=1, le=200), only_alerts: bool = True):
    s = _state()["sample"].reset_index()
    tau = _state()["meta"]["threshold_xgboost"]
    if only_alerts:
        s = s[s["score"] >= tau]
    s = s.sort_values("score", ascending=False).head(n)
    return [{"TransactionID": int(r.TransactionID), "score": float(r.score), "amount": float(r.amount),
             "is_fraud_ground_truth": int(r.y)} for r in s.itertuples()]


@app.get("/transactions/{transaction_id}", response_model=ScoreResponse, tags=["démo"],
         summary="Scorer et expliquer une transaction de l'échantillon")
def transaction(transaction_id: int, top_reasons: int = Query(5, ge=1, le=10)):
    smp = _state()["sample"]
    if transaction_id not in smp.index:
        raise HTTPException(404, "TransactionID absent de l'échantillon de démonstration.")
    feats = _state()["feats"]
    X = smp.loc[[transaction_id], feats].astype("float32")
    X.index = [0]
    return _score_frame(X, [[]], [{}], None, top_reasons)[0]
