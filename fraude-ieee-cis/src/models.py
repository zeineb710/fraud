"""Modèles supervisés et non supervisés, stratégies contre le déséquilibre."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import OneClassSVM

# clé -> (nom affiché, famille)
MODELS = {
    "xgboost": ("XGBoost", "supervisé"),
    "random_forest": ("Random Forest", "supervisé"),
    "logreg": ("Régression logistique", "supervisé"),
    "iforest": ("Isolation Forest", "non supervisé"),
    "lof": ("LOF", "non supervisé"),
    "ocsvm": ("One-Class SVM", "non supervisé"),
    "autoencoder": ("Autoencodeur", "non supervisé"),
}

# Réglages par défaut (valeurs utilisées par le pipeline) ; l'application permet de les modifier
DEFAULT_PARAMS = {
    "xgboost": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.08},
    "random_forest": {"n_estimators": 80, "max_depth": 10, "min_samples_leaf": 100},
    "logreg": {"C": 0.5},
    "iforest": {"n_estimators": 100, "max_samples": 256},
    "lof": {"n_neighbors": 20},
    "ocsvm": {"nu": 0.02},
    "autoencoder": {"bottleneck": 16, "max_iter": 40},
}

STRATEGIES = [
    {"name": "aucune", "kind": "none"},
    {"name": "poids des classes", "kind": "weights"},
    {"name": "sous-éch. 10:1", "kind": "under", "ratio": 10},
    {"name": "sous-éch. 3:1", "kind": "under", "ratio": 3},
]


def label(key: str) -> str:
    return MODELS[key][0]


def is_supervised(key: str) -> bool:
    return MODELS[key][1] == "supervisé"


# ----------------------------------------------------------------------------
# Pré-traitement commun aux modèles qui n'acceptent pas les NaN
# ----------------------------------------------------------------------------
def clip10(X):
    return np.clip(X, -10, 10)


def _prep():
    return [("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("sc", StandardScaler()),
            ("clip", FunctionTransformer(clip10))]


class AEDetector(BaseEstimator):
    """Autoencodeur (MLP avec goulot) : score = erreur de reconstruction."""

    def __init__(self, hidden=(64, 16, 64), max_iter=40, seed=42):
        self.hidden = hidden
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X, y=None):
        self.mlp_ = MLPRegressor(hidden_layer_sizes=self.hidden, activation="relu",
                                 max_iter=self.max_iter, early_stopping=True,
                                 n_iter_no_change=5, random_state=self.seed)
        self.mlp_.fit(X, X)
        return self

    def score_samples(self, X):
        return -((X - self.mlp_.predict(X)) ** 2).mean(axis=1)


class XGBScorer:
    """Enveloppe d'un Booster XGBoost chargé depuis JSON (probabilités + contributions SHAP)."""

    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    @classmethod
    def load(cls, path):
        b = xgb.Booster()
        b.load_model(str(path))
        return cls(b)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.booster.predict(xgb.DMatrix(X))
        return np.c_[1 - p, p]

    def contribs(self, X: pd.DataFrame) -> np.ndarray:
        """Valeurs de Shapley exactes (TreeSHAP natif) ; dernière colonne = valeur de base."""
        return self.booster.predict(xgb.DMatrix(X), pred_contribs=True)


# ----------------------------------------------------------------------------
# Constructeurs
# ----------------------------------------------------------------------------
def build_xgb(spw=1.0, n_trees=300, seed=42, max_depth=6, learning_rate=0.08):
    return xgb.XGBClassifier(
        n_estimators=n_trees, max_depth=max_depth, learning_rate=learning_rate, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=2, tree_method="hist",
        eval_metric="aucpr", scale_pos_weight=spw, random_state=seed, n_jobs=-1)


def build_supervised(key: str, spw=1.0, cw=None, n_trees=300, seed=42, params=None):
    p = {**DEFAULT_PARAMS.get(key, {}), **(params or {})}
    if key == "xgboost":
        return build_xgb(spw, p["n_estimators"] if params else n_trees, seed, p["max_depth"], p["learning_rate"])
    if key == "logreg":
        return Pipeline(_prep() + [("clf", LogisticRegression(max_iter=300, C=p["C"], class_weight=cw))])
    if key == "random_forest":
        n_est = p["n_estimators"] if params else (80 if n_trees >= 100 else 20)
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("clf", RandomForestClassifier(
                             n_estimators=n_est, max_depth=p["max_depth"],
                             min_samples_leaf=p["min_samples_leaf"], max_samples=0.5, n_jobs=-1,
                             class_weight=cw, random_state=seed))])
    raise KeyError(key)


def build_unsupervised(key: str, n_features: int, seed=42, fast=False, params=None):
    p = {**DEFAULT_PARAMS.get(key, {}), **(params or {})}
    nc = max(2, min(20, n_features))
    if key == "iforest":
        return Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("clf", IsolationForest(n_estimators=p["n_estimators"], max_samples=p["max_samples"],
                                                 random_state=seed, n_jobs=-1))])
    if key == "lof":
        return Pipeline(_prep() + [("pca", PCA(nc, random_state=seed)),
                                   ("clf", LocalOutlierFactor(n_neighbors=p["n_neighbors"], novelty=True, n_jobs=-1))])
    if key == "ocsvm":
        return Pipeline(_prep() + [("pca", PCA(nc, random_state=seed)),
                                   ("clf", OneClassSVM(kernel="rbf", nu=p["nu"], gamma="scale"))])
    if key == "autoencoder":
        it = p["max_iter"] if params else (15 if fast else 40)
        return Pipeline(_prep() + [("clf", AEDetector(hidden=(64, int(p["bottleneck"]), 64), max_iter=it, seed=seed))])
    raise KeyError(key)


# ----------------------------------------------------------------------------
# Entraînement avec stratégie anti-déséquilibre (appliquée au TRAIN seulement)
# ----------------------------------------------------------------------------
def undersample_idx(y: np.ndarray, ratio: int, seed: int):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m = min(len(neg), int(ratio * len(pos)))
    keep = rng.choice(neg, m, replace=False)
    idx = np.concatenate([pos, keep])
    rng.shuffle(idx)
    return idx, m / len(neg)


def fit_supervised(key, X: pd.DataFrame, y: np.ndarray, strategy: dict, seed=42, n_trees=300, params=None):
    """Retourne (modèle, beta, nb de lignes d'entraînement).

    beta = part des cotes conservée par la stratégie : cote réelle = beta * cote apprise
    (voir cours : p = beta*ps / (beta*ps - ps + 1)).
    """
    spw, cw, beta, idx = 1.0, None, 1.0, None
    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
    if strategy["kind"] == "weights":
        spw, cw, beta = n_neg / n_pos, "balanced", n_pos / n_neg
    elif strategy["kind"] == "under":
        idx, beta = undersample_idx(y, strategy["ratio"], seed)
    Xf, yf = (X, y) if idx is None else (X.iloc[idx], y[idx])
    model = build_supervised(key, spw=spw, cw=cw, n_trees=n_trees, seed=seed, params=params)
    model.fit(Xf, yf)
    return model, float(beta), int(len(yf))


def calibrate(p, beta: float):
    """Corrige les probabilités après rééchantillonnage / pondération."""
    p = np.asarray(p, dtype=float)
    return beta * p / (beta * p - p + 1.0)


def raw_score(key: str, model, X) -> np.ndarray:
    """Score « plus grand = plus suspect »."""
    if is_supervised(key):
        return model.predict_proba(X)[:, 1]
    return -model.score_samples(X)
