"""Métriques adaptées aux événements rares, budget d'alertes et coût métier."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def pr_auc(y, s) -> float:
    return float(average_precision_score(y, s))


def roc_auc(y, s) -> float:
    return float(roc_auc_score(y, s))


def topk_flag(s: np.ndarray, k: int) -> np.ndarray:
    """Marque les k transactions au score le plus élevé."""
    k = int(max(0, min(k, len(s))))
    flag = np.zeros(len(s), dtype=bool)
    if k:
        flag[np.argsort(-s, kind="stable")[:k]] = True
    return flag


def confusion(y: np.ndarray, flag: np.ndarray) -> tuple[int, int, int, int]:
    pos = y == 1
    tp = int((flag & pos).sum())
    fp = int((flag & ~pos).sum())
    fn = int((~flag & pos).sum())
    tn = int((~flag & ~pos).sum())
    return tp, fp, fn, tn


def scores_from_confusion(tp: int, fp: int, fn: int, tn: int) -> dict:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    d2 = 5 * tp + 4 * fn + fp
    f2 = 5 * tp / d2 if d2 else 0.0
    den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"precision": prec, "recall": rec, "f2": f2, "mcc": mcc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate_flag(y, amt, flag, cost_per_alert: float) -> dict:
    """Métriques + coût pour un ensemble d'alertes donné."""
    y = np.asarray(y)
    tp, fp, fn, tn = confusion(y, flag)
    out = scores_from_confusion(tp, fp, fn, tn)
    n_alerts = int(flag.sum())
    missed = float((amt * ((y == 1) & ~flag)).sum())
    caught = float((amt * ((y == 1) & flag)).sum())
    out.update({
        "n_alerts": n_alerts, "missed_amount": missed, "caught_amount": caught,
        "analysis_cost": cost_per_alert * n_alerts,
        "cost": missed + cost_per_alert * n_alerts,
        "no_model_cost": float((amt * (y == 1)).sum()),
    })
    out["saving"] = out["no_model_cost"] - out["cost"]
    return out


def metrics_at_k(y, s, amt, k: int, cost_per_alert: float) -> dict:
    return evaluate_flag(y, amt, topk_flag(s, k), cost_per_alert)


def metrics_at_threshold(y, s, amt, tau: float, cost_per_alert: float) -> dict:
    return evaluate_flag(y, amt, s >= tau, cost_per_alert)


def cost_curve(y, s, amt, cost_per_alert: float):
    """Coût C(k) = montants des fraudes manquées + Ca * k, pour tous les budgets k."""
    y = np.asarray(y)
    order = np.argsort(-s, kind="stable")
    caught = np.r_[0.0, np.cumsum((amt * (y == 1))[order])]
    total = float((amt * (y == 1)).sum())
    k = np.arange(len(s) + 1)
    return k, total - caught + cost_per_alert * k, total


def best_budget(y, s, amt, cost_per_alert: float) -> dict:
    """Budget k* et seuil tau* qui minimisent le coût (à calculer sur la VALIDATION)."""
    k, cost, total = cost_curve(y, s, amt, cost_per_alert)
    i = int(np.argmin(cost))
    srt = np.sort(s)[::-1]
    tau = float(srt[i - 1]) if i > 0 else float(srt[0]) + 1e-9
    return {"k": i, "tau": tau, "cost": float(cost[i]), "no_model_cost": total}


def recall_at_budgets(y, s, ks) -> np.ndarray:
    y = np.asarray(y)
    order = np.argsort(-s, kind="stable")
    cum = np.cumsum(y[order])
    ks = np.clip(np.asarray(ks, dtype=int), 1, len(s))
    return cum[ks - 1] / max(int(y.sum()), 1)


def max_recall(y, k: int) -> float:
    """Rappel maximal théorique avec un budget k (modèle parfait)."""
    return min(1.0, k / max(int(np.sum(y)), 1))
