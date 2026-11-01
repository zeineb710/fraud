"""Pipeline complet : téléchargement -> variables -> modèles -> évaluation -> artefacts.

Usage :
    python -m src.pipeline                 # données réelles IEEE-CIS (téléchargement Kaggle auto)
    python -m src.pipeline --max-rows 150000
    python -m src.pipeline --synthetic --fast     # démo / test rapide (données fictives)
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import eda as EDA
from . import features as F
from . import metrics as M
from . import models as MD
from .download import ensure_raw_data


def _default_log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _summary(y, s, amt, k, ca) -> dict:
    m = M.metrics_at_k(y, s, amt, k, ca)
    return {"pr_auc": M.pr_auc(y, s), "roc_auc": M.roc_auc(y, s), "k": int(k),
            "precision_at_k": m["precision"], "recall_at_k": m["recall"],
            "f2_at_k": m["f2"], "mcc_at_k": m["mcc"], "cost_at_k": m["cost"]}


def run(max_rows: int | None = None, fast: bool = False, synthetic: bool = False,
        seeds: int = 2, log=_default_log) -> dict:
    t0 = time.time()
    max_rows = C.max_rows() if max_rows is None else max_rows
    ca = C.default_cost_per_alert()
    for d in (C.ART, C.MODELS_DIR, C.EDA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1. données
    if synthetic:
        from .synthetic import make_synthetic
        log("Génération d'un jeu SYNTHÉTIQUE (mode démo, pas les vraies données)…")
        df = make_synthetic(n=max_rows or 40000)
        source = "synthetic"
    else:
        ensure_raw_data(log)
        log("Chargement et fusion des CSV…")
        df = F.load_raw(max_rows)
        source = "kaggle:ieee-fraud-detection"
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    n = len(df)
    n_tr, n_va = C.split_points(n)
    log(f"{n:,} transactions, taux de fraude {df['isFraud'].mean():.2%}")

    # ---------------------------------------------------------------- 2. variables (passé uniquement)
    log("Variables temporelles et comportementales (passé uniquement)…")
    df = F.add_time_features(df.copy())
    df = F.add_behavior_features(df)

    tr_raw = df.iloc[:n_tr]
    v_cols = F.select_v_cols(tr_raw)                        # choisi sur le train
    cat_cols = [c for c in F.CAT_COLS if c in df.columns]
    maps = F.fit_cat_maps(tr_raw, cat_cols)                 # appris sur le train
    feats = F.feature_list(df.columns, v_cols)
    log(f"{len(feats)} variables retenues.")

    log("Tables d'exploration…")
    EDA.build_eda(df, n_tr, n_va)

    X = F.encode(df, feats, maps)
    y = df["isFraud"].to_numpy().astype(int)
    amt = df["TransactionAmt"].to_numpy().astype(float)
    tid = df["TransactionID"].to_numpy()
    dt = df["TransactionDT"].to_numpy()
    del df
    gc.collect()

    sl = {"train": slice(0, n_tr), "validation": slice(n_tr, n_va), "test": slice(n_va, n)}
    Xtr, ytr = X.iloc[sl["train"]], y[sl["train"]]
    Xva, yva, ava = X.iloc[sl["validation"]], y[sl["validation"]], amt[sl["validation"]]
    Xte, yte, ate = X.iloc[sl["test"]], y[sl["test"]], amt[sl["test"]]
    k_val, k_test = int(C.BUDGET_FRAC * len(yva)), int(C.BUDGET_FRAC * len(yte))
    n_trees = 60 if fast else 300

    # ---------------------------------------------------------------- 3. stratégie anti-déséquilibre
    log(f"Comparaison des stratégies contre le déséquilibre (XGBoost, {seeds} graine(s))…")
    rows = []
    for st in MD.STRATEGIES:
        for sd in range(seeds):
            mdl, _, n_rows = MD.fit_supervised("xgboost", Xtr, ytr, st, seed=C.SEED + sd, n_trees=n_trees)
            s = MD.raw_score("xgboost", mdl, Xva)
            r = _summary(yva, s, ava, k_val, ca)
            rows.append({"strategie": st["name"], "graine": sd, "lignes_train": n_rows, **r})
            log(f"  {st['name']:<18} graine {sd}: PR-AUC {r['pr_auc']:.4f}")
    raw = pd.DataFrame(rows)
    order = [s["name"] for s in MD.STRATEGIES]
    agg = raw.groupby("strategie").agg(
        lignes_train=("lignes_train", "first"),
        pr_auc=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        roc_auc=("roc_auc", "mean"), recall_at_k=("recall_at_k", "mean"),
        precision_at_k=("precision_at_k", "mean")).reindex(order).fillna(0.0)
    best_name = agg["pr_auc"].idxmax()
    tol = max(0.002, float(agg.loc[best_name, "pr_auc_std"]))
    chosen = "aucune" if agg.loc["aucune", "pr_auc"] >= agg.loc[best_name, "pr_auc"] - tol else best_name
    strategy = next(s for s in MD.STRATEGIES if s["name"] == chosen)
    log(f"Stratégie retenue : « {chosen} » (règle : on ne rééchantillonne que si le gain dépasse le bruit).")
    imbalance = agg.reset_index().to_dict(orient="records")

    # ---------------------------------------------------------------- 4. modèles finaux
    fitted, betas, scores_val, scores_test = {}, {}, {}, {}
    for key in ("xgboost", "random_forest", "logreg"):
        log(f"Entraînement : {MD.label(key)}…")
        mdl, beta, _ = MD.fit_supervised(key, Xtr, ytr, strategy, seed=C.SEED, n_trees=n_trees)
        fitted[key], betas[key] = mdl, beta

    rng = np.random.default_rng(C.SEED)
    normal_idx = np.where(ytr == 0)[0]
    sizes = {"lof": 3000 if fast else 20000, "ocsvm": 3000 if fast else 10000,
             "autoencoder": 20000 if fast else 100000}
    for key in ("iforest", "lof", "ocsvm", "autoencoder"):
        log(f"Entraînement : {MD.label(key)}…")
        mdl = MD.build_unsupervised(key, X.shape[1], seed=C.SEED, fast=fast)
        if key == "iforest":
            mdl.fit(Xtr)                                    # détection d'anomalies (non étiqueté)
        else:                                               # détection de nouveauté : normales seulement
            idx = rng.choice(normal_idx, min(sizes[key], len(normal_idx)), replace=False)
            mdl.fit(Xtr.iloc[idx])
        fitted[key], betas[key] = mdl, 1.0

    log("Calcul des scores sur validation et test…")
    for key, mdl in fitted.items():
        sv, st_ = MD.raw_score(key, mdl, Xva), MD.raw_score(key, mdl, Xte)
        if MD.is_supervised(key):
            sv, st_ = MD.calibrate(sv, betas[key]), MD.calibrate(st_, betas[key])
        scores_val[key], scores_test[key] = sv, st_

    # ---------------------------------------------------------------- 5. évaluation + seuil optimal (coût)
    log("Évaluation, budget k et seuil optimal en coût…")
    res_models = {}
    for key in fitted:
        sv, st_ = scores_val[key], scores_test[key]
        opt = M.best_budget(yva, sv, ava, ca)
        res_models[key] = {
            "label": MD.label(key), "kind": MD.MODELS[key][1],
            "validation": _summary(yva, sv, ava, k_val, ca),
            "test": _summary(yte, st_, ate, k_test, ca),
            "optimal": {
                "k_val": opt["k"], "tau": opt["tau"],
                "validation": M.metrics_at_threshold(yva, sv, ava, opt["tau"], ca),
                "test": M.metrics_at_threshold(yte, st_, ate, opt["tau"], ca),
            },
        }
    best_model = max(res_models, key=lambda k: res_models[k]["validation"]["pr_auc"])
    log(f"Meilleur modèle (PR-AUC validation) : {MD.label(best_model)}")

    # ---------------------------------------------------------------- 6. artefacts
    sc_rows = []
    for split, ids, yy, aa, dd, scs in (
            ("validation", tid[sl["validation"]], yva, ava, dt[sl["validation"]], scores_val),
            ("test", tid[sl["test"]], yte, ate, dt[sl["test"]], scores_test)):
        d = pd.DataFrame({"split": split, "TransactionID": ids, "y": yy, "amount": aa, "dt": dd})
        for key, v in scs.items():
            d[f"score_{key}"] = v.astype("float32")
        sc_rows.append(d)
    pd.concat(sc_rows, ignore_index=True).to_parquet(C.ART / "scores.parquet", index=False)

    for key, mdl in fitted.items():
        if key == "xgboost":
            mdl.get_booster().save_model(str(C.MODELS_DIR / "xgboost.json"))
        else:
            joblib.dump(mdl, C.MODELS_DIR / f"{key}.joblib", compress=3)

    # échantillon pour l'explicabilité (alertes les plus fortes + tirage aléatoire du test)
    xgb_s = scores_test["xgboost"]
    n_top, n_rand = (300, 1200) if fast else (3000, 12000)
    top = np.argsort(-xgb_s)[:n_top]
    rest = np.setdiff1d(np.arange(len(yte)), top)
    pick = np.sort(np.concatenate([top, rng.choice(rest, min(n_rand, len(rest)), replace=False)]))
    smp = Xte.iloc[pick].copy()
    smp.insert(0, "TransactionID", tid[sl["test"]][pick])
    smp.insert(1, "y", yte[pick])
    smp.insert(2, "amount", ate[pick])
    smp.insert(3, "score", xgb_s[pick].astype("float32"))
    smp.to_parquet(C.ART / "explain_sample.parquet", index=False)

    scorer = MD.XGBScorer(fitted["xgboost"].get_booster())
    gi = rng.choice(len(Xte), min(5000, len(Xte)), replace=False)
    contrib = scorer.contribs(Xte.iloc[gi])[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=feats).sort_values(ascending=False).head(25)
    (C.ART / "shap_importance.json").write_text(json.dumps(
        [{"feature": f, "label": F.label(f), "mean_abs_shap": float(v)} for f, v in imp.items()],
        ensure_ascii=False, indent=1))

    (C.ART / "cat_maps.json").write_text(json.dumps(maps, ensure_ascii=False))
    results = {"imbalance": imbalance, "imbalance_runs": raw.to_dict(orient="records"),
               "selected_strategy": chosen, "models": res_models, "best_model": best_model,
               "k_val": k_val, "k_test": k_test, "cost_per_alert": ca,
               "no_model_cost": {"validation": float((ava * yva).sum()), "test": float((ate * yte).sum())}}
    (C.ART / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

    split_info = {}
    for name, s_ in sl.items():
        split_info[name] = {"rows": int(s_.stop - s_.start), "fraud_rate": float(y[s_].mean() * 100),
                            "day_start": float(dt[s_].min() / 86400), "day_end": float(dt[s_].max() / 86400)}
    meta = {
        "source": source, "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_rows": int(n), "n_fraud": int(y.sum()), "fraud_rate": float(y.mean() * 100),
        "splits": split_info, "k_val": k_val, "k_test": k_test, "budget_frac": C.BUDGET_FRAC,
        "cost_per_alert": ca, "features": feats, "cat_cols": cat_cols, "v_cols": v_cols,
        "selected_strategy": chosen, "beta_xgboost": betas["xgboost"], "best_model": best_model,
        "threshold_xgboost": res_models["xgboost"]["optimal"]["tau"],
        "fast_mode": bool(fast), "elapsed_s": round(time.time() - t0),
    }
    (C.ART / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    log(f"Terminé en {meta['elapsed_s']} s. Artefacts écrits dans artifacts/.")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-rows", type=int, default=None, help="tranche chronologique (0 = tout)")
    ap.add_argument("--fast", action="store_true", help="modèles allégés pour tester vite")
    ap.add_argument("--synthetic", action="store_true", help="données fictives (démo)")
    ap.add_argument("--seeds", type=int, default=2, help="graines pour la comparaison des stratégies")
    a = ap.parse_args()
    run(max_rows=a.max_rows, fast=a.fast, synthetic=a.synthetic, seeds=a.seeds)


if __name__ == "__main__":
    main()
