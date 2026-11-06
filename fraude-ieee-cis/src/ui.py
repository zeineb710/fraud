"""Éléments d'interface Streamlit communs à toutes les pages."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SECRET_KEYS = ("KAGGLE_USERNAME", "KAGGLE_KEY", "API_URL", "MAX_ROWS", "COST_PER_ALERT", "AUTO_BUILD")


def sync_secrets() -> None:
    """Copie les secrets Streamlit dans les variables d'environnement (Kaggle, API_URL…)."""
    try:
        for k in _SECRET_KEYS:
            if k in st.secrets and k not in os.environ:
                os.environ[k] = str(st.secrets[k])
    except Exception:  # noqa: BLE001  (pas de secrets.toml en local)
        pass


def setup(title: str, icon: str = "🕵️") -> None:
    st.set_page_config(page_title=f"{title} · Fraude IEEE-CIS", page_icon=icon, layout="wide")
    sync_secrets()
    st.sidebar.markdown("### 🕵️ Fraude explicable\nIEEE-CIS · Data Mining · ENSI")


def _run_build(**kw) -> None:
    from src import pipeline, store
    with st.status("Pipeline en cours (téléchargement, variables, modèles)…", expanded=True) as status:
        try:
            pipeline.run(log=lambda m: st.write(m), **kw)
            status.update(label="Terminé", state="complete")
        except Exception as e:  # noqa: BLE001
            st.session_state["build_failed"] = True
            status.update(label="Échec", state="error")
            st.error(str(e))
            return
    store.clear_caches()
    st.rerun()


def require_artifacts() -> None:
    """Arrête la page si les artefacts n'existent pas ; propose de les construire."""
    from src import store
    if store.artifacts_ready():
        meta = store.meta()
        if meta["source"] == "synthetic":
            st.warning("**Mode démo** : données SYNTHÉTIQUES générées pour tester l'application. "
                       "Les résultats ne reflètent pas le jeu IEEE-CIS réel.")
        return

    st.title("Construction du projet")
    st.info("Les artefacts (modèles, scores, tables) sont absents. Ils sont produits par le pipeline, "
            "qui **télécharge automatiquement** les données IEEE-CIS depuis Kaggle.")
    st.markdown("Prérequis : règles de la compétition acceptées sur Kaggle + secrets "
                "`KAGGLE_USERNAME` et `KAGGLE_KEY`. Sur le cloud, ajoutez `MAX_ROWS` (ex. 150000) pour limiter la mémoire.")
    auto = os.getenv("AUTO_BUILD", "").lower() in ("1", "true", "yes")
    if (auto and not st.session_state.get("build_failed")) or st.button("🚀 Télécharger les données et construire", type="primary"):
        _run_build()
    if st.button("🧪 Mode démo (données synthétiques, 1 minute)"):
        _run_build(synthetic=True, fast=True, max_rows=40000, seeds=1)
    st.stop()


def money(x: float) -> str:
    return f"{x:,.0f} $".replace(",", " ")

def active_run():
    """Scores, méta et résultats à utiliser : ceux de l'entraînement fait dans l'application, sinon la référence."""
    run = st.session_state.get("run")
    if run:
        m = run["meta"]
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        n_txt = f"{m['n_rows']:,}".replace(",", " ")
        c1.info(f"Vous utilisez **vos modèles** ({n_txt} transactions, {len(run['keys'])} modèle(s), "
                f"stratégie « {m['strategy']} »).")
        if c2.button("Voir la référence"):
            del st.session_state["run"]
            st.rerun()
        return run["scores"], run["meta"], run["res"]
    require_artifacts()
    from src import store
    st.caption("Résultats de référence (pipeline complet). Pour utiliser vos propres choix de préparation et de modèles, "
               "passez par les pages Préparation et Modélisation.")
    return store.scores(), store.meta(), store.results()
