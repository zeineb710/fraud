"""Synthèse : résultats clés et lecture rapide."""
import pandas as pd
import streamlit as st

from src import ui

ui.setup("Synthèse", "")
ui.require_artifacts()

from src import models as MD  # noqa: E402
from src import store  # noqa: E402

meta, res = store.meta(), store.results()
sup = [k for k in res["models"] if MD.is_supervised(k)]
uns = [k for k in res["models"] if not MD.is_supervised(k)]
best_sup = max(sup, key=lambda k: res["models"][k]["validation"]["pr_auc"])
best_uns = max(uns, key=lambda k: res["models"][k]["validation"]["pr_auc"])
bs, bu = res["models"][best_sup], res["models"][best_uns]
o = bs["optimal"]["test"]

st.title(" Synthèse des résultats")
st.caption(f"Détection de fraude explicable · IEEE-CIS · {meta['n_rows']:,} transactions · "
           f"{meta['fraud_rate']:.2f} % de fraudes · coût d'analyse supposé {res['cost_per_alert']:.0f} $ par alerte".replace(",", " "))

c = st.columns(4)
c[0].metric("Meilleur supervisé", MD.label(best_sup), f"PR-AUC test {bs['test']['pr_auc']:.3f}")
c[1].metric("Meilleur non supervisé", MD.label(best_uns), f"PR-AUC test {bu['test']['pr_auc']:.3f}")
c[2].metric("Économie vs sans modèle (test)", ui.money(o["saving"]),
            f"{100 * o['saving'] / max(o['no_model_cost'], 1):.0f} %")
c[3].metric(f"Précision / rappel au seuil optimal", f"{o['precision']:.2f} / {o['recall']:.2f}",
            f"{o['n_alerts']:,} alertes".replace(",", " "), delta_color="off")

st.subheader("Tous les modèles sur le jeu de test")
rows = []
for k, v in res["models"].items():
    t, ot = v["test"], v["optimal"]["test"]
    rows.append({"Modèle": v["label"], "Famille": v["kind"], "PR-AUC": t["pr_auc"], "ROC-AUC": t["roc_auc"],
                 f"Précision@{res['k_test']}": t["precision_at_k"], f"Rappel@{res['k_test']}": t["recall_at_k"],
                 "Économie au seuil optimal ($)": ot["saving"]})
tab = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
st.dataframe(tab.style.format({"PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}", f"Précision@{res['k_test']}": "{:.3f}",
                               f"Rappel@{res['k_test']}": "{:.3f}", "Économie au seuil optimal ($)": "{:,.0f}"})
             .highlight_max(subset=["PR-AUC", "Économie au seuil optimal ($)"], color="#d4edda"),
             width="stretch", hide_index=True)

st.subheader("À retenir")
gap = bs["test"]["pr_auc"] - bu["test"]["pr_auc"]
st.markdown(f"""
- **Stratégie contre le déséquilibre** : « {res['selected_strategy']} » (comparée en validation, appliquée au train seul).
- **Supervisé vs non supervisé** : {MD.label(best_sup)} devance {MD.label(best_uns)} de {gap:.2f} point(s) de PR-AUC. 
  Un comportement inhabituel n'est pas forcément frauduleux ; le non supervisé reste utile pour les schémas inédits.
- **Seuil** : choisi sur la validation (coût minimal), puis appliqué au test : {o['n_alerts']:,} alertes, 
  {o['tp']:,} fraudes détectées pour {o['fn']:,} manquées.
- **Coût** : {ui.money(o['cost'])} avec le modèle, contre {ui.money(o['no_model_cost'])} sans modèle.
""".replace(",", " "))

st.caption(f"Artefacts construits le {meta['built_at']} · source : {meta['source']}. "
           "Les pages de l'application sont dans le menu de gauche.")