import pandas as pd
import plotly.express as px
import streamlit as st

from src import features as F
from src import ui

ui.setup("Préparation", "🛠️")
ui.require_artifacts()
from src import store  # noqa: E402

meta, res = store.meta(), store.results()
st.title("2 · Préparation et variables")

st.subheader("Découpage chronologique 60 / 20 / 20")
sp = store.eda("split")
st.dataframe(sp.rename(columns={"jeu": "Jeu", "lignes": "Lignes", "fraudes": "Fraudes", "taux_fraude": "Taux de fraude (%)",
                                "jour_debut": "Jour début", "jour_fin": "Jour fin"}).round(2),
             width="stretch", hide_index=True)
st.caption("On apprend sur le passé et on évalue sur le futur. Toute statistique (sélection des variables V, encodage des "
           "catégories, imputation, normalisation) est apprise sur le **train uniquement**.")

st.subheader("Variables créées")
st.markdown("""
| Variable | Idée | Piège évité |
|---|---|---|
| `hour`, `h_sin`, `h_cos`, `dow` | heure en codage cyclique (23 h proche de 1 h), jour de la semaine | – |
| `tx_count_1h`, `tx_count_24h` | **vélocité** : nombre d'achats de la même identité sur la dernière heure / 24 h | seulement les achats **antérieurs** |
| `amt_vs_mean_30d` | montant ÷ moyenne des achats des 30 derniers jours (écart au profil) | historique passé uniquement |
| `time_since_prev` | secondes depuis l'achat précédent | idem |
| `amt_log`, `amt_cents`, `email_match` | montant (log), partie décimale, emails identiques | – |
""")
st.warning("**Identité approximative** : IEEE-CIS n'a pas de colonne client. L'identité est reconstituée par "
           "`card1 | card2 | card3 | card5 | addr1 | (jour − D1)`. C'est une heuristique, pas une vérité : une même personne "
           "peut être éclatée en plusieurs identités, ou des personnes distinctes regroupées.")

b = store.eda("behav")
var = st.selectbox("Taux de fraude selon une variable comportementale", b["variable"].unique(),
                   format_func=lambda v: f"{v} · {F.label(v)}")
d = b[b["variable"] == var]
fig = px.bar(d, x="groupe", y="fraud_rate", hover_data=["n"], labels={"groupe": var, "fraud_rate": "Taux de fraude (%)"})
fig.add_hline(y=meta["fraud_rate"], line_dash="dash", annotation_text="taux global")
st.plotly_chart(fig, width="stretch")

st.subheader("Stratégie contre le déséquilibre (appliquée au train seul)")
imb = pd.DataFrame(res["imbalance"])
show = imb.rename(columns={"strategie": "stratégie", "lignes_train": "lignes train", "pr_auc": "PR-AUC",
                           "pr_auc_std": "± écart-type", "roc_auc": "ROC-AUC",
                           "recall_at_k": f"rappel@{res['k_val']}", "precision_at_k": f"précision@{res['k_val']}"})
st.dataframe(show.style.format({"PR-AUC": "{:.4f}", "± écart-type": "{:.4f}", "ROC-AUC": "{:.4f}",
                                f"rappel@{res['k_val']}": "{:.4f}", f"précision@{res['k_val']}": "{:.4f}"})
             .highlight_max(subset=["PR-AUC"], color="#d4edda"), width="stretch", hide_index=True)
st.success(f"Stratégie retenue : **{res['selected_strategy']}**. Règle : on ne rééchantillonne que si le gain de PR-AUC en "
           "validation dépasse le bruit entre tirages (le cours recommande de commencer sans rééchantillonnage).")
st.caption("Comparaison faite avec XGBoost sur la validation, qui garde la proportion réelle de fraudes. "
           "Après rééchantillonnage ou pondération, les probabilités sont corrigées avec "
           "p = β·ps / (β·ps − ps + 1).")
with st.expander("Détail par graine"):
    st.dataframe(pd.DataFrame(res["imbalance_runs"]), width="stretch", hide_index=True)
