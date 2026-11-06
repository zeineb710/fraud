import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from src import config as C
from src import metrics as M
from src import models as MD
from src import ui

ui.setup("Comparaison des modèles", "⚖️")
ui.require_artifacts()
from src import store  # noqa: E402

st.title("4 · Comparaison des modèles")
sc, meta, res = ui.active_run()
keys = [k for k in MD.MODELS if f"score_{k}" in sc.columns]
st.caption("Choisissez les modèles, le jeu d'évaluation et le budget d'alertes : tout se met à jour en direct.")

with st.sidebar:
    st.header("Paramètres")
    split = st.radio("Jeu d'évaluation", ["validation", "test"], horizontal=True)
    chosen = st.multiselect("Modèles à comparer", keys, default=[k for k in ["xgboost", "logreg", "iforest", "autoencoder"] if k in keys] or keys[:1],
                            format_func=MD.label)
    d = sc[sc["split"] == split]
    n = len(d)
    k_default = int(C.BUDGET_FRAC * n)
    k = st.slider("Budget d'alertes k", 50, max(int(0.10 * n), 100), k_default, step=1,
                  help=f"Par défaut {C.BUDGET_FRAC:.0%} des transactions du jeu.")
    ca = st.slider("Coût d'analyse par alerte ($)", 0.5, 50.0, float(meta["cost_per_alert"]), 0.5)
    ens = st.checkbox("Ajouter un ensemble (moyenne des rangs des modèles choisis)", value=False,
                      disabled=len(chosen) < 2)

if not chosen:
    st.info("Sélectionnez au moins un modèle dans la barre latérale.")
    st.stop()

y, amt = d["y"].to_numpy(), d["amount"].to_numpy()
S = {MD.label(m): d[f"score_{m}"].to_numpy() for m in chosen}
kind = {MD.label(m): MD.MODELS[m][1] for m in chosen}
if ens and len(chosen) >= 2:
    S["Ensemble (rangs)"] = np.mean([pd.Series(v).rank(pct=True).to_numpy() for v in S.values()], axis=0)
    kind["Ensemble (rangs)"] = "hybride"

rows = []
for name, s in S.items():
    mk = M.metrics_at_k(y, s, amt, k, ca)
    kk, cost, tot = M.cost_curve(y, s, amt, ca)
    rows.append({"Modèle": name, "Famille": kind[name], "PR-AUC": M.pr_auc(y, s), "ROC-AUC": M.roc_auc(y, s),
                 f"Précision@{k}": mk["precision"], f"Rappel@{k}": mk["recall"], "F2": mk["f2"], "MCC": mk["mcc"],
                 f"Coût@{k} ($)": mk["cost"], "Coût optimal ($)": float(cost.min()),
                 "Économie max (%)": 100 * (1 - float(cost.min()) / max(tot, 1))})
tab = pd.DataFrame(rows)

c = st.columns(3)
c[0].metric("Fraudes dans ce jeu", f"{int(y.sum()):,}".replace(",", " "), f"{100 * y.mean():.2f} %")
c[1].metric("Rappel maximal théorique", f"{100 * M.max_recall(y, k):.0f} %", help="Un modèle parfait ne peut pas dépasser k / nb de fraudes.")
c[2].metric("Coût sans modèle", ui.money(float((amt * y).sum())))

pos = [col for col in tab.columns if col in ("PR-AUC", "ROC-AUC", f"Précision@{k}", f"Rappel@{k}", "F2", "MCC", "Économie max (%)")]
neg = [f"Coût@{k} ($)", "Coût optimal ($)"]
fmt = {col: "{:.3f}" for col in pos if col != "Économie max (%)"}
fmt.update({"Économie max (%)": "{:.1f}", f"Coût@{k} ($)": "{:,.0f}", "Coût optimal ($)": "{:,.0f}"})
st.dataframe(tab.style.format(fmt).highlight_max(subset=pos, color="#d4edda").highlight_min(subset=neg, color="#d4edda"),
             width="stretch", hide_index=True)

t1, t2, t3, t4 = st.tabs(["Précision-rappel", "ROC", "Rappel selon le budget", "Comparer une métrique"])
with t1:
    fig = go.Figure()
    for name, s in S.items():
        p, r, _ = precision_recall_curve(y, s)
        idx = np.unique(np.linspace(0, len(p) - 1, 600).astype(int))
        fig.add_trace(go.Scatter(x=r[idx], y=p[idx], mode="lines", name=name))
    fig.add_hline(y=y.mean(), line_dash="dot", annotation_text="hasard (taux de fraude)")
    fig.update_layout(xaxis_title="Rappel", yaxis_title="Précision", yaxis_range=[0, 1.02], height=480)
    st.plotly_chart(fig, width="stretch")
with t2:
    fig = go.Figure()
    for name, s in S.items():
        f, t, _ = roc_curve(y, s)
        idx = np.unique(np.linspace(0, len(f) - 1, 600).astype(int))
        fig.add_trace(go.Scatter(x=f[idx], y=t[idx], mode="lines", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="grey"), name="hasard"))
    fig.update_layout(xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs", height=480)
    st.plotly_chart(fig, width="stretch")
with t3:
    ks = np.unique(np.linspace(10, max(int(0.10 * n), 100), 250).astype(int))
    fig = go.Figure()
    for name, s in S.items():
        fig.add_trace(go.Scatter(x=ks, y=M.recall_at_budgets(y, s, ks), mode="lines", name=name))
    fig.add_trace(go.Scatter(x=ks, y=np.minimum(1, ks / max(int(y.sum()), 1)), mode="lines",
                             line=dict(dash="dot", color="grey"), name="modèle parfait"))
    fig.add_vline(x=k, line_dash="dash", annotation_text=f"k = {k}")
    fig.update_layout(xaxis_title="Budget d'alertes k", yaxis_title="Rappel@k", yaxis_range=[0, 1.02], height=480)
    st.plotly_chart(fig, width="stretch")
    st.caption("Avec un petit budget, le rappel est plafonné par k / nombre de fraudes (courbe pointillée).")
with t4:
    metric = st.selectbox("Métrique", [c_ for c_ in tab.columns if c_ not in ("Modèle", "Famille")])
    fig = go.Figure(go.Bar(x=tab["Modèle"], y=tab[metric], marker_color=["#8E2431" if f == "supervisé" else "#2F6690" if f == "non supervisé" else "#6A994E" for f in tab["Famille"]]))
    fig.update_layout(yaxis_title=metric, height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption("Rouge : supervisé · Bleu : non supervisé · Vert : ensemble.")

st.info("Lecture : les méthodes non supervisées repèrent ce qui est *inhabituel*, pas forcément ce qui est *frauduleux*. "
        "Elles sont utiles pour les schémas inédits, mais produisent plus de fausses alertes que le modèle supervisé.")
