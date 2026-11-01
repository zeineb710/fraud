import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src import metrics as M
from src import models as MD
from src import ui

ui.setup("Évaluation et seuil", "🎚️")
ui.require_artifacts()
from src import store  # noqa: E402

meta, res = store.meta(), store.results()
sc = store.scores()

st.title("4 · Évaluation et seuil d'alerte")
st.markdown("Coût = **montants des fraudes manquées** + **coût d'analyse × nombre d'alertes**. Le seuil optimal est choisi "
            "sur la **validation**, puis confirmé sur le **test**.")

with st.sidebar:
    st.header("Paramètres")
    key = st.selectbox("Modèle", list(MD.MODELS), index=list(MD.MODELS).index(res["best_model"]), format_func=MD.label)
    split = st.radio("Jeu affiché", ["validation", "test"], horizontal=True)
    ca = st.slider("Coût d'analyse par alerte ($)", 0.5, 50.0, float(meta["cost_per_alert"]), 0.5)
    mode = st.radio("Piloter par", ["Budget d'alertes k", "Seuil τ (quantile de la validation)", "Seuil optimal en coût"])

dv = sc[sc["split"] == "validation"]
sv, yv, av = dv[f"score_{key}"].to_numpy(), dv["y"].to_numpy(), dv["amount"].to_numpy()
opt = M.best_budget(yv, sv, av, ca)

d = sc[sc["split"] == split]
s, y, amt = d[f"score_{key}"].to_numpy(), d["y"].to_numpy(), d["amount"].to_numpy()
n = len(d)

with st.sidebar:
    if mode == "Budget d'alertes k":
        k = st.slider("Budget k", 0, max(int(0.15 * n), 10), int(C.BUDGET_FRAC * n))
        flag, tau = M.topk_flag(s, k), float(np.sort(s)[::-1][max(k - 1, 0)])
    elif mode.startswith("Seuil τ"):
        q = st.slider("Quantile du seuil (%)", 80.0, 99.99, 98.0, 0.01)
        tau = float(np.quantile(sv, q / 100))
        flag = s >= tau
    else:
        tau = opt["tau"]
        flag = s >= tau
        st.caption(f"Seuil optimal (validation) : τ = {tau:.4g}")

m = M.evaluate_flag(y, amt, flag, ca)

c = st.columns(5)
c[0].metric("Alertes", f"{m['n_alerts']:,}".replace(",", " "), f"{100 * m['n_alerts'] / n:.2f} % du flux")
c[1].metric("Précision", f"{m['precision']:.3f}")
c[2].metric("Rappel", f"{m['recall']:.3f}")
c[3].metric("F2", f"{m['f2']:.3f}")
c[4].metric("MCC", f"{m['mcc']:.3f}")
c = st.columns(5)
c[0].metric("Fraudes manquées", f"{m['fn']:,}".replace(",", " "), ui.money(m["missed_amount"]), delta_color="off")
c[1].metric("Fraudes détectées", f"{m['tp']:,}".replace(",", " "), ui.money(m["caught_amount"]), delta_color="off")
c[2].metric("Coût d'analyse", ui.money(m["analysis_cost"]))
c[3].metric("Coût total", ui.money(m["cost"]))
c[4].metric("Économie vs sans modèle", ui.money(m["saving"]), f"{100 * m['saving'] / max(m['no_model_cost'], 1):.0f} %")

k_arr, cost, tot = M.cost_curve(y, s, amt, ca)
step = max(1, len(k_arr) // 1500)
fig = go.Figure(go.Scatter(x=k_arr[::step], y=cost[::step], mode="lines", name="coût total"))
fig.add_hline(y=tot, line_dash="dot", annotation_text="sans modèle (aucune alerte)")
fig.add_vline(x=m["n_alerts"], line_color="#C0392B", annotation_text="choix actuel")
ib = int(np.argmin(cost))
fig.add_trace(go.Scatter(x=[ib], y=[cost[ib]], mode="markers", marker=dict(size=11, color="green"),
                         name=f"optimum sur ce jeu (k={ib})"))
fig.update_layout(xaxis_title="Nombre d'alertes k", yaxis_title="Coût ($)", height=420,
                  xaxis_range=[0, min(len(k_arr), max(int(0.15 * n), ib * 3))])
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.markdown("**Matrice de confusion**")
    st.dataframe(pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], index=["Réel : honnête", "Réel : fraude"],
                              columns=["Prédit : normal", "Prédit : alerte"]), width="stretch")
with right:
    st.markdown("**Seuil choisi en validation, confirmé sur le test**")
    rows = []
    for name, dd in (("validation", dv), ("test", sc[sc["split"] == "test"])):
        mm = M.metrics_at_threshold(dd["y"].to_numpy(), dd[f"score_{key}"].to_numpy(), dd["amount"].to_numpy(), opt["tau"], ca)
        rows.append({"Jeu": name, "Alertes": mm["n_alerts"], "Précision": mm["precision"], "Rappel": mm["recall"],
                     "F2": mm["f2"], "MCC": mm["mcc"], "Coût ($)": mm["cost"], "Sans modèle ($)": mm["no_model_cost"],
                     "Économie ($)": mm["saving"]})
    st.dataframe(pd.DataFrame(rows).style.format({"Précision": "{:.3f}", "Rappel": "{:.3f}", "F2": "{:.3f}", "MCC": "{:.3f}",
                                                  "Coût ($)": "{:,.0f}", "Sans modèle ($)": "{:,.0f}", "Économie ($)": "{:,.0f}"}),
                 width="stretch", hide_index=True)
st.caption("Le seuil ne doit jamais être choisi sur le test : il est fixé en validation (τ optimal ci-dessus) "
           "puis simplement appliqué au test pour vérifier qu'il généralise.")
