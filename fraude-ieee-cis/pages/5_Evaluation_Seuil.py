import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src import metrics as M
from src import models as MD
from src import ui

ui.setup("Évaluation et seuil", "🎚️")



st.title("5 · Évaluation et seuil d'alerte")
sc, meta, res = ui.active_run()
keys = [k for k in MD.MODELS if f"score_{k}" in sc.columns]
st.markdown("Coût = **montants des fraudes manquées** + **coût d'analyse × nombre d'alertes**. Le seuil optimal est choisi "
            "sur la **validation**, puis confirmé sur le **test**.")
with st.expander("Comprendre le seuil d'alerte", expanded=False):
    st.markdown("""
**1. Le modèle ne décide pas, il note.** Chaque transaction reçoit un score. Pour un modèle supervisé, c'est une probabilité
estimée de fraude ; pour un modèle non supervisé, un score d'anomalie (plus élevé = plus inhabituel).

**2. Le seuil τ transforme la note en décision.** Si score ≥ τ, la transaction devient une **alerte** (un analyste la vérifie) ;
sinon elle passe. Deux erreurs coûtent de l'argent : une **fraude manquée** (on perd son montant) et une **fausse alerte**
(on paie l'analyse).

**3. Baisser τ ou le monter.** Un τ bas lève beaucoup d'alertes : plus de fraudes attrapées (rappel élevé), mais plus d'analyses
et plus de fausses alertes (précision basse). Un τ haut fait l'inverse. Le meilleur compromis dépend du coût d'analyse (curseur à gauche).

**4. Trois façons de fixer le même réglage** (elles sont équivalentes : k alertes ⇔ un seuil τ ⇔ un quantile des scores) :
- **Budget d'alertes k** : « mon équipe peut traiter k alertes ». C'est la contrainte opérationnelle.
- **Seuil τ (quantile de la validation)** : on fixe τ sur la validation puis on le réutilise tel quel sur de nouvelles données.
- **Seuil optimal en coût** : l'application cherche le τ qui minimise le coût total sur la validation.

**5. Pourquoi pas 0,5 ?** La fraude est rare ; les probabilités prédites sont en général petites, et 0,5 ne tient pas compte du coût
des erreurs. Le tableau « Trois règles de seuil » ci-dessous compare 0,5 aux autres règles.

**6. Validation puis test.** On choisit τ sur la validation, puis on l'applique **sans le modifier** au test. Choisir τ sur le test
donnerait un résultat trop optimiste.
""")

with st.sidebar:
    st.header("Paramètres")
    key = st.selectbox("Modèle", keys, index=keys.index(res["best_model"]), format_func=MD.label)    
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
c[0].metric("Fraudes manquées", f"{m['fn']:,}".replace(",", " "))
c[0].caption(f"Montant perdu : {ui.money(m['missed_amount'])}")
c[1].metric("Fraudes détectées", f"{m['tp']:,}".replace(",", " "))
c[1].caption(f"Montant récupéré : {ui.money(m['caught_amount'])}")
c[2].metric("Coût d'analyse", ui.money(m["analysis_cost"]))
c[3].metric("Coût total", ui.money(m["cost"]))
c[4].metric("Économie vs sans modèle", ui.money(m["saving"]), f"{100 * m['saving'] / max(m['no_model_cost'], 1):.0f} %")

fr = lambda v: f"{v:,}".replace(",", " ")
st.info(f"**En clair** : avec τ = {tau:.4g}, {fr(m['n_alerts'])} transactions ({100 * m['n_alerts'] / n:.2f} % du flux) partent à l'analyse. "
        f"{fr(m['tp'])} sont de vraies fraudes (précision {m['precision']:.0%}) ; {fr(m['fn'])} fraudes passent inaperçues "
        f"(rappel {m['recall']:.0%}), soit {ui.money(m['missed_amount'])}. Coût total : {ui.money(m['cost'])} "
        f"contre {ui.money(m['no_model_cost'])} sans modèle.")
st.caption(f"Équivalence : k = {fr(m['n_alerts'])} alertes ⇔ τ = {tau:.4g} ⇔ quantile {100 * (1 - m['n_alerts'] / n):.2f} % des scores de ce jeu.")

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

st.markdown("**Où se place le seuil ?**")
pct = pd.Series(s).rank(pct=True).to_numpy() * 100
tau_pct = 100 * float((s < tau).mean())
fig2 = go.Figure()
for cls, name, col in ((0, "Honnête", "#2F6690"), (1, "Fraude", "#C0392B")):
    fig2.add_trace(go.Histogram(x=pct[y == cls], histnorm="percent", nbinsx=50, name=name, opacity=0.65, marker_color=col))
fig2.add_vline(x=tau_pct, line_color="black", line_dash="dash", annotation_text="τ : à droite = alerte")
fig2.update_layout(barmode="overlay", height=320, xaxis_title="Rang du score dans ce jeu (0 = le plus faible, 100 = le plus suspect)",
                   yaxis_title="Part de la classe (%)")
st.plotly_chart(fig2, width="stretch")
st.caption("Si le modèle est bon, les fraudes (rouge) se concentrent à droite. Déplacer τ vers la gauche ajoute des alertes : "
           "on attrape plus de fraudes mais on analyse plus d'honnêtes transactions.")

st.markdown("**Trois règles de seuil, sur le jeu affiché**")
k_bud = int(C.BUDGET_FRAC * n)
rules = [("Budget de 2 % d'alertes", M.metrics_at_k(y, s, amt, k_bud, ca), float(np.sort(s)[::-1][max(k_bud - 1, 0)])),
         ("Seuil optimal en coût (choisi en validation)", M.metrics_at_threshold(y, s, amt, opt["tau"], ca), opt["tau"])]
if MD.is_supervised(key):
    rules.insert(0, ("Seuil par défaut 0,5", M.metrics_at_threshold(y, s, amt, 0.5, ca), 0.5))
st.dataframe(pd.DataFrame([{"Règle": r_, "τ": t_, "Alertes": mm["n_alerts"], "Précision": mm["precision"], "Rappel": mm["recall"],
                            "Coût ($)": mm["cost"], "Économie ($)": mm["saving"]} for r_, mm, t_ in rules])
             .style.format({"τ": "{:.4g}", "Précision": "{:.3f}", "Rappel": "{:.3f}", "Coût ($)": "{:,.0f}", "Économie ($)": "{:,.0f}"}),
             width="stretch", hide_index=True)

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
