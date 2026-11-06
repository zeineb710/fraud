import streamlit as st

from src import config as C
from src import models as MD
from src import trainer as T
from src import ui

ui.setup("Modélisation", "🤖")
st.title("3 · Modélisation")

p = st.session_state.get("prep")
if p is None:
    st.info("Appliquez d'abord la préparation des données.")
    st.page_link("pages/2_Preparation.py", label="← Aller à la préparation", icon="🛠️")
    st.stop()

keys = st.multiselect("Modèles à entraîner", list(MD.MODELS), default=["xgboost", "random_forest", "logreg", "iforest"],
                      format_func=MD.label)
c1, c2 = st.columns(2)
with c1:
    strat = st.selectbox("Compenser le déséquilibre des classes (modèles supervisés)", [s["name"] for s in MD.STRATEGIES],
                         help="Appliqué à l'entraînement seul. Les probabilités sont ensuite corrigées pour rester comparables.")
with c2:
    ca = st.slider("Coût d'analyse par alerte ($)", 0.5, 50.0, float(C.default_cost_per_alert()), 0.5,
                   help="Valeur par défaut utilisée à la page Évaluation et seuil.")

d, params = MD.DEFAULT_PARAMS, {}
for k in keys:
    with st.expander(f"Réglages : {MD.label(k)}"):
        if k == "xgboost":
            params[k] = {"n_estimators": st.slider("Nombre d'arbres", 50, 600, d[k]["n_estimators"], 50, key=f"{k}1"),
                         "max_depth": st.slider("Profondeur maximale", 2, 10, d[k]["max_depth"], key=f"{k}2"),
                         "learning_rate": st.select_slider("Taux d'apprentissage", [0.02, 0.05, 0.08, 0.1, 0.2, 0.3],
                                                           d[k]["learning_rate"], key=f"{k}3")}
        elif k == "random_forest":
            params[k] = {"n_estimators": st.slider("Nombre d'arbres", 20, 300, d[k]["n_estimators"], 10, key=f"{k}1"),
                         "max_depth": st.slider("Profondeur maximale", 3, 20, d[k]["max_depth"], key=f"{k}2"),
                         "min_samples_leaf": st.slider("Taille minimale d'une feuille", 5, 300, d[k]["min_samples_leaf"], 5, key=f"{k}3")}
        elif k == "logreg":
            params[k] = {"C": st.select_slider("Régularisation C (petit = plus régularisé)", [0.01, 0.1, 0.5, 1.0, 5.0], d[k]["C"], key=f"{k}1")}
        elif k == "iforest":
            params[k] = {"n_estimators": st.slider("Nombre d'arbres", 50, 400, d[k]["n_estimators"], 50, key=f"{k}1"),
                         "max_samples": st.select_slider("Taille de sous-échantillon", [128, 256, 512, 1024], d[k]["max_samples"], key=f"{k}2")}
        elif k == "lof":
            params[k] = {"n_neighbors": st.slider("Nombre de voisins", 5, 100, d[k]["n_neighbors"], key=f"{k}1")}
        elif k == "ocsvm":
            params[k] = {"nu": st.select_slider("nu (part d'anomalies tolérée)", [0.005, 0.01, 0.02, 0.05, 0.1], d[k]["nu"], key=f"{k}1")}
        else:
            params[k] = {"bottleneck": st.select_slider("Taille du goulot", [4, 8, 16, 32], d[k]["bottleneck"], key=f"{k}1"),
                         "max_iter": st.slider("Itérations maximales", 10, 100, d[k]["max_iter"], 10, key=f"{k}2")}

if st.button("Entraîner les modèles", type="primary", disabled=not keys):
    with st.status("Entraînement en cours…", expanded=True) as status:
        try:
            st.session_state["run"] = T.train(p, keys, strat, params, ca, log=st.write)
            status.update(label="Entraînement terminé", state="complete")
        except ValueError as e:
            status.update(label="Échec", state="error")
            st.error(str(e))

run = st.session_state.get("run")
if run is None:
    st.stop()

ca_run = run["meta"]["cost_per_alert"]
st.subheader("Comparaison des modèles (jeu de test, seuil 0,5)")
tab = T.summary_at_threshold(run, ca_run)
fmt = {c: "{:.3f}" for c in ("PR-AUC", "ROC-AUC", "Précision", "Rappel", "F2", "MCC")}
fmt.update({"Alertes": "{:,.0f}", "Économie ($)": "{:,.0f}"})
st.dataframe(tab.style.format(fmt, na_rep="–").highlight_max(subset=["PR-AUC"], color="#d4edda"),
             width="stretch", hide_index=True)
sup = tab.dropna(subset=["Alertes"])
if not sup.empty:
    b = sup.sort_values("PR-AUC", ascending=False).iloc[0]
    st.info(f"Au seuil 0,5, {b['Modèle']} lève {int(b['Alertes'])} alertes (précision {b['Précision']:.2f}, rappel {b['Rappel']:.2f}). "
            "Ce seuil par défaut n'a pas été choisi selon le coût des erreurs : la page Évaluation et seuil montre comment le fixer.")
st.caption("PR-AUC et ROC-AUC ne dépendent d'aucun seuil. Les modèles non supervisés donnent un score d'anomalie, pas une probabilité : "
           "le seuil 0,5 n'a pas de sens pour eux (ils sont comparés au budget d'alertes k).")
st.page_link("pages/5_Evaluation_Seuil.py", label="Choisir le seuil d'alerte →", icon="🎚️")