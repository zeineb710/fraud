import streamlit as st

from src import config as C
from src import data as D
from src import prep as P
from src import ui

ui.setup("Préparation", "🛠️")
st.title("2 · Préparation des données")

n_max = D.n_available()
if n_max == 0:
    st.error("Aucune donnée disponible : ni CSV dans `data/raw/`, ni échantillon dans `data/sample/`.")
    st.code("python -m src.data --rows 40000   # à lancer là où se trouvent les CSV Kaggle, puis commit de data/sample/", language="bash")
    st.stop()

st.caption(f"Cible : **isFraud** (classe positive : 1, très rare) · source : {D.source_label()}")
st.info("Identifiants exclus par défaut : TransactionID. Le découpage est **chronologique** (entraînement → validation → test) : "
        "on apprend sur le passé et on évalue sur le futur. Toutes les statistiques (médianes, codes, échelles) sont apprises "
        "sur l'entraînement seul.")

groups = st.multiselect("Groupes de variables explicatives", list(P.GROUPS), default=list(P.GROUPS))
n_v = st.slider("Variables Vesta V conservées (les moins manquantes sur l'entraînement)", 0, 100, 40, step=5)

c1, c2, c3 = st.columns(3)
with c1:
    num_imp = st.selectbox("Numériques manquantes", P.NUM_IMPUTE,
                           help="« Laisser manquant » convient à XGBoost ; les autres modèles appliquent alors une médiane de sécurité.")
    cat_imp = st.selectbox("Catégorielles manquantes", P.CAT_IMPUTE)
with c2:
    enc = st.selectbox("Encodage des catégories", P.ENCODINGS,
                       help="Ordinal : un code entier par catégorie (utilisé par le pipeline de référence). "
                            "One-hot : une colonne par catégorie. Fréquence : part de chaque catégorie dans l'entraînement.")
    rare = st.slider("Regrouper les catégories rares (one-hot)", 0.001, 0.10, 0.01, format="%.3f",
                     disabled=enc != P.ENCODINGS[1])
    scal = st.selectbox("Normalisation", P.SCALINGS, help="Inutile pour les arbres, utile pour la régression logistique et LOF.")
with c3:
    n_rows = st.slider("Transactions utilisées", min(5000, n_max), n_max, min(30000, n_max), step=1000 if n_max > 6000 else 100,
                       help="Tranche chronologique : plus il y a de lignes, plus l'entraînement est long.")
    val_f = st.slider("Part de validation", 0.10, 0.30, 0.20, 0.05, help="Sert à choisir le seuil d'alerte.")
    test_f = st.slider("Part de test", 0.10, 0.30, 0.20, 0.05, help="Sert uniquement à confirmer.")
    seed = st.number_input("Graine aléatoire", 0, 10_000, C.SEED)

if st.button("Appliquer la préparation", type="primary"):
    if not groups:
        st.error("Choisissez au moins un groupe de variables.")
        st.stop()
    with st.spinner("Préparation en cours…"):
        prepared = P.prepare(D.load(n_rows), P.PrepConfig(groups=groups, n_v=n_v, n_rows=n_rows, val_frac=val_f,
                                                          test_frac=test_f, num_impute=num_imp, cat_impute=cat_imp,
                                                          encoding=enc, rare=rare, scaling=scal, seed=int(seed)))
    st.session_state["prep"] = prepared
    st.session_state.pop("run", None)      # un nouvel entraînement est nécessaire après une nouvelle préparation

p = st.session_state.get("prep")
if p is None:
    st.stop()

r = p.report
st.success("Préparation appliquée. Passez à la page Modélisation.")
m = st.columns(4)
m[0].metric("Variables d'entrée", r["n_raw_features"])
m[1].metric("Colonnes après encodage", r["n_features"])
m[2].metric("Manquants avant", f"{r['missing_before']:.1f} %")
m[3].metric("Manquants après", f"{r['missing_after']:.1f} %")
st.dataframe(r["splits"], width="stretch", hide_index=True)
if (r["splits"]["Fraudes"] < 50).any():
    st.warning("Moins de 50 fraudes dans un des jeux : les métriques seront très instables. Augmentez le nombre de transactions.")
with st.expander("Aperçu des données préparées (entraînement)"):
    st.dataframe(p.X["train"].head(10), width="stretch")
st.page_link("pages/3_Modelisation.py", label="Aller à la modélisation →", icon="🤖")