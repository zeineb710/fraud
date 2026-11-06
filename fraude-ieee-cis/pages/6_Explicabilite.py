import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import shap  # noqa: E402
import streamlit as st  # noqa: E402

from src import api_client  # noqa: E402
from src import features as F  # noqa: E402
from src import ui  # noqa: E402

ui.setup("Explicabilité", "🔍")
ui.require_artifacts()
from src import store  # noqa: E402

meta = store.meta()
feats = meta["features"]
smp = store.sample()
scorer = store.load_model("xgboost")

st.title("5 · Explicabilité (SHAP)")
st.caption("Modèle expliqué : XGBoost. Valeurs de Shapley exactes (TreeSHAP) exprimées en log-odds : "
           "positif = pousse vers la fraude.")


@st.cache_data(show_spinner="Calcul des valeurs SHAP…")
def global_contribs(n: int = 2000):
    idx = np.random.default_rng(0).choice(len(smp), min(n, len(smp)), replace=False)
    X = smp.iloc[idx][feats]
    return X, scorer.contribs(X)


tab1, tab2 = st.tabs(["Explication globale", "Expliquer une alerte"])

with tab1:
    X, cb = global_contribs()
    exp = shap.Explanation(values=cb[:, :-1], base_values=cb[:, -1], data=X.to_numpy(), feature_names=feats)
    st.markdown("**Beeswarm** : chaque point est une transaction ; couleur = valeur de la variable, position = effet sur le score.")
    plt.figure()
    shap.plots.beeswarm(exp, max_display=15, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    st.pyplot(fig, clear_figure=True)
    imp = pd.DataFrame(store.shap_importance()).head(15)
    st.markdown("**Importance moyenne |SHAP|** (calculée sur 5 000 transactions du test)")
    st.plotly_chart(px.bar(imp, x="mean_abs_shap", y="label", orientation="h",
                           labels={"mean_abs_shap": "Moyenne de |SHAP|", "label": ""}).update_yaxes(autorange="reversed"),
                    width="stretch")
    st.caption("Échantillon : 2 000 transactions tirées au hasard parmi celles du test mises en réserve pour la démonstration "
               "(les alertes les plus fortes y sont sur-représentées).")

with tab2:
    top = smp.sort_values("score", ascending=False).head(300).reset_index(drop=True)
    info = {int(r.TransactionID): r for r in top.itertuples()}
    tid = st.selectbox("Alerte à expliquer (classées par score décroissant)", list(info),
                       format_func=lambda i: f"{i} · score {info[i].score:.3f} · {info[i].amount:,.0f} $ · "
                                             f"{'fraude réelle' if info[i].y == 1 else 'honnête'}")
    n_reasons = st.slider("Nombre de codes motifs", 3, 5, 5)
    row = smp[smp["TransactionID"] == tid].iloc[0]
    x = row[feats].astype("float32").to_frame().T
    c = scorer.contribs(x)[0]

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Waterfall SHAP**")
        e1 = shap.Explanation(values=c[:-1], base_values=c[-1], data=x.to_numpy()[0], feature_names=feats)
        plt.figure()
        shap.plots.waterfall(e1, max_display=12, show=False)
        fig = plt.gcf()
        fig.set_size_inches(7, 5)
        st.pyplot(fig, clear_figure=True)
    with right:
        st.markdown("**Codes motifs (via l'API FastAPI)**")
        try:
            r = api_client.call("GET", f"/transactions/{tid}", params={"top_reasons": n_reasons})
            r.raise_for_status()
            out = r.json()
            st.metric("Score de fraude", f"{out['score']:.3f}", "ALERTE" if out["alert"] else "pas d'alerte",
                      delta_color="inverse" if out["alert"] else "normal")
            st.caption(f"Seuil d'alerte : {out['threshold']:.3f} · niveau de risque : {out['risk_level']}")
            if out["reasons"]:
                st.dataframe(pd.DataFrame([{"Motif": q["label"], "Valeur": q["value"], "Contribution": round(q["contribution"], 3)}
                                           for q in out["reasons"]]), width="stretch", hide_index=True)
            else:
                st.info("Aucune variable ne pousse significativement vers la fraude.")
        except Exception as e:  # noqa: BLE001
            st.error(f"API injoignable : {e}")
    st.caption("Les valeurs catégorielles sont des codes ordinaux appris sur le train ; l'API renvoie les 3 à 5 variables "
               "qui ont le plus contribué à l'alerte.")
