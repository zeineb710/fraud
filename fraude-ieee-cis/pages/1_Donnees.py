import plotly.express as px
import streamlit as st

from src import ui

ui.setup("Données", "📊")
ui.require_artifacts()
from src import store  # noqa: E402

st.title("1 · Données")
ov = store.eda_overview()
c = st.columns(5)
c[0].metric("Transactions", f"{ov['n_rows']:,}".replace(",", " "))
c[1].metric("Fraudes", f"{ov['n_fraud']:,}".replace(",", " "))
c[2].metric("Taux de fraude", f"{ov['fraud_rate']:.2f} %")
c[3].metric("Période couverte", f"{ov['days']:.0f} jours")
c[4].metric("Montant médian (fraude / honnête)", f"{ov['median_amt_fraud']:.0f} $ / {ov['median_amt_legit']:.0f} $")
st.info("`TransactionDT` est un décalage en secondes depuis une origine inconnue (pas une date réelle) : "
        "on s'en sert pour ordonner et découper chronologiquement, et pour l'heure et le jour.")

t1, t2, t3, t4, t5 = st.tabs(["Évolution dans le temps", "Par heure", "Montants", "Par catégorie", "Valeurs manquantes"])
with t1:
    w = store.eda("week")
    fig = px.line(w, x="week", y="fraud_rate", markers=True, labels={"week": "Semaine", "fraud_rate": "Taux de fraude (%)"})
    fig.add_hline(y=ov["fraud_rate"], line_dash="dash", annotation_text="taux global")
    st.plotly_chart(fig, width="stretch")
    st.caption("Un taux qui varie d'une semaine à l'autre est un signe de dérive (les fraudeurs s'adaptent).")
with t2:
    h = store.eda("hour")
    st.plotly_chart(px.bar(h, x="hour", y="fraud_rate", labels={"hour": "Heure (décalage depuis l'origine)", "fraud_rate": "Taux de fraude (%)"}),
                    width="stretch")
with t3:
    a = store.eda("amount")
    st.plotly_chart(px.bar(a, x="log10_montant", y="part", color="classe", barmode="overlay", opacity=0.65,
                           labels={"log10_montant": "log10(montant en $)", "part": "Part des transactions de la classe (%)"}),
                    width="stretch")
with t4:
    cat = store.eda("cat")
    var = st.selectbox("Variable", cat["variable"].unique())
    d = cat[cat["variable"] == var]
    st.plotly_chart(px.bar(d, x="categorie", y="fraud_rate", hover_data=["n"],
                           labels={"categorie": var, "fraud_rate": "Taux de fraude (%)"}), width="stretch")
    st.caption("Les 12 catégories les plus fréquentes ; « (manquant) » = valeur absente.")
with t5:
    m = store.eda("missing")
    st.plotly_chart(px.bar(m, x="manquant_pct", y="colonne", orientation="h",
                           labels={"manquant_pct": "% de valeurs manquantes"}).update_yaxes(autorange="reversed"),
                    width="stretch")
    st.caption("Être manquant peut être un indice en soi (ex. pas d'information d'appareil).")
