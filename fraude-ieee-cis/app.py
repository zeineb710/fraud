"""Point d'entrée : navigation entre les pages (la page « Synthèse » est supprimée)."""
import streamlit as st

pages = [
    st.Page("pages/1_Donnees.py", title="Données", icon="📊", url_path="donnees", default=True),
    st.Page("pages/2_Preparation.py", title="Préparation", icon="🛠️", url_path="preparation"),
    st.Page("pages/3_Modelisation.py", title="Modélisation", icon="🤖", url_path="modelisation"),
    st.Page("pages/4_Comparaison_Modeles.py", title="Comparaison des modèles", icon="⚖️", url_path="comparaison"),
    st.Page("pages/5_Evaluation_Seuil.py", title="Évaluation et seuil", icon="🎚️", url_path="seuil"),
    st.Page("pages/6_Explicabilite.py", title="Explicabilité", icon="🔍", url_path="explicabilite"),
    st.Page("pages/7_API.py", title="API", icon="🔌", url_path="api"),
]
st.navigation(pages).run()