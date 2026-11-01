# Détection de fraude explicable : IEEE-CIS

Application web interactive qui charge les transactions, entraîne et compare plusieurs modèles de détection de fraude (supervisés et non supervisés), règle le seuil de décision selon le coût des erreurs, puis explique les scores (globalement et transaction par transaction) avec SHAP, dans un langage compréhensible par un analyste métier.

## Liens

| Livrable | Lien |
|---|---|
| Application déployée (Streamlit Community Cloud) | https://e24fw7lsqrtmw7fcmo3gkn.streamlit.app |
| API d'interprétation (FastAPI, documentation interactive) | https://fraud-0kzn.onrender.com/docs |
| Dépôt GitHub | https://github.com/zeineb710/fraud |
| Jeu de données (Kaggle) | https://www.kaggle.com/competitions/ieee-fraud-detection |

## Problème métier

Une banque ou un acteur du paiement en ligne veut repérer les transactions frauduleuses pour les bloquer ou les faire vérifier. Les deux erreurs n'ont pas le même coût : une fraude manquée est une perte directe, une fausse alerte dégrade l'expérience du client et mobilise des analystes. Le modèle doit donc être performant sur une classe très rare, et ses alertes doivent pouvoir être justifiées.

**Données** : échantillon de 150 000 transactions du jeu IEEE-CIS, dont 2,65 % de fraudes (jeu très déséquilibré). Le coût d'analyse d'une alerte est supposé égal à 5 $.

## Fonctionnalités

L'application comporte six pages.

1. **Données** : aperçu, statistiques descriptives, valeurs manquantes, proportion de fraudes, distribution des montants, fraude par heure et par semaine, variables catégorielles.
2. **Préparation** : traitement des valeurs manquantes, encodage des catégories, construction de variables, séparation entraînement / test [préciser : chronologique ou aléatoire].
3. **Comparaison des modèles** : régression logistique, Random Forest, XGBoost (supervisés), Isolation Forest, LOF, One-Class SVM et autoencodeur (détection d'anomalies), avec tableau comparatif.
4. **Évaluation et seuil** : AUC-ROC, PR-AUC, précision, rappel et F1 sur la classe fraude, courbes ROC et précision-rappel, matrice de confusion, curseur de seuil de décision.
5. **Explicabilité** : importance globale SHAP, explication d'une transaction, phrase de synthèse.
6. **API** : service FastAPI documenté (Swagger), appelé par l'application.

## Résultats principaux

Jeu de test ([X XXX] transactions). Tableau trié par PR-AUC décroissante. Précision@600 et Rappel@600 sont calculés sur les 600 transactions les plus suspectes. L'économie est mesurée au seuil optimal de chaque modèle.

| Modèle | Famille | PR-AUC | ROC-AUC | Précision@600 | Rappel@600 | Économie ($) |
|---|---|---|---|---|---|---|
| XGBoost | supervisé | **0,599** | **0,905** | 0,803 | 0,438 | **92 530** |
| Random Forest | supervisé | 0,445 | 0,874 | 0,613 | 0,335 | 82 699 |
| Régression logistique | supervisé | 0,232 | 0,820 | 0,345 | 0,188 | 64 131 |
| Isolation Forest | non supervisé | 0,129 | 0,720 | 0,192 | 0,105 | 19 197 |
| Autoencodeur | non supervisé | 0,092 | 0,688 | 0,163 | 0,089 | 21 540 |
| One-Class SVM | non supervisé | 0,079 | 0,622 | 0,160 | 0,087 | 5 207 |
| LOF | non supervisé | 0,069 | 0,652 | 0,098 | 0,054 | 65 |

- **Modèle recommandé : XGBoost**. Il est premier sur toutes les métriques du tableau, avec une PR-AUC de 0,599 contre 0,445 pour le Random Forest.
- Au seuil optimal, XGBoost économise 92 530 $ par rapport à l'absence de modèle (−60 %), avec une précision de 0,28 et un rappel de 0,71 pour 2 760 alertes.
- Les modèles supervisés dominent nettement les modèles non supervisés. Le meilleur non supervisé, Isolation Forest, atteint une PR-AUC de 0,129, ce qui reste utile comme point de comparaison mais insuffisant seul.
- Le classement par économie ne suit pas exactement celui de la PR-AUC : l'autoencodeur économise davantage qu'Isolation Forest malgré une PR-AUC plus faible.


### Explications

- Les variables les plus influentes (SHAP) sont [variable 1], [variable 2] et [variable 3].
- SHAP décrit ce que le modèle utilise, pas ce qui cause la fraude.

## Structure du dépôt

```
fraude-ieee-cis/
├── app.py                # application Streamlit (page d'accueil)
├── pages/                # pages 1 à 6 de l'application
├── api/main.py           # API FastAPI
├── src/                  # code métier
│   ├── download.py       # téléchargement Kaggle
│   ├── eda.py            # analyse exploratoire
│   ├── features.py       # préparation et variables
│   ├── models.py         # modèles
│   ├── metrics.py        # métriques
│   ├── pipeline.py       # exécution complète
│   ├── store.py          # lecture des artefacts
│   ├── api_client.py     # client de l'API
│   └── ...
├── artifacts/            # résultats du pipeline (modèles, scores, SHAP)
├── notebooks/            # notebook d'exploration
├── data/raw/             # données brutes (non versionnées)
├── requirements.txt
└── .streamlit/
```

## Installation et lancement en local

```bash
git clone https://github.com/zeineb710/fraud.git
cd fraud/fraude-ieee-cis
pip install -r requirements.txt
```

Télécharger les données (jeton Kaggle requis) puis lancer le pipeline :

```bash
python -m src.download
python -m src.pipeline [options à préciser, par exemple --max-rows]
```

Lancer l'API, puis l'application, dans deux terminaux :

```bash
uvicorn api.main:app --reload
streamlit run app.py
```

L'API est alors disponible sur `http://127.0.0.1:8000`, avec sa documentation sur `http://127.0.0.1:8000/docs`.

## Notebook

`notebooks/01_exploration.ipynb` présente l'exploration des données. Il s'ouvre avec Jupyter, VS Code, Colab ou GitHub Codespaces.

## Déploiement

- **Application** : Streamlit Community Cloud, fichier principal `fraude-ieee-cis/app.py`, Python 3.11. Elle lit les résultats du dossier `artifacts/`, sans données brutes ni clé Kaggle.
- **API** : Render, dossier racine `fraude-ieee-cis`. Si l'API est en veille, le premier appel peut prendre environ une minute.
- L'application trouve l'API grâce au secret `API_URL`.

## Limites

- Résultats obtenus sur un échantillon de 150 000 transactions (le jeu d'entraînement Kaggle en compte 590 540), avec un taux de fraude de 2,65 %.
- L'économie dépend de l'hypothèse de 5 $ par alerte [et de : préciser comment est valorisée une fraude manquée].
- Données très déséquilibrées et avec beaucoup de valeurs manquantes.
- Jeu de données de compétition Kaggle, sans validation externe.