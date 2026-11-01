#  Détection de fraude explicable — IEEE-CIS

Projet Challenge 2 · Data Mining · ENSI. Application web **Streamlit** (6 modules) + **API FastAPI** + one-pager PDF.

## Livrables

| Livrable | Où |
|---|---|
| Application web (6 modules) | `app.py` + `pages/` — déployable sur Streamlit Community Cloud ou Hugging Face Spaces |
| API FastAPI documentée | `api/main.py` — Swagger sur `/docs`, appelée par l'application (pages 5 et 6) |
| Dépôt GitHub | ce dépôt (code, notebook, `requirements.txt`, README) — **ajoutez vos captures d'écran dans `docs/`** |
| One-pager PDF (1 page) | bouton de téléchargement sur la page d'accueil (`src/onepager.py`) |

## Démarrage dans GitHub Codespaces

1. **Kaggle** : acceptez les règles de la compétition *IEEE-CIS Fraud Detection*, puis créez une clé API (Settings > API).
2. **Secrets Codespaces** (GitHub > Settings > Codespaces > Secrets) : `KAGGLE_USERNAME` et `KAGGLE_KEY`, associés à ce dépôt.
3. Ouvrez le Codespace (les dépendances s'installent seules via `.devcontainer/`).
4. Construisez le projet (téléchargement **automatique** des données + entraînement) :
   ```bash
   python -m src.pipeline                   # jeu complet (≈ 20-40 min selon la machine)
   python -m src.pipeline --max-rows 150000 # version plus légère
   python -m src.pipeline --synthetic --fast  # démo en 1 min, données FICTIVES
   ```
5. Lancez l'application : `streamlit run app.py` (port 8501). L'API démarre automatiquement dans le même processus ;
   pour la lancer seule : `uvicorn api.main:app --port 8000` puis ouvrez `/docs`.

Sans artefacts, l'application elle-même propose un bouton « Télécharger les données et construire ».

## Déploiement

**Option A (recommandée)** : construire dans Codespaces, puis `git add artifacts/ && git commit && git push`
(≈ 30-60 Mo, sans Git LFS). Déployez `app.py` sur Streamlit Community Cloud : aucune donnée brute ni clé Kaggle nécessaire.

**Option B (téléchargement automatique au déploiement)** : dans les *Secrets* du service, définir
`KAGGLE_USERNAME`, `KAGGLE_KEY`, `AUTO_BUILD = "true"` et `MAX_ROWS = "150000"` (mémoire limitée sur le cloud).
Le premier lancement télécharge les données et construit les artefacts.

**API publique** : l'API tourne par défaut dans l'app (non exposée). Pour une URL publique, déployez-la séparément
(`uvicorn api.main:app --host 0.0.0.0 --port $PORT` sur Render/Hugging Face Spaces, avec `artifacts/` dans le dépôt)
et ajoutez le secret `API_URL` à l'application.

## Contenu

```
app.py                     accueil + one-pager PDF
pages/1_Donnees.py         Module 1 · exploration
pages/2_Preparation.py     Module 2 · split chronologique, variables, stratégie anti-déséquilibre
pages/3_Comparaison_Modeles.py   Module 3 · choix dynamique des modèles, courbes PR / ROC / rappel@k
pages/4_Evaluation_Seuil.py      Module 4 · curseurs seuil + budget, coût en direct
pages/5_Explicabilite.py   Module 5 · SHAP beeswarm, waterfall, codes motifs (via l'API)
pages/6_API.py             Module 6 · documentation et tests de l'API
src/                       download, features, models, metrics, pipeline, onepager
api/main.py                FastAPI (/score, /score/batch, /transactions/..., /model/info, /health)
```

## Choix méthodologiques (à reprendre dans le rapport)

- **Découpage chronologique 60/20/20** ; statistiques (sélection des V, encodage, imputation, normalisation) apprises sur le train seul.
- **Variables comportementales sans fuite** : pour chaque transaction, seuls les achats *antérieurs* de la même identité sont comptés.
- **Identité reconstituée** (`card1|card2|card3|card5|addr1|jour − D1`) : approximation, à présenter comme telle.
- **Stratégie anti-déséquilibre** comparée en validation (XGBoost, plusieurs graines) ; on ne rééchantillonne que si le gain dépasse le bruit.
  Probabilités corrigées par `p = β·ps / (β·ps − ps + 1)`.
- **Modèles** : XGBoost, Random Forest, régression logistique ; Isolation Forest, LOF, One-Class SVM, autoencodeur
  (LOF/OCSVM sur ACP à 20 dimensions pour la vitesse). DBSCAN n'est pas inclus : il ne sait pas scorer de nouvelles transactions.
- **Coût** = montants des fraudes manquées + coût d'analyse × alertes ; le coût par alerte (5 $ par défaut) est une **hypothèse** : faites une analyse de sensibilité.
- **SHAP** : valeurs exactes TreeSHAP natives de XGBoost (log-odds).

## Limites

Heuristique d'identité, variables Vesta anonymisées, étiquettes tardives, non supervisé nettement moins performant que le supervisé,
montants supposés en dollars. Le mode `--synthetic` ne sert qu'à tester l'application.
