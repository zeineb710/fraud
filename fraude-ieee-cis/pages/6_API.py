import json

import pandas as pd
import streamlit as st

from src import api_client, ui

ui.setup("API", "🔌")
ui.require_artifacts()

st.title("6 · API FastAPI")
with st.spinner("Démarrage / connexion à l'API…"):
    try:
        base = api_client.api_base()
        ok = api_client.call("GET", "/health").ok
    except Exception as e:  # noqa: BLE001
        base, ok = "?", False
        st.error(f"API injoignable : {e}")
c = st.columns(3)
c[0].metric("État", "En ligne ✅" if ok else "Hors ligne ❌")
c[1].metric("Mode", "API externe" if api_client.is_external() else "Intégrée au processus de l'app")
c[2].link_button("📘 Documentation Swagger (/docs)", api_client.public_docs_url())
st.caption(f"URL utilisée par l'application : `{base}`")
if not ok:
    st.stop()

st.subheader("Endpoints (lus dynamiquement depuis /openapi.json)")
spec = api_client.call("GET", "/openapi.json").json()
rows = [{"Méthode": m.upper(), "Chemin": p, "Description": d.get("summary", "")}
        for p, ops in spec["paths"].items() for m, d in ops.items()]
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

tab1, tab2, tab3, tab4 = st.tabs(["Transaction de l'échantillon", "Transaction personnalisée", "Lot", "Appeler depuis Python"])

with tab1:
    sample = api_client.call("GET", "/transactions/sample", params={"n": 30, "only_alerts": False}).json()
    tid = st.selectbox("TransactionID", [s_["TransactionID"] for s_ in sample],
                       format_func=lambda i: next(f"{i} · score {s_['score']:.3f} · {s_['amount']:,.0f} $" for s_ in sample if s_["TransactionID"] == i))
    if st.button("GET /transactions/{id}", key="b1"):
        r = api_client.call("GET", f"/transactions/{tid}")
        st.json(r.json())

with tab2:
    example = spec["components"]["schemas"]["ScoreRequest"]["example"]
    txt = st.text_area("Corps JSON de POST /score", json.dumps(example, indent=2, ensure_ascii=False), height=300)
    if st.button("POST /score", key="b2"):
        try:
            r = api_client.call("POST", "/score", json=json.loads(txt))
            out = r.json()
            if r.ok:
                st.metric("Score de fraude", f"{out['score']:.3f}", "ALERTE" if out["alert"] else "pas d'alerte",
                          delta_color="inverse" if out["alert"] else "normal")
                if out["reasons"]:
                    st.dataframe(pd.DataFrame(out["reasons"]), width="stretch", hide_index=True)
            st.json(out)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    st.caption("Les variables de vélocité doivent être calculées côté appelant (historique passé de l'identité). "
               "Les variables absentes sont traitées comme manquantes.")

with tab3:
    from src import store
    smp = store.sample()
    nb = st.slider("Taille du lot", 5, 200, 50)
    if st.button("POST /score/batch", key="b3"):
        feats = store.meta()["features"]
        items = [{k: float(v) for k, v in r.items() if pd.notna(v)} for r in smp.head(nb)[feats].to_dict("records")]
        r = api_client.call("POST", "/score/batch", json={"items": items}).json()
        st.success(f"{r['count']} transactions scorées, {r['alerts']} alertes.")
        st.dataframe(pd.DataFrame([{"score": x["score"], "alerte": x["alert"], "risque": x["risk_level"]} for x in r["results"]]),
                     width="stretch", hide_index=True)

with tab4:
    st.code(f'''import requests

API = "{base}"
payload = {json.dumps(example, ensure_ascii=False)}
r = requests.post(f"{{API}}/score", json=payload, timeout=30)
print(r.json()["score"], r.json()["alert"])
for reason in r.json()["reasons"]:
    print(reason["label"], reason["contribution"])
''', language="python")
    st.code(f"curl -X POST {base}/score -H 'Content-Type: application/json' -d '{json.dumps(example, ensure_ascii=False)}'",
            language="bash")
