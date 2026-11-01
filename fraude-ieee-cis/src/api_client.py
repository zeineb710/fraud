"""Client de l'API FastAPI. Si API_URL n'est pas défini, l'API démarre dans le même processus."""
from __future__ import annotations

import os
import threading
import time

import requests

_lock = threading.Lock()


def _health(base: str) -> bool:
    try:
        return requests.get(base + "/health", timeout=1.5).ok
    except requests.RequestException:
        return False


def ensure_local_api(port: int | None = None) -> str:
    port = port or int(os.getenv("API_PORT", "8000"))
    base = f"http://127.0.0.1:{port}"
    if _health(base):
        return base
    with _lock:
        if not _health(base):
            import uvicorn

            from api.main import app

            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
            threading.Thread(target=server.run, daemon=True).start()
            for _ in range(60):
                if _health(base):
                    break
                time.sleep(0.5)
    return base


def api_base() -> str:
    url = os.getenv("API_URL", "").strip()
    return url.rstrip("/") if url else ensure_local_api()


def is_external() -> bool:
    return bool(os.getenv("API_URL", "").strip())


def public_docs_url() -> str:
    """Lien vers la documentation Swagger visible depuis le navigateur."""
    if os.getenv("API_URL"):
        return os.environ["API_URL"].rstrip("/") + "/docs"
    if os.getenv("API_PUBLIC_URL"):
        return os.environ["API_PUBLIC_URL"].rstrip("/") + "/docs"
    cs, dom = os.getenv("CODESPACE_NAME"), os.getenv("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN")
    if cs and dom:
        return f"https://{cs}-{os.getenv('API_PORT', '8000')}.{dom}/docs"
    return f"http://localhost:{os.getenv('API_PORT', '8000')}/docs"


def call(method: str, path: str, **kw) -> requests.Response:
    return requests.request(method, api_base() + path, timeout=60, **kw)
