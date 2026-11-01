"""Téléchargement automatique des données IEEE-CIS depuis Kaggle.

Prérequis (une seule fois) :
  1. Accepter les règles de la compétition « IEEE-CIS Fraud Detection » sur Kaggle.
  2. Fournir UN des jeux d'identifiants suivants :
       - KAGGLE_API_TOKEN (nouveau jeton, commence par « KGAT_ »)
       - KAGGLE_USERNAME + KAGGLE_KEY (ancien format)
       - ou le fichier ~/.kaggle/kaggle.json ou ~/.kaggle/access_token
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import requests

from .config import COMPETITION, DATA_RAW, RAW_FILES


def raw_ready() -> bool:
    return all((DATA_RAW / f).exists() for f in RAW_FILES)


def _unzip_if_needed(name: str) -> None:
    z = DATA_RAW / (name + ".zip")
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DATA_RAW)
        z.unlink()


def _token() -> str | None:
    tok = os.getenv("KAGGLE_API_TOKEN", "").strip()
    if tok:
        return tok
    f = Path.home() / ".kaggle" / "access_token"
    return f.read_text().strip() if f.exists() else None


def _has_credentials() -> bool:
    return bool(_token()
                or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
                or (Path.home() / ".kaggle" / "kaggle.json").exists())


def _download_rest(name: str, token: str, log) -> None:
    """Secours : téléchargement direct par l'API REST avec le jeton (Bearer)."""
    url = f"https://www.kaggle.com/api/v1/competitions/data/download/{COMPETITION}/{name}"
    dest = DATA_RAW / (name + ".part")
    with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=120,
                      allow_redirects=True) as r:
        if r.status_code in (401, 403):
            raise RuntimeError(
                f"Accès refusé ({r.status_code}). Vérifiez le jeton et surtout que vous avez accepté les règles : "
                f"https://www.kaggle.com/c/{COMPETITION}/rules")
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with open(dest, "rb") as f:
        is_zip = f.read(2) == b"PK"
    if is_zip:
        dest.rename(DATA_RAW / (name + ".zip"))
        _unzip_if_needed(name)
    else:
        dest.rename(DATA_RAW / name)


def ensure_raw_data(log=print) -> None:
    """Télécharge train_transaction.csv et train_identity.csv s'ils sont absents."""
    if raw_ready():
        log("Données brutes déjà présentes dans data/raw/")
        return
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if not _has_credentials():
        raise RuntimeError(
            "Identifiants Kaggle introuvables. Définissez KAGGLE_API_TOKEN (jeton KGAT_…) ou "
            "KAGGLE_USERNAME + KAGGLE_KEY, ou déposez train_transaction.csv et train_identity.csv dans data/raw/.")

    api = None
    try:  # 1) bibliothèque officielle
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except Exception as e:  # noqa: BLE001
        log(f"Bibliothèque kaggle indisponible ou trop ancienne ({e}) : passage à l'API REST.")
        api = None

    for name in RAW_FILES:
        _unzip_if_needed(name)
        if (DATA_RAW / name).exists():
            continue
        log(f"Téléchargement de {name} depuis Kaggle (plusieurs minutes)…")
        done = False
        if api is not None:
            try:
                api.competition_download_file(COMPETITION, name, path=str(DATA_RAW), quiet=False)
                _unzip_if_needed(name)
                done = (DATA_RAW / name).exists()
            except Exception as e:  # noqa: BLE001
                log(f"Échec via la bibliothèque kaggle ({e}).")
        if not done and _token():
            _download_rest(name, _token(), log)
            done = (DATA_RAW / name).exists()
        if not done:
            raise RuntimeError(
                f"Impossible de télécharger {name}. Vérifiez que vous avez accepté les règles de la compétition : "
                f"https://www.kaggle.com/c/{COMPETITION}/rules")

    if not raw_ready():
        raise RuntimeError("Téléchargement incomplet : fichiers manquants dans data/raw/.")
    log("Téléchargement terminé.")