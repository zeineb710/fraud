"""Jeu SYNTHÉTIQUE au format IEEE-CIS (mode démo / tests). Ce ne sont PAS les vraies données."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(n: int = 40000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_cards = max(n // 6, 100)
    fraud = rng.random(n) < 0.035
    t = rng.integers(86400, 86400 * 182, n)

    # les fraudes arrivent en rafales sur des cartes compromises
    n_burst = max(int(fraud.sum() // 6), 5)
    centers = rng.integers(86400, 86400 * 182, n_burst)
    burst_card = rng.integers(0, n_cards, n_burst)
    card = rng.integers(0, n_cards, n)
    fi = np.where(fraud)[0]
    b = rng.integers(0, n_burst, len(fi))
    t[fi] = centers[b] + rng.integers(0, 3 * 3600, len(fi))
    card[fi] = burst_card[b]
    order = np.argsort(t, kind="stable")
    t, card, fraud = t[order], card[order], fraud[order]

    c2 = rng.integers(100, 600, n_cards)
    c4 = rng.choice(["visa", "mastercard", "american express", "discover"], n_cards)
    c5 = rng.choice([100, 126, 166, 195, 226], n_cards)
    c6 = rng.choice(["debit", "credit"], n_cards, p=[0.7, 0.3])
    a1 = rng.integers(100, 500, n_cards)
    first_day = rng.integers(0, 60, n_cards)
    day = t // 86400

    amt = np.where(fraud, rng.lognormal(5.0, 1.1, n), rng.lognormal(3.8, 1.0, n)).round(2)
    prod = np.where(fraud, rng.choice(list("CWHRS"), n, p=[.5, .3, .1, .05, .05]),
                    rng.choice(list("CWHRS"), n, p=[.1, .7, .1, .05, .05]))
    em_l = rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", None],
                      n, p=[.4, .2, .15, .02, .23])
    em_f = rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", None],
                      n, p=[.3, .1, .1, .2, .3])
    p_email = np.where(fraud, em_f, em_l)
    r_email = np.where(rng.random(n) < 0.3, p_email, None)

    df = pd.DataFrame({
        "TransactionID": 2987000 + np.arange(n),
        "isFraud": fraud.astype(int),
        "TransactionDT": t,
        "TransactionAmt": amt,
        "ProductCD": prod,
        "card1": 1000 + card, "card2": c2[card].astype(float), "card3": 150.0,
        "card4": c4[card], "card5": c5[card].astype(float), "card6": c6[card],
        "addr1": a1[card].astype(float), "addr2": 87.0,
        "dist1": np.where(rng.random(n) < 0.6, np.nan, rng.exponential(50, n)),
        "dist2": np.where(rng.random(n) < 0.9, np.nan, rng.exponential(200, n)),
        "P_emaildomain": p_email, "R_emaildomain": r_email,
    })
    for i in range(1, 15):
        df[f"C{i}"] = rng.poisson(np.where(fraud, 3.0, 1.0)).astype("float32")
    D1 = np.maximum(day - first_day[card], 0).astype("float32")
    for i in range(1, 16):
        if i == 1:
            df["D1"] = D1
        else:
            v = rng.exponential(30, n).astype("float32")
            v[rng.random(n) < min(0.1 + 0.05 * i, 0.9)] = np.nan
            df[f"D{i}"] = v
    for i in range(1, 10):
        pT = np.where(fraud, 0.3, 0.7)
        m = np.where(rng.random(n) < pT, "T", "F").astype(object)
        m[rng.random(n) < 0.35] = None
        df[f"M{i}"] = m
    for i in range(1, 101):
        shift = np.where(fraud, 1.0, 0.0) if i <= 10 else 0.0
        v = (rng.normal(0, 1, n) + shift).astype("float32")
        v[rng.random(n) < min(0.005 * i, 0.8)] = np.nan
        df[f"V{i}"] = v

    has_id = rng.random(n) < np.where(fraud, 0.6, 0.22)
    for i in range(1, 12):
        v = rng.normal(0, 50, n).astype("float32")
        v[~has_id] = np.nan
        df[f"id_{i:02d}"] = v
    dev = np.where(rng.random(n) < np.where(fraud, 0.7, 0.4), "mobile", "desktop").astype(object)
    brow = rng.choice(["chrome 63.0", "safari 11.0", "firefox 57.0", "edge 16.0"], n).astype(object)
    for col, vals in {"DeviceType": dev, "id_31": brow,
                      "id_12": np.where(rng.random(n) < .9, "Found", "NotFound").astype(object),
                      "DeviceInfo": rng.choice(["Windows", "iOS Device", "SM-G960F", "MacOS"], n).astype(object)}.items():
        v = vals.copy()
        v[~has_id] = None
        df[col] = v
    return df
