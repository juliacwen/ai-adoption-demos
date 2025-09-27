"""
ai_payment_data.py
Author: Julia Wen
Date: 2025-09-26
Description: 
Generates synthetic transactions and trains a RandomForest pipeline if no artifact exists.
Exports: train_or_load_pipeline, generate_synthetic_transactions, label_synthetic, ARTIFACT_PATH
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "fraud_model_artifact.joblib")

def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)

# Official-ish Stripe test cards (good)
_TEST_CARD_SET = {
    "4242424242424242",
    "4000000000000002",
    "4012888888881881",
    "5555555555554444",
    "4000000000009995",
    "4000000000000341",
}

def generate_synthetic_transactions(n: int = 4000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    names = ["Alice","Bob","Carol","Dave","Eve","Frank","Grace","Hank","Ivy","Jack","Karen","Leo","Mia","Nina","Oscar","Pam"]
    payment_methods = ["Card", "Apple Pay", "Google Pay", "PayPal"]
    devices = ["mobile","desktop","tablet"]
    countries = ["US","GB","CA","AU","IN","NG","RU"]

    rows = []
    for i in range(n):
        name = rng.choice(names)
        email = f"{name.lower()}{rng.integers(1,500)}@example.com"
        method = rng.choice(payment_methods, p=[0.6,0.13,0.13,0.14])
        amount = float(round(np.exp(rng.normal(np.log(30), 1.0)),2))
        device = rng.choice(devices, p=[0.6,0.3,0.1])
        country = rng.choice(countries, p=[0.5,0.12,0.1,0.08,0.08,0.06,0.06])
        hour = int(rng.integers(0,24))
        ip_risk = 0
        if country in ["NG","RU"]:
            ip_risk += 1
        if amount > 300:
            ip_risk += 1
        if hour in (0,1,2,3,4):
            ip_risk += 1
        past_txns = int(rng.poisson(1.2))

        if method == "Card":
            # mostly valid Stripe test cards, occasionally random invalid
            if rng.random() < 0.8:
                card_number = rng.choice(list(_TEST_CARD_SET))
            else:
                # random 16-digit (not guaranteed Luhn-valid)
                card_number = str(rng.integers(4_000_000_000_000_000, 4_999_999_999_999_999))
            cvv = f"{rng.integers(100,999)}"
            expiry = f"{rng.integers(1,12):02d}/{rng.integers(2025,2032)}"
            token = ""
        else:
            token_suffix = rng.choice(["success","fail","flagged"], p=[0.75,0.18,0.07])
            token = f"tok_{method.replace(' ','').lower()}_{token_suffix}_{rng.integers(1000,9999)}"
            card_number = ""
            cvv = ""
            expiry = ""

        rows.append({
            "name": name,
            "email": email,
            "amount": amount,
            "payment_method": method,
            "card_number": card_number,
            "cvv": cvv,
            "expiry": expiry,
            "token": token,
            "device": device,
            "country": country,
            "hour": hour,
            "ip_risk": ip_risk,
            "past_txns": past_txns
        })
    return pd.DataFrame(rows)

def label_synthetic(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = df.copy()
    probs = []
    for _, r in df.iterrows():
        score = 0.0
        if r["amount"] > 200:
            score += 0.25
        if r["amount"] > 1000:
            score += 0.5
        score += 0.15 * r.get("ip_risk",0)
        tkn = _safe_str(r.get("token","")).lower()
        if "fail" in tkn:
            score += 0.6
        if "flagged" in tkn:
            score += 0.8
        if r.get("country") in ["NG","RU"]:
            score += 0.4
        if r.get("past_txns",0) > 5:
            score -= 0.2
        if r.get("device") == "mobile" and r["amount"] < 5:
            score += 0.15
        # jitter
        score += float(rng.normal(0, 0.05))
        probs.append(float(min(max(score,0.0),1.0)))
    df["fraud_prob_label"] = probs
    def map_outcome(p):
        if p >= 0.7: return "rejected"
        if p >= 0.4: return "flagged"
        return "approved"
    df["outcome"] = df["fraud_prob_label"].apply(map_outcome)
    return df

def train_or_load_pipeline() -> Dict[str, Any]:
    """
    Returns artifact dict: {"pipeline": pipeline, "label_encoder": lbl}
    If ARTIFACT_PATH exists and loads, returns it; otherwise generates data, trains, saves, and returns new artifact.
    """
    # Attempt to load existing artifact
    if os.path.exists(ARTIFACT_PATH):
        try:
            art = joblib.load(ARTIFACT_PATH)
            # quick sanity: expect keys pipeline and label_encoder
            if isinstance(art, dict) and "pipeline" in art and "label_encoder" in art:
                return art
        except Exception:
            pass

    # Train new pipeline from synthetic data
    df = generate_synthetic_transactions(n=6000)
    df = label_synthetic(df)
    X = []
    for _, r in df.iterrows():
        X.append({
            "amount": float(r.get("amount",0.0)),
            "payment_method": r.get("payment_method","Card"),
            "device": r.get("device","desktop"),
            "country": r.get("country","US"),
            "hour": int(r.get("hour",0)),
            "ip_risk": int(r.get("ip_risk",0)),
            "past_txns": int(r.get("past_txns",0)),
            "token_has_fail": 1 if ("fail" in _safe_str(r.get("token","")).lower()) else 0,
            "token_has_flagged": 1 if ("flagged" in _safe_str(r.get("token","")).lower()) else 0,
            "card_present": 1 if str(r.get("card_number","")).strip() != "" else 0
        })
    y = df["outcome"].astype(str).values
    lbl = LabelEncoder()
    y_enc = lbl.fit_transform(y)
    pipeline = Pipeline([
        ("vec", DictVectorizer(sparse=False)),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    pipeline.fit(X, y_enc)
    artifact = {"pipeline": pipeline, "label_encoder": lbl}
    joblib.dump(artifact, ARTIFACT_PATH)
    return artifact

