"""
ai_payment_data.py
Author: Julia Wen
Date: 2025-09-28
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

# ---------------- GPU/CPU Detection ---------------- #
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

# ---------------- Constants ---------------- #

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "fraud_model_artifact.joblib")

NUM_SYNTHETIC_TRANSACTIONS_CPU = 2000
NUM_SYNTHETIC_TRANSACTIONS_GPU = 10000  # larger dataset if GPU available
DEFAULT_RANDOM_STATE = 42

# Stripe official test cards (good ones)
TEST_CARD_SET = {
    "4242424242424242",  # Visa
    "4000056655665556",  # Visa debit
    "5555555555554444",  # Mastercard
    "2223003122003222",  # Mastercard (2-series)
    "378282246310005",   # Amex
    "6011111111111117",  # Discover
    "30569309025904",    # Diners Club
    "3566002020360505",  # JCB
}

# Names and payment methods
NAMES = ["Alice","Bob","Carol","Dave","Eve","Frank","Grace","Hank","Ivy","Jack",
         "Karen","Leo","Mia","Nina","Oscar","Pam"]
PAYMENT_METHODS = ["Card", "Apple Pay", "Google Pay", "PayPal"]
PAYMENT_PROBS = [0.6, 0.13, 0.13, 0.14]

DEVICES = ["mobile","desktop","tablet"]
DEVICE_PROBS = [0.6, 0.3, 0.1]

COUNTRIES = ["US","GB","CA","AU","IN","NG","RU"]
COUNTRY_PROBS = [0.5,0.12,0.1,0.08,0.08,0.06,0.06]

TOKEN_SUFFIXES = ["success","fail","flagged"]
TOKEN_PROBS = [0.75,0.18,0.07]

# Risk scoring thresholds
SCORE_FLAGGED = 0.4
SCORE_REJECTED = 0.7

# Scoring weights
AMOUNT_HIGH_SCORE = 0.25
AMOUNT_VERY_HIGH_SCORE = 0.5
IP_RISK_WEIGHT = 0.15
FAIL_SCORE = 0.6
FLAGGED_SCORE = 0.8
COUNTRY_RISK_SCORE = 0.4
PAST_TXNS_PENALTY = 0.2
MOBILE_LOW_AMOUNT_BONUS = 0.15
JITTER_STD = 0.05

# ---------------- Helpers ---------------- #

def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)

# ---------------- Data Generation ---------------- #

def generate_synthetic_transactions(n: int = None, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """
    Generates synthetic payment transactions.
    Automatically chooses CPU/GPU default sizes if n is None.
    """
    if n is None:
        n = NUM_SYNTHETIC_TRANSACTIONS_GPU if GPU_AVAILABLE else NUM_SYNTHETIC_TRANSACTIONS_CPU

    rng = np.random.default_rng(random_state)
    rows = []

    for _ in range(n):
        name = rng.choice(NAMES)
        email = f"{name.lower()}{rng.integers(1,500)}@example.com"
        method = rng.choice(PAYMENT_METHODS, p=PAYMENT_PROBS)
        amount = float(round(np.exp(rng.normal(np.log(30), 1.0)), 2))
        device = rng.choice(DEVICES, p=DEVICE_PROBS)
        country = rng.choice(COUNTRIES, p=COUNTRY_PROBS)
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
            if rng.random() < 0.8:
                card_number = rng.choice(list(TEST_CARD_SET))
            else:
                card_number = str(rng.integers(4_000_000_000_000_000, 4_999_999_999_999_999))
            cvv = f"{rng.integers(100,999)}"
            expiry = f"{rng.integers(1,12):02d}/{rng.integers(2025,2032)}"
            token = ""
        else:
            token_suffix = rng.choice(TOKEN_SUFFIXES, p=TOKEN_PROBS)
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

# ---------------- Labeling ---------------- #

def label_synthetic(df: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = df.copy()
    probs = []

    for _, r in df.iterrows():
        score = 0.0
        if r["amount"] > 200:
            score += AMOUNT_HIGH_SCORE
        if r["amount"] > 1000:
            score += AMOUNT_VERY_HIGH_SCORE
        score += IP_RISK_WEIGHT * r.get("ip_risk",0)
        tkn = _safe_str(r.get("token","")).lower()
        if "fail" in tkn:
            score += FAIL_SCORE
        if "flagged" in tkn:
            score += FLAGGED_SCORE
        if r.get("country") in ["NG","RU"]:
            score += COUNTRY_RISK_SCORE
        if r.get("past_txns",0) > 5:
            score -= PAST_TXNS_PENALTY
        if r.get("device") == "mobile" and r["amount"] < 5:
            score += MOBILE_LOW_AMOUNT_BONUS
        score += float(rng.normal(0, JITTER_STD))
        probs.append(float(min(max(score,0.0),1.0)))

    df["fraud_prob_label"] = probs

    def map_outcome(p):
        if p >= SCORE_REJECTED: return "rejected"
        if p >= SCORE_FLAGGED: return "flagged"
        return "approved"

    df["outcome"] = df["fraud_prob_label"].apply(map_outcome)
    return df

# ---------------- Pipeline ---------------- #

def train_or_load_pipeline() -> Dict[str, Any]:
    """
    Returns artifact dict: {"pipeline": pipeline, "label_encoder": lbl}.
    If ARTIFACT_PATH exists and loads, returns it; otherwise generates data, trains, saves, and returns new 
artifact.
    """
    if os.path.exists(ARTIFACT_PATH):
        try:
            art = joblib.load(ARTIFACT_PATH)
            if isinstance(art, dict) and "pipeline" in art and "label_encoder" in art:
                return art
        except Exception:
            pass

    df = generate_synthetic_transactions(n=None)
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
        ("clf", RandomForestClassifier(n_estimators=200, random_state=DEFAULT_RANDOM_STATE, n_jobs=-1))
    ])
    pipeline.fit(X, y_enc)
    artifact = {"pipeline": pipeline, "label_encoder": lbl}
    joblib.dump(artifact, ARTIFACT_PATH)
    return artifact

# ---------------- Run as script ---------------- #

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "sandbox_transactions.csv")
    
    # Generate CSV only if missing
    if not os.path.exists(csv_path):
        print("Generating synthetic transactions database...")
        df = generate_synthetic_transactions(n=None)
        df = label_synthetic(df)
        df.to_csv(csv_path, index=False)
        print(f"Synthetic transactions CSV saved to {csv_path}")
    else:
        print(f"Synthetic transactions CSV already exists at {csv_path}")

    # Train or load pipeline artifact
    print("Training or loading pipeline artifact...")
    artifact = train_or_load_pipeline()
    print(f"Pipeline artifact saved to {ARTIFACT_PATH}")
    print("Done.")

