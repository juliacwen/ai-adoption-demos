"""ai_payment_data.py
Refactored to read constants from ai_payment_config.yaml
Exports: train_or_load_pipeline, generate_synthetic_transactions, label_synthetic, ARTIFACT_PATH
"""

import os
import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import concurrent.futures

# Load config
CONFIG_PATH = Path(__file__).parent / "ai_payment_config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# Exports and shortcuts
ARTIFACT_PATH = str(Path(__file__).parent / CONFIG["fraud_model"]["artifact_path"])
TRAIN_CSV = str(Path(__file__).parent / CONFIG["fraud_model"]["train_csv"])
DEFAULT_RANDOM_STATE = int(CONFIG["fraud_model"].get("random_state", 42))

# Data constants from config
TEST_CARD_SET = set(CONFIG["data"]["test_card_set"])
NAMES = CONFIG["data"]["names"]
PAYMENT_METHODS = CONFIG["data"]["payment_methods"]["values"]
PAYMENT_PROBS = CONFIG["data"]["payment_methods"]["probs"]
DEVICES = CONFIG["data"]["devices"]["values"]
DEVICE_PROBS = CONFIG["data"]["devices"]["probs"]
COUNTRIES = CONFIG["data"]["countries"]["values"]
COUNTRY_PROBS = CONFIG["data"]["countries"]["probs"]
TOKEN_SUFFIXES = CONFIG["data"]["tokens"]["suffixes"]
TOKEN_PROBS = CONFIG["data"]["tokens"]["probs"]

# thresholds & weights
THRESHOLDS = CONFIG.get("thresholds", {})
WEIGHTS = CONFIG.get("weights", {})

# training settings
NUM_SYNTHETIC_TRANSACTIONS_CPU = int(CONFIG["fraud_model"].get("num_synthetic_transactions_cpu", 2000))
NUM_SYNTHETIC_TRANSACTIONS_GPU = int(CONFIG["fraud_model"].get("num_synthetic_transactions_gpu", 10000))
RF_N_ESTIMATORS = int(CONFIG["fraud_model"].get("rf_n_estimators", 200))
TIMEOUT_SECONDS = int(CONFIG["fraud_model"].get("timeout_seconds", 30))

# GPU detection
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except Exception:
    GPU_AVAILABLE = False

# Helpers
def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)

def _run_with_timeout(func, *args, timeout=TIMEOUT_SECONDS, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Operation failed: {e}") from e

# Data generation
def generate_synthetic_transactions(n: int = None, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    if n is None:
        n = NUM_SYNTHETIC_TRANSACTIONS_GPU if GPU_AVAILABLE else NUM_SYNTHETIC_TRANSACTIONS_CPU
    else:
        if not GPU_AVAILABLE and n > NUM_SYNTHETIC_TRANSACTIONS_CPU:
            print(f"Warning: Requested {n} transactions on CPU. Capping to {NUM_SYNTHETIC_TRANSACTIONS_CPU} for stability.")
            n = NUM_SYNTHETIC_TRANSACTIONS_CPU
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
        if country in CONFIG.get("fraud", {}).get("high_risk_countries", ["NG","RU"]):
            ip_risk += 1
        if amount > THRESHOLDS.get("amount_ip_risk", 300):
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
            card_number = cvv = expiry = ""
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

# Labeling
def label_synthetic(df: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = df.copy()
    probs = []
    for _, r in df.iterrows():
        score = 0.0
        if r["amount"] > THRESHOLDS.get("amount_high", 200):
            score += WEIGHTS.get("amount_high_score", 0.25)
        if r["amount"] > THRESHOLDS.get("amount_very_high", 1000):
            score += WEIGHTS.get("amount_very_high_score", 0.5)
        score += WEIGHTS.get("ip_risk_weight", 0.15) * r.get("ip_risk", 0)
        tkn = _safe_str(r.get("token","")).lower()
        if "fail" in tkn:
            score += WEIGHTS.get("fail_score", 0.6)
        if "flagged" in tkn:
            score += WEIGHTS.get("flagged_score", 0.8)
        if r.get("country") in CONFIG.get("fraud", {}).get("high_risk_countries", ["NG","RU"]):
            score += WEIGHTS.get("country_risk_score", 0.4)
        if r.get("past_txns",0) > THRESHOLDS.get("past_txns", 5):
            score -= WEIGHTS.get("past_txns_penalty", 0.2)
        if r.get("device") == "mobile" and r["amount"] < THRESHOLDS.get("mobile_low_amount", 5):
            score += WEIGHTS.get("mobile_low_amount_bonus", 0.15)
        score += float(rng.normal(0, WEIGHTS.get("jitter_std", 0.05)))
        probs.append(float(min(max(score,0.0),1.0)))
    df["fraud_prob_label"] = probs
    def map_outcome(p):
        if p >= WEIGHTS.get("score_rejected", 0.7): return "rejected"
        if p >= WEIGHTS.get("score_flagged", 0.4): return "flagged"
        return "approved"
    df["outcome"] = df["fraud_prob_label"].apply(map_outcome)
    return df

# Pipeline training/loading
def train_or_load_pipeline() -> Dict[str, Any]:
    if os.path.exists(ARTIFACT_PATH):
        try:
            art = joblib.load(ARTIFACT_PATH)
            if isinstance(art, dict) and "pipeline" in art and "label_encoder" in art:
                return art
        except Exception as e:
            print(f"Failed to load artifact: {e}")
    df = generate_synthetic_transactions()
    df = label_synthetic(df)
    X = []
    for _, r in df.iterrows():
        X.append({
            "amount": float(r.get("amount",0.0)),
            "payment_method": str(r.get("payment_method","Card")),
            "device": str(r.get("device","desktop")),
            "country": str(r.get("country","US")),
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
        ("clf", RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=DEFAULT_RANDOM_STATE, n_jobs=-1))
    ])
    pipeline.fit(X, y_enc)
    artifact = {"pipeline": pipeline, "label_encoder": lbl}
    joblib.dump(artifact, ARTIFACT_PATH)
    return artifact
