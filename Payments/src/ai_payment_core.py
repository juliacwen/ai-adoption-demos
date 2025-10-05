"""
ai_payment_core.py
Author: Julia Wen
Date: 2025-10-01

Description
-----------
Core payment API handling:
- input normalization & validation (cards & wallet tokens)
- feature extraction for fraud model
- scoring (calculate_fraud_prob) and mapping to human decisions (fraud_decision)
- single payment processing (process_payment), refunds (process_refund), batch processing 
(process_batch)
- CSV loader helper (load_transactions_csv)

Design notes
------------
- Validation returns an explicit (bool, reason) so GUI can show user-facing reasons.
- Detailed debug traces are appended to returned dicts when `debug=True`.
- The module imports model/data functions from ai_payment_data.py
"""

import os
import uuid
import math
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, IO

import numpy as np
import pandas as pd

from ai_payment_data import (
    generate_synthetic_transactions,
    label_synthetic,
    train_or_load_pipeline,
    ARTIFACT_PATH,
    TEST_CARD_SET,
)

# -----------------------
# Logging
# -----------------------
logger = logging.getLogger("ai_payment_core")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("ai_payment_core.log", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)
logger.debug("ai_payment_core logger initialized")

# -----------------------
# Constants
# -----------------------
MIN_CARD_DIGITS = 12
MAX_CARD_DIGITS = 19
CVV_LENGTHS = (3, 4)
EXPIRY_FORMAT = "MM/YYYY"
EXPIRY_TWO_DIGIT_YEAR_BASE = 2000
DEFAULT_COUNTRY = "US"
FRAUD_THRESHOLDS = (0.25, 0.6)
LARGE_AMOUNT_RISK_THRESHOLD = 500.0

# -----------------------
# Utilities
# -----------------------
def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)

def _digits_only_from_string(s: Any) -> str:
    if s is None:
        return ""
    return "".join(ch for ch in str(s) if ch.isdigit())

def sanitize_card_number(raw_value: Any) -> str:
    """
    Normalize a card number into digits-only string.
    Handles ints, floats with trailing .0, numpy numeric types, and strings with spaces/dashes.
    """
    if raw_value is None:
        return ""
    try:
        import numpy as _np
        if isinstance(raw_value, (_np.integer, _np.floating)):
            try:
                return str(int(raw_value))
            except Exception:
                raw_value = str(raw_value)
    except Exception:
        pass

    if isinstance(raw_value, float):
        try:
            return str(int(raw_value))
        except Exception:
            raw_value = str(raw_value)

    if isinstance(raw_value, int):
        return str(raw_value)

    s = str(raw_value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = s.replace(" ", "").replace("-", "")
    return _digits_only_from_string(s)

def luhn_checksum(card_number: str) -> bool:
    """
    Luhn algorithm; returns True if card_number (digits-only string) is valid.
    """
    cn = _digits_only_from_string(card_number)
    if not cn or len(cn) < MIN_CARD_DIGITS:
        return False
    total = 0
    reverse = cn[::-1]
    for i, ch in enumerate(reverse):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0

# -----------------------
# Load / train pipeline
# -----------------------
_model_artifact = train_or_load_pipeline()
_model_pipeline = _model_artifact["pipeline"]
_label_encoder = _model_artifact["label_encoder"]

# -----------------------
# Feature extraction
# -----------------------
def _extract_features_for_model(txn: Dict[str, Any], history_list: Optional[Union[List[dict], 
pd.DataFrame]] = None) -> Dict[str, Any]:
    tok = _safe_str(txn.get("token","")).lower()
    features = {
        "amount": float(txn.get("amount",0.0) or 0.0),
        "payment_method": txn.get("payment_method","Card"),
        "device": txn.get("device","desktop"),
        "country": txn.get("country", DEFAULT_COUNTRY),
        "hour": int(txn.get("hour", datetime.utcnow().hour)),
        "ip_risk": int(txn.get("ip_risk", 0) or 0),
        "past_txns": int(txn.get("past_txns", 0) or 0),
        "token_has_fail": 1 if ("fail" in tok) else 0,
        "token_has_flagged": 1 if ("flagged" in tok) else 0,
        "card_present": 1 if str(txn.get("card_number","")).strip() != "" else 0
    }
    if history_list is not None:
        try:
            if isinstance(history_list, list):
                features["past_txns"] = sum(1 for t in history_list if t.get("email","")==txn.get("email",""))
            elif isinstance(history_list, pd.DataFrame):
                features["past_txns"] = int(history_list[history_list.get("email","")==txn.get("email","")].shape[0])
        except Exception:
            pass
    return features

# -----------------------
# Public API: scoring & decision
# -----------------------
def calculate_fraud_prob(transaction: Dict[str, Any], history_df: Optional[Union[List[dict], 
pd.DataFrame]] = None) -> float:
    """
    Returns a fraud score in [0,1].
    Uses model predict_proba and maps to scalar score:
      fraud_score = P(rejected) + 0.5 * P(flagged)
    """
    features = _extract_features_for_model(transaction, history_df)
    probs = _model_pipeline.predict_proba([features])[0]
    clf_classes = _model_pipeline.named_steps["clf"].classes_
    try:
        class_labels = _label_encoder.inverse_transform(clf_classes)
    except Exception:
        class_labels = np.array(["approved","flagged","rejected"])
    prob_map = {lab: float(probs[idx]) for idx, lab in enumerate(class_labels)}
    return float(min(max(prob_map.get("rejected",0.0) + 0.5*prob_map.get("flagged",0.0),0.0),1.0))

def fraud_decision(fraud_prob: float, thresholds: Tuple[float,float] = FRAUD_THRESHOLDS) -> str:
    """Return human-readable decision string based on thresholds."""
    flag_th, reject_th = thresholds
    if fraud_prob >= reject_th:
        return "Rejected ❌"
    if fraud_prob >= flag_th:
        return "Flagged ⚠️"
    return "Approved ✅"

# -----------------------
# Validation
# -----------------------
def is_valid_payment(payment_method: str,
                     card_number: Any = "",
                     cvv: Any = "",
                     expiry: str = "",
                     email: str = "",
                     token: str = "",
                     debug_msgs: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Validate a payment. Returns (True, "OK") or (False, reason).
    If debug_msgs list provided, append step-by-step traces.
    """
    if debug_msgs is None:
        debug_msgs = []
    pm = (payment_method or "").strip()
    debug_msgs.append(f"Validation: payment_method='{pm}'")
    if pm == "Card":
        cn = sanitize_card_number(card_number)
        debug_msgs.append(f"Validation: card_number_sanitized='{cn}'")
        if cn in TEST_CARD_SET:
            debug_msgs.append("Validation: known test card -> format accepted")
            if not email or "@" not in _safe_str(email):
                debug_msgs.append("Validation failed: email missing/invalid for test card")
                return False, "Email missing or invalid for test card"
            return True, "OK"
        if not cn or not (MIN_CARD_DIGITS <= len(cn) <= MAX_CARD_DIGITS):
            debug_msgs.append("Validation failed: card digits missing or invalid length")
            return False, "Card number missing or length invalid"
        if not luhn_checksum(cn):
            debug_msgs.append("Validation failed: Luhn checksum failed")
            return False, "Card failed Luhn check"
        debug_msgs.append("Validation: Luhn checksum passed")
        if expiry:
            try:
                parts = expiry.split("/")
                if len(parts) != 2:
                    debug_msgs.append("Validation failed: expiry format invalid")
                    return False, f"Expiry must be in {EXPIRY_FORMAT} format"
                m = int(parts[0]); y = int(parts[1])
                if y < 100:
                    y += EXPIRY_TWO_DIGIT_YEAR_BASE
                if not (1 <= m <= 12):
                    debug_msgs.append("Validation failed: expiry month invalid")
                    return False, "Expiry month invalid"
                debug_msgs.append("Validation: expiry parsed")
            except Exception:
                debug_msgs.append("Validation failed: expiry parse error")
                return False, f"Expiry must be in {EXPIRY_FORMAT} format"
        cvv_d = _digits_only_from_string(cvv)
        debug_msgs.append(f"Validation: cvv digits='{cvv_d}'")
        if not cvv_d or len(cvv_d) not in CVV_LENGTHS:
            debug_msgs.append("Validation failed: CVV invalid")
            return False, "CVV invalid"
        if not email or "@" not in _safe_str(email):
            debug_msgs.append("Validation failed: Email missing/invalid")
            return False, "Email missing or invalid"
        debug_msgs.append("Validation: card payment validated")
        return True, "OK"
    else:
        tok = _safe_str(token)
        debug_msgs.append(f"Validation: wallet token='{tok[:40]}'")
        if not tok or not tok.startswith("tok_"):
            debug_msgs.append("Validation failed: token missing or malformed")
            return False, "Wallet token missing or malformed"
        if not email or "@" not in _safe_str(email):
            debug_msgs.append("Validation failed: Email missing/invalid for wallet")
            return False, "Email missing or invalid for wallet"
        debug_msgs.append("Validation: wallet payment validated")
        return True, "OK"

# -----------------------
# Explain helper
# -----------------------
def _explain_transaction(transaction: Dict[str,Any], fraud_prob: float) -> str:
    reasons = []
    if float(transaction.get("amount",0.0)) > LARGE_AMOUNT_RISK_THRESHOLD:
        reasons.append("High amount")
    tok = _safe_str(transaction.get("token","")).lower()
    if "fail" in tok:
        reasons.append("Token indicates failure")
    if "flagged" in tok:
        reasons.append("Token indicates flagged")
    if transaction.get("country") in ["NG","RU"]:
        reasons.append("High-risk country")
    if transaction.get("device") == "mobile" and transaction.get("amount",0.0) < 5:
        reasons.append("Mobile tiny-amount anomaly")
    if not reasons:
        reasons.append("No obvious signals; model score used")
    return "; ".join(reasons)

# -----------------------
# Processing
# -----------------------
def process_payment(transaction: Dict[str,Any],
                    transactions_db: Optional[Union[dict,pd.DataFrame]] = None,
                    debug: bool = False) -> Dict[str,Any]:
    """
    Validate -> score -> decision -> store if Approved.
    Returns dict: {txn_id, decision, fraud_prob, reason, valid, debug?}
    """
    debug_msgs: List[str] = []
    txn = dict(transaction)
    txn["token"] = _safe_str(txn.get("token",""))
    txn["card_number"] = txn.get("card_number","")
    txn["cvv"] = txn.get("cvv","")
    txn["expiry"] = txn.get("expiry","")
    txn["email"] = _safe_str(txn.get("email",""))

    debug_msgs.append(f"process_payment: starting for email='{txn.get('email','')}' amount={txn.get('amount',0)} method={txn.get('payment_method','')}")

    valid, reason = is_valid_payment(
        payment_method=txn.get("payment_method",""),
        card_number=txn.get("card_number",""),
        cvv=txn.get("cvv",""),
        expiry=txn.get("expiry",""),
        email=txn.get("email",""),
        token=txn.get("token",""),
        debug_msgs=debug_msgs
    )

    if not valid:
        reason_parts = [m for m in debug_msgs if "Validation failed" in m or m.startswith("Validation:")]
        if not reason_parts:
            reason_parts = [reason or "Validation failed"]
        reason_text = "; ".join(reason_parts)
        out = {
            "txn_id": None,
            "decision": "Rejected ❌",
            "fraud_prob": 1.0,
            "reason": reason_text,
            "valid": False
        }
        if debug:
            out["debug"] = debug_msgs
            logger.debug("process_payment debug: %s", debug_msgs)
        return out

    hist_df = None
    if transactions_db is not None and isinstance(transactions_db, pd.DataFrame):
        hist_df = transactions_db

    try:
        fraud_prob = calculate_fraud_prob(txn, hist_df)
        debug_msgs.append(f"Model fraud_prob={fraud_prob:.4f}")
    except Exception as e:
        logger.exception("Model scoring failed: %s", e)
        fraud_prob = 1.0
        debug_msgs.append(f"Model scoring exception: {e}")

    decision = fraud_decision(fraud_prob)
    reason_text = _explain_transaction(txn, fraud_prob)
    debug_msgs.append(f"Decision: {decision} reason: {reason_text}")

    txn_id = None
    # ALWAYS assign a transaction ID
    txn_id = "TXN-" + uuid.uuid4().hex[:10].upper()
    if decision == "Approved ✅":
        record = dict(txn)
        record.update({
            "txn_id": txn_id,
            "decision": decision,
            "fraud_prob": float(fraud_prob),
            "reason": reason_text,
            "timestamp": datetime.utcnow().isoformat(),
            "refunded": False
        })
        if isinstance(transactions_db, dict):
            transactions_db[txn_id] = record
            debug_msgs.append(f"Stored approved txn in transactions_db under {txn_id}")

    out = {
        "txn_id": txn_id,
        "decision": decision,
        "fraud_prob": float(fraud_prob),
        "reason": reason_text,
        "valid": True
    }
    if debug:
        out["debug"] = debug_msgs
        logger.debug("process_payment debug: %s", debug_msgs)
    return out

def process_refund(txn_id: str, transactions_db: dict) -> bool:
    """Mark stored transaction as refunded."""
    if not txn_id:
        return False
    if txn_id in transactions_db:
        rec = transactions_db[txn_id]
        if rec.get("decision") != "Approved ✅":
            # Only allow refund for Approved transactions
            return False
        if rec.get("refunded", False):
            return False
        rec["refunded"] = True
        rec["refund_timestamp"] = datetime.utcnow().isoformat()
        rec["decision"] = "Refunded ♻️"
        transactions_db[txn_id] = rec
        logger.debug("process_refund: refunded %s", txn_id)
        return True
    logger.debug("process_refund: txn_id %s not found", txn_id)
    return False

def process_batch(transactions_list: List[dict], transactions_db: Optional[dict] = None, debug: bool = 
False) -> Tuple[List[dict], dict]:
    """
    Process batch transactions; returns (results_list, counts_dict)
    Ensures all transactions (Approved, Flagged, Rejected) have txn_id.
    """
    results = []
    counts = {"Approved ✅":0, "Flagged ⚠️":0, "Rejected ❌":0}
    for t in transactions_list:
        # Process single transaction
        res = process_payment(t, transactions_db, debug=debug)
        decision = res.get("decision")
        if decision in counts:
            counts[decision] += 1

        # Include all txn_ids, even for rejected/flagged
        txn_id = res.get("txn_id")
        if not txn_id:
            # Safety fallback (should not happen with fixed process_payment)
            txn_id = "TXN-" + uuid.uuid4().hex[:10].upper()

        results.append({
            "txn_id": res.get("txn_id"),
            "name": t.get("name",""),
            "email": t.get("email",""),
            "amount": float(t.get("amount",0.0) or 0.0),
            "payment_method": t.get("payment_method",""),
            "fraud_prob": float(res.get("fraud_prob", 1.0)),
            "decision": decision,
            "reason": res.get("reason",""),
            "debug": res.get("debug") if debug else None
        })
    return results, counts

# -----------------------
# CSV helper
# -----------------------
def load_transactions_csv(file_or_path: Union[str, IO]) -> List[dict]:
    """
    Load transactions CSV (path or file-like) into list of dicts.
    """
    if isinstance(file_or_path, str):
        df = pd.read_csv(file_or_path, dtype=str)
    else:
        df = pd.read_csv(file_or_path, dtype=str)
    df = df.fillna("")
    txns = []
    for _, row in df.iterrows():
        amt_raw = row.get("amount","")
        try:
            amt = float(amt_raw) if amt_raw != "" else 0.0
        except Exception:
            try:
                amt = float(_safe_str(amt_raw).replace(",",""))
            except Exception:
                amt = 0.0
        txns.append({
            "name": row.get("name",""),
            "email": row.get("email",""),
            "amount": amt,
            "payment_method": row.get("payment_method","Card"),
            "card_number": row.get("card_number",""),
            "cvv": row.get("cvv",""),
            "expiry": row.get("expiry",""),
            "token": row.get("token",""),
            "device": "batch_upload",
            "country": row.get("country", DEFAULT_COUNTRY) or DEFAULT_COUNTRY,
            "hour": int(datetime.utcnow().hour),
        })
    return txns

