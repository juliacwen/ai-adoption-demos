"""
# ai_payment_api.py
Author: Julia Wen
Date: 2025-10-04
Description
-----------
This FastAPI service wraps both ai_payment_core and ai_payment_db, providing
REST endpoints for:
- Single transaction processing
- Batch processing
- Refunds
- CSV uploads
- Retrieving latest or specific transactions

GUI or other clients should only call this API; no direct access to DB or core logic.
"""

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import io
import json
import uuid
from datetime import datetime
import logging
import pandas as pd
import torch  # For GPU detection

import ai_payment_core as core
import ai_payment_db as db

# -----------------------
# Constants & Device Detection
# -----------------------
TXN_PREFIX = "TXN-"
DECISION_APPROVED = "Approved ✅"
DECISION_FLAGGED = "Flagged ⚠️"
DECISION_REJECTED = "Rejected ❌"

# Detect CPU vs GPU
DEVICE = "GPU" if torch.cuda.is_available() else "CPU"

# Dynamic limits based on device
if DEVICE == "GPU":
    CSV_LATEST_PREDICTIONS_LIMIT = 5000
    DEFAULT_LATEST_LIMIT = 500
else:
    CSV_LATEST_PREDICTIONS_LIMIT = 1000
    DEFAULT_LATEST_LIMIT = 50

# -----------------------
# Logging
# -----------------------
logger = logging.getLogger("ai_payment_api")
logging.basicConfig(level=logging.INFO)
logger.info(f"Running on {DEVICE}, CSV_LATEST_PREDICTIONS_LIMIT={CSV_LATEST_PREDICTIONS_LIMIT}, DEFAULT_LATEST_LIMIT={DEFAULT_LATEST_LIMIT}")

# -----------------------
# FastAPI App
# -----------------------
app = FastAPI(title="AI Payment API")

# -----------------------
# Pydantic Models
# -----------------------
class PaymentRequest(BaseModel):
    txn_id: Optional[str] = None
    name: Optional[str] = ""
    email: Optional[str] = ""
    amount: Optional[float] = 0.0
    payment_method: Optional[str] = "Card"
    card_number: Optional[str] = ""
    cvv: Optional[str] = ""
    expiry: Optional[str] = ""
    token: Optional[str] = ""
    device: Optional[str] = "web"
    country: Optional[str] = "US"

class BatchRequest(BaseModel):
    transactions: List[PaymentRequest]

class RefundRequest(BaseModel):
    txn_id: str

# -----------------------
# Helper Functions
# -----------------------
def _as_dict_from_row(r: Any) -> Dict[str, Any]:
    if isinstance(r, dict):
        return r
    if hasattr(r, "_mapping"):
        return dict(r._mapping)
    if isinstance(r, (list, tuple)):
        out = {}
        try:
            out["transaction_id"] = r[0]
            out["prediction"] = r[1] if len(r) > 1 else None
            out["fraud_prob"] = r[2] if len(r) > 2 else None
            out["decision_text"] = r[3] if len(r) > 3 else None
            if len(r) > 4:
                out["created_at"] = r[4]
            if len(r) > 5:
                out["amount"] = r[5]
        except Exception:
            out["raw"] = str(r)
        return out
    if isinstance(r, str):
        try:
            parsed = json.loads(r)
            return parsed if isinstance(parsed, dict) else {"raw": parsed}
        except Exception:
            return {"raw": r}
    return {"raw": str(r)}

def _ensure_result_fields(res: Dict[str,Any], txn_source: Dict[str,Any]):
    if "name" not in res or res.get("name") in (None, ""):
        res["name"] = txn_source.get("name","") or ""
    if "amount" not in res or res.get("amount") in (None,):
        res["amount"] = float(txn_source.get("amount", 0.0) or 0.0)
    if "payment_method" not in res:
        res["payment_method"] = txn_source.get("payment_method","")
    if "txn_id" not in res or not res.get("txn_id"):
        res["txn_id"] = txn_source.get("txn_id") or (TXN_PREFIX + uuid.uuid4().hex[:10].upper())
    return res

def _save_any_decision(txn_id: str, decision: str, fraud_prob: float, amount: float = 0.0, name: str = ""):
    try:
        db.save_single_prediction(txn_id, decision, fraud_prob, amount, name)
        return True
    except Exception:
        logger.exception("db.save_single_prediction failed for txn_id %s", txn_id)
        return False

def _get_txn_from_db(txn_id: str) -> Optional[Dict[str,Any]]:
    candidate_names = [
        "get_transaction", "get_prediction_by_txn_id", "get_txn_by_id", "get_transaction_by_id", "get_by_txn_id"
    ]
    for nm in candidate_names:
        if hasattr(db, nm):
            try:
                rec = getattr(db, nm)(txn_id)
                if rec:
                    return _as_dict_from_row(rec)
            except Exception:
                logger.exception("db.%s failed", nm)
    if hasattr(db, "get_latest_predictions"):
        try:
            rows = db.get_latest_predictions(CSV_LATEST_PREDICTIONS_LIMIT)
            for r in rows:
                d = _as_dict_from_row(r)
                tid = d.get("transaction_id") or d.get("txn_id")
                if tid == txn_id:
                    return d
        except Exception:
            logger.exception("searching latest_predictions failed")
    if hasattr(db, "engine"):
        try:
            engine = getattr(db, "engine")
            q = "SELECT transaction_id, prediction, prediction_prob, decision_text, amount, refunded, refunded_amount, refunded_at, created_at FROM payment_predictions WHERE transaction_id = :tid LIMIT 1"
            df = pd.read_sql_query(q, con=engine, params={"tid": txn_id})
            if not df.empty:
                return df.to_dict(orient="records")[0]
        except Exception:
            logger.exception("SQL get_transaction fallback failed")
    return None

def _mark_refunded_in_db(txn_id: str, amount: float):
    for nm in ("mark_refunded", "set_refunded", "mark_transaction_refunded"):
        if hasattr(db, nm):
            try:
                getattr(db, nm)(txn_id, amount)
                return True
            except TypeError:
                try:
                    getattr(db, nm)(txn_id)
                    return True
                except Exception:
                    logger.exception("db.%s attempted but failed", nm)
            except Exception:
                logger.exception("db.%s failed", nm)
    if hasattr(db, "engine"):
        try:
            engine = getattr(db, "engine")
            with engine.begin() as conn:
                conn.execute(
                    "UPDATE payment_predictions SET refunded = TRUE, refunded_amount = :amt, refunded_at = :ts WHERE transaction_id = :tid",
                    {"amt": float(amount or 0.0), "ts": datetime.utcnow(), "tid": txn_id}
                )
            return True
        except Exception:
            logger.exception("SQL mark_refunded failed")
    return False

# -----------------------
# Endpoints
# -----------------------

# --- Single Payment Prediction ---
@app.post("/api/payments/predict")
def predict_payment(req: PaymentRequest):
    txn = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    txn["txn_id"] = txn.get("txn_id") or (TXN_PREFIX + uuid.uuid4().hex[:10].upper())
    try:
        result = core.process_payment(txn)
    except Exception as e:
        logger.exception("core.process_payment failed")
        return {"error": str(e)}

    result = _ensure_result_fields(result, txn)

    if result.get("txn_id") and result.get("decision"):
        _save_any_decision(result["txn_id"], result["decision"], result.get("fraud_prob", 0.0),
                           float(result.get("amount", 0.0) or 0.0), result.get("name",""))

    return result

# --- Batch Payment Prediction ---
@app.post("/api/payments/batch")
def batch_payment(req: BatchRequest):
    results = []
    counts = {DECISION_APPROVED:0, DECISION_FLAGGED:0, DECISION_REJECTED:0}
    for t in req.transactions:
        txn = t.model_dump() if hasattr(t, "model_dump") else t.dict()
        txn["txn_id"] = txn.get("txn_id") or (TXN_PREFIX + uuid.uuid4().hex[:10].upper())
        try:
            res = core.process_payment(txn)
        except Exception as e:
            logger.exception("core.process_payment failed for batch item")
            res = {"txn_id": txn["txn_id"], "decision": DECISION_REJECTED, "fraud_prob": 1.0, "reason": f"processing error: {e}"}
        res = _ensure_result_fields(res, txn)
        results.append(res)
        d = res.get("decision", "")
        if d in counts:
            counts[d] += 1
        if res.get("txn_id") and res.get("decision"):
            _save_any_decision(res["txn_id"], res["decision"], res.get("fraud_prob", 0.0),
                               float(res.get("amount", 0.0) or 0.0), res.get("name",""))

    return {"results": results, "summary": counts}

# --- CSV Upload for Batch Processing ---
@app.post("/api/payments/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        logger.exception("CSV read failed")
        return {"error": f"Failed to parse CSV: {e}", "results": [], "summary": {}}

    results = []
    counts = {DECISION_APPROVED:0, DECISION_FLAGGED:0, DECISION_REJECTED:0}

    # Apply CPU/GPU dynamic limit
    max_rows = CSV_LATEST_PREDICTIONS_LIMIT
    df = df.head(max_rows)

    for _, row in df.iterrows():
        txn = row.to_dict()
        txn["txn_id"] = txn.get("txn_id") or (TXN_PREFIX + uuid.uuid4().hex[:10].upper())
        try:
            res = core.process_payment(txn)
        except Exception as e:
            logger.exception("processing CSV row failed")
            res = {"txn_id": txn["txn_id"], "decision": DECISION_REJECTED, "fraud_prob": 1.0, "reason": f"processing error: {e}"}
        res = _ensure_result_fields(res, txn)
        results.append(res)
        d = res.get("decision", "")
        if d in counts:
            counts[d] += 1
        if res.get("txn_id") and res.get("decision"):
            _save_any_decision(res["txn_id"], res["decision"], res.get("fraud_prob", 0.0),
                               float(res.get("amount", 0.0) or 0.0), res.get("name",""))

    return {"results": results, "summary": counts}

# --- Refund Payment ---
@app.post("/api/payments/refund")
def refund_payment(req: RefundRequest):
    txn_id = req.txn_id
    tx = _get_txn_from_db(txn_id)
    if not tx:
        return {"refunded": False, "txn_id": txn_id, "error": "Transaction not found."}

    decision = tx.get("decision_text") or tx.get("decision") or ""
    if DECISION_APPROVED not in decision:
        return {
            "refunded": False,
            "txn_id": txn_id,
            "error": f"Only '{DECISION_APPROVED}' transactions can be refunded. Current status: {decision}"
        }

    if tx.get("refunded") or tx.get("refunded", False):
        return {"refunded": False, "txn_id": txn_id, "error": "Transaction already refunded."}

    amount = float(tx.get("amount", tx.get("refunded_amount", 0.0) or 0.0))
    ok = _mark_refunded_in_db(txn_id, amount)
    if not ok:
        return {"refunded": False, "txn_id": txn_id, "error": "Failed to mark transaction as refunded in DB."}

    try:
        try:
            core.process_refund(txn_id, {})
        except TypeError:
            try:
                core.process_refund(txn_id)
            except Exception:
                logger.exception("core.process_refund alt call failed")
        except Exception:
            logger.exception("core.process_refund failed")
    except Exception:
        logger.exception("core.process_refund overall failure")

    return {"refunded": True, "txn_id": txn_id, "amount": amount}

# --- Latest Transactions ---
@app.get("/api/payments/latest")
def latest_transactions(limit: int = None):
    # Dynamic default based on CPU/GPU
    if limit is None:
        limit = DEFAULT_LATEST_LIMIT

    rows = []
    try:
        rows = db.get_latest_predictions(limit)
    except Exception:
        logger.exception("db.get_latest_predictions failed")

    normalized = []
    for r in rows:
        d = _as_dict_from_row(r)
        normalized.append({
            "transaction_id": d.get("transaction_id") or d.get("txn_id"),
            "prediction": d.get("prediction"),
            "fraud_prob": float(d.get("fraud_prob") or d.get("prediction_prob") or 0.0),
            "decision_text": d.get("decision_text") or d.get("decision"),
            "amount": float(d.get("amount") or 0.0),
            "refunded": bool(d.get("refunded", False)),
            "refunded_amount": float(d.get("refunded_amount") or 0.0),
            "refunded_at": (d.get("refunded_at").isoformat() if hasattr(d.get("refunded_at"), "isoformat") else d.get("refunded_at")),
            "created_at": (d.get("created_at").isoformat() if hasattr(d.get("created_at"), "isoformat") else d.get("created_at"))
        })
    return normalized

# --- Get Specific Transaction ---
@app.get("/api/payments/{txn_id}")
def get_transaction(txn_id: str):
    tx = _get_txn_from_db(txn_id)
    if not tx:
        return {"error": "Transaction not found"}
    return {
        "transaction_id": tx.get("transaction_id") or tx.get("txn_id"),
        "prediction": tx.get("prediction"),
        "fraud_prob": float(tx.get("fraud_prob") or tx.get("prediction_prob") or 0.0),
        "decision_text": tx.get("decision_text") or tx.get("decision"),
        "amount": float(tx.get("amount", 0.0)),
        "refunded": bool(tx.get("refunded", False)),
        "refunded_amount": float(tx.get("refunded_amount", 0.0)),
        "refunded_at": tx.get("refunded_at"),
        "created_at": (tx.get("created_at").isoformat() if hasattr(tx.get("created_at"), "isoformat") else tx.get("created_at"))
    }

