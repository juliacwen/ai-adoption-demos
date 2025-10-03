"""
# ai_payment_api.py
Author: Julia Wen
Date: 2025-10-02
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
from typing import List
import io

import pandas as pd
import ai_payment_core as core
import ai_payment_db as db
from sqlalchemy import text

# -----------------------
# Constants
# -----------------------
DEFAULT_LATEST_LIMIT = 20  # Default number of transactions to fetch for latest predictions

# -----------------------
# FastAPI app instance
# -----------------------
app = FastAPI(title="AI Payment Fraud Full API")

# -----------------------
# Pydantic request/response schemas
# -----------------------
class PaymentRequest(BaseModel):
    """Single payment request"""
    name: str
    email: str
    amount: float
    payment_method: str = "Card"
    card_number: str = ""
    cvv: str = ""
    expiry: str = ""
    token: str = ""
    device: str = "web"
    country: str = "US"

class RefundRequest(BaseModel):
    """Refund request by transaction ID"""
    txn_id: str

class BatchRequest(BaseModel):
    """Batch of transactions"""
    transactions: List[PaymentRequest]

# -----------------------
# Helper DB functions
# -----------------------
def refund_transaction_in_db(txn_id: str) -> bool:
    """
    Mark a transaction as refunded in the Postgres DB.
    Returns True if successful, False if txn_id not found.
    """
    try:
        sql = """
        UPDATE payment_predictions
        SET decision_text = 'Refunded ♻️'
        WHERE transaction_id = :tid
        RETURNING transaction_id;
        """
        with db.engine.begin() as conn:
            row = conn.execute(text(sql), {"tid": txn_id}).fetchone()
        return row is not None
    except Exception as e:
        print(f"[DB ERROR] Failed to refund txn {txn_id}: {e}")
        return False

# -----------------------
# API Endpoints
# -----------------------

@app.post("/api/payments/predict")
def predict_payment(req: PaymentRequest):
    """
    Process a single payment transaction:
    1. Validate payment info
    2. Score fraud probability
    3. Map to decision (Approved/Flagged/Rejected)
    4. Save approved transactions to database
    """
    txn_dict = req.model_dump()  # Pydantic V2
    result = core.process_payment(txn_dict)

    # Save approved transactions to DB
    if result["decision"] == "Approved ✅" and result["txn_id"]:
        db.save_single_prediction(result["txn_id"], result["decision"], result["fraud_prob"])

    return result


@app.post("/api/payments/batch")
def batch_payment(req: BatchRequest):
    """
    Process multiple transactions in a batch:
    - Each transaction is validated, scored, and decisioned
    - Approved transactions are saved to DB
    - Returns results list and a summary count per decision
    """
    txn_list = [t.model_dump() for t in req.transactions]
    results, counts = core.process_batch(txn_list)

    # Save approved transactions
    for r in results:
        if r["decision"] == "Approved ✅" and r["txn_id"]:
            db.save_single_prediction(r["txn_id"], r["decision"], r["fraud_prob"])

    return {"results": results, "summary": counts}


@app.post("/api/payments/refund")
def refund_payment(req: RefundRequest):
    """
    Refund a transaction by txn_id stored in DB.
    Returns {"txn_id": ..., "refunded": True/False}.
    """
    refunded = refund_transaction_in_db(req.txn_id)
    return {"txn_id": req.txn_id, "refunded": refunded}


@app.post("/api/payments/upload_csv")
def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV of transactions:
    - Reads CSV file
    - Validates, scores, and decides each transaction
    - Saves approved transactions to DB
    - Returns results and summary
    """
    content = file.file.read()
    txns = core.load_transactions_csv(io.BytesIO(content))
    results, counts = core.process_batch(txns)

    # Save approved transactions
    for r in results:
        if r["decision"] == "Approved ✅" and r["txn_id"]:
            db.save_single_prediction(r["txn_id"], r["decision"], r["fraud_prob"])

    return {"results": results, "summary": counts}


@app.get("/api/payments/latest")
def latest_predictions(limit: int = DEFAULT_LATEST_LIMIT):
    """
    Fetch the latest 'limit' transactions saved in the database.
    Returns transaction ID, prediction code, fraud probability, decision text, and timestamp.
    """
    rows = db.get_latest_predictions(limit)
    return [
        {
            "transaction_id": r[0],
            "prediction": r[1],
            "fraud_prob": r[2],
            "decision_text": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]


@app.get("/api/payments/{txn_id}")
def single_transaction(txn_id: str):
    """
    Fetch a specific transaction by its txn_id.
    Returns transaction details if found, else an error message.
    """
    rows = db.get_latest_predictions(limit=1000)  # Fetch a reasonable window; replace with DB query if needed
    for r in rows:
        if r[0] == txn_id:
            return {
                "transaction_id": r[0],
                "prediction": r[1],
                "fraud_prob": r[2],
                "decision_text": r[3],
                "created_at": r[4],
            }
    return {"error": "Transaction not found"}

