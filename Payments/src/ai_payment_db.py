# ai_payment_db.py
"""
ai_payment_db.py
Author: Julia Wen (patched)
Date: 2025-10-04

Postgres helper for AI payment demo.

Minimal patch:
- save_batch_predictions now assigns a generated txn_id for items missing txn_id
  instead of skipping them. This avoids "Skipping batch item without txn_id"
  which caused only Approved rows (which had txn_id) to appear in the DB.
- No other logic changed.
"""
import os
import uuid
from datetime import datetime
import logging

from sqlalchemy import create_engine, text

logger = logging.getLogger("ai_payment_db")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# DB URI: allow override with env var
DB_URI = os.environ.get("PAYMENTS_DB_URI", "postgresql+psycopg2://demo_user:demo_pass@localhost:5432/payments")
engine = create_engine(DB_URI, pool_pre_ping=True)

# Decision mapping: numeric -> text
DECISION_MAP = {1: "Approved ✅", 2: "Flagged ⚠️", 3: "Rejected ❌"}

def _create_table_and_columns():
    create_sql = """
    CREATE TABLE IF NOT EXISTS payment_predictions (
        transaction_id TEXT PRIMARY KEY,
        name TEXT,
        prediction INT NOT NULL,
        prediction_prob FLOAT NOT NULL,
        decision_text TEXT,
        amount FLOAT DEFAULT 0.0,
        refunded BOOLEAN DEFAULT FALSE,
        refunded_amount FLOAT DEFAULT 0.0,
        refunded_at TIMESTAMP NULL,
        created_at TIMESTAMP NOT NULL
    );
    """
    alters = [
        "ALTER TABLE payment_predictions ADD COLUMN IF NOT EXISTS name TEXT;",
        "ALTER TABLE payment_predictions ADD COLUMN IF NOT EXISTS decision_text TEXT;",
        "ALTER TABLE payment_predictions ADD COLUMN IF NOT EXISTS amount FLOAT DEFAULT 0.0;",
        "ALTER TABLE payment_predictions ADD COLUMN IF NOT EXISTS refunded BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE payment_predictions ADD COLUMN IF NOT EXISTS refunded_amount FLOAT DEFAULT 0.0;",
        "ALTER TABLE payment_predictions ADD COLUMN IF NOT EXISTS refunded_at TIMESTAMP NULL;"
    ]
    try:
        with engine.begin() as conn:
            conn.execute(text(create_sql))
            for a in alters:
                conn.execute(text(a))
        logger.info("[DB] payment_predictions table ready")
    except Exception as e:
        logger.exception("Failed to ensure payment_predictions table: %s", e)

_create_table_and_columns()

def _normalize_decision(decision):
    """
    Normalize arbitrary decision input to (decision_text, decision_code).
    Accepts ints, floats, strings, booleans, and emoji-containing strings.
    """
    if decision is None:
        return DECISION_MAP[3], 3

    # numeric code
    try:
        d = int(decision)
        if d in DECISION_MAP:
            return DECISION_MAP[d], d
    except Exception:
        pass

    # boolean
    if isinstance(decision, bool):
        return (DECISION_MAP[1], 1) if decision else (DECISION_MAP[3], 3)

    s = str(decision).strip().lower()

    # reject keywords
    if any(k in s for k in ["❌", "reject", "declin", "fail", "fraud", "no", "deny"]):
        return DECISION_MAP[3], 3
    # flagged keywords
    if any(k in s for k in ["⚠", "flag", "warning", "suspicious", "review"]):
        return DECISION_MAP[2], 2
    # approve keywords
    if any(k in s for k in ["✅", "approve", "approved", "yes", "ok", "legit", "pass"]):
        return DECISION_MAP[1], 1

    # emoji catches
    if "⚠" in s:
        return DECISION_MAP[2], 2
    if "❌" in s:
        return DECISION_MAP[3], 3
    if "✅" in s:
        return DECISION_MAP[1], 1

    # fallback -> Rejected (safe default)
    return DECISION_MAP[3], 3

def _row_to_dict(row):
    """Convert SQLAlchemy Row or mapping to a plain dict with expected keys."""
    try:
        if hasattr(row, "_mapping"):
            m = dict(row._mapping)
        else:
            try:
                m = dict(row)
            except Exception:
                # fallback to attribute access
                m = {k: getattr(row, k) for k in getattr(row, "keys", lambda: [])()}
        return {
            "transaction_id": m.get("transaction_id") or m.get("txn_id"),
            "name": m.get("name"),
            "prediction": m.get("prediction"),
            "fraud_prob": float(m.get("prediction_prob") or m.get("fraud_prob") or 0.0),
            "decision_text": m.get("decision_text") or m.get("decision"),
            "amount": float(m.get("amount") or 0.0),
            "refunded": bool(m.get("refunded") or False),
            "refunded_amount": float(m.get("refunded_amount") or 0.0),
            "refunded_at": m.get("refunded_at"),
            "created_at": m.get("created_at")
        }
    except Exception as e:
        logger.exception("_row_to_dict failed: %s", e)
        return {"raw": str(row)}

def save_single_prediction(txn_id: str, decision, fraud_prob: float, amount: float = 0.0, name: str = None):
    """
    Save or update a single prediction.
    - decision: can be "Approved ✅", 1, "Flagged ⚠️", etc.
    """
    try:
        decision_text, decision_code = _normalize_decision(decision)
        created_at = datetime.utcnow()

        sql = """
        INSERT INTO payment_predictions
          (transaction_id, name, prediction, prediction_prob, decision_text, amount, refunded, refunded_amount, refunded_at, created_at)
        VALUES (:tid, :name, :pred, :prob, :dt, :amt, :ref, :refamt, :refts, :ts)
        ON CONFLICT (transaction_id) DO UPDATE
          SET name = COALESCE(EXCLUDED.name, payment_predictions.name),
              prediction = COALESCE(EXCLUDED.prediction, payment_predictions.prediction),
              prediction_prob = COALESCE(EXCLUDED.prediction_prob, payment_predictions.prediction_prob),
              decision_text = COALESCE(EXCLUDED.decision_text, payment_predictions.decision_text),
              amount = COALESCE(EXCLUDED.amount, payment_predictions.amount),
              created_at = EXCLUDED.created_at;
        """
        with engine.begin() as conn:
            conn.execute(
                text(sql),
                {
                    "tid": txn_id,
                    "name": name,
                    "pred": int(decision_code),
                    "prob": float(fraud_prob),
                    "dt": str(decision_text),
                    "amt": float(amount or 0.0),
                    "ref": False,
                    "refamt": 0.0,
                    "refts": None,
                    "ts": created_at
                }
            )
        logger.debug("Saved txn %s -> %s (code=%s) amount=%.2f", txn_id, decision_text, decision_code, float(amount or 0.0))
        return True
    except Exception as e:
        logger.exception("save_single_prediction failed: %s", e)
        return False

def save_batch_predictions(results: list):
    """
    Save a batch of results into the DB.

    Minimal, targeted fix:
    - If an item is missing txn_id, generate one and attach it to the item so it
      will be saved (Rejected/Flagged rows often arrived with txn_id=None).
    - This prevents skipping valid rows and ensures GUI sees all decisions.
    """
    for r in results:
        try:
            txn_id = r.get("txn_id") or r.get("transaction_id")
            if not txn_id:
                # generate a stable-looking txn id and attach to the record
                txn_id = "TXN-" + uuid.uuid4().hex[:10].upper()
                r["txn_id"] = txn_id
                logger.info("Assigned generated txn_id for batch item: %s (name=%s)", txn_id, r.get("name"))
            decision = r.get("decision") or r.get("decision_text") or r.get("status")
            prob = r.get("fraud_prob", r.get("prediction_prob", 1.0))
            amt = r.get("amount", 0.0)
            name = r.get("name", None)
            save_single_prediction(txn_id, decision, prob, amt, name)
        except Exception:
            logger.exception("save_batch_predictions item failed: %s", r)

def get_latest_predictions(limit: int = 50):
    """
    Returns list[dict] with keys:
    transaction_id, name, prediction, fraud_prob, decision_text, amount, refunded, refunded_amount, refunded_at, created_at
    """
    try:
        sql = """
        SELECT transaction_id, name, prediction, prediction_prob, decision_text, amount, refunded, refunded_amount, refunded_at, created_at
        FROM payment_predictions
        ORDER BY created_at DESC
        LIMIT :limit;
        """
        with engine.begin() as conn:
            rows = conn.execute(text(sql), {"limit": limit}).fetchall()
        out = []
        for r in rows:
            out.append(_row_to_dict(r))
        return out
    except Exception as e:
        logger.exception("get_latest_predictions failed: %s", e)
        return []

def get_transaction(txn_id: str):
    """
    Return normalized dict for single txn_id or None.
    """
    try:
        sql = """
        SELECT transaction_id, name, prediction, prediction_prob, decision_text, amount, refunded, refunded_amount, refunded_at, created_at
        FROM payment_predictions WHERE transaction_id = :tid LIMIT 1;
        """
        with engine.begin() as conn:
            row = conn.execute(text(sql), {"tid": txn_id}).fetchone()
        if not row:
            return None
        return _row_to_dict(row)
    except Exception as e:
        logger.exception("get_transaction failed: %s", e)
        return None

def mark_refunded(txn_id: str, amount_refunded: float = 0.0):
    """
    Mark transaction as refunded and set refunded_amount and refunded_at.
    """
    try:
        ts = datetime.utcnow()
        sql = """
        UPDATE payment_predictions
        SET refunded = TRUE, refunded_amount = :amt, refunded_at = :ts
        WHERE transaction_id = :tid;
        """
        with engine.begin() as conn:
            conn.execute(text(sql), {"amt": float(amount_refunded), "ts": ts, "tid": txn_id})
        logger.info("Marked refunded %s amount=%.2f", txn_id, float(amount_refunded))
        return True
    except Exception as e:
        logger.exception("mark_refunded failed: %s", e)
        return False

