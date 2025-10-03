"""
ai_payment_db.py
Author: Julia Wen
Date: 2025-10-01
Description: Handles Postgres operations for AI payment demo.
Auto-creates table and normalizes decisions (Approved/Flagged/Rejected).
"""

from sqlalchemy import create_engine, text
from datetime import datetime

# Postgres connection (adjust if needed)
DB_URI = "postgresql+psycopg2://demo_user:demo_pass@localhost:5432/payments"
engine = create_engine(DB_URI, pool_pre_ping=True)

# Ensure table + column exist
def _create_table_and_columns():
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS payment_predictions (
        transaction_id TEXT PRIMARY KEY,
        prediction INT NOT NULL,
        prediction_prob FLOAT NOT NULL,
        decision_text TEXT,
        created_at TIMESTAMP NOT NULL
    );
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(create_table_sql))
            conn.execute(
                text("ALTER TABLE payment_predictions ADD COLUMN IF NOT EXISTS decision_text TEXT;")
            )
        print("[DB INFO] payment_predictions table and columns ready.")
    except Exception as e:
        print(f"[DB ERROR] Failed to create/alter table: {e}")

_create_table_and_columns()

# mapping codes
# 1 = Approved, 2 = Flagged, 3 = Rejected
DECISION_MAP = {
    1: ("Approved ✅", 1),
    2: ("Flagged ⚠️", 2),
    3: ("Rejected ❌", 3),
}

def _normalize_decision(decision):
    """
    Normalize arbitrary decision input to (decision_text, decision_code).
    Accepts ints, floats, strings, booleans, and emojis.
    """
    if decision is None:
        return DECISION_MAP[3]  # Default reject

    # Already numeric
    try:
        d = int(decision)
        if d in DECISION_MAP:
            return DECISION_MAP[d]
    except Exception:
        pass

    # Handle booleans
    if isinstance(decision, bool):
        return DECISION_MAP[1] if decision else DECISION_MAP[3]

    # Normalize string
    s = str(decision).strip().lower()

    # Match keywords / emojis
    if any(x in s for x in ["✅", "approve", "approved", "yes", "ok", "legit", "pass"]):
        return DECISION_MAP[1]
    if any(x in s for x in ["⚠", "flag", "warning", "suspicious", "review"]):
        return DECISION_MAP[2]
    if any(x in s for x in ["❌", "reject", "declin", "fail", "fraud", "no", "deny"]):
        return DECISION_MAP[3]

    # Last fallback → reject
    return DECISION_MAP[3]

def _save_prediction(txn_id: str, raw_decision, fraud_prob: float):
    try:
        print(f"[DEBUG] Input -> txn_id={txn_id}, decision={raw_decision}, prob={fraud_prob}")
        decision_text, decision_code = _normalize_decision(raw_decision)
        created_at = datetime.utcnow()

        sql = """
        INSERT INTO payment_predictions
          (transaction_id, prediction, prediction_prob, decision_text, created_at)
        VALUES (:tid, :pred, :prob, :dt, :ts)
        ON CONFLICT (transaction_id) DO UPDATE
          SET prediction = COALESCE(EXCLUDED.prediction, payment_predictions.prediction),
              prediction_prob = COALESCE(EXCLUDED.prediction_prob, payment_predictions.prediction_prob),
              decision_text = COALESCE(EXCLUDED.decision_text, payment_predictions.decision_text),
              created_at = EXCLUDED.created_at;
        """
        with engine.begin() as conn:
            conn.execute(
                text(sql),
                {"tid": txn_id, "pred": decision_code, "prob": float(fraud_prob),
                 "dt": decision_text, "ts": created_at}
            )
        print(f"[DB INFO] Saved {txn_id}: {decision_text} ({decision_code}), prob={fraud_prob:.4f}")
        return True
    except Exception as e:
        print(f"[DB WARNING] Failed to save transaction {txn_id}: {e}")
        return False
def get_latest_predictions(limit=50):
    try:
        sql = """
        SELECT transaction_id, prediction, prediction_prob, decision_text, created_at
        FROM payment_predictions
        ORDER BY created_at DESC
        LIMIT :limit;
        """
        with engine.begin() as conn:
            rows = conn.execute(text(sql), {"limit": limit}).fetchall()
        print(f"[DB INFO] Retrieved {len(rows)} rows")
        for r in rows:
            print(r)
        return rows
    except Exception as e:
        print(f"[DB ERROR] Failed to fetch latest predictions: {e}")
        return []

# Public functions
def save_single_prediction(txn_id: str, decision, fraud_prob: float):
    return _save_prediction(txn_id, decision, fraud_prob)

def save_batch_predictions(results: list):
    """
    results: list of dict-like objects; expects keys 'txn_id', 'decision', 'fraud_prob'
    Function tolerant to missing keys.
    """
    for r in results:
        txn_id = r.get("txn_id")
        if not txn_id:
            print("[DB WARN] skipping result without txn_id:", r)
            continue
        raw_dec = r.get("decision", None)
        prob = r.get("fraud_prob", r.get("prob", r.get("prediction_prob", 1.0)))
        save_single_prediction(txn_id, raw_dec, prob)

if __name__ == "__main__":
    # quick test
    rows = get_latest_predictions()
    for r in rows:
        print(r)
