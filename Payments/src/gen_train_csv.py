"""
gen_train_csv.py
Author: Julia Wen
Date: 2025-09-28
Description:
Generate synthetic payment training data with Card, Apple Pay, Google Pay, PayPal.
Labels follow risk scoring logic: approved, flagged, rejected.
"""

import csv
import random
from collections import Counter, defaultdict

# ---------------- Constants ---------------- #

# Dataset
N_ROWS = 500
OUTPUT_FILE = "training_data.csv"

# Payment methods
PAYMENT_METHODS = ["Card", "Apple Pay", "Google Pay", "PayPal"]
PAYMENT_WEIGHTS = [0.6, 0.13, 0.13, 0.14]

# Stripe official test cards
STRIPE_TEST_CARDS = [
    "4242424242424242",  # Visa
    "4000056655665556",  # Visa debit
    "5555555555554444",  # Mastercard
    "2223003122003222",  # Mastercard (2-series)
    "378282246310005",   # Amex
    "6011111111111117",  # Discover
    "30569309025904",    # Diners Club
    "3566002020360505",  # JCB
]

# Countries
COUNTRIES = ["US", "GB", "CA", "AU", "IN", "NG", "RU"]
COUNTRY_PROBS = [0.3, 0.1, 0.1, 0.05, 0.05, 0.2, 0.2]  # boosted NG/RU

# Token suffixes
TOKEN_SUFFIXES = ["success", "fail", "flagged"]
TOKEN_SUFFIX_PROBS = [0.65, 0.25, 0.10]

# Risk scoring thresholds
SCORE_FLAGGED = 0.4
SCORE_REJECTED = 0.7

# Amount generation parameters (lognormal)
AMOUNT_MU = 5.0
AMOUNT_SIGMA = 1.1

# Random seed
RANDOM_SEED = 42

# ---------------- Helpers ---------------- #

def generate_card(rng: random.Random) -> str:
    """Return a Stripe test card or random 16-digit card."""
    if rng.random() < 0.8:
        return rng.choice(STRIPE_TEST_CARDS)
    return str(rng.randint(4_000_000_000_000_000, 4_999_999_999_999_999))


def generate_token(method: str, rng: random.Random) -> str:
    """Generate a synthetic token for non-Card payments."""
    suffix = rng.choices(TOKEN_SUFFIXES, TOKEN_SUFFIX_PROBS, k=1)[0]
    return f"tok_{method.replace(' ','').lower()}_{suffix}_{rng.randint(1000,9999)}"


def risk_score(row: dict, rng: random.Random) -> float:
    """Calculate a synthetic risk score for a transaction."""
    score = 0.0
    if row["amount"] > 200:
        score += 0.25
    if row["amount"] > 1000:
        score += 0.5
    if row["country"] in ["NG", "RU"]:
        score += 0.4
    if row["hour"] in (0, 1, 2, 3, 4):
        score += 0.15
    if "fail" in row["token"]:
        score += 0.6
    if "flagged" in row["token"]:
        score += 0.8
    score += rng.normalvariate(0, 0.05)  # jitter
    return min(max(score, 0.0), 1.0)


def map_outcome(score: float) -> str:
    """Map risk score to approved/flagged/rejected."""
    if score >= SCORE_REJECTED:
        return "rejected"
    if score >= SCORE_FLAGGED:
        return "flagged"
    return "approved"


# ---------------- Main ---------------- #

def generate_dataset(filename: str = OUTPUT_FILE, n_rows: int = N_ROWS):
    rng = random.Random(RANDOM_SEED)
    fieldnames = ["email", "amount", "payment_method", "card_number", "token", "country", "hour", 
"label"]
    counts = defaultdict(Counter)

    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(n_rows):
            email = f"user{i}@example.com"
            method = rng.choices(PAYMENT_METHODS, weights=PAYMENT_WEIGHTS)[0]
            amount = round(rng.lognormvariate(mu=AMOUNT_MU, sigma=AMOUNT_SIGMA), 2)
            country = rng.choices(COUNTRIES, COUNTRY_PROBS, k=1)[0]
            hour = rng.randint(0, 23)

            if method == "Card":
                card_number = generate_card(rng)
                token = ""
            else:
                card_number = ""
                token = generate_token(method, rng)

            row = {
                "email": email,
                "amount": amount,
                "payment_method": method,
                "card_number": card_number,
                "token": token,
                "country": country,
                "hour": hour,
            }

            score = risk_score(row, rng)
            outcome = map_outcome(score)
            row["label"] = outcome

            writer.writerow(row)
            counts[method][outcome] += 1

    print(f"✅ Generated {n_rows} rows in {filename}")
    print("Summary by payment method:")
    for method in PAYMENT_METHODS:
        total = sum(counts[method].values())
        approved = counts[method].get("approved", 0)
        flagged = counts[method].get("flagged", 0)
        rejected = counts[method].get("rejected", 0)
        print(f"  - {method}: total={total}, approved={approved}, flagged={flagged}, rejected={rejected}")


# ---------------- Entry ---------------- #

if __name__ == "__main__":
    generate_dataset()

