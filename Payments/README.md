# AI Adoption Demos — Payments Module
=======

# AI Adoption Demos

**Author:** Julia Wen (wendigilane@gmail.com)  
**License:** MIT

## Overview
This repository demonstrates AI adoption in payment workflows. The `Payments` module provides a **Streamlit GUI** for processing payments, including single transactions, refund and batch CSV uploads, with fraud detection powered by a trained model, PostgreSQL database, and yaml config, and RESTful API with FastAPI.

### Features
- Single transaction input (card or wallet)
- Batch upload of CSV files
- Displays fraud probability, decision, and reason
- Admin refund panel
- Retrain or inspect the fraud detection model
- Generates synthetic transactions for testing and training

The fraud detection model is stored in `Payments/fraud_model_artifact.joblib`.  
**Delete this file to force retraining on the next run.**  

### Transaction Storage

Transactions are stored in a **PostgreSQL database**, which is used by the Streamlit GUI for both single and batch transaction processing.  

For testing and demonstration, a **synthetic CSV database** (`Payments/sandbox_transactions.csv`) is also generated automatically if missing:  
- Used for batch upload demos and model training.  
- Safe for testing — all card numbers are test-only.  
- Delete the CSV to regenerate new synthetic transactions.

The PostgreSQL database stores the same fields as the CSV:  
`name, email, amount, payment_method, card_number, cvv, expiry, token, device, country, hour, ip_risk, past_txns, fraud_prob_label, outcome`  
It is required for the GUI to display transaction history and fraud decisions.

### Notes
- All card numbers are test-only (see [Stripe testing](https://stripe.com/docs/testing)).
- No real payments are processed.
- The demo is for educational purposes and illustrating AI adoption in payment workflows.

## Installation
1. Clone the repository:
```bash
git clone <your-repo-url>
cd ai-adoption-demos
```
2. Create a virtual environment:
```bash
python3 -m venv venv_demo
source venv_demo/bin/activate  # macOS/Linux
venv_demo\Scripts\activate     # Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the GUI
```bash
streamlit run Payments/ai_payment_gui.py
```

Open the displayed URL in your browser to interact with the dashboard.

## File Structure
```
Payments/
├── ai_payment_api.py            # RESTful API
├── ai_payment_core.py           # Core API and payment processing logic
├── ai_payment_db.py             # PostgreSQL database
├── ai_payment_data.py           # Data generation and model training helper
├── ai_payment_gui.py            # Streamlit GUI
├── gen_train_csv.py             # Script to generate synthetic training CSVs
├── sandbox_transactions.csv     # Synthetic transactions database (auto-generated)
└── fraud_model_artifact.joblib  # Trained model artifact (auto-generated)
```

## Usage
- Enter single payments through the GUI or upload a CSV for batch processing.
- Fraud decisions include `Approved ✅`, `Flagged ⚠️`, `Rejected ❌`.
- Refunds can be processed in the admin panel.
- Regenerate synthetic transactions by deleting `sandbox_transactions.csv`.
- Retrain the model by deleting `fraud_model_artifact.joblib`.
- PostgreSQL must be running for transaction history display and GUI functionality.

## Notes
- AI tools were used in assisting with reviewing, refining, and enhancing portions of the codebase.
- Synthetic transaction generation and labeling are fully configurable via `Payments/ai_payment_data.py`.

---

**MIT License**  
This repo contains **practical AI demos** showing how AI can be applied in real-world workflows.
