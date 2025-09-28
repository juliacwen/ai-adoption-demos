# AI Adoption Demos — Payments Module
=======

# AI Adoption Demos

**Author:** Julia Wen (wendigilane@gmail.com)  
**License:** MIT

## Overview
This repository demonstrates AI adoption in payment workflows. The `Payments` module provides a **Streamlit GUI** for processing payments, including single transactions, refund and batch CSV uploads, with fraud detection powered by a trained model.

### Features
- Single transaction input (card or wallet)
- Batch upload of CSV files
- Displays fraud probability, decision, and reason
- Admin refund panel
- Retrain or inspect the fraud detection model

The fraud detection model is stored in `Payments/fraud_model_artifact.joblib`.  
**Delete this file to force retraining on the next run.**

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
├── ai_payment_core.py      # Core API and payment processing logic
├── ai_payment_data.py      # Data generation and model training helper
├── ai_payment_gui.py       # Streamlit GUI
├── gen_train_csv.py        # Script to generate synthetic training CSVs
└── fraud_model_artifact.joblib  # Trained model artifact (auto-generated)
```

## Usage
- Enter single payments through the GUI or upload a CSV for batch processing.
- Fraud decisions include `Approved ✅`, `Flagged ⚠️`, `Rejected ❌`.
- Refunds can be processed in the admin panel.

## Notes
- Some code assisted by AI tools; core logic verified manually

---

**MIT License**
This repo contains **practical AI demos** showing how AI can be applied in real-world workflows.
