# ai_payment_gui_restfulapi.py
"""
AI Payment GUI (REST API client)
Author: Julia Wen  (wendigilane@gmail.com)
Date: 2025-10-03

Streamlit GUI that calls the REST API. Preserves original layout, role selector,
batch table, refund amount display, and DB panel.
"""
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import requests
import io

API_BASE = "http://127.0.0.1:8000/api/payments"

st.set_page_config(page_title="AI Payment Demo (API)", layout="wide")

# Sidebar role selector restored
role = st.sidebar.selectbox("Select Role", ["Admin", "Customer"], index=0, key="role_selector")
st.session_state['role'] = role

# ---------------------------
# Top page title
# ---------------------------
st.title("AI Payment Demo")

# ---------------------------
# Session state initialization
# ---------------------------
if 'transactions' not in st.session_state:
    st.session_state['transactions'] = {}
if 'active_panel' not in st.session_state:
    st.session_state['active_panel'] = "Payment"

# ---------------------------
# CSS for buttons and cards
# ---------------------------
st.markdown("""
<style>
/* --- Primary action buttons (submit, process, download) --- */
div.stButton > button,
div.stDownloadButton > button,
div.stFormSubmitButton > button,
button[data-baseweb="button"] {
    background-color: #1976D2 !important;
    color: white !important;
    font-weight: 800 !important;
    border-radius: 12px !important;
    padding: 14px 24px !important;
    border: none !important;
    cursor: pointer !important;
    font-size: 24px !important;
    min-width: 130px !important;
    white-space: nowrap;
    transition: background-color 0.2s ease;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover,
div.stFormSubmitButton > button:hover,
button[data-baseweb="button"]:hover {
    background-color: #1565C0 !important;
}

/* --- Navigation buttons (Payment / Refund / Batch) --- */
div[data-testid="stHorizontalBlock"] div.stButton > button {
    background-color: #1976D2 !important;  /* primary shade  */
}
div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
    background-color: #2196F3 !important;  /* hover = lighter blue */
}

/* --- Active nav button --- */
.active-nav {
    background-color: #0D47A1 !important;  /* darkest blue */
    font-weight: 800 !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 14px rgba(0,0,0,0.4) !important;
    transform: scale(1.02);  /* subtle emphasis */
}

/* --- Cards for AI decision messages --- */
.success-card, .warning-card, .error-card {
    padding: 12px;
    border-radius: 8px;
    font-weight: 700;
    margin-top: 10px;
}
.success-card { background-color: #4CAF50; color: white; }
.warning-card { background-color: #FFC107; color: black; }
.error-card   { background-color: #F44336; color: white; }

/* --- Panel header styling --- */
.panel-header {
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Top navigation bar
# ---------------------------
col1, spacer1, col2, spacer2, col3, col4 = st.columns([1.8,0.2,1.8,0.2,1.8,6])

with col1:
    nav_payment = st.button("💰 Payment", key="nav_payment_btn")
    if nav_payment:
        st.session_state['active_panel'] = "Payment"
    if st.session_state['active_panel'] == "Payment":
        st.markdown("<style>#nav_payment_btn button{background-color:#0D47A1 !important; font-weight:900 !important; color:white !important; box-shadow:0 0 12px rgba(0,0,0,0.4) !important; transform:scale(1.02);}</style>", unsafe_allow_html=True)

if st.session_state['role'] == "Admin":
    with col2:
        nav_refund = st.button("↩️ Refund", key="nav_refund_btn")
        if nav_refund:
            st.session_state['active_panel'] = "Refund"
        if st.session_state['active_panel'] == "Refund":
            st.markdown("<style>#nav_refund_btn button{background-color:#0D47A1 !important; font-weight:900 !important; color:white !important; box-shadow:0 0 12px rgba(0,0,0,0.4) !important; transform:scale(1.02);}</style>", unsafe_allow_html=True)

    with col3:
        nav_batch = st.button("📊 Batch", key="nav_batch_btn")
        if nav_batch:
            st.session_state['active_panel'] = "Batch Processing"
        if st.session_state['active_panel'] == "Batch Processing":
            st.markdown("<style>#nav_batch_btn button{background-color:#0D47A1 !important; font-weight:900 !important; color:white !important; box-shadow:0 0 12px rgba(0,0,0,0.4) !important; transform:scale(1.02);}</style>", unsafe_allow_html=True)

# Right-aligned session counter (Admin only)
if st.session_state['role'] == "Admin":
    with col4:
        st.markdown(
            f"<div style='text-align:right;'>Session Transactions: <b>{len(st.session_state['transactions'])}</b></div>",
            unsafe_allow_html=True
        )

st.markdown("---")

# Helper to call API defensively
def call_api(method: str, path: str, **kwargs):
    url = f"{API_BASE}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=15, **kwargs)
    except Exception as e:
        st.error(f"API request failed: {e}")
        return None
    # If status >=400 show helpful info
    if resp.status_code >= 400:
        try:
            j = resp.json()
            st.error(f"API error ({resp.status_code}): {j}")
        except Exception:
            st.error(f"API error ({resp.status_code}): {resp.text[:800]}")
        return None
    text = resp.text or ""
    # If it's JSON-like, parse
    try:
        if resp.headers.get("content-type","").startswith("application/json") or text.strip().startswith("{") or text.strip().startswith("["):
            return resp.json()
    except Exception as e:
        st.error(f"Failed to parse JSON response: {e} — raw: {text[:800]}")
        return None
    # fallback: if non-empty text, try to parse as JSON string list
    if text:
        try:
            parsed = json.loads(text)
            return parsed
        except Exception:
            st.error(f"API returned non-JSON: {text[:800]}")
            return None
    st.error("API returned empty response")
    return None

# ---------------------------
# Payment panel
# ---------------------------
if st.session_state['active_panel'] == "Payment":
    st.markdown('<div class="panel-header">Single Payment — Customer View</div>', unsafe_allow_html=True)
    payment_type = st.selectbox("Select Payment Type", ["Card","Apple Pay","Google Pay","PayPal"], key="payment_type_select")
    with st.form("payment_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Full name", placeholder="Jane Doe")
            email = st.text_input("Email", placeholder="janedoe@example.com")
        with c2:
            amount = st.number_input("Amount ($)", min_value=0.5, step=0.5, value=10.0)
            if payment_type == "Card":
                card_number = st.text_input("Card number", placeholder="4242424242424242")
                cvv = st.text_input("CVV", type="password", placeholder="123")
                expiry = st.text_input("Expiry (MM/YYYY)", placeholder="12/2030")
                token = ""
            else:
                token = st.text_input("Wallet token (tok_...)", placeholder="tok_apple_success123")
                card_number = cvv = expiry = ""
        submitted = st.form_submit_button("💥 Submit Payment")
    if submitted:
        payload = {
            "name": name, "email": email, "amount": float(amount),
            "payment_method": payment_type, "card_number": card_number,
            "cvv": cvv, "expiry": expiry, "token": token,
            "device": "customer_ui", "country": "US"
        }
        res = call_api("post", "/predict", json=payload)
        if res:
            txn_id = res.get("txn_id")
            if txn_id:
                st.session_state['transactions'][txn_id] = res
            if not res.get("valid", True):
                st.markdown(f'<div class="error-card">{res.get("reason","Invalid payment details")}</div>', unsafe_allow_html=True)
            else:
                fraud_prob = res.get("fraud_prob",1.0)
                decision = res.get("decision","Rejected ❌")
                color_class = "success-card" if decision=="Approved ✅" else ("warning-card" if "Flagged" in decision else "error-card")
                st.markdown(f'<div class="{color_class}">AI Decision: {decision} | Fraud Probability: {fraud_prob:.2f}</div>', unsafe_allow_html=True)

# ---------------------------
# Refund panel (Admin)
# ---------------------------
elif st.session_state['active_panel'] == "Refund" and st.session_state['role'] == "Admin":
    st.markdown('<div class="panel-header">Refund (Admin)</div>', unsafe_allow_html=True)
    with st.form("refund_form"):
        refund_txn = st.text_input("Transaction ID to refund (TXN-...)", key="refund_txn_input")
        refund_submit = st.form_submit_button("💥 Process Refund", key="refund_submit_btn")
    if refund_submit:
        res = call_api("post", "/refund", json={"txn_id": refund_txn})
        if res:
            if res.get("refunded"):
                amt = res.get("amount", res.get("amount_refunded", 0.0))
                st.markdown(f'<div class="success-card">Refund processed for {refund_txn} | Amount Refunded: ${float(amt):.2f}</div>', 
unsafe_allow_html=True)
                if refund_txn in st.session_state['transactions']:
                    st.session_state['transactions'][refund_txn]['refunded'] = True
            else:
                err = res.get("error") or f"Refund failed for {refund_txn}"
                st.markdown(f'<div class="error-card">Refund failed: {err}</div>', unsafe_allow_html=True)

# ---------------------------
# Batch panel (Admin)
# ---------------------------
elif st.session_state['active_panel'] == "Batch Processing" and st.session_state['role'] == "Admin":
    st.markdown('<div class="panel-header">Batch Processing (Upload CSV)</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv_uploader")
    if st.button("Download example CSV", key="download_example_btn"):
        # try to generate synthetic if available
        try:
            from ai_payment_core import generate_synthetic_transactions
            df_example = generate_synthetic_transactions(n=200)
        except Exception:
            df_example = pd.DataFrame([{
                "name":"Jane Doe","email":"janedoe@example.com","amount":10.0,
                "payment_method":"Card","card_number":"4242424242424242","cvv":"123","expiry":"12/2030","token":""
            }])
        st.download_button("Download CSV", data=df_example.to_csv(index=False).encode(), file_name="example_batch_transactions.csv", 
key="download_csv_btn")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.markdown("### Preview uploaded file")
            st.dataframe(df.head(50), use_container_width=True)
            if st.button("💥 Process Batch Payments"):
                # preferred: upload the file bytes to /upload_csv so API retains original parsing
                files = {"file": ("upload.csv", uploaded.getvalue(), "text/csv")}
                res = call_api("post", "/upload_csv", files=files)
                if res:
                    results = res.get("results", []) if isinstance(res, dict) else res.get("results", []) if isinstance(res, dict) else res
                    # normalize: ensure results is a list of dicts
                    if isinstance(results, dict) and "results" in results:
                        results = results["results"]
                    # ensure name column exists
                    result_df = pd.DataFrame(results)
                    if "name" not in result_df.columns:
                        result_df["name"] = ""
                    # update session transactions for approved
                    for r in results:
                        tid = r.get("txn_id") or r.get("transaction_id")
                        if tid:
                            st.session_state['transactions'][tid] = r
                    st.success("Batch processed")
                    st.dataframe(result_df, use_container_width=True)
                    # chart
                    try:
                        chart_df = result_df.copy()
                        chart_df["NameShort"] = chart_df["name"].astype(str).str.slice(0,20)
                        decision_color_scale = alt.Scale(
                            domain=["Approved ✅", "Flagged ⚠️", "Rejected ❌"],
                            range=["#4CAF50", "#FFC107", "#F44336"]
                        )
                        chart = alt.Chart(chart_df).mark_bar().encode(
                            x=alt.X("NameShort:N", sort=None),
                            y=alt.Y("fraud_prob:Q"),
                            color=alt.Color("decision:N", scale=decision_color_scale),
                            tooltip=["txn_id","name","amount","payment_method","fraud_prob","decision","reason"]
                        ).properties(height=360)
                        st.altair_chart(chart, use_container_width=True)
                    except Exception:
                        # chart optional
                        pass
        except Exception as e:
            st.markdown(f'<div class="error-card">Failed to process CSV locally: {e}</div>', unsafe_allow_html=True)

# ---------------------------
# DB Stored Predictions panel (Admin)
# ---------------------------
if st.session_state['role'] == "Admin" and st.session_state['active_panel'] != "Payment":
    st.markdown("<hr><h3>DB Stored Predictions (Latest 50)</h3>", unsafe_allow_html=True)
    res = call_api("get", "/latest", params={"limit": 50})
    if res:
        # res should be a list (API returns list). If API returns dict with key, handle.
        if isinstance(res, dict) and "transactions" in res:
            txns = res["transactions"]
        else:
            txns = res
        try:
            df_db = pd.DataFrame(txns)
            if not df_db.empty:
                if "created_at" in df_db.columns:
                    try:
                        df_db['created_at'] = pd.to_datetime(df_db['created_at']).dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass
                st.dataframe(df_db, use_container_width=True)
            else:
                st.info("No predictions saved yet.")
        except Exception:
            st.write(txns)

