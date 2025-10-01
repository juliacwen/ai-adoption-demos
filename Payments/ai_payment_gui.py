# ai_payment_gui.py
"""
ai_payment_gui.py
Author: Julia Wen
Date: 2025-09-29
Description: GUI for AI payment demo. All Postgres DB saving is handled by ai_payment_db.py.
Includes Admin panel to view saved predictions.
"""
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from ai_payment_core import (
    process_payment,
    process_refund,
    process_batch,
    generate_synthetic_transactions,
)
from ai_payment_db import save_single_prediction, save_batch_predictions, engine

st.set_page_config(page_title="AI Payment Demo", layout="wide")

# ---------------------------
# Sidebar role selector
# ---------------------------
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
# CSS for professional buttons + cards
# ---------------------------
st.markdown("""
<style>
div.stButton > button, div.stDownloadButton > button {
    background-color:#1976D2 !important;
    color:white !important;
    font-weight:600 !important;
    border-radius:8px !important;
    padding:10px 20px !important;
    border:none !important;
    cursor:pointer !important;
    font-size:14px !important;
    min-width:110px !important;
    white-space: nowrap;
    transition: all 0.2s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    background-color:#1565C0 !important;
}
.success-card, .warning-card, .error-card {
    padding:12px; border-radius:8px; font-weight:700; margin-top:10px;
}
.success-card { background-color:#4CAF50; }
.warning-card { background-color:#FFC107; color:black; }
.error-card { background-color:#F44336; }
.panel-header { font-size:24px; font-weight:700; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Top navigation bar
# ---------------------------
col1, spacer1, col2, spacer2, col3, col4 = st.columns([1.5,0.2,1.5,0.2,1.5,6])
with col1:
    if st.button("💰 Payment", key="nav_payment_btn"):
        st.session_state['active_panel'] = "Payment"
if st.session_state['role'] == "Admin":
    with col2:
        if st.button("↩️ Refund", key="nav_refund_btn"):
            st.session_state['active_panel'] = "Refund"
    with col3:
        if st.button("📊 Batch", key="nav_batch_btn"):
            st.session_state['active_panel'] = "Batch Processing"

# Right-aligned session counter (Admin only)
if st.session_state['role'] == "Admin":
    with col4:
        st.markdown(
            f"<div style='text-align:right; display:inline-block;'>"
            f"Session Transactions: <b>{len(st.session_state['transactions'])}</b>"
            f"</div>",
            unsafe_allow_html=True
        )

st.markdown("---")

# ---------------------------
# Payment panel
# ---------------------------
if st.session_state['active_panel'] == "Payment":
    st.markdown('<div class="panel-header">Single Payment — Customer View</div>', unsafe_allow_html=True)
    payment_type = st.selectbox("Select Payment Type", ["Card","Apple Pay","Google Pay","PayPal"], key="payment_type_select")
    if payment_type:
        with st.form("payment_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Full name", placeholder="Jane Doe", key="payment_name")
                email = st.text_input("Email", placeholder="janedoe@example.com", key="payment_email")
            with c2:
                amount = st.number_input("Amount ($)", min_value=0.5, step=0.5, value=10.0, key="payment_amount")
                if payment_type == "Card":
                    card_number = st.text_input("Card number", placeholder="4242424242424242", key="payment_card")
                    cvv = st.text_input("CVV", type="password", placeholder="123", key="payment_cvv")
                    expiry = st.text_input("Expiry (MM/YYYY)", placeholder="12/2030", key="payment_expiry")
                    token = ""
                else:
                    token = st.text_input("Wallet token (e.g. tok_apple_success...)", key="payment_token")
                    card_number = cvv = expiry = ""
            submitted = st.form_submit_button("💥 Submit Payment", key="submit_payment_btn")
        if submitted:
            transaction = {
                "name": name,
                "email": email,
                "amount": float(amount),
                "payment_method": payment_type,
                "card_number": card_number.strip() if payment_type=="Card" else "",
                "cvv": cvv.strip() if payment_type=="Card" else "",
                "expiry": expiry.strip() if payment_type=="Card" else "",
                "token": token.strip() if payment_type!="Card" else "",
                "device": "customer_ui",
                "country": "US",
                "hour": datetime.utcnow().hour,
                "ip_risk": 0,
                "past_txns": 0
            }
            res = process_payment(transaction, st.session_state['transactions'], debug=True)
            txn_id = res.get("txn_id")
            if txn_id:
                st.session_state['transactions'][txn_id] = res
                save_single_prediction(txn_id, res.get("decision","Rejected ❌"), res.get("fraud_prob",1.0))

            if not res.get("valid", True):
                st.markdown(f'<div class="error-card">{res.get("reason","Invalid payment details")}</div>', unsafe_allow_html=True)
            else:
                fraud_prob = res.get("fraud_prob",1.0)
                decision = res.get("decision","Rejected ❌")
                color_class = "success-card" if decision=="Approved ✅" else ("warning-card" if decision=="Flagged ⚠️" else "error-card")
                st.markdown(f'<div class="{color_class}">AI Decision: {decision} | Fraud Probability: {fraud_prob:.2f}</div>', unsafe_allow_html=True)

# ---------------------------
# Refund panel (Admin only)
# ---------------------------
elif st.session_state['active_panel'] == "Refund" and st.session_state['role'] == "Admin":
    st.markdown('<div class="panel-header">Refund (Admin)</div>', unsafe_allow_html=True)

    with st.form("refund_form_unique"):
        refund_txn = st.text_input("Transaction ID to refund (TXN-...)", key="refund_txn_input")
        refund_submit = st.form_submit_button("💥 Process Refund", key="refund_submit_btn")

    if refund_submit:
        txn_record = st.session_state['transactions'].get(refund_txn)

        if txn_record is None:
            msg = f"Refund failed: transaction {refund_txn} not found."
            success = False
        elif txn_record.get("refunded", False):
            msg = f"Refund failed: transaction {refund_txn} was already refunded."
            success = False
        elif txn_record.get("decision") != "Approved ✅":
            msg = f"Refund failed: transaction {refund_txn} is {txn_record.get('decision')} and cannot be refunded."
            success = False
        else:
            success = process_refund(refund_txn, st.session_state['transactions'])
            msg = f"Refund processed for {refund_txn}" if success else f"Refund failed: transaction {refund_txn} could not be refunded."

        st.markdown(f'<div class="{"success-card" if success else "error-card"}">{msg}</div>', unsafe_allow_html=True)

# ---------------------------
# Batch Processing panel (Admin only)
# ---------------------------
elif st.session_state['active_panel'] == "Batch Processing" and st.session_state['role'] == "Admin":
    st.markdown('<div class="panel-header">Batch Processing (Upload CSV)</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv_uploader")

    if st.button("Download example CSV", key="download_example_btn"):
        df_example = generate_synthetic_transactions(n=200)
        st.download_button("Download CSV", data=df_example.to_csv(index=False).encode(), 
                           file_name="example_batch_transactions.csv", key="download_csv_btn")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.markdown("### Preview uploaded file")
            st.dataframe(df.head(50), use_container_width=True)
            with st.form("batch_form_unique"):
                submit = st.form_submit_button("💥 Process Batch Payments", key="batch_submit_btn")
            if submit:
                tx_list = []
                for _, r in df.iterrows():
                    tx_list.append({
                        "name": str(r.get("name","")),
                        "email": str(r.get("email","")),
                        "amount": float(r.get("amount",0.0) or 0.0),
                        "payment_method": str(r.get("payment_method","Card")),
                        "card_number": str(r.get("card_number","") or ""),
                        "cvv": str(r.get("cvv","") or ""),
                        "expiry": str(r.get("expiry","") or ""),
                        "token": str(r.get("token","") or ""),
                        "device": "batch_upload",
                        "country": "US",
                        "hour": datetime.utcnow().hour,
                        "ip_risk": 0,
                        "past_txns": 0
                    })
                results, counts = process_batch(tx_list, st.session_state['transactions'])
                st.session_state['transactions'].update({r.get("txn_id"): r for r in results if r.get("txn_id")})

                save_batch_predictions(results)

                result_df = pd.DataFrame(results)
                st.success("Batch processed")
                st.dataframe(result_df, use_container_width=True)

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

        except Exception as e:
            st.markdown(f'<div class="error-card">Failed to process CSV: {e}</div>', unsafe_allow_html=True)

# ---------------------------
# DB Stored Predictions panel (Admin only)
# ---------------------------
if st.session_state['role'] == "Admin" and st.session_state['active_panel'] != "Payment":
    st.markdown("<hr><h3>DB Stored Predictions (Latest 50)</h3>", unsafe_allow_html=True)
    try:
        df_db = pd.read_sql("SELECT * FROM payment_predictions ORDER BY created_at DESC LIMIT 50", con=engine)
        if not df_db.empty:
            df_db['created_at'] = pd.to_datetime(df_db['created_at']).dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(df_db, use_container_width=True)
        else:
            st.info("No predictions saved yet.")
    except Exception as e:
        st.warning(f"Failed to fetch from DB: {e}")

