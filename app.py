import sqlite3
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from src.features import build_features

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "churn.db"
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
SHAP_IMG_PATH = BASE_DIR / "reports" / "shap_summary.png"

st.set_page_config(page_title="Telco Churn & Retention", layout="wide")
st.title("Telco Customer Churn & Retention Strategy")
st.caption("SQLite Analytics • Feature Engineering • XGBoost + SHAP")


@st.cache_data
def load_customers_df() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        raw = pd.read_sql_query("SELECT * FROM customers;", conn)
    return build_features(raw)


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def recommend_action(prob: float, intl_plan: str, service_calls: int) -> str:
    if prob >= 0.70 or service_calls >= 4:
        return "Priority retention outreach: escalate open support tickets, assign specialist follow-up, and offer a loyalty credit."
    if intl_plan == "yes" and prob >= 0.40:
        return "Proactive plan check-in: offer a discounted international bundle or usage alert before high charges build up."
    if prob >= 0.40:
        return "Moderate risk: share tailored bundle options and check recent usage or billing changes."
    return "Low risk: maintain standard engagement and periodic check-ins."


df = load_customers_df()
bundle = load_model_bundle()

if df.empty or bundle is None:
    st.error("Missing data/churn.db or models/churn_model.joblib. Run load_to_sql.py and train.py first.")
    st.stop()

# Top KPI row
total_cust = len(df)
churn_rate = df["churn_flag"].mean() * 100 if "churn_flag" in df.columns else 0.0
intl_churn = (
    df.loc[df["intl_plan_flag"] == 1, "churn_flag"].mean() * 100
    if {"intl_plan_flag", "churn_flag"}.issubset(df.columns)
    else 0.0
)
high_call_churn = (
    df.loc[df["high_service_calls_flag"] == 1, "churn_flag"].mean() * 100
    if {"high_service_calls_flag", "churn_flag"}.issubset(df.columns)
    else 0.0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{total_cust:,}")
c2.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
c3.metric("Intl Plan Churn", f"{intl_churn:.2f}%")
c4.metric("4+ Service Calls Churn", f"{high_call_churn:.2f}%")

tab1, tab2, tab3 = st.tabs(["Live Churn Scorer", "SQL & Cohort Insights", "SHAP Explainability"])

with tab1:
    st.subheader("Score a Customer Profile")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        state = st.selectbox("State", sorted(df["state"].dropna().unique().tolist()) if "state" in df.columns else ["CA"])
        account_length = st.slider("Account Length (days)", 1, 250, int(df["account_length"].median()) if "account_length" in df.columns else 100)
        intl_plan = st.selectbox("International Plan", ["no", "yes"])
        vmail_plan = st.selectbox("Voice Mail Plan", ["no", "yes"])
        vmail_msgs = st.slider("Voicemail Messages", 0, 60, int(df["number_vmail_messages"].median()) if "number_vmail_messages" in df.columns else 0)

    with col_b:
        day_mins = st.number_input("Total Day Minutes", 0.0, 400.0, float(df["total_day_minutes"].median()) if "total_day_minutes" in df.columns else 180.0)
        day_calls = st.number_input("Total Day Calls", 0, 200, int(df["total_day_calls"].median()) if "total_day_calls" in df.columns else 100)
        eve_mins = st.number_input("Total Eve Minutes", 0.0, 400.0, float(df["total_eve_minutes"].median()) if "total_eve_minutes" in df.columns else 200.0)
        eve_calls = st.number_input("Total Eve Calls", 0, 200, int(df["total_eve_calls"].median()) if "total_eve_calls" in df.columns else 100)

    with col_c:
        night_mins = st.number_input("Total Night Minutes", 0.0, 450.0, float(df["total_night_minutes"].median()) if "total_night_minutes" in df.columns else 200.0)
        night_calls = st.number_input("Total Night Calls", 0, 200, int(df["total_night_calls"].median()) if "total_night_calls" in df.columns else 100)
        intl_mins = st.number_input("Total Intl Minutes", 0.0, 30.0, float(df["total_intl_minutes"].median()) if "total_intl_minutes" in df.columns else 10.0)
        intl_calls = st.number_input("Total Intl Calls", 0, 25, int(df["total_intl_calls"].median()) if "total_intl_calls" in df.columns else 4)
        service_calls = st.slider("Customer Service Calls", 0, 10, 1)

    # Approximate standard BigML per-minute rates when scoring custom inputs
    sample_row = pd.DataFrame(
        [
            {
                "state": state,
                "account_length": account_length,
                "area_code": int(df["area_code"].mode()[0]) if "area_code" in df.columns else 415,
                "phone_number": "000-0000",
                "international_plan": intl_plan,
                "voice_mail_plan": vmail_plan,
                "number_vmail_messages": vmail_msgs,
                "total_day_minutes": day_mins,
                "total_day_calls": day_calls,
                "total_day_charge": round(day_mins * 0.17, 2),
                "total_eve_minutes": eve_mins,
                "total_eve_calls": eve_calls,
                "total_eve_charge": round(eve_mins * 0.085, 2),
                "total_night_minutes": night_mins,
                "total_night_calls": night_calls,
                "total_night_charge": round(night_mins * 0.045, 2),
                "total_intl_minutes": intl_mins,
                "total_intl_calls": intl_calls,
                "total_intl_charge": round(intl_mins * 0.27, 2),
                "customer_service_calls": service_calls,
                "churn": "False",
            }
        ]
    )

    engineered_row = build_features(sample_row)
    X_input = engineered_row[bundle["feature_cols"]]
    churn_prob = float(bundle["pipeline"].predict_proba(X_input)[0, 1])

    st.divider()
    r1, r2 = st.columns([1, 2])
    r1.metric("Predicted Churn Probability", f"{churn_prob * 100:.1f}%")
    r2.info(f"**Recommended Retention Action:** {recommend_action(churn_prob, intl_plan, service_calls)}")

with tab2:
    st.subheader("Churn by Service Calls")
    if {"customer_service_calls", "churn_flag"}.issubset(df.columns):
        by_calls = (
            df.groupby("customer_service_calls", as_index=False)
            .agg(customers=("churn_flag", "count"), churn_rate_pct=("churn_flag", lambda s: round(s.mean() * 100, 2)))
        )
        st.dataframe(by_calls, use_container_width=True, hide_index=True)
        st.bar_chart(by_calls.set_index("customer_service_calls")["churn_rate_pct"])

with tab3:
    st.subheader("Global Feature Importance (SHAP)")
    if SHAP_IMG_PATH.exists():
        st.image(str(SHAP_IMG_PATH), caption="SHAP Summary Plot from src/explain.py", use_container_width=True)
    else:
        st.warning("reports/shap_summary.png not found yet. Run src/explain.py first.")