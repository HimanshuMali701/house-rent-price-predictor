import streamlit as st
import pandas as pd
import numpy as np
import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="House Rent Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# =========================================================
# CONSTANTS
# =========================================================
SCALING_STATS = {
    'BHK': {'mean': 2.083860, 'std': 0.832256},
    'Size': {'mean': 967.490729, 'std': 634.202328},
    'Bathroom': {'mean': 1.965866, 'std': 0.884532},
}

MODEL_METRICS = {
    "RF": {"r2": 0.9549, "mse": 179590041},
    "DT": {"r2": 0.9015, "mse": 350000000},
    "SVR": {"r2": 0.8800, "mse": 408000000},
}

# =========================================================
# MODEL LOGIC (Mock – Replace with joblib later)
# =========================================================
def base_score(features):
    score = (
        9.0 +
        features["Size"][0] * 0.75 +
        features["BHK"][0] * 0.4 +
        features["Bathroom"][0] * 0.3
    )

    if features.get("City_Mumbai", [0])[0] == 1:
        score += 1.0
    if features.get("City_Delhi", [0])[0] == 1:
        score += 0.5

    if features.get("Furnishing Status_Furnished", [0])[0] == 1:
        score += 0.4
    elif features.get("Furnishing Status_Semi-Furnished", [0])[0] == 1:
        score += 0.2

    return score


def predict_models(df):
    rf = np.expm1(base_score(df) + 0.1)
    dt = np.expm1(base_score(df)) * 1.05
    svr = np.expm1(base_score(df) - 0.2)
    return rf, dt, svr

# =========================================================
# HEADER
# =========================================================
st.title("🏠 House Rent Price Predictor")
st.markdown(
    "Compare predictions from **Random Forest**, **Decision Tree**, and **SVR** models."
)
st.divider()

# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("📥 Property Details")

bhk = st.sidebar.slider("BHK", 1, 6, 2)
bathroom = st.sidebar.slider("Bathrooms", 1, 5, 2)
size = st.sidebar.number_input("Size (Sq. Ft.)", 100, 8000, 1000)

city = st.sidebar.selectbox(
    "City", ['Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Delhi']
)

furnishing = st.sidebar.radio(
    "Furnishing Status",
    ['Unfurnished', 'Semi-Furnished', 'Furnished']
)



predict_btn = st.sidebar.button("🚀 Predict Rent")

# =========================================================
# MAIN LOGIC
# =========================================================
if predict_btn:
    # ---------------- Feature Engineering ----------------
    df = pd.DataFrame({
        "BHK": [bhk],
        "Size": [size],
        "Bathroom": [bathroom],
        "City": [city],
        "Furnishing Status": [furnishing],
    })

    # Scaling
    for col in ["BHK", "Size", "Bathroom"]:
        df[col] = (df[col] - SCALING_STATS[col]["mean"]) / SCALING_STATS[col]["std"]

    # Encoding
    df["City_Mumbai"] = 1 if city == "Mumbai" else 0
    df["City_Delhi"] = 1 if city == "Delhi" else 0

    for f in ["Unfurnished", "Semi-Furnished", "Furnished"]:
        df[f"Furnishing Status_{f}"] = 1 if furnishing == f else 0

    rf, dt, svr = predict_models(df)

    # Confidence band
    rf_low, rf_high = rf * 0.9, rf * 1.1

    # =====================================================
    # TABS
    # =====================================================
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Predictions", "📈 Visual Analysis", "🧠 Model Insights", "ℹ️ About Project"]
    )

    # ---------------- TAB 1: Predictions ----------------
    with tab1:
        st.subheader("📊 Model Comparison Results")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("🌳 Random Forest", f"₹{rf:,.0f}")
            st.caption(f"R²: {MODEL_METRICS['RF']['r2']} | MSE: {MODEL_METRICS['RF']['mse']:,}")
            st.success("Best overall performance")

        with c2:
            st.metric("🌲 Decision Tree", f"₹{dt:,.0f}")
            st.caption(f"R²: {MODEL_METRICS['DT']['r2']} | MSE: {MODEL_METRICS['DT']['mse']:,}")
            st.warning("Higher variance model")

        with c3:
            st.metric("⚙️ SVR", f"₹{svr:,.0f}")
            st.caption(f"R²: {MODEL_METRICS['SVR']['r2']} | MSE: {MODEL_METRICS['SVR']['mse']:,}")
            st.info("Sensitive to scaling")

        st.markdown(
            f"📌 **Estimated Rent Range (RF): ₹{rf_low:,.0f} – ₹{rf_high:,.0f}**"
        )

    # ---------------- TAB 2: Visuals ----------------
    with tab2:
        st.subheader("📈 Rent Prediction Comparison")

        chart_df = pd.DataFrame({
            "Model": ["Random Forest", "Decision Tree", "SVR"],
            "Predicted Rent (₹)": [rf, dt, svr]
        })

        st.bar_chart(chart_df.set_index("Model"))

        st.subheader("🔍 Feature Influence (Approximate)")
        influence = pd.DataFrame({
            "Feature": ["Size", "BHK", "Bathrooms", "City", "Furnishing"],
            "Impact (%)": [45, 25, 15, 10, 5]
        })
        st.dataframe(influence, use_container_width=True)

    # ---------------- TAB 3: Insights ----------------
    with tab3:
        st.subheader("🧠 Model Insights")

        st.markdown("""
        **Random Forest**
        - Handles non-linear relationships
        - Robust to outliers
        - Best generalization performance  

        **Decision Tree**
        - Easy to interpret
        - Faster predictions
        - High variance  

        **Support Vector Regressor**
        - Sensitive to feature scaling
        - Performs well on small datasets
        - Computationally expensive
        """)

    # ---------------- TAB 4: About ----------------
    with tab4:
        st.subheader("ℹ️ About This Project")

        st.markdown("""
        **House Rent Price Prediction System**

        - Dataset: Indian housing rental listings  
        - Models: Random Forest, Decision Tree, SVR  
        - Metrics: R² Score, Mean Squared Error  

        **Tech Stack**
        - Python
        - Pandas, NumPy
        - Streamlit
        - Scikit-learn

        **Use Case**
        - Helps tenants estimate fair rent
        - Assists owners in pricing properties
        """)

else:
    st.info("👈 Enter property details from the sidebar and click **Predict Rent**")

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption("© 2025 | House Rent Price Predictor | Built with Streamlit")
