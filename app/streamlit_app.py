"""
app/streamlit_app.py
────────────────────
Streamlit web app — River Water BOD Predictor

Run with:  streamlit run app/streamlit_app.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BOD Predictor — River Water Quality",
    page_icon="💧",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "best_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")
DATA_PATH    = os.path.join(BASE_DIR, "data", "river_water_quality.csv")
PLOTS_DIR    = os.path.join(BASE_DIR, "plots")

@st.cache_resource
def load_model():
    model    = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features

model, FEATURES = load_model()

# ── Helpers ───────────────────────────────────────────────────────────────────
def bod_category(bod):
    if bod < 20:
        return "🟢 Clean / Acceptable", "#27A85F"
    elif bod < 50:
        return "🟡 Moderately Polluted", "#E0A020"
    elif bod < 100:
        return "🟠 Polluted — Attention Required", "#E06020"
    else:
        return "🔴 Highly Polluted — Action Required", "#C0152A"


def gauge_color(bod):
    if bod < 20:   return "#27A85F"
    elif bod < 50: return "#E0A020"
    elif bod < 100:return "#E06020"
    else:          return "#C0152A"


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background: linear-gradient(135deg, #1B3A5C 0%, #2563A8 100%);
            padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;'>
  <h1 style='color: white; margin: 0; font-size: 2.1rem;'>💧 River Water BOD Predictor</h1>
  <p style='color: #C8DDF5; margin: 0.4rem 0 0 0; font-size: 1.05rem;'>
    Predict Biochemical Oxygen Demand from observable water quality parameters
    using Machine Learning (SVR / PLS Regression)
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔬 Predict BOD", "📊 Model Performance", "📖 About"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Enter Water Quality Parameters")
    st.caption("Adjust the sliders to match your water sample measurements, then click Predict.")

    col1, col2 = st.columns(2)

    with col1:
        pH           = st.slider("pH",              min_value=4.0,  max_value=10.0, value=7.4,  step=0.1)
        temperature  = st.slider("Temperature (°C)", min_value=5.0, max_value=42.0, value=26.0, step=0.5)
        dissolved_o2 = st.slider("Dissolved Oxygen (mg/L)", min_value=0.5, max_value=14.0, value=7.0, step=0.1)
        turbidity    = st.slider("Turbidity (NTU)",  min_value=0.5, max_value=80.0, value=12.0, step=0.5)
        conductivity = st.slider("Conductivity (µS/cm)", min_value=50.0, max_value=1000.0, value=450.0, step=10.0)

    with col2:
        total_solids = st.slider("Total Dissolved Solids (mg/L)", min_value=50.0, max_value=750.0, value=320.0, step=5.0)
        nitrates     = st.slider("Nitrates (mg/L)",   min_value=0.1, max_value=25.0, value=4.0,   step=0.1)
        phosphates   = st.slider("Phosphates (mg/L)", min_value=0.01,max_value=10.0, value=1.5,   step=0.05)
        ammonia      = st.slider("Ammonia (mg/L)",    min_value=0.05,max_value=15.0, value=2.0,   step=0.05)
        cod          = st.slider("COD (mg/L)",        min_value=10.0,max_value=300.0,value=90.0,  step=1.0)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict BOD", type="primary", use_container_width=True)

    if predict_btn:
        input_array = np.array([[pH, temperature, dissolved_o2, turbidity,
                                  conductivity, total_solids, nitrates,
                                  phosphates, ammonia, cod]])
        prediction = model.predict(input_array)
        bod_val = float(np.array(prediction).ravel()[0])
        bod_val = max(1.0, bod_val)

        label, color = bod_category(bod_val)

        st.markdown("---")
        res1, res2, res3 = st.columns(3)

        with res1:
            st.markdown(f"""
            <div style='background:{color}22; border-left:5px solid {color};
                        padding:1.2rem; border-radius:8px; text-align:center;'>
              <p style='font-size:0.9rem; color:#555; margin:0;'>Predicted BOD</p>
              <h1 style='color:{color}; font-size:3rem; margin:0.2rem 0;'>{bod_val:.1f}</h1>
              <p style='font-size:1rem; color:#555; margin:0;'>mg/L</p>
            </div>""", unsafe_allow_html=True)

        with res2:
            st.markdown(f"""
            <div style='background:#F5F8FF; border:1px solid #D0DDF5;
                        padding:1.2rem; border-radius:8px; text-align:center;'>
              <p style='font-size:0.9rem; color:#555; margin:0;'>Water Quality Status</p>
              <h3 style='color:{color}; margin:0.6rem 0;'>{label}</h3>
            </div>""", unsafe_allow_html=True)

        with res3:
            # Simple bar gauge
            pct = min(bod_val / 150.0, 1.0)
            fig, ax = plt.subplots(figsize=(3.5, 1.2))
            ax.barh([0], [1.0], color="#E8EFF8", height=0.5)
            ax.barh([0], [pct], color=gauge_color(bod_val), height=0.5)
            ax.set_xlim(0, 1)
            ax.axis("off")
            ax.set_facecolor("none")
            fig.patch.set_alpha(0)
            plt.tight_layout(pad=0.1)
            st.pyplot(fig, use_container_width=True)
            st.caption(f"BOD scale: 0 → 150+ mg/L")

        # WHO guideline note
        if bod_val < 6:
            note = "Below WHO drinking water threshold (≤ 6 mg/L). Excellent quality."
        elif bod_val < 20:
            note = "Suitable for most aquatic life. Acceptable for treated use."
        elif bod_val < 50:
            note = "Moderate organic load. Treatment recommended before use."
        else:
            note = "High organic pollution. Not suitable for human or aquatic use without treatment."

        st.info(f"**Interpretation:** {note}")

        # Input summary
        with st.expander("View input summary"):
            input_df = pd.DataFrame({
                "Parameter": FEATURES,
                "Your Input": [pH, temperature, dissolved_o2, turbidity,
                               conductivity, total_solids, nitrates,
                               phosphates, ammonia, cod]
            })
            st.dataframe(input_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Model Training Results")

    # Metrics table
    metrics_data = {
        "Model":           ["SVR (RBF kernel)", "PLS Regression", "Ridge Regression"],
        "R² Score":        [0.892, 0.945, 0.945],
        "RMSE (mg/L)":     [6.047, 4.297, 4.298],
        "MAE (mg/L)":      [4.840, 3.400, 3.399],
        "CV R² (5-fold)":  [0.865, 0.945, 0.945]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(
        metrics_df.style.highlight_max(subset=["R² Score", "CV R² (5-fold)"], color="#C8F0D0")
                        .highlight_min(subset=["RMSE (mg/L)", "MAE (mg/L)"], color="#C8F0D0"),
        use_container_width=True, hide_index=True
    )
    st.caption("✅ Best model (PLS Regression) saved and used in the predictor above.")
    st.markdown("---")

    # Load plots
    plot_files = {
        "Actual vs Predicted":   "actual_vs_predicted.png",
        "Model Comparison":      "model_comparison.png",
        "Feature Importance":    "feature_importance.png",
        "Correlation Heatmap":   "correlation_heatmap.png"
    }
    for title, fname in plot_files.items():
        fpath = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(fpath):
            st.markdown(f"**{title}**")
            st.image(fpath, use_container_width=True)
            st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("About This Project")
    st.markdown("""
**Biochemical Oxygen Demand (BOD)** is the most critical indicator of organic pollution
in river and surface water systems. Conventional lab-based BOD tests take **5 days** and
require expensive equipment. This project builds an ML pipeline that estimates BOD
**instantly** from 10 easily measurable water quality parameters.

### What this project demonstrates
- End-to-end ML pipeline: data → preprocessing → model training → evaluation → deployment
- Comparison of three algorithms: **SVR**, **PLS Regression**, **Ridge Regression**
- Feature importance analysis using permutation importance
- Production-ready model serialisation with `joblib`
- Interactive web app built with **Streamlit**

### Models used
| Model | Why it was chosen |
|-------|-------------------|
| **SVR (RBF kernel)** | Handles non-linear relationships, robust to outliers |
| **PLS Regression** | Handles multicollinearity between features — ideal for correlated water quality params |
| **Ridge Regression** | Linear baseline with L2 regularisation |

### Input parameters
The model uses 10 observable water quality measurements:
`pH`, `Temperature`, `Dissolved Oxygen`, `Turbidity`, `Conductivity`,
`Total Dissolved Solids`, `Nitrates`, `Phosphates`, `Ammonia`, `COD`

### References
- BVS Praveen et al., *Machine Learning in Water Treatment*, Wiley (2025)
- WHO Water Quality Guidelines, 2022
- UCI Machine Learning Repository — Water Quality datasets

### Author
**B.V.S. Praveen** | Ph.D. IIT Madras | M.Tech IIT Kharagpur  
[Google Scholar](https://scholar.google.com/citations?user=QWW0uwwAAAAJ)
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#AAA; font-size:0.8rem; margin-top:2rem;'>
  BOD Predictor v1.0 · Built with Python, scikit-learn & Streamlit
</div>
""", unsafe_allow_html=True)
