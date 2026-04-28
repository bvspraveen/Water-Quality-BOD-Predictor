"""
train_models.py
────────────────
Trains three models on the river water quality dataset and compares performance:
  1. Support Vector Regression (SVR)
  2. Partial Least Squares Regression (PLS)
  3. Ridge Regression (baseline)

Saves the best model to models/best_model.pkl for use in the Streamlit app.
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "river_water_quality.csv")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

FEATURES = [
    "pH", "Temperature_C", "Dissolved_O2", "Turbidity_NTU",
    "Conductivity", "Total_Solids", "Nitrates", "Phosphates",
    "Ammonia", "COD"
]
TARGET = "BOD"


# ── Load & split ───────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES].values
    y  = df[TARGET].values
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ── Build pipelines ────────────────────────────────────────────────────────────
def build_pipelines():
    return {
        "SVR (RBF kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.5))
        ]),
        "PLS Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  PLSRegression(n_components=5))
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0))
        ])
    }


# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # PLSRegression returns 2D array
    if hasattr(y_pred, "ravel"):
        y_pred = y_pred.ravel()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    # 5-fold cross-val R²
    cv   = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2").mean()

    return {"RMSE": round(rmse, 3), "MAE": round(mae, 3),
            "R2": round(r2, 3), "CV_R2": round(cv_r2, 3),
            "y_pred": y_pred}


# ── Plot: Actual vs Predicted ──────────────────────────────────────────────────
def plot_actual_vs_predicted(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Actual vs Predicted BOD (mg/L)", fontsize=14, fontweight="bold", y=1.02)

    colors = ["#2563A8", "#E05A2B", "#27A85F"]
    for ax, (name, res), color in zip(axes, results.items(), colors):
        y_pred = res["y_pred"]
        ax.scatter(y_test, y_pred, alpha=0.55, s=25, color=color, edgecolors="white", linewidths=0.4)
        lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Ideal fit")
        ax.set_xlabel("Actual BOD (mg/L)", fontsize=11)
        ax.set_ylabel("Predicted BOD (mg/L)", fontsize=11)
        ax.set_title(f"{name}\nR² = {res['R2']}  |  RMSE = {res['RMSE']}", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → plots/actual_vs_predicted.png")


# ── Plot: Model comparison bar chart ──────────────────────────────────────────
def plot_comparison(results):
    names   = list(results.keys())
    r2_vals = [r["R2"]   for r in results.values()]
    rmse_vals = [r["RMSE"] for r in results.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Comparison — BOD Prediction", fontsize=13, fontweight="bold")

    bars1 = ax1.bar(names, r2_vals, color=["#2563A8", "#E05A2B", "#27A85F"], edgecolor="white", linewidth=0.8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("R² Score (higher = better)", fontsize=11)
    ax1.set_title("R² Score by Model")
    ax1.bar_label(bars1, fmt="%.3f", padding=3, fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(names, rmse_vals, color=["#2563A8", "#E05A2B", "#27A85F"], edgecolor="white", linewidth=0.8)
    ax2.set_ylabel("RMSE — mg/L (lower = better)", fontsize=11)
    ax2.set_title("RMSE by Model")
    ax2.bar_label(bars2, fmt="%.2f", padding=3, fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → plots/model_comparison.png")


# ── Plot: Feature importance (via SVR permutation proxy) ──────────────────────
def plot_feature_importance(svr_pipeline, X_train, y_train):
    from sklearn.inspection import permutation_importance
    result = permutation_importance(svr_pipeline, X_train, y_train,
                                    n_repeats=15, random_state=42, scoring="r2")
    imp_mean = result.importances_mean
    imp_std  = result.importances_std

    sorted_idx = np.argsort(imp_mean)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2563A8" if v > 0 else "#E05A2B" for v in imp_mean[sorted_idx]]
    ax.barh(range(len(FEATURES)), imp_mean[sorted_idx],
            xerr=imp_std[sorted_idx], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels([FEATURES[i] for i in sorted_idx], fontsize=11)
    ax.set_xlabel("Permutation Importance (R² drop)", fontsize=11)
    ax.set_title("Feature Importance — SVR Model\n(how much R² drops when feature is shuffled)", fontsize=12)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → plots/feature_importance.png")


# ── Plot: Correlation heatmap ──────────────────────────────────────────────────
def plot_correlation():
    df = pd.read_csv(DATA_PATH)
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[FEATURES + [TARGET]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → plots/correlation_heatmap.png")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n══════════════════════════════════════════════")
    print("   BOD PREDICTOR — Model Training Pipeline    ")
    print("══════════════════════════════════════════════\n")

    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples\n")

    print("Training and evaluating models...")
    pipelines = build_pipelines()
    results   = {}

    for name, pipe in pipelines.items():
        print(f"  → {name}")
        res = evaluate(pipe, X_train, X_test, y_train, y_test)
        results[name] = res
        print(f"     R²={res['R2']}  RMSE={res['RMSE']} mg/L  MAE={res['MAE']}  CV-R²={res['CV_R2']}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n── Results Summary ───────────────────────────")
    summary = pd.DataFrame(
        {k: {m: v for m, v in v.items() if m != "y_pred"} for k, v in results.items()}
    ).T
    print(summary.to_string())

    # ── Save best model (highest R²) ─────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["R2"])
    best_pipe = pipelines[best_name]
    best_pipe.fit(X_train, y_train)           # refit on full train set
    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(FEATURES,  os.path.join(MODELS_DIR, "feature_names.pkl"))
    print(f"\n  Best model: {best_name}  (R²={results[best_name]['R2']})")
    print("  Saved → models/best_model.pkl")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_actual_vs_predicted(results, y_test)
    plot_comparison(results)
    plot_feature_importance(pipelines["SVR (RBF kernel)"], X_train, y_train)
    plot_correlation()

    print("\nDone. ✓\n")
    return results


if __name__ == "__main__":
    main()
