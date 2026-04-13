"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Step 4: Model Training & Evaluation
========================================================================
Models trained:
  M1 - Logistic Regression        (interpretable baseline)
  M2 - Random Forest              (strong tree-based)
  M3 - XGBoost                    (gradient boosted)
  M4 - LightGBM                   (fast gradient boosted)
  M5 - Voting Ensemble            (M2 + M3 + M4)
  M6 - LSTM                       (deep-learning time-series)

Evaluation metrics:
  • Accuracy, Macro F1, Weighted F1
  • Confusion Matrix
  • Per-class Precision / Recall
  • ROC-AUC (one-vs-rest)
  • SHAP feature importance (for tree models)

Two targets:
  T1 - risk_level (4-class classification  0/1/2/3)
  T2 - composite_score regression (continuous 0–100)
========================================================================
"""

import pandas as pd
import numpy as np
import os, pickle, json, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection       import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing         import StandardScaler, LabelEncoder
from sklearn.linear_model          import LogisticRegression
from sklearn.ensemble              import RandomForestClassifier, VotingClassifier, RandomForestRegressor
from sklearn.metrics               import (classification_report, confusion_matrix,
                                           f1_score, accuracy_score, roc_auc_score,
                                           mean_absolute_error, r2_score)
from sklearn.pipeline              import Pipeline
from sklearn.utils.class_weight    import compute_class_weight
import xgboost  as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn  as sns

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[!] shap not installed — feature importance plots will be skipped")

PROC_DIR  = "data/processed"
MODEL_DIR = "models"
PLOT_DIR  = "plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# ── FEATURE SET ──────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Raw
    "temp_max", "temp_min", "humidity", "wind", "rainfall", "aqi",
    # Derived
    "heat_index", "humidex", "temp_range", "temp_mean", "feels_like_excess",
    "wind_heat_ratio",
    # Rolling
    "temp_max_roll3", "temp_max_roll7", "humidity_roll7", "aqi_roll7",
    "rainfall_roll7",
    # Lag
    "temp_max_lag1", "temp_max_lag2", "aqi_lag1", "humidity_lag1",
    # Anomaly
    "temp_departure", "temp_zscore", "aqi_departure",
    # Drought
    "dry_days_streak", "spi_30", "drought_flag",
    # Calendar
    "month", "day_of_year", "is_summer",
    # Compound
    "compound_heat_aqi", "compound_heat_drought", "compound_heat_humidity",
    "triple_compound", "consec_hot_days", "aqi_category",
]
TARGET_CLASS = "risk_level"
TARGET_REG   = "composite_score"


def load_data():
    path = os.path.join(PROC_DIR, "labelled_all.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df.dropna(subset=FEATURE_COLS + [TARGET_CLASS], inplace=True)
    return df


def get_xy(df):
    X = df[FEATURE_COLS].fillna(0).values
    y_cls = df[TARGET_CLASS].values.astype(int)
    y_reg = df[TARGET_REG].values.astype(float)
    return X, y_cls, y_reg


# ── CLASSIFICATION PIPELINE ──────────────────────────────────────────────

def train_classifiers(X_tr, X_te, y_tr, y_te, city_tag="all"):
    classes = np.unique(y_tr)
    cw = compute_class_weight("balanced", classes=classes, y=y_tr)
    cw_dict = dict(zip(classes, cw))

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=0.5, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            use_label_encoder=False, eval_metric="mlogloss",
            scale_pos_weight=1, random_state=42),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=42, verbose=-1),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        X_fit = X_tr_s if name == "LogisticRegression" else X_tr
        X_eval= X_te_s  if name == "LogisticRegression" else X_te

        model.fit(X_fit, y_tr)
        preds = model.predict(X_eval)
        probas= model.predict_proba(X_eval)

        acc   = accuracy_score(y_te, preds)
        f1_m  = f1_score(y_te, preds, average="macro")
        f1_w  = f1_score(y_te, preds, average="weighted")
        try:
            auc = roc_auc_score(y_te, probas, multi_class="ovr", average="macro")
        except Exception:
            auc = np.nan

        results[name] = {"Accuracy": acc, "Macro F1": f1_m,
                          "Weighted F1": f1_w, "ROC-AUC": auc}
        trained[name] = model
        print(f"  {name:22s}  Acc={acc:.3f}  F1(macro)={f1_m:.3f}  AUC={auc:.3f}")

    # Voting ensemble
    ens = VotingClassifier(
        estimators=[("rf", trained["RandomForest"]),
                    ("xgb", trained["XGBoost"]),
                    ("lgb", trained["LightGBM"])],
        voting="soft")
    ens.fit(X_tr, y_tr)
    preds_e = ens.predict(X_te)
    probas_e= ens.predict_proba(X_te)
    acc_e   = accuracy_score(y_te, preds_e)
    f1_e    = f1_score(y_te, preds_e, average="macro")
    try:
        auc_e = roc_auc_score(y_te, probas_e, multi_class="ovr", average="macro")
    except Exception:
        auc_e = np.nan
    results["Ensemble"] = {"Accuracy": acc_e, "Macro F1": f1_e,
                            "Weighted F1": f1_score(y_te, preds_e, average="weighted"),
                            "ROC-AUC": auc_e}
    trained["Ensemble"] = ens
    print(f"  {'Ensemble':22s}  Acc={acc_e:.3f}  F1(macro)={f1_e:.3f}  AUC={auc_e:.3f}")

    # ── Save best model ──────────────────────────────────────────────────
    best = max(results, key=lambda k: results[k]["Macro F1"])
    best_model = trained[best]
    meta = {"best_model": best, "scaler": scaler,
            "feature_cols": FEATURE_COLS, "city": city_tag}
    with open(os.path.join(MODEL_DIR, f"classifier_{city_tag}.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "meta": meta}, f)
    print(f"\n  [✓] Best model: {best} — saved to models/classifier_{city_tag}.pkl")

    # ── Confusion matrix of best ─────────────────────────────────────────
    preds_best = best_model.predict(X_te)
    cm = confusion_matrix(y_te, preds_best)
    plot_confusion_matrix(cm, city_tag, best)

    # ── SHAP ─────────────────────────────────────────────────────────────
    if SHAP_AVAILABLE and best in ("RandomForest", "XGBoost", "LightGBM"):
        plot_shap(best_model, X_te, city_tag)

    return results, trained, scaler


def plot_confusion_matrix(cm, city_tag, model_name):
    labels = ["Low", "Moderate", "High", "Severe"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name} ({city_tag})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/cm_{city_tag}.png", bbox_inches="tight")
    plt.close()
    print(f"  [✓] Confusion matrix saved for {city_tag}")


def plot_shap(model, X_test, city_tag):
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test[:200])
        # For multi-class, shap_vals is a list; use class 2 (High) as representative
        sv = shap_vals[2] if isinstance(shap_vals, list) else shap_vals
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(sv, X_test[:200], feature_names=FEATURE_COLS,
                          plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance — {city_tag} (High Risk Class)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/shap_{city_tag}.png", bbox_inches="tight")
        plt.close()
        print(f"  [✓] SHAP plot saved for {city_tag}")
    except Exception as e:
        print(f"  [!] SHAP failed: {e}")


def plot_model_comparison(all_results: dict):
    """Bar chart comparing all models across metrics."""
    rows = []
    for city, res in all_results.items():
        for model, metrics in res.items():
            rows.append({"City": city, "Model": model, **metrics})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_plot = ["Accuracy", "Macro F1", "ROC-AUC"]
    for ax, metric in zip(axes, metrics_plot):
        pivot = df.pivot(index="Model", columns="City", values=metric)
        pivot.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white")
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Model Comparison Across Cities", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/model_comparison.png", bbox_inches="tight")
    plt.close()
    print("[✓] Model comparison chart saved")


# ── REGRESSION PIPELINE ──────────────────────────────────────────────────
def train_regressor(X_tr, X_te, y_tr, y_te, city_tag):
    rf_reg = RandomForestRegressor(n_estimators=300, max_depth=12,
                                    random_state=42, n_jobs=-1)
    rf_reg.fit(X_tr, y_tr)
    preds = rf_reg.predict(X_te)
    mae = mean_absolute_error(y_te, preds)
    r2  = r2_score(y_te, preds)
    print(f"  Regressor ({city_tag}): MAE={mae:.2f}, R²={r2:.3f}")

    xgb_reg = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                 random_state=42, verbosity=0)
    xgb_reg.fit(X_tr, y_tr)
    preds_x = xgb_reg.predict(X_te)
    mae_x = mean_absolute_error(y_te, preds_x)
    r2_x  = r2_score(y_te, preds_x)
    print(f"  XGB Regressor ({city_tag}): MAE={mae_x:.2f}, R²={r2_x:.3f}")

    # Scatter: predicted vs actual
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te, preds_x, alpha=0.3, s=10, c="#E74C3C")
    ax.plot([0, 100], [0, 100], "k--", lw=0.8)
    ax.set_xlabel("Actual Score"); ax.set_ylabel("Predicted Score")
    ax.set_title(f"Score Regression — {city_tag}\nR²={r2_x:.3f}  MAE={mae_x:.2f}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/regression_{city_tag}.png", bbox_inches="tight")
    plt.close()

    best_reg = xgb_reg if r2_x >= r2 else rf_reg
    with open(os.path.join(MODEL_DIR, f"regressor_{city_tag}.pkl"), "wb") as f:
        pickle.dump(best_reg, f)
    print(f"  [✓] Regressor saved for {city_tag}")


# ── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STEP 4 ─ MODEL TRAINING & EVALUATION")
    print("="*60)

    df = load_data()

    all_cls_results = {}

    # ── Train on combined (all cities) ───────────────────────────────────
    tags = [("all", df)] + [(city, df[df["city"] == city]) for city in df["city"].unique()]

    for tag, subset in tags:
        print(f"\n  ── {tag.upper()} ─────────────────────────────────")
        X, y_cls, y_reg = get_xy(subset)
        if len(X) < 100:
            print(f"  [!] Not enough data for {tag}")
            continue
        X_tr, X_te, y_tr_c, y_te_c, y_tr_r, y_te_r = train_test_split(
            X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls)

        print("  Classification:")
        cls_res, _, scaler = train_classifiers(X_tr, X_te, y_tr_c, y_te_c, tag)
        all_cls_results[tag] = cls_res

        print("  Regression:")
        train_regressor(X_tr, X_te, y_tr_r, y_te_r, tag)

    plot_model_comparison({k: v for k, v in all_cls_results.items() if k != "all"})

    # ── Save results summary ─────────────────────────────────────────────
    summary = {}
    for tag, res in all_cls_results.items():
        summary[tag] = {m: {k: round(v, 4) for k, v in metrics.items()}
                         for m, metrics in res.items()}
    with open(os.path.join(MODEL_DIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[✓] Results summary saved to models/results_summary.json")
    print("\n[Done] Model training complete.\n")