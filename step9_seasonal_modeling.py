"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Step 9: Seasonal Model Training & Evaluation
========================================================================
Input  : data/processed/monthly_features_all.csv  (from step 8)
Output : models/seasonal_classifier_{city}.pkl
         models/seasonal_regressor_highdays_{city}.pkl
         models/seasonal_climate_normals_{city}.json
         plots/seasonal_*.png

FOUR PREDICTION TARGETS:
  T1 — monthly_risk_level     (4-class classification: 0/1/2/3)
  T2 — n_high_risk_days       (regression: how many risky days)
  T3 — has_severe_day         (binary: will any day be Severe?)
  T4 — has_compound_event     (binary: will any compound event occur?)

MODELS:
  Random Forest, XGBoost, LightGBM → Voting Ensemble (T1)
  XGBoost Regressor (T2)
  XGBoost Classifier (T3, T4)

CROSS-VALIDATION:
  Time-series aware — uses TimeSeriesSplit (no data leakage).
  Monthly data must never have future months in training set.

CLIMATE NORMALS SAVED:
  For each city × month, the historical mean of all input features
  is saved as the baseline for forecasting future months.
  A warming trend slope (°C/year) is also computed and saved.
========================================================================
"""

import os, json, pickle, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.ensemble          import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.model_selection   import TimeSeriesSplit, cross_val_score
from sklearn.metrics           import (f1_score, accuracy_score, classification_report,
                                        mean_absolute_error, r2_score,
                                        roc_auc_score, confusion_matrix)
from sklearn.metrics           import make_scorer
from sklearn.preprocessing     import LabelEncoder
import xgboost  as xgb
import lightgbm as lgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

PROC_DIR  = "data/processed"
MODEL_DIR = "models"
PLOT_DIR  = "plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

CITIES      = ["Delhi", "Hyderabad", "Nagpur"]
RISK_LABELS = {0:"Low", 1:"Moderate", 2:"High", 3:"Severe"}

# ── FEATURE COLUMNS USED FOR MODELLING ───────────────────────────────────
# Excludes targets and identifiers
BASE_FEATURES = [
    "month", "year", "is_premonsoon", "is_monsoon",
    "temp_max_mean", "temp_max_max", "temp_max_p90", "temp_max_p95",
    "temp_min_mean", "temp_range_mean", "temp_departure_mean",
    "heat_index_mean", "heat_index_max", "humidex_mean",
    "humidity_mean", "wind_mean",
    "rainfall_total", "rainfall_days", "drought_days", "dry_streak_max",
    "aqi_mean", "aqi_max", "aqi_p75",
    "n_heatwave_days", "n_severe_hw_days",
    "n_compound_days", "n_triple_compound", "max_consec_hot",
    "composite_mean", "composite_max", "composite_p75",
    "high_risk_fraction", "warming_signal",
]

LAG_FEATURES = [
    "temp_max_mean_lag1", "temp_max_mean_lag2", "temp_max_mean_lag12",
    "composite_mean_lag1", "composite_mean_lag2", "composite_mean_lag12",
    "n_heatwave_days_lag1", "n_heatwave_days_lag12",
    "n_high_risk_days_lag1", "n_high_risk_days_lag12",
    "aqi_mean_lag1", "aqi_mean_lag12",
    "rainfall_total_lag1", "rainfall_total_lag12",
    "high_risk_fraction_lag1", "high_risk_fraction_lag12",
]

ALL_FEATURES = BASE_FEATURES + LAG_FEATURES


def get_features(df: pd.DataFrame) -> list[str]:
    """Returns only feature columns that actually exist in the dataframe."""
    return [c for c in ALL_FEATURES if c in df.columns]


# ── SAFE F1 SCORER ───────────────────────────────────────────────────────
# When a CV fold has only 1 class in predictions (common for imbalanced
# seasonal data like Hyderabad with few Severe months), sklearn's built-in
# 'f1_macro' returns NaN due to 0/0 in per-class averaging.
# This custom scorer suppresses that warning and returns 0.0 instead.
def _safe_f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

SAFE_F1 = make_scorer(_safe_f1_macro)


# ════════════════════════════════════════════════════════════════════════
# TARGET 1: MONTHLY RISK CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════

def train_classifier(X: np.ndarray, y: np.ndarray,
                     city_tag: str) -> tuple:
    """
    Trains RF + XGB + LGB + Ensemble. Time-series CV.
    Returns (best_model, results_dict).
    """
    tscv  = TimeSeriesSplit(n_splits=5)
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            class_weight="balanced", random_state=42, verbose=-1),
    }

    results = {}
    fitted  = {}

    print(f"\n  ── {city_tag} — Classification (monthly risk level) ──")
    for name, model in models.items():
        # Use safe F1 scorer to avoid NaN when a CV fold has only 1 class
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring=SAFE_F1)
        cv_f1 = float(np.nanmean(cv_scores))   # nanmean: ignore any remaining NaN folds
        model.fit(X, y)
        preds = model.predict(X)
        train_f1 = f1_score(y, preds, average="macro", zero_division=0)
        results[name] = {"cv_macro_f1": round(cv_f1, 4),
                          "train_macro_f1": round(train_f1, 4)}
        fitted[name] = model
        print(f"    {name:16s}  CV F1={cv_f1:.3f}  Train F1={train_f1:.3f}")

    # Ensemble
    ens = VotingClassifier(
        estimators=[("rf", fitted["RandomForest"]),
                    ("xgb", fitted["XGBoost"]),
                    ("lgb", fitted["LightGBM"])],
        voting="soft")
    cv_ens_scores = cross_val_score(ens, X, y, cv=tscv, scoring=SAFE_F1)
    cv_ens = float(np.nanmean(cv_ens_scores))
    ens.fit(X, y)
    preds_e = ens.predict(X)
    ens_f1  = f1_score(y, preds_e, average="macro", zero_division=0)
    results["Ensemble"] = {"cv_macro_f1": round(cv_ens, 4),
                            "train_macro_f1": round(ens_f1, 4)}
    fitted["Ensemble"] = ens
    print(f"    {'Ensemble':16s}  CV F1={cv_ens:.3f}  Train F1={ens_f1:.3f}")

    # NaN-safe best model selection
    best_name  = max(results, key=lambda k: results[k]["cv_macro_f1"]
                     if not np.isnan(results[k]["cv_macro_f1"]) else -1.0)
    best_model = fitted[best_name]
    print(f"    [✓] Best: {best_name}  (CV F1={results[best_name]['cv_macro_f1']:.3f})")
    return best_model, results


# ════════════════════════════════════════════════════════════════════════
# TARGET 2: NUMBER OF HIGH-RISK DAYS (Regression)
# ════════════════════════════════════════════════════════════════════════

def train_regressor_highdays(X: np.ndarray, y: np.ndarray,
                              city_tag: str):
    """Predicts expected number of high-risk days in a month."""
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        random_state=42, verbosity=0)
    tscv  = TimeSeriesSplit(n_splits=5)
    cv_mae = -cross_val_score(model, X, y, cv=tscv,
                               scoring="neg_mean_absolute_error").mean()
    model.fit(X, y)
    preds = model.predict(X)
    mae   = mean_absolute_error(y, preds)
    r2    = r2_score(y, preds)
    print(f"\n  ── {city_tag} — Regression (n_high_risk_days) ──")
    print(f"    XGB Regressor  CV MAE={cv_mae:.2f}  Train MAE={mae:.2f}  R²={r2:.3f}")
    return model


# ════════════════════════════════════════════════════════════════════════
# TARGET 3+4: BINARY CLASSIFIERS
# ════════════════════════════════════════════════════════════════════════

def train_binary(X: np.ndarray, y: np.ndarray,
                 target_name: str, city_tag: str):
    """Binary XGB for has_severe_day and has_compound_event."""
    if y.sum() < 3:
        print(f"    [!] {target_name}: too few positive samples — skipping")
        return None
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42, verbosity=0,
        scale_pos_weight=max(1, (y==0).sum()/(y==1).sum()))
    tscv  = TimeSeriesSplit(n_splits=5)
    try:
        cv_auc = cross_val_score(model, X, y, cv=tscv,
                                  scoring="roc_auc").mean()
    except Exception:
        cv_auc = np.nan
    model.fit(X, y)
    preds = model.predict(X)
    acc   = accuracy_score(y, preds)
    print(f"    {target_name:22s}  CV AUC={cv_auc:.3f}  Train Acc={acc:.3f}")
    return model


# ════════════════════════════════════════════════════════════════════════
# CLIMATE NORMALS
# ════════════════════════════════════════════════════════════════════════

def compute_climate_normals(df: pd.DataFrame, city: str) -> dict:
    """
    For each month (1–12), computes the historical mean of all input
    features. Also computes a warming trend slope.
    These are used by step10 to generate forecasted monthly feature vectors.
    """
    feat_cols = [c for c in BASE_FEATURES if c in df.columns and c != "year"]

    normals = {}
    for month in range(1, 13):
        sub = df[df["month"] == month]
        if sub.empty:
            continue
        normals[str(month)] = {
            col: float(sub[col].mean()) for col in feat_cols
            if col in sub.columns and not pd.isna(sub[col].mean())
        }

    # Warming trend: linear slope of temp_max_mean vs year
    if "temp_max_mean" in df.columns and "year" in df.columns:
        from numpy.polynomial.polynomial import polyfit as npfit
        annual_mean = df.groupby("year")["temp_max_mean"].mean().dropna()
        if len(annual_mean) >= 3:
            years  = annual_mean.index.values.astype(float)
            temps  = annual_mean.values
            coeffs = np.polyfit(years, temps, 1)
            warming_slope = float(coeffs[0])   # °C per year
        else:
            warming_slope = 0.0
    else:
        warming_slope = 0.0

    return {
        "city"          : city,
        "normals"       : normals,
        "warming_slope" : warming_slope,
        "base_year"     : int(df["year"].mean()) if "year" in df.columns else 2022,
        "feature_cols"  : feat_cols,
    }


# ════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════

def plot_monthly_risk_calendar(df: pd.DataFrame, city: str):
    """Heatmap calendar: year × month → monthly_risk_level."""
    pivot = df.pivot_table(
        index="year", columns="month",
        values="monthly_risk_level", aggfunc="first")
    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot)*0.6)))
    cmap = plt.cm.colors.ListedColormap(["#2ECC71","#F1C40F","#E67E22","#E74C3C"])
    im   = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                     vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.astype(int), fontsize=9)
    ax.set_title(f"{city} — Monthly Risk Level Calendar", fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, ticks=[0,1,2,3], fraction=0.03)
    cbar.ax.set_yticklabels(["Low","Moderate","High","Severe"])
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, f"seasonal_calendar_{city}.png")
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [✓] Calendar plot → {p}")


def plot_seasonal_feature_importance(model, feat_cols: list, city: str):
    """Bar chart of top feature importances from XGBoost."""
    if not hasattr(model, "feature_importances_"):
        # Ensemble — extract XGBoost estimator
        for name, est in model.estimators:
            if hasattr(est, "feature_importances_"):
                model = est
                break
        else:
            return

    imp  = pd.Series(model.feature_importances_, index=feat_cols)
    top  = imp.nlargest(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    top.sort_values().plot(kind="barh", ax=ax, color="#3498DB")
    ax.set_title(f"{city} — Top Seasonal Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, f"seasonal_importance_{city}.png")
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [✓] Feature importance plot → {p}")


def plot_confusion_monthly(model, X, y, city: str):
    preds = model.predict(X)
    cm    = confusion_matrix(y, preds)
    labels = ["Low","Moderate","High","Severe"]
    present = sorted(list(set(y)))
    labels_present = [labels[i] for i in present]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_present, yticklabels=labels_present, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{city} — Monthly Risk Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    p = os.path.join(PLOT_DIR, f"seasonal_cm_{city}.png")
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STEP 9 — SEASONAL MODEL TRAINING")
    print("="*60)

    path = os.path.join(PROC_DIR, "monthly_features_all.csv")
    if not os.path.exists(path):
        print("[!] monthly_features_all.csv not found. Run step8 first.")
        exit(1)

    df_all = pd.read_csv(path)
    print(f"\n  Monthly data: {df_all.shape[0]} rows, "
          f"{df_all['city'].nunique()} cities")

    summary = {}

    for city in CITIES:
        city_df = df_all[df_all["city"] == city].copy()
        if city_df.empty:
            print(f"\n  [!] {city}: no data")
            continue

        city_df = city_df.sort_values(["year","month"]).reset_index(drop=True)
        feat_cols = get_features(city_df)
        X = city_df[feat_cols].fillna(0).values

        print(f"\n{'='*50}")
        print(f"  {city}  ({len(city_df)} months, {len(feat_cols)} features)")
        print(f"{'='*50}")

        # ── T1: Classification ─────────────────────────────────────────
        y_cls = city_df["monthly_risk_level"].fillna(0).astype(int).values
        clf, cls_results = train_classifier(X, y_cls, city)
        plot_confusion_monthly(clf, X, y_cls, city)
        plot_seasonal_feature_importance(clf, feat_cols, city)
        plot_monthly_risk_calendar(city_df, city)

        # ── T2: Regression — n_high_risk_days ─────────────────────────
        if "n_high_risk_days" in city_df.columns:
            y_reg = city_df["n_high_risk_days"].fillna(0).values
            reg   = train_regressor_highdays(X, y_reg, city)
        else:
            reg = None

        # ── T3: Binary — has_severe_day ───────────────────────────────
        print(f"\n  ── {city} — Binary classifiers ──")
        binary_models = {}
        for target in ["has_severe_day", "has_compound_event"]:
            if target in city_df.columns:
                y_bin = city_df[target].fillna(0).astype(int).values
                bmod  = train_binary(X, y_bin, target, city)
                binary_models[target] = bmod

        # ── Climate normals ────────────────────────────────────────────
        normals = compute_climate_normals(city_df, city)
        normals_path = os.path.join(MODEL_DIR,
                                     f"seasonal_climate_normals_{city}.json")
        with open(normals_path, "w") as f:
            json.dump(normals, f, indent=2)
        print(f"\n  [✓] Climate normals saved → {normals_path}")
        print(f"      Warming trend: {normals['warming_slope']:.4f} °C/year")

        # ── Save models ────────────────────────────────────────────────
        bundle = {
            "classifier"    : clf,
            "regressor"     : reg,
            "binary_models" : binary_models,
            "feature_cols"  : feat_cols,
            "city"          : city,
            "cls_results"   : cls_results,
        }
        pkl_path = os.path.join(MODEL_DIR, f"seasonal_classifier_{city}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"  [✓] Models saved → {pkl_path}")

        summary[city] = {
            "months"          : len(city_df),
            "features"        : len(feat_cols),
            # NaN-safe: ignore any model whose CV F1 is NaN
            "best_clf_cv_f1"  : float(np.nanmax([v["cv_macro_f1"]
                                                  for v in cls_results.values()])),
            "warming_slope"   : normals["warming_slope"],
        }

    # ── Print summary ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for city, s in summary.items():
        print(f"  {city:12s}: {s['months']:3d} months  "
              f"{s['features']:2d} features  "
              f"CV F1={s['best_clf_cv_f1']:.3f}  "
              f"warming={s['warming_slope']:+.4f}°C/yr")

    print("\n[Done] Step 9 complete. Run step10_seasonal_forecast.py next.\n")