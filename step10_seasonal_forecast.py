"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Step 10: Seasonal Forecast — Monthly Risk Outlook
========================================================================
Input  : models/seasonal_classifier_{city}.pkl  (from step 9)
         models/seasonal_climate_normals_{city}.json
         data/processed/monthly_features_{city}.csv

Output : data/predictions/seasonal_forecast_{city}_{date}.json
         Console table + dashboard-ready dict

HOW FORECASTING WORKS FOR FUTURE MONTHS:
─────────────────────────────────────────────────────────────────────
  Problem: Unlike daily prediction, weather APIs don't give monthly
  forecasts. For "what will July 2026 look like?", we have no
  raw temperature or AQI reading.

  Solution — Climate Normals + Trend Adjustment:

  Step A: Baseline  = historical average for that month
    e.g., Delhi May: avg Tmax = 41.2°C, avg AQI = 180, etc.
    (computed from all May months in 2020–2025)

  Step B: Trend     = warming_slope × (target_year - base_year)
    e.g., +0.045 °C/year × 4 years = +0.18°C added to Tmax

  Step C: Lag features = from the actual past months in the CSV
    lag1  = last month's actual statistics
    lag12 = same month last year's actual statistics
    These are REAL values from the CSV, not estimated.

  Step D: Predict using the trained seasonal model

  Result: Each month in the 12-month outlook gets:
    - risk_level (Low/Moderate/High/Severe)
    - expected n_high_risk_days
    - P(at least one Severe day)
    - P(compound event occurring)
    - confidence score

CONFIDENCE MODEL:
  Months 1–2 ahead : High    (recent lag features are accurate)
  Months 3–6 ahead : Medium  (normals reliable, lag gets stale)
  Months 7–12 ahead: Low     (indicative trend only)
========================================================================
"""

import os, json, pickle, argparse
import pandas as pd
import numpy as np
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings("ignore")

PROC_DIR  = "data/processed"
MODEL_DIR = "models"
LOG_DIR   = "data/predictions"
CITIES    = ["Delhi", "Hyderabad", "Nagpur"]
os.makedirs(LOG_DIR, exist_ok=True)

RISK_LABELS  = {0:"Low", 1:"Moderate", 2:"High", 3:"Severe"}
RISK_EMOJIS  = {0:"🟢",  1:"🟡",       2:"🟠",   3:"🔴"}
MONTH_NAMES  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

SEASONAL_CONFIDENCE = {
    1: ("High",       1.00),
    2: ("High",       0.95),
    3: ("Medium",     0.85),
    4: ("Medium",     0.78),
    5: ("Medium",     0.72),
    6: ("Medium",     0.66),
    7: ("Low",        0.58),
    8: ("Low",        0.52),
    9: ("Low",        0.48),
    10:("Indicative", 0.44),
    11:("Indicative", 0.40),
    12:("Indicative", 0.38),
}

SEASON_MAP = {
    12:"Winter", 1:"Winter", 2:"Winter",
    3:"Pre-Monsoon", 4:"Pre-Monsoon", 5:"Pre-Monsoon",
    6:"Monsoon", 7:"Monsoon", 8:"Monsoon", 9:"Monsoon",
    10:"Post-Monsoon", 11:"Post-Monsoon",
}


# ════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════

def load_seasonal_bundle(city: str) -> dict | None:
    p = os.path.join(MODEL_DIR, f"seasonal_classifier_{city}.pkl")
    if not os.path.exists(p):
        print(f"  [!] No seasonal model for {city}. Run step9 first.")
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def load_climate_normals(city: str) -> dict | None:
    p = os.path.join(MODEL_DIR, f"seasonal_climate_normals_{city}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_monthly_history(city: str) -> pd.DataFrame | None:
    p = os.path.join(PROC_DIR, f"monthly_features_{city}.csv")
    if not os.path.exists(p):
        p2 = os.path.join(PROC_DIR, "monthly_features_all.csv")
        if os.path.exists(p2):
            df = pd.read_csv(p2)
            return df[df["city"] == city].copy()
        return None
    return pd.read_csv(p)


# ════════════════════════════════════════════════════════════════════════
# FEATURE VECTOR BUILDER FOR A FUTURE MONTH
# ════════════════════════════════════════════════════════════════════════

def build_forecast_feature_vector(target_year: int,
                                   target_month: int,
                                   normals: dict,
                                   history_df: pd.DataFrame,
                                   warming_slope: float,
                                   base_year: float,
                                   feat_cols: list) -> np.ndarray:
    """
    Constructs the feature vector for a future month using:
      1. Climate normals for that month (baseline)
      2. Warming trend adjustment on temperature features
      3. Real lag features from actual history (where available)
    """
    month_key = str(target_month)
    if month_key not in normals:
        month_key = str(target_month)

    baseline = normals.get(month_key, {})

    # Trend adjustment (applied to temperature-related features)
    trend_adj = warming_slope * (target_year - base_year)

    # Start with the baseline feature dict
    fvec = {}

    # Calendar features
    fvec["month"]        = target_month
    fvec["year"]         = target_year
    fvec["is_premonsoon"]= int(target_month in [3, 4, 5])
    fvec["is_monsoon"]   = int(target_month in [6, 7, 8, 9])
    fvec["warming_signal"]= trend_adj   # represents departure from base year

    # Fill from climate normals
    temp_related = ["temp_max_mean","temp_max_max","temp_max_p90","temp_max_p95",
                    "temp_min_mean","temp_range_mean","temp_departure_mean",
                    "heat_index_mean","heat_index_max","humidex_mean"]

    for col, val in baseline.items():
        if col in ["month", "year", "is_premonsoon", "is_monsoon"]:
            continue
        if col in temp_related:
            fvec[col] = val + trend_adj    # adjust temperature features
        else:
            fvec[col] = val                # non-temperature: use as-is

    # ── Lag features from actual history ─────────────────────────────
    if history_df is not None and not history_df.empty:
        # lag1 = previous month's actual stats
        prev1_year  = target_year if target_month > 1 else target_year - 1
        prev1_month = target_month - 1 if target_month > 1 else 12
        prev1 = history_df[
            (history_df["year"] == prev1_year) &
            (history_df["month"] == prev1_month)]
        if not prev1.empty:
            r = prev1.iloc[-1]
            for lag_col in ["temp_max_mean","composite_mean","n_heatwave_days",
                            "n_high_risk_days","aqi_mean","rainfall_total",
                            "high_risk_fraction"]:
                if lag_col in r.index:
                    fvec[f"{lag_col}_lag1"] = float(r[lag_col])

        # lag2 = two months ago
        prev2_year  = target_year if target_month > 2 else target_year - 1
        prev2_month = target_month - 2 if target_month > 2 else 12 + target_month - 2
        prev2 = history_df[
            (history_df["year"] == prev2_year) &
            (history_df["month"] == prev2_month)]
        if not prev2.empty:
            r2 = prev2.iloc[-1]
            for lag_col in ["temp_max_mean","composite_mean"]:
                if lag_col in r2.index:
                    fvec[f"{lag_col}_lag2"] = float(r2[lag_col])

        # lag12 = same month last year (most important seasonal lag)
        prev12 = history_df[
            (history_df["year"] == target_year - 1) &
            (history_df["month"] == target_month)]
        if not prev12.empty:
            r12 = prev12.iloc[-1]
            for lag_col in ["temp_max_mean","composite_mean","n_heatwave_days",
                            "n_high_risk_days","aqi_mean","rainfall_total",
                            "high_risk_fraction"]:
                if lag_col in r12.index:
                    fvec[f"{lag_col}_lag12"] = float(r12[lag_col])

    # Build array in the correct feature order
    X = np.array([fvec.get(col, 0.0) for col in feat_cols],
                  dtype=float).reshape(1, -1)
    # Replace any NaN with column mean from normals or 0
    X = np.nan_to_num(X, nan=0.0)
    return X, fvec


# ════════════════════════════════════════════════════════════════════════
# SINGLE MONTH PREDICTION
# ════════════════════════════════════════════════════════════════════════

def predict_month(year: int, month: int,
                   bundle: dict, normals: dict,
                   history_df: pd.DataFrame,
                   months_ahead: int) -> dict:
    """
    Predicts risk for a single future month.
    Returns a dict with all prediction outputs.
    """
    clf      = bundle["classifier"]
    reg      = bundle.get("regressor")
    binary   = bundle.get("binary_models", {})
    feat_cols= bundle["feature_cols"]
    w_slope  = normals["warming_slope"]
    base_yr  = normals["base_year"]

    X, fvec = build_forecast_feature_vector(
        year, month, normals["normals"], history_df,
        w_slope, base_yr, feat_cols)

    # T1: risk classification
    probas    = clf.predict_proba(X)[0]
    risk_lvl  = int(np.argmax(probas))
    raw_conf  = float(probas[risk_lvl])

    # Adjust confidence by temporal distance
    rel_label, decay = SEASONAL_CONFIDENCE.get(months_ahead, ("Indicative", 0.38))
    adj_conf  = round(raw_conf * decay * 100, 1)

    # T2: expected high-risk days
    n_high_days = int(round(float(reg.predict(X)[0]))) if reg else None

    # T3/T4: binary probabilities
    p_severe   = None
    p_compound = None
    for target, bmod in binary.items():
        if bmod is None:
            continue
        prob = float(bmod.predict_proba(X)[0][1]) if hasattr(bmod, "predict_proba") else None
        if "severe" in target:
            p_severe = round(prob * 100, 1) if prob is not None else None
        elif "compound" in target:
            p_compound = round(prob * 100, 1) if prob is not None else None

    # Approximate expected temperature and AQI from feature vector
    exp_tmax = round(float(fvec.get("temp_max_mean", 0)), 1)
    exp_aqi  = round(float(fvec.get("aqi_mean", 0)), 0)

    return {
        "year"              : year,
        "month"             : month,
        "month_name"        : MONTH_NAMES[month - 1],
        "season"            : SEASON_MAP.get(month, ""),
        "months_ahead"      : months_ahead,
        "risk_level"        : risk_lvl,
        "risk_label"        : RISK_LABELS[risk_lvl],
        "emoji"             : RISK_EMOJIS[risk_lvl],
        "probabilities"     : {RISK_LABELS[i]: round(float(p)*100, 1)
                                for i, p in enumerate(probas)},
        "raw_confidence"    : round(raw_conf * 100, 1),
        "adj_confidence"    : adj_conf,
        "reliability"       : rel_label,
        "decay_factor"      : decay,
        "n_high_risk_days"  : n_high_days,
        "p_severe_day"      : p_severe,
        "p_compound_event"  : p_compound,
        "exp_tmax"          : exp_tmax,
        "exp_aqi"           : exp_aqi,
        "trend_adj_applied" : round(w_slope * (year - base_yr), 3),
        "lag12_available"   : f"{year-1}-{month:02d}" in (
            [f"{int(r['year'])}-{int(r['month']):02d}"
             for _, r in history_df.iterrows()]
            if history_df is not None else []),
    }


# ════════════════════════════════════════════════════════════════════════
# FULL CITY OUTLOOK
# ════════════════════════════════════════════════════════════════════════

def forecast_city_outlook(city: str, n_months: int = 12) -> list[dict]:
    """
    Generates a monthly risk outlook for city for the next n_months.
    """
    bundle     = load_seasonal_bundle(city)
    normals    = load_climate_normals(city)
    history_df = load_monthly_history(city)

    if bundle is None or normals is None:
        print(f"  [!] Cannot forecast {city} — missing model or normals.")
        return []

    today       = date.today()
    forecasts   = []
    last_hist_str = ""

    if history_df is not None and not history_df.empty:
        last_yr  = int(history_df["year"].max())
        last_mon = int(history_df[history_df["year"]==last_yr]["month"].max())
        last_hist_str = f"{last_yr}-{MONTH_NAMES[last_mon-1]}"

    print(f"\n  {city}: last history = {last_hist_str or 'unknown'}, "
          f"forecasting {n_months} months ahead")

    for i in range(1, n_months + 1):
        target_date  = today + relativedelta(months=i)
        target_year  = target_date.year
        target_month = target_date.month

        result = predict_month(
            target_year, target_month, bundle, normals,
            history_df, months_ahead=i)
        result["city"] = city
        forecasts.append(result)

        print(f"  {target_year}-{MONTH_NAMES[target_month-1]:3s}  "
              f"{result['emoji']} {result['risk_label']:10s}  "
              f"score≈{result['adj_confidence']:5.1f}%  "
              f"highdays≈{result['n_high_risk_days'] or '?':>3}  "
              f"P(sev)={result['p_severe_day'] or '?':>5}%  "
              f"[{result['reliability']}]")

    return forecasts


# ════════════════════════════════════════════════════════════════════════
# PRINT & SAVE
# ════════════════════════════════════════════════════════════════════════

def print_outlook_table(forecasts: list[dict], city: str):
    print(f"\n  {'━'*72}")
    print(f"  {city.upper()} — MONTHLY RISK OUTLOOK")
    print(f"  {'━'*72}")
    print(f"  {'Month':12s} {'Risk':10s} {'Conf':>6s} {'HighDays':>9s} "
          f"{'P(Sev%)':>8s} {'Tmax°C':>7s} {'AQI':>5s} {'Rely':>12s}")
    print(f"  {'-'*70}")
    for f in forecasts:
        print(f"  {f['year']}-{f['month_name']:3s}   "
              f"{f['emoji']} {f['risk_label']:8s}  "
              f"{f['adj_confidence']:5.1f}%  "
              f"{str(f['n_high_risk_days'] or '?'):>9s}  "
              f"{str(f['p_severe_day'] or '?'):>7s}%  "
              f"{f['exp_tmax']:>7.1f}  "
              f"{f['exp_aqi']:>5.0f}  "
              f"[{f['reliability']:>11s}]")
    print(f"  {'━'*72}\n")


def save_outlook(forecasts: list[dict], city: str):
    p = os.path.join(LOG_DIR,
        f"seasonal_forecast_{city}_{date.today()}.json")
    with open(p, "w") as f:
        json.dump(forecasts, f, indent=2, default=str)
    print(f"  [✓] Saved → {p}")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seasonal monthly risk outlook")
    parser.add_argument("--city",   default="all",
                        help="Delhi / Hyderabad / Nagpur / all")
    parser.add_argument("--months", type=int, default=12,
                        help="Months ahead to forecast (default 12)")
    parser.add_argument("--no-save",action="store_true")
    args = parser.parse_args()

    cities = CITIES if args.city == "all" else [args.city]

    print("\n" + "="*60)
    print("  STEP 10 — SEASONAL MONTHLY OUTLOOK")
    print(f"  Months ahead  : {args.months}")
    print(f"  Cities        : {', '.join(cities)}")
    print(f"  Run date      : {date.today()}")
    print("="*60)
    print("""
  How forecasted features are built:
    Baseline  : historical monthly avg (climate normal)
    Trend     : +warming_slope × (target_year - base_year)
    Lag1/2    : real values from last 1-2 months in CSV
    Lag12     : same month last year (real value from CSV)
    """)

    all_forecasts = {}
    for city in cities:
        fc = forecast_city_outlook(city, n_months=args.months)
        if fc:
            print_outlook_table(fc, city)
            all_forecasts[city] = fc
            if not args.no_save:
                save_outlook(fc, city)

    print("[Done] Seasonal forecast complete.\n")