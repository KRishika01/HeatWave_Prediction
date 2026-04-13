"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Step 8: Seasonal Feature Engineering — Daily → Monthly Aggregation
========================================================================
Input  : data/processed/labelled_all.csv  (from steps 1–2)
Output : data/processed/monthly_features_all.csv
         data/processed/monthly_features_{city}.csv  (per city)

What this does:
  Collapses daily rows into monthly summary statistics.
  Each output row = one city-month combination.

Monthly features engineered (30 total):
  ── Temperature ──────────────────────────────────────────
  temp_max_mean      : avg daily Tmax in month
  temp_max_max       : hottest single day
  temp_max_p90       : 90th percentile Tmax (sustained heat)
  temp_max_p95       : 95th percentile Tmax
  temp_min_mean      : avg daily Tmin (night cooling)
  temp_range_mean    : avg diurnal range (DTR)
  temp_departure_mean: avg departure from climatological norm

  ── Heat Stress ──────────────────────────────────────────
  heat_index_mean    : avg apparent temperature
  heat_index_max     : peak apparent temperature
  humidex_mean       : avg Humidex

  ── Humidity & Wind ──────────────────────────────────────
  humidity_mean      : avg relative humidity
  wind_mean          : avg wind speed

  ── Rainfall & Drought ───────────────────────────────────
  rainfall_total     : total monthly rainfall (mm)
  rainfall_days      : days with rain > 1mm
  drought_days       : days with drought_flag = 1
  dry_streak_max     : longest consecutive dry spell

  ── AQI ──────────────────────────────────────────────────
  aqi_mean           : avg AQI
  aqi_max            : worst single-day AQI
  aqi_p75            : 75th percentile AQI

  ── Risk Events ──────────────────────────────────────────
  n_heatwave_days    : days with Tmax ≥ 40°C (IMD heat wave)
  n_severe_hw_days   : days with Tmax ≥ 45°C
  n_compound_days    : days with any compound flag active
  n_triple_compound  : days with triple compound event
  max_consec_hot     : longest hot streak (consec_hot_days max)
  n_high_risk_days   : days with risk_level ≥ 2
  n_severe_risk_days : days with risk_level = 3
  high_risk_fraction : n_high_risk_days / total_days

  ── Composite Score ──────────────────────────────────────
  composite_mean     : avg daily composite score
  composite_max      : peak daily composite score

  ── Lag Features ─────────────────────────────────────────
  All key features lagged by 1 month, 2 months, and 12 months
  (same month last year — captures seasonal persistence)

  ── Climate Trend ────────────────────────────────────────
  year               : year (captures long-term warming)
  warming_signal     : temp_max_mean departure from multi-year mean

Monthly target labels:
  monthly_risk_level : 0=Low  1=Moderate  2=High  3=Severe
    Logic: based on composite_mean + hard overrides for
           n_heatwave_days and n_severe_risk_days counts
  has_severe_day     : 1 if any day in month was Severe
  has_compound_event : 1 if any compound event in month
========================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

PROC_DIR  = "data/processed"
CITIES    = ["Delhi", "Hyderabad", "Nagpur"]

# ── FEATURE SPEC: (output_name, agg_function) ─────────────────────────
# These are computed from daily columns in the labelled CSV.

MONTHLY_AGG = {
    # Temperature
    "temp_max_mean"       : ("temp_max",   "mean"),
    "temp_max_max"        : ("temp_max",   "max"),
    "temp_max_p90"        : ("temp_max",   lambda x: x.quantile(0.90)),
    "temp_max_p95"        : ("temp_max",   lambda x: x.quantile(0.95)),
    "temp_min_mean"       : ("temp_min",   "mean"),
    "temp_range_mean"     : ("temp_range", "mean"),
    "temp_departure_mean" : ("temp_departure", "mean"),
    # Heat stress
    "heat_index_mean"     : ("heat_index", "mean"),
    "heat_index_max"      : ("heat_index", "max"),
    "humidex_mean"        : ("humidex",    "mean"),
    # Humidity / Wind
    "humidity_mean"       : ("humidity",   "mean"),
    "wind_mean"           : ("wind",       "mean"),
    # Rainfall / Drought
    "rainfall_total"      : ("rainfall",   "sum"),
    "rainfall_days"       : ("rainfall",   lambda x: (x > 1.0).sum()),
    "drought_days"        : ("drought_flag","sum"),
    "dry_streak_max"      : ("dry_days_streak","max"),
    # AQI
    "aqi_mean"            : ("aqi",        "mean"),
    "aqi_max"             : ("aqi",        "max"),
    "aqi_p75"             : ("aqi",        lambda x: x.quantile(0.75)),
    # Risk events
    "n_heatwave_days"     : ("temp_max",   lambda x: (x >= 40).sum()),
    "n_severe_hw_days"    : ("temp_max",   lambda x: (x >= 45).sum()),
    "n_compound_days"     : ("compound_heat_aqi",
                             lambda x: ((x == 1) | False).sum()),  # patched below
    "n_triple_compound"   : ("triple_compound", "sum"),
    "max_consec_hot"      : ("consec_hot_days",  "max"),
    "n_high_risk_days"    : ("risk_level", lambda x: (x >= 2).sum()),
    "n_severe_risk_days"  : ("risk_level", lambda x: (x == 3).sum()),
    # Composite
    "composite_mean"      : ("composite_score", "mean"),
    "composite_max"       : ("composite_score", "max"),
    "composite_p75"       : ("composite_score", lambda x: x.quantile(0.75)),
    "n_days"              : ("temp_max",   "count"),
}


def aggregate_monthly(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """
    Aggregates a city's daily labelled dataframe into monthly features.
    Returns one row per year-month.
    """
    sub = df[df["city"] == city].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date").reset_index(drop=True)

    # Fill missing columns with 0 to avoid KeyErrors
    for col in ["temp_range", "temp_departure", "heat_index", "humidex",
                "drought_flag", "dry_days_streak", "compound_heat_aqi",
                "compound_heat_drought", "compound_heat_humidity",
                "triple_compound", "consec_hot_days", "composite_score",
                "risk_level"]:
        if col not in sub.columns:
            sub[col] = 0

    sub["year_month"] = sub["date"].dt.to_period("M")

    rows = []
    for ym, grp in sub.groupby("year_month"):
        row = {"year": ym.year, "month": ym.month, "city": city,
               "year_month": str(ym)}

        for feat, (src_col, agg_fn) in MONTHLY_AGG.items():
            if src_col not in grp.columns:
                row[feat] = np.nan
                continue
            if callable(agg_fn):
                row[feat] = agg_fn(grp[src_col])
            else:
                row[feat] = getattr(grp[src_col], agg_fn)()

        # Compound days: any of the three flags active on same day
        compound_any = (
            (grp.get("compound_heat_aqi", 0) == 1) |
            (grp.get("compound_heat_drought", 0) == 1) |
            (grp.get("compound_heat_humidity", 0) == 1)
        )
        row["n_compound_days"] = int(compound_any.sum())

        # Fraction of high-risk days
        row["high_risk_fraction"] = (
            row["n_high_risk_days"] / row["n_days"]
            if row["n_days"] > 0 else 0
        )

        # Binary targets
        row["has_severe_day"]     = int(row["n_severe_risk_days"] > 0)
        row["has_compound_event"] = int(row["n_compound_days"]    > 0)

        # Calendar / season
        row["season"] = {
            12:"Winter",1:"Winter",2:"Winter",
            3:"Pre-Monsoon",4:"Pre-Monsoon",5:"Pre-Monsoon",
            6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",
            10:"Post-Monsoon",11:"Post-Monsoon"
        }.get(row["month"], "Unknown")
        row["is_premonsoon"] = int(row["month"] in [3, 4, 5])
        row["is_monsoon"]    = int(row["month"] in [6, 7, 8, 9])

        rows.append(row)

    monthly_df = pd.DataFrame(rows)
    if monthly_df.empty:
        return monthly_df

    monthly_df = monthly_df.sort_values(["year", "month"]).reset_index(drop=True)

    # ── LAG FEATURES ────────────────────────────────────────────────────
    # Lag 1 month, 2 months, 12 months (same month last year)
    lag_cols = ["temp_max_mean", "temp_max_max", "composite_mean",
                "n_heatwave_days", "n_high_risk_days", "aqi_mean",
                "rainfall_total", "drought_days", "heat_index_mean",
                "high_risk_fraction"]

    for col in lag_cols:
        if col not in monthly_df.columns:
            continue
        monthly_df[f"{col}_lag1"]  = monthly_df[col].shift(1)
        monthly_df[f"{col}_lag2"]  = monthly_df[col].shift(2)
        monthly_df[f"{col}_lag12"] = monthly_df[col].shift(12)

    # ── YEAR-OVER-YEAR DEPARTURE ─────────────────────────────────────────
    # For each month, compute its historical mean across all years
    clim_mean = monthly_df.groupby("month")["temp_max_mean"].transform("mean")
    clim_std  = monthly_df.groupby("month")["temp_max_mean"].transform("std").replace(0, 1e-6)
    monthly_df["warming_signal"]  = monthly_df["temp_max_mean"] - clim_mean
    monthly_df["temp_max_zscore"] = (monthly_df["temp_max_mean"] - clim_mean) / clim_std

    # ── MONTHLY RISK LABEL ───────────────────────────────────────────────
    monthly_df["monthly_risk_level"] = assign_monthly_risk(monthly_df)

    return monthly_df


def assign_monthly_risk(df: pd.DataFrame) -> pd.Series:
    """
    Assigns a 4-level monthly risk label based on:
      1. composite_mean → base classification (thresholds differ from daily)
      2. Hard upgrades if # of event-days exceeds thresholds
    """
    score = df["composite_mean"].fillna(0)

    # Monthly thresholds (lower than daily because we're using averages)
    risk = pd.cut(
        score,
        bins=[-0.1, 20, 35, 55, 100],
        labels=[0, 1, 2, 3]
    ).astype(float).fillna(0).astype(int)

    # Hard upgrade rules aligned with IMD standards
    # ≥ 5 heatwave days in month → at least High
    hw_mask = df.get("n_heatwave_days", pd.Series(0, index=df.index)) >= 5
    risk = risk.where(~hw_mask, risk.clip(lower=2))

    # ≥ 3 severe risk days in month → at least High
    sev_mask = df.get("n_severe_risk_days", pd.Series(0, index=df.index)) >= 3
    risk = risk.where(~sev_mask, risk.clip(lower=2))

    # Any severe heat wave day (Tmax ≥ 45) → at least High
    shw_mask = df.get("n_severe_hw_days", pd.Series(0, index=df.index)) >= 1
    risk = risk.where(~shw_mask, risk.clip(lower=2))

    # high_risk_fraction > 0.5 (majority of month is High) → force Severe
    hrf_mask = df.get("high_risk_fraction", pd.Series(0, index=df.index)) > 0.5
    risk = risk.where(~hrf_mask, risk.clip(lower=3))

    return risk


# ── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STEP 8 — SEASONAL FEATURE ENGINEERING")
    print("="*60)

    path = os.path.join(PROC_DIR, "labelled_all.csv")
    if not os.path.exists(path):
        print("[!] labelled_all.csv not found. Run steps 1–2 first.")
        exit(1)

    df_all = pd.read_csv(path, parse_dates=["date"])
    print(f"\n  Daily data: {df_all.shape[0]:,} rows, "
          f"{df_all['city'].nunique()} cities, "
          f"{df_all['date'].min().date()} → {df_all['date'].max().date()}")

    all_monthly = []
    for city in CITIES:
        if city not in df_all["city"].values:
            print(f"  [!] {city} not found in data — skipping")
            continue
        monthly = aggregate_monthly(df_all, city)
        if monthly.empty:
            print(f"  [!] {city}: no monthly data produced")
            continue

        out = os.path.join(PROC_DIR, f"monthly_features_{city}.csv")
        monthly.to_csv(out, index=False)

        dist = monthly["monthly_risk_level"].value_counts().sort_index()
        labels = {0:"Low",1:"Moderate",2:"High",3:"Severe"}
        print(f"\n  {city}: {len(monthly)} months  "
              f"({monthly['year'].min()}–{monthly['year'].max()})")
        for lvl, cnt in dist.items():
            bar = "█" * int(cnt / max(dist.max(), 1) * 20)
            print(f"    {labels.get(int(lvl),'?'):10s}: {cnt:3d} months  {bar}")
        print(f"  [✓] Saved → {out}")
        print(f"  Features: {len(monthly.columns)} columns")
        all_monthly.append(monthly)

    if all_monthly:
        combined = pd.concat(all_monthly, ignore_index=True)
        out_all  = os.path.join(PROC_DIR, "monthly_features_all.csv")
        combined.to_csv(out_all, index=False)
        print(f"\n[✓] Combined: {combined.shape} → {out_all}")

    print("\n[Done] Step 8 complete. Run step9_seasonal_modeling.py next.\n")