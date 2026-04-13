# """
# ========================================================================
# HEATWAVE RISK PREDICTION SYSTEM
# Step 1: Data Preprocessing & Feature Engineering
# ========================================================================
# Input  : Raw CSV files for Delhi, Hyderabad, Nagpur
#          Columns: DATE, TEMPERATURE_MAX, TEMPERATURE_MIN, HUMIDITY, WIND, RAINFALL, AQI
# Output : Preprocessed DataFrames with 25+ engineered features saved as
#          processed_<city>.csv in /processed/ folder
# ========================================================================
# """

# import pandas as pd
# import numpy as np
# import os
# import warnings
# warnings.filterwarnings("ignore")

# # ── CONFIG ──────────────────────────────────────────────────────────────
# CITIES      = ["Delhi", "Hyderabad", "Nagpur"]
# RAW_DIR     = "data/raw"          # put your CSVs here
# PROC_DIR    = "data/processed"
# os.makedirs(PROC_DIR, exist_ok=True)

# # Expected column name map (rename your CSV headers if different)
# COL_MAP = {
#     "DATE"     : "date",
#     "TEMPMAX"  : "temp_max",
#     "TEMPMIN"  : "temp_min",
#     "HUMIDITY" : "humidity",
#     "WIND"     : "wind",     
#     "RAINFALL" : "rainfall",
#     "AQI"      : "aqi",
# }

# # ── IMD HEATWAVE THRESHOLDS (°C) ────────────────────────────────────────
# # Plains: Heat Wave ≥ 40 °C  |  Severe ≥ 45 °C
# HEAT_WAVE_THRESHOLD   = 40.0
# SEVERE_HW_THRESHOLD   = 45.0

# # ── HELPER: HEAT INDEX (Steadman / NWS formula) ─────────────────────────
# def compute_heat_index(T, RH):
#     """
#     Rothfusz regression (NWS).
#     T  : air temperature in °C
#     RH : relative humidity in %
#     Returns apparent temperature (°C)
#     """
#     # Convert to Fahrenheit for formula, convert result back
#     TF = T * 9/5 + 32
#     HI = (-42.379
#           + 2.04901523 * TF
#           + 10.14333127 * RH
#           - 0.22475541 * TF * RH
#           - 0.00683783 * TF**2
#           - 0.05481717 * RH**2
#           + 0.00122874 * TF**2 * RH
#           + 0.00085282 * TF * RH**2
#           - 0.00000199 * TF**2 * RH**2)
#     HI_C = (HI - 32) * 5/9
#     return HI_C

# # ── HELPER: HUMIDEX (Canadian formula) ─────────────────────────────────
# def compute_humidex(T, RH):
#     """
#     Humidex = T + 0.5555 * (6.11 * e^(5417.7530*(1/273.16 - 1/(273.15+Td))) - 10)
#     Simplified using dew-point approximation.
#     """
#     Td = T - ((100 - RH) / 5)          # Magnus dew-point approx
#     e  = 6.105 * np.exp(25.22 * (Td - 273.16) / Td)   # vapour pressure (hPa)
#     humidex = T + 0.5555 * (e - 10)
#     return humidex

# # ── HELPER: STANDARDISED PRECIPITATION INDEX (drought proxy) ────────────
# def compute_spi_30(rainfall_series):
#     """
#     Rolling 30-day SPI proxy.
#     Positive → wetter than normal; Negative → drier (drought risk).
#     """
#     rolling = rainfall_series.rolling(30, min_periods=7)
#     mean    = rolling.mean()
#     std     = rolling.std().replace(0, 1e-6)
#     return (rainfall_series - mean) / std

# # ── MAIN PREPROCESSING FUNCTION ─────────────────────────────────────────
# def preprocess_city(city: str) -> pd.DataFrame:
#     """
#     Full preprocessing pipeline for one city.
#     Returns enriched DataFrame with 25+ features.
#     """
#     path = os.path.join(RAW_DIR, f"{city}.csv")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"CSV not found: {path}")

#     # ── 1. LOAD ──────────────────────────────────────────────────────────
#     df = pd.read_csv(path)

#     # Flexible column renaming (handles case / spacing variation)
#     df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
#     df.rename(columns=COL_MAP, inplace=True)

#     df["date"] = pd.to_datetime(df["date"])
#     df.sort_values("date", inplace=True)
#     ###------------------ADDED------------------###
#     df.set_index("date", inplace=True)
#     # df.reset_index(drop=True, inplace=True)
#     df["city"] = city

#     # ── 2. HANDLE MISSING VALUES ─────────────────────────────────────────
#     numeric_cols = ["temp_max", "temp_min", "humidity", "wind", "rainfall", "aqi"]
#     for col in numeric_cols:
#         if col == "rainfall":
#             df[col].fillna(0, inplace=True)          # missing rainfall = no rain
#         else:
#             df[col].interpolate(method="time", inplace=True)
#             df[col].fillna(method="bfill", inplace=True)
#             df[col].fillna(method="ffill", inplace=True)

#     df.reset_index(inplace=True)

#     # ── 3. DATE FEATURES ─────────────────────────────────────────────────
#     df["year"]       = df["date"].dt.year
#     df["month"]      = df["date"].dt.month
#     df["day_of_year"]= df["date"].dt.dayofyear
#     df["season"]     = df["month"].map({
#         12:"Winter", 1:"Winter", 2:"Winter",
#         3:"Pre-Monsoon", 4:"Pre-Monsoon", 5:"Pre-Monsoon",
#         6:"Monsoon", 7:"Monsoon", 8:"Monsoon", 9:"Monsoon",
#         10:"Post-Monsoon", 11:"Post-Monsoon"
#     })
#     df["is_summer"]  = df["month"].isin([4, 5, 6]).astype(int)

#     # ── 4. BASIC DERIVED FEATURES ─────────────────────────────────────────
#     df["temp_range"]  = df["temp_max"] - df["temp_min"]          # Diurnal Temp Range
#     df["temp_mean"]   = (df["temp_max"] + df["temp_min"]) / 2

#     df["heat_index"]  = compute_heat_index(df["temp_max"], df["humidity"])
#     df["humidex"]     = compute_humidex(df["temp_max"], df["humidity"])
#     df["feels_like_excess"] = df["heat_index"] - df["temp_max"]  # How much hotter it feels

#     # ── 5. ROLLING / TREND FEATURES ──────────────────────────────────────
#     for window in [3, 7, 14]:
#         df[f"temp_max_roll{window}"]  = df["temp_max"].rolling(window, min_periods=1).mean()
#         df[f"humidity_roll{window}"]  = df["humidity"].rolling(window, min_periods=1).mean()
#         df[f"aqi_roll{window}"]       = df["aqi"].rolling(window, min_periods=1).mean()
#         df[f"rainfall_roll{window}"]  = df["rainfall"].rolling(window, min_periods=1).sum()

#     # ── 6. LAG FEATURES (yesterday / 2 days ago) ──────────────────────────
#     for lag in [1, 2, 3]:
#         df[f"temp_max_lag{lag}"]  = df["temp_max"].shift(lag)
#         df[f"aqi_lag{lag}"]       = df["aqi"].shift(lag)
#         df[f"humidity_lag{lag}"]  = df["humidity"].shift(lag)

#     # ── 7. ANOMALY / DEPARTURE FEATURES ──────────────────────────────────
#     # Monthly climatological mean
#     monthly_mean = df.groupby("month")["temp_max"].transform("mean")
#     monthly_std  = df.groupby("month")["temp_max"].transform("std").replace(0, 1e-6)
#     df["temp_departure"]  = df["temp_max"] - monthly_mean       # °C above monthly mean
#     df["temp_zscore"]     = (df["temp_max"] - monthly_mean) / monthly_std

#     aqi_monthly_mean = df.groupby("month")["aqi"].transform("mean")
#     df["aqi_departure"] = df["aqi"] - aqi_monthly_mean

#     # ── 8. DROUGHT / RAINFALL FEATURES ────────────────────────────────────
#     df["dry_days_streak"]  = (df["rainfall"] < 1.0).groupby(
#                                 (df["rainfall"] >= 1.0).cumsum()).cumcount()
#     df["spi_30"]           = compute_spi_30(df["rainfall"])
#     df["cumrain_7"]        = df["rainfall"].rolling(7, min_periods=1).sum()
#     df["drought_flag"]     = (df["cumrain_7"] < 2.0).astype(int)  # <2mm in 7 days

#     # ── 9. AQI CATEGORY ──────────────────────────────────────────────────
#     bins   = [0, 50, 100, 200, 300, 400, 500]
#     labels = [1, 2, 3, 4, 5, 6]                                 # 1=Good … 6=Severe
#     df["aqi_category"] = pd.cut(df["aqi"], bins=bins, labels=labels).astype(float)

#     # ── 10. CONSECUTIVE HOT DAYS ──────────────────────────────────────────
#     hot = (df["temp_max"] >= HEAT_WAVE_THRESHOLD).astype(int)
#     df["consec_hot_days"] = hot.groupby((hot != hot.shift()).cumsum()).cumcount() + hot

#     # ── 11. COMPOUND RISK FLAGS ───────────────────────────────────────────
#     df["compound_heat_aqi"]      = ((df["temp_max"] >= 38) & (df["aqi"] >= 200)).astype(int)
#     df["compound_heat_drought"]  = ((df["temp_max"] >= 38) & (df["drought_flag"] == 1)).astype(int)
#     df["compound_heat_humidity"] = ((df["temp_max"] >= 38) & (df["humidity"] >= 60)).astype(int)
#     df["triple_compound"]        = (
#         (df["temp_max"] >= 38) & (df["aqi"] >= 150) & (df["drought_flag"] == 1)
#     ).astype(int)

#     # ── 12. WIND HEAT INTERACTION ─────────────────────────────────────────
#     # Low wind + high temp = trapping heat (bad)
#     df["wind_heat_ratio"] = df["temp_max"] / (df["wind"].replace(0, 0.1))

#     # ── SAVE ──────────────────────────────────────────────────────────────
#     out_path = os.path.join(PROC_DIR, f"processed_{city}.csv")
#     df.to_csv(out_path, index=False)
#     print(f"[✓] {city}: {len(df)} rows, {len(df.columns)} features → {out_path}")
#     return df


# # ── RUN ──────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     print("\n" + "="*60)
#     print("  STEP 1 ─ DATA PREPROCESSING & FEATURE ENGINEERING")
#     print("="*60)

#     all_dfs = []
#     for city in CITIES:
#         try:
#             df = preprocess_city(city)
#             all_dfs.append(df)
#         except FileNotFoundError as e:
#             print(f"[!] Skipping {city}: {e}")

#     if all_dfs:
#         combined = pd.concat(all_dfs, ignore_index=True)
#         combined.to_csv(os.path.join(PROC_DIR, "all_cities.csv"), index=False)
#         print(f"\n[✓] Combined dataset: {combined.shape}")
#         print(f"\n  Features generated:")
#         base = ["date","city","temp_max","temp_min","humidity","wind","rainfall","aqi"]
#         new_features = [c for c in combined.columns if c not in base]
#         for i, f in enumerate(new_features, 1):
#             print(f"    {i:2d}. {f}")
#     print("\n[Done] Preprocessing complete.\n")


"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Step 1: Data Preprocessing & Feature Engineering
========================================================================
Input  : Raw CSV files for Delhi, Hyderabad, Nagpur
         Columns: DATE, TEMPERATURE_MAX, TEMPERATURE_MIN, HUMIDITY, WIND, RAINFALL, AQI
Output : Preprocessed DataFrames with 25+ engineered features saved as
         processed_<city>.csv in /processed/ folder
========================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────
CITIES      = ["Delhi", "Hyderabad", "Nagpur"]
RAW_DIR     = "data/raw"          # put your CSVs here
PROC_DIR    = "data/processed"
os.makedirs(PROC_DIR, exist_ok=True)

# Expected column name map (rename your CSV headers if different)
COL_MAP = {
    "DATE"           : "date",
    "TEMPMAX": "temp_max",
    "TEMPMIN": "temp_min",
    "HUMIDITY"       : "humidity",
    "WIND"           : "wind",
    "RAINFALL"       : "rainfall",
    "AQI"            : "aqi",
}

# ── IMD HEATWAVE THRESHOLDS (°C) ────────────────────────────────────────
# Plains: Heat Wave ≥ 40 °C  |  Severe ≥ 45 °C
HEAT_WAVE_THRESHOLD   = 40.0
SEVERE_HW_THRESHOLD   = 45.0

# ── HELPER: HEAT INDEX (Steadman / NWS formula) ─────────────────────────
def compute_heat_index(T, RH):
    """
    Rothfusz regression (NWS).
    T  : air temperature in °C
    RH : relative humidity in %
    Returns apparent temperature (°C)
    """
    # Convert to Fahrenheit for formula, convert result back
    TF = T * 9/5 + 32
    HI = (-42.379
          + 2.04901523 * TF
          + 10.14333127 * RH
          - 0.22475541 * TF * RH
          - 0.00683783 * TF**2
          - 0.05481717 * RH**2
          + 0.00122874 * TF**2 * RH
          + 0.00085282 * TF * RH**2
          - 0.00000199 * TF**2 * RH**2)
    HI_C = (HI - 32) * 5/9
    return HI_C

# ── HELPER: HUMIDEX (Canadian formula) ─────────────────────────────────
def compute_humidex(T, RH):
    """
    Humidex = T + 0.5555 * (6.11 * e^(5417.7530*(1/273.16 - 1/(273.15+Td))) - 10)
    Simplified using dew-point approximation.
    """
    Td = T - ((100 - RH) / 5)          # Magnus dew-point approx
    e  = 6.105 * np.exp(25.22 * (Td - 273.16) / Td)   # vapour pressure (hPa)
    humidex = T + 0.5555 * (e - 10)
    return humidex

# ── HELPER: STANDARDISED PRECIPITATION INDEX (drought proxy) ────────────
def compute_spi_30(rainfall_series):
    """
    Rolling 30-day SPI proxy.
    Positive → wetter than normal; Negative → drier (drought risk).
    """
    rolling = rainfall_series.rolling(30, min_periods=7)
    mean    = rolling.mean()
    std     = rolling.std().replace(0, 1e-6)
    return (rainfall_series - mean) / std

# ── MAIN PREPROCESSING FUNCTION ─────────────────────────────────────────
def preprocess_city(city: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline for one city.
    Returns enriched DataFrame with 25+ features.
    """
    path = os.path.join(RAW_DIR, f"{city}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    # ── 1. LOAD ──────────────────────────────────────────────────────────
    df = pd.read_csv(path)

    # Flexible column renaming (handles case / spacing variation)
    df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
    df.rename(columns=COL_MAP, inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    ###-------------------------ADDED-------------------------###
    df.set_index("date", inplace=True)
    # df.reset_index(drop=True, inplace=True)
    df["city"] = city

    # ── 2. HANDLE MISSING VALUES ─────────────────────────────────────────
    numeric_cols = ["temp_max", "temp_min", "humidity", "wind", "rainfall", "aqi"]
    for col in numeric_cols:
        if col == "rainfall":
            df[col].fillna(0, inplace=True)          # missing rainfall = no rain
        else:
            df[col].interpolate(method="time", inplace=True)
            df[col].fillna(method="bfill", inplace=True)
            df[col].fillna(method="ffill", inplace=True)
    
    ###-------------------------ADDED-------------------------###
    df.reset_index(inplace=True)

    # ── 3. DATE FEATURES ─────────────────────────────────────────────────
    df["year"]       = df["date"].dt.year
    df["month"]      = df["date"].dt.month
    df["day_of_year"]= df["date"].dt.dayofyear
    df["season"]     = df["month"].map({
        12:"Winter", 1:"Winter", 2:"Winter",
        3:"Pre-Monsoon", 4:"Pre-Monsoon", 5:"Pre-Monsoon",
        6:"Monsoon", 7:"Monsoon", 8:"Monsoon", 9:"Monsoon",
        10:"Post-Monsoon", 11:"Post-Monsoon"
    })
    df["is_summer"]  = df["month"].isin([4, 5, 6]).astype(int)

    # ── 4. BASIC DERIVED FEATURES ─────────────────────────────────────────
    df["temp_range"]  = df["temp_max"] - df["temp_min"]          # Diurnal Temp Range
    df["temp_mean"]   = (df["temp_max"] + df["temp_min"]) / 2

    df["heat_index"]  = compute_heat_index(df["temp_max"], df["humidity"])
    df["humidex"]     = compute_humidex(df["temp_max"], df["humidity"])
    df["feels_like_excess"] = df["heat_index"] - df["temp_max"]  # How much hotter it feels

    # ── 5. ROLLING / TREND FEATURES ──────────────────────────────────────
    for window in [3, 7, 14]:
        df[f"temp_max_roll{window}"]  = df["temp_max"].rolling(window, min_periods=1).mean()
        df[f"humidity_roll{window}"]  = df["humidity"].rolling(window, min_periods=1).mean()
        df[f"aqi_roll{window}"]       = df["aqi"].rolling(window, min_periods=1).mean()
        df[f"rainfall_roll{window}"]  = df["rainfall"].rolling(window, min_periods=1).sum()

    # ── 6. LAG FEATURES (yesterday / 2 days ago) ──────────────────────────
    for lag in [1, 2, 3]:
        df[f"temp_max_lag{lag}"]  = df["temp_max"].shift(lag)
        df[f"aqi_lag{lag}"]       = df["aqi"].shift(lag)
        df[f"humidity_lag{lag}"]  = df["humidity"].shift(lag)

    # ── 7. ANOMALY / DEPARTURE FEATURES ──────────────────────────────────
    # Monthly climatological mean
    monthly_mean = df.groupby("month")["temp_max"].transform("mean")
    monthly_std  = df.groupby("month")["temp_max"].transform("std").replace(0, 1e-6)
    df["temp_departure"]  = df["temp_max"] - monthly_mean       # °C above monthly mean
    df["temp_zscore"]     = (df["temp_max"] - monthly_mean) / monthly_std

    aqi_monthly_mean = df.groupby("month")["aqi"].transform("mean")
    df["aqi_departure"] = df["aqi"] - aqi_monthly_mean

    # ── 8. DROUGHT / RAINFALL FEATURES ────────────────────────────────────
    df["dry_days_streak"]  = (df["rainfall"] < 1.0).groupby(
                                (df["rainfall"] >= 1.0).cumsum()).cumcount()
    df["spi_30"]           = compute_spi_30(df["rainfall"])
    df["cumrain_7"]        = df["rainfall"].rolling(7, min_periods=1).sum()
    df["drought_flag"]     = (df["cumrain_7"] < 2.0).astype(int)  # <2mm in 7 days

    # ── 9. AQI CATEGORY ──────────────────────────────────────────────────
    bins   = [0, 50, 100, 200, 300, 400, 500]
    labels = [1, 2, 3, 4, 5, 6]                                 # 1=Good … 6=Severe
    df["aqi_category"] = pd.cut(df["aqi"], bins=bins, labels=labels).astype(float)

    # ── 10. CONSECUTIVE HOT DAYS ──────────────────────────────────────────
    hot = (df["temp_max"] >= HEAT_WAVE_THRESHOLD).astype(int)
    df["consec_hot_days"] = hot.groupby((hot != hot.shift()).cumsum()).cumcount() + hot

    # ── 11. COMPOUND RISK FLAGS ───────────────────────────────────────────
    df["compound_heat_aqi"]      = ((df["temp_max"] >= 38) & (df["aqi"] >= 200)).astype(int)
    df["compound_heat_drought"]  = ((df["temp_max"] >= 38) & (df["drought_flag"] == 1)).astype(int)
    df["compound_heat_humidity"] = ((df["temp_max"] >= 38) & (df["humidity"] >= 60)).astype(int)
    df["triple_compound"]        = (
        (df["temp_max"] >= 38) & (df["aqi"] >= 150) & (df["drought_flag"] == 1)
    ).astype(int)

    # ── 12. WIND HEAT INTERACTION ─────────────────────────────────────────
    # Low wind + high temp = trapping heat (bad)
    df["wind_heat_ratio"] = df["temp_max"] / (df["wind"].replace(0, 0.1))

    # ── SAVE ──────────────────────────────────────────────────────────────
    # Guard: if a processed file already exists and contains rows NEWER
    # than the raw CSV (added by step7b or step7 daily appends), preserve
    # those rows by merging them in before saving.
    out_path = os.path.join(PROC_DIR, f"processed_{city}.csv")
    if os.path.exists(out_path):
        existing_proc = pd.read_csv(out_path, parse_dates=["date"])
        raw_last_date = df["date"].max()
        newer_rows    = existing_proc[existing_proc["date"] > raw_last_date]
        if len(newer_rows) > 0:
            # Re-engineer features for newer rows using the full df as history
            # (simple approach: concat raw-based features + preserve newer rows)
            df = pd.concat([df, newer_rows], ignore_index=True)
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"[i] {city}: preserved {len(newer_rows)} newer rows "
                  f"(from step7b/step7) beyond raw CSV end date "
                  f"({raw_last_date.date()})")

    df.to_csv(out_path, index=False)
    print(f"[✓] {city}: {len(df)} rows, {len(df.columns)} features → {out_path}")
    return df


# ── RUN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STEP 1 ─ DATA PREPROCESSING & FEATURE ENGINEERING")
    print("="*60)

    all_dfs = []
    for city in CITIES:
        try:
            df = preprocess_city(city)
            all_dfs.append(df)
        except FileNotFoundError as e:
            print(f"[!] Skipping {city}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(PROC_DIR, "all_cities.csv"), index=False)
        print(f"\n[✓] Combined dataset: {combined.shape}")
        print(f"\n  Features generated:")
        base = ["date","city","temp_max","temp_min","humidity","wind","rainfall","aqi"]
        new_features = [c for c in combined.columns if c not in base]
        for i, f in enumerate(new_features, 1):
            print(f"    {i:2d}. {f}")
    print("\n[Done] Preprocessing complete.\n")