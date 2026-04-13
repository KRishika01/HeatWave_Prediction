# """
# ========================================================================
# HEATWAVE RISK PREDICTION SYSTEM
# Step 2: Risk Label Definition & Composite Scoring
# ========================================================================
# Defines a multi-dimensional heatwave risk score (0–100) and categorises
# each day into one of four risk levels:
#     0 = Low      (score  0–24)
#     1 = Moderate (score 25–49)
#     2 = High     (score 50–74)
#     3 = Severe   (score 75–100)

# Scoring pillars (weighted):
#     Temperature   35 %
#     Heat Index    25 %
#     AQI           20 %
#     Drought       10 %
#     Compounding   10 %
# ========================================================================
# """

# import pandas as pd
# import numpy as np
# import os

# PROC_DIR = "data/processed"
# CITIES   = ["Delhi", "Hyderabad", "Nagpur"]

# # ── PILLAR WEIGHTS ───────────────────────────────────────────────────────
# W_TEMP    = 0.35
# W_HI      = 0.25
# W_AQI     = 0.20
# W_DROUGHT = 0.10
# W_COMPOUND= 0.10

# # ── SCORING HELPERS ──────────────────────────────────────────────────────

# def temp_score(temp_max: pd.Series) -> pd.Series:
#     """
#     0 pts  → <35°C
#     25 pts → 35–37.9°C  (warm)
#     50 pts → 38–39.9°C  (hot)
#     75 pts → 40–44.9°C  (heat wave)
#     100 pts→ ≥45°C      (severe heat wave)
#     """
#     s = pd.cut(temp_max,
#                bins=[-np.inf, 35, 38, 40, 45, np.inf],
#                labels=[0, 25, 50, 75, 100]).astype(float)
#     return s

# def heat_index_score(heat_index: pd.Series) -> pd.Series:
#     """
#     Apparent temperature score (how body feels).
#     """
#     s = pd.cut(heat_index,
#                bins=[-np.inf, 27, 32, 41, 54, np.inf],
#                labels=[0, 25, 50, 75, 100]).astype(float)
#     return s

# def aqi_score(aqi: pd.Series) -> pd.Series:
#     """
#     AQI bands → risk score.
#     0–50 Good, 51–100 Satisfactory, 101–200 Moderate,
#     201–300 Poor, 301–400 Very Poor, 401+ Severe
#     """
#     s = pd.cut(aqi,
#                bins=[-np.inf, 50, 100, 200, 300, 400, np.inf],
#                labels=[0, 15, 40, 60, 80, 100]).astype(float)
#     return s

# def drought_score(drought_flag: pd.Series, dry_days: pd.Series,
#                   spi_30: pd.Series) -> pd.Series:
#     """
#     Composite drought component.
#     """
#     score = np.zeros(len(drought_flag))
#     score += drought_flag * 40                      # active drought +40
#     score += (dry_days.clip(0, 15) / 15) * 30       # consecutive dry streak up to 30 pts
#     score += (-spi_30.clip(-3, 0) / 3) * 30         # negative SPI (dry) up to 30 pts
#     return pd.Series(np.clip(score, 0, 100), index=drought_flag.index)

# def compound_score(df: pd.DataFrame) -> pd.Series:
#     """
#     Extra penalty for compound events.
#     """
#     score = np.zeros(len(df))
#     score += df["compound_heat_aqi"]      * 30
#     score += df["compound_heat_drought"]  * 25
#     score += df["compound_heat_humidity"] * 20
#     score += df["triple_compound"]        * 25      # extra penalty if all three
#     score += (df["consec_hot_days"].clip(0, 7) / 7) * 20  # sustained heat
#     return pd.Series(np.clip(score, 0, 100), index=df.index)

# # ── MAIN LABELLING FUNCTION ──────────────────────────────────────────────

# def assign_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Adds columns:
#         score_temp, score_hi, score_aqi, score_drought, score_compound
#         composite_score  (0–100)
#         risk_level       (0=Low, 1=Moderate, 2=High, 3=Severe)
#         risk_label       (text)
#     """
#     df = df.copy()

#     df["score_temp"]     = temp_score(df["temp_max"])
#     df["score_hi"]       = heat_index_score(df["heat_index"])
#     df["score_aqi"]      = aqi_score(df["aqi"])
#     df["score_drought"]  = drought_score(df["drought_flag"],
#                                          df["dry_days_streak"],
#                                          df["spi_30"])
#     df["score_compound"] = compound_score(df)

#     df["composite_score"] = (
#         W_TEMP     * df["score_temp"]     +
#         W_HI       * df["score_hi"]       +
#         W_AQI      * df["score_aqi"]      +
#         W_DROUGHT  * df["score_drought"]  +
#         W_COMPOUND * df["score_compound"]
#     ).round(2)

#     # ── IMD OVERRIDE: Force "High" if actual heat-wave conditions met ────
#     hw_flag  = df["temp_max"] >= 40.0
#     shw_flag = df["temp_max"] >= 45.0
#     df.loc[hw_flag,  "composite_score"] = df.loc[hw_flag,  "composite_score"].clip(lower=50)
#     df.loc[shw_flag, "composite_score"] = df.loc[shw_flag, "composite_score"].clip(lower=75)

#     # ── CLASSIFY ─────────────────────────────────────────────────────────
#     df["risk_level"] = pd.cut(df["composite_score"],
#                               bins=[-0.1, 24.9, 49.9, 74.9, 100.1],
#                               labels=[0, 1, 2, 3])
    
#     ###------------------------------ADDED-------------------------------###
#     df["risk_level"] = df["risk_level"].astype(float).fillna(0).astype(int)

#     label_map = {0: "Low", 1: "Moderate", 2: "High", 3: "Severe"}
#     df["risk_label"] = df["risk_level"].map(label_map)

#     return df


# def print_label_stats(df: pd.DataFrame, city: str):
#     print(f"\n  {city}:")
#     dist = df["risk_label"].value_counts()
#     total = len(df)
#     for lvl in ["Low", "Moderate", "High", "Severe"]:
#         cnt = dist.get(lvl, 0)
#         pct = cnt / total * 100
#         bar = "█" * int(pct / 2)
#         print(f"    {lvl:10s}: {cnt:5d} days  ({pct:5.1f}%)  {bar}")


# # ── RUN ──────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     print("\n" + "="*60)
#     print("  STEP 2 ─ RISK LABEL ASSIGNMENT")
#     print("="*60)
#     print("""
#   Scoring Pillars & Weights:
#     Temperature Component  : 35%
#     Heat Index Component   : 25%
#     AQI Component          : 20%
#     Drought Component      : 10%
#     Compound Risk Bonus    : 10%

#   Risk Levels:
#     0 Low      → composite score  0–24
#     1 Moderate → composite score 25–49
#     2 High     → composite score 50–74
#     3 Severe   → composite score 75–100
#     """)

#     labelled_dfs = []
#     for city in CITIES:
#         path = os.path.join(PROC_DIR, f"processed_{city}.csv")
#         if not os.path.exists(path):
#             print(f"[!] {city}: processed file not found. Run step1 first.")
#             continue
#         df = pd.read_csv(path, parse_dates=["date"])
#         df = assign_risk_labels(df)
#         out = os.path.join(PROC_DIR, f"labelled_{city}.csv")
#         df.to_csv(out, index=False)
#         print_label_stats(df, city)
#         labelled_dfs.append(df)
#         print(f"  [✓] Saved → {out}")

#     if labelled_dfs:
#         combined = pd.concat(labelled_dfs, ignore_index=True)
#         combined.to_csv(os.path.join(PROC_DIR, "labelled_all.csv"), index=False)
#         print(f"\n[✓] Combined labelled dataset: {combined.shape}")
#     print("\n[Done] Risk labelling complete.\n")



"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Step 2: Risk Label Definition & Composite Scoring
========================================================================
Defines a multi-dimensional heatwave risk score (0–100) and categorises
each day into one of four risk levels:
    0 = Low      (score  0–24)
    1 = Moderate (score 25–49)
    2 = High     (score 50–74)
    3 = Severe   (score 75–100)

Scoring pillars (weighted):
    Temperature   35 %
    Heat Index    25 %
    AQI           20 %
    Drought       10 %
    Compounding   10 %
========================================================================
"""

import pandas as pd
import numpy as np
import os, json

PROC_DIR = "data/processed"
CITIES   = ["Delhi", "Hyderabad", "Nagpur"]

# ── PILLAR WEIGHTS ───────────────────────────────────────────────────────
# Loads optimised weights from step2b if available; else uses manual values.
_weights_path = os.path.join(PROC_DIR, "optimal_weights.json")
if os.path.exists(_weights_path):
    with open(_weights_path) as _f:
        _wdata = json.load(_f)
    _w = _wdata["best_weights"]
    W_TEMP, W_HI, W_AQI, W_DROUGHT, W_COMPOUND = _w
    print(f"[step2] Optimal weights loaded (method: {_wdata['best_method']})")
    print(f"        T={W_TEMP:.4f}  HI={W_HI:.4f}  AQI={W_AQI:.4f}  "
          f"Drought={W_DROUGHT:.4f}  Compound={W_COMPOUND:.4f}")
else:
    W_TEMP    = 0.35
    W_HI      = 0.25
    W_AQI     = 0.20
    W_DROUGHT = 0.10
    W_COMPOUND= 0.10
    print("[step2] Using manual pillar weights (run step2b to optimise)")

# ── SCORING HELPERS ──────────────────────────────────────────────────────

def temp_score(temp_max: pd.Series) -> pd.Series:
    """
    0 pts  → <35°C
    25 pts → 35–37.9°C  (warm)
    50 pts → 38–39.9°C  (hot)
    75 pts → 40–44.9°C  (heat wave)
    100 pts→ ≥45°C      (severe heat wave)
    """
    s = pd.cut(temp_max,
               bins=[-np.inf, 35, 38, 40, 45, np.inf],
               labels=[0, 25, 50, 75, 100]).astype(float)
    return s

def heat_index_score(heat_index: pd.Series) -> pd.Series:
    """
    Apparent temperature score (how body feels).
    """
    s = pd.cut(heat_index,
               bins=[-np.inf, 27, 32, 41, 54, np.inf],
               labels=[0, 25, 50, 75, 100]).astype(float)
    return s

def aqi_score(aqi: pd.Series) -> pd.Series:
    """
    AQI bands → risk score.
    0–50 Good, 51–100 Satisfactory, 101–200 Moderate,
    201–300 Poor, 301–400 Very Poor, 401+ Severe
    """
    s = pd.cut(aqi,
               bins=[-np.inf, 50, 100, 200, 300, 400, np.inf],
               labels=[0, 15, 40, 60, 80, 100]).astype(float)
    return s

def drought_score(drought_flag: pd.Series, dry_days: pd.Series,
                  spi_30: pd.Series) -> pd.Series:
    """
    Composite drought component.
    """
    score = np.zeros(len(drought_flag))
    score += drought_flag * 40                      # active drought +40
    score += (dry_days.clip(0, 15) / 15) * 30       # consecutive dry streak up to 30 pts
    score += (-spi_30.clip(-3, 0) / 3) * 30         # negative SPI (dry) up to 30 pts
    return pd.Series(np.clip(score, 0, 100), index=drought_flag.index)

def compound_score(df: pd.DataFrame) -> pd.Series:
    """
    Extra penalty for compound events.
    """
    score = np.zeros(len(df))
    score += df["compound_heat_aqi"]      * 30
    score += df["compound_heat_drought"]  * 25
    score += df["compound_heat_humidity"] * 20
    score += df["triple_compound"]        * 25      # extra penalty if all three
    score += (df["consec_hot_days"].clip(0, 7) / 7) * 20  # sustained heat
    return pd.Series(np.clip(score, 0, 100), index=df.index)

# ── MAIN LABELLING FUNCTION ──────────────────────────────────────────────

def assign_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns:
        score_temp, score_hi, score_aqi, score_drought, score_compound
        composite_score  (0–100)
        risk_level       (0=Low, 1=Moderate, 2=High, 3=Severe)
        risk_label       (text)
    """
    df = df.copy()

    df["score_temp"]     = temp_score(df["temp_max"])
    df["score_hi"]       = heat_index_score(df["heat_index"])
    df["score_aqi"]      = aqi_score(df["aqi"])
    df["score_drought"]  = drought_score(df["drought_flag"],
                                         df["dry_days_streak"],
                                         df["spi_30"])
    df["score_compound"] = compound_score(df)

    df["composite_score"] = (
        W_TEMP     * df["score_temp"]     +
        W_HI       * df["score_hi"]       +
        W_AQI      * df["score_aqi"]      +
        W_DROUGHT  * df["score_drought"]  +
        W_COMPOUND * df["score_compound"]
    ).round(2)

    # ── IMD OVERRIDE: Force "High" if actual heat-wave conditions met ────
    hw_flag  = df["temp_max"] >= 40.0
    shw_flag = df["temp_max"] >= 45.0
    df.loc[hw_flag,  "composite_score"] = df.loc[hw_flag,  "composite_score"].clip(lower=50)
    df.loc[shw_flag, "composite_score"] = df.loc[shw_flag, "composite_score"].clip(lower=75)

    # ── CLASSIFY ─────────────────────────────────────────────────────────
    # Clip score to [0, 100] so no value falls outside bin edges
    # df["composite_score"] = df["composite_score"].clip(0, 100)

    df["risk_level"] = pd.cut(
        df["composite_score"],
        bins=[-0.1, 24.9, 49.9, 74.9, 100.1],
        labels=[0, 1, 2, 3]
    )
    df["risk_level"] = df["risk_level"].astype(float).fillna(0).astype(int)

    label_map = {0: "Low", 1: "Moderate", 2: "High", 3: "Severe"}
    df["risk_label"] = df["risk_level"].map(label_map)

    return df


def print_label_stats(df: pd.DataFrame, city: str):
    print(f"\n  {city}:")
    dist = df["risk_label"].value_counts()
    total = len(df)
    for lvl in ["Low", "Moderate", "High", "Severe"]:
        cnt = dist.get(lvl, 0)
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {lvl:10s}: {cnt:5d} days  ({pct:5.1f}%)  {bar}")


# ── RUN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STEP 2 ─ RISK LABEL ASSIGNMENT")
    print("="*60)
    print("""
  Scoring Pillars & Weights:
    Temperature Component  : 35%
    Heat Index Component   : 25%
    AQI Component          : 20%
    Drought Component      : 10%
    Compound Risk Bonus    : 10%

  Risk Levels:
    0 Low      → composite score  0–24
    1 Moderate → composite score 25–49
    2 High     → composite score 50–74
    3 Severe   → composite score 75–100
    """)

    labelled_dfs = []
    for city in CITIES:
        path = os.path.join(PROC_DIR, f"processed_{city}.csv")
        if not os.path.exists(path):
            print(f"[!] {city}: processed file not found. Run step1 first.")
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        df = assign_risk_labels(df)
        out = os.path.join(PROC_DIR, f"labelled_{city}.csv")
        df.to_csv(out, index=False)
        print_label_stats(df, city)
        labelled_dfs.append(df)
        print(f"  [✓] Saved → {out}")

    if labelled_dfs:
        combined = pd.concat(labelled_dfs, ignore_index=True)
        combined.to_csv(os.path.join(PROC_DIR, "labelled_all.csv"), index=False)
        print(f"\n[✓] Combined labelled dataset: {combined.shape}")
    print("\n[Done] Risk labelling complete.\n")