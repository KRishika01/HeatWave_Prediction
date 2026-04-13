# """
# ========================================================================
# HEATWAVE RISK PREDICTION SYSTEM
# Step 5: Daily Prediction Inference Engine
# ========================================================================
# Given today's raw observations (temp_max, temp_min, humidity,
# wind, rainfall, aqi) + the last N days of history:

# Outputs:
#   • Predicted risk level (Low / Moderate / High / Severe)
#   • Composite risk score (0–100)
#   • Per-pillar score breakdown (Temperature, Heat Index, AQI, ...)
#   • Compound risk flags active today
#   • Prediction confidence (class probabilities)
#   • Natural-language advisory message
#   • Tomorrow's predicted risk (next-day forward prediction)
# ========================================================================
# """

# import pandas as pd
# import numpy as np
# import pickle
# import os
# import json
# from datetime import datetime, timedelta

# MODEL_DIR = "models"
# PROC_DIR  = "data/processed"

# FEATURE_COLS = [
#     "temp_max", "temp_min", "humidity", "wind", "rainfall", "aqi",
#     "heat_index", "humidex", "temp_range", "temp_mean", "feels_like_excess",
#     "wind_heat_ratio",
#     "temp_max_roll3", "temp_max_roll7", "humidity_roll7", "aqi_roll7",
#     "rainfall_roll7",
#     "temp_max_lag1", "temp_max_lag2", "aqi_lag1", "humidity_lag1",
#     "temp_departure", "temp_zscore", "aqi_departure",
#     "dry_days_streak", "spi_30", "drought_flag",
#     "month", "day_of_year", "is_summer",
#     "compound_heat_aqi", "compound_heat_drought", "compound_heat_humidity",
#     "triple_compound", "consec_hot_days", "aqi_category",
# ]

# RISK_LABELS   = {0: "Low", 1: "Moderate", 2: "High", 3: "Severe"}
# RISK_EMOJIS   = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}

# # ── ADVISORY TEMPLATES ────────────────────────────────────────────────────
# ADVISORIES = {
#     0: "No significant heat risk. Normal outdoor activities are safe.",
#     1: ("Moderate heat conditions. Stay hydrated, avoid prolonged outdoor "
#         "activity during peak hours (12–4 PM). Monitor elderly and children."),
#     2: ("HIGH HEAT RISK. Limit outdoor exertion. Drink water regularly. "
#         "Avoid direct sun exposure. Seek air-conditioned spaces if possible. "
#         "Watch for signs of heat exhaustion."),
#     3: ("⚠ SEVERE HEAT WAVE CONDITIONS. AVOID all non-essential outdoor activity. "
#         "Drink 2–3 litres of water daily. Stay in cool/shaded environments. "
#         "Emergency services should be on alert for heat-related illness. "
#         "Vulnerable populations (elderly, infants, outdoor workers) at extreme risk."),
# }

# COMPOUND_ADVISORIES = {
#     "compound_heat_aqi"     : "High pollution compounds heat stress — avoid outdoor exercise.",
#     "compound_heat_drought" : "Drought conditions increase heat island effect and fire risk.",
#     "compound_heat_humidity": "High humidity reduces body's cooling ability — risk of heat stroke.",
#     "triple_compound"       : "⚠ TRIPLE COMPOUND EVENT: Extreme combined health and environmental risk.",
# }


# # ── FEATURE BUILDER ───────────────────────────────────────────────────────
# def build_features_from_history(history_df: pd.DataFrame,
#                                  today: dict) -> pd.DataFrame:
#     """
#     Appends today's observation to history and engineers features.
#     history_df : last 30 days of raw data (date, temp_max, temp_min,
#                  humidity, wind, rainfall, aqi)
#     today      : dict with same keys for the current day
#     Returns a 1-row DataFrame with all feature columns.
#     """
#     from step1_preprocessing import (
#         compute_heat_index, compute_humidex, compute_spi_30
#     )
#     from step2_risk_labeling import assign_risk_labels

#     today_row = pd.DataFrame([today])
#     df = pd.concat([history_df, today_row], ignore_index=True)
#     df["date"] = pd.to_datetime(df["date"])
#     df.sort_values("date", inplace=True)
#     df.reset_index(drop=True, inplace=True)

#     # Calendar
#     df["month"]       = df["date"].dt.month
#     df["day_of_year"] = df["date"].dt.dayofyear
#     df["is_summer"]   = df["month"].isin([4, 5, 6]).astype(int)

#     # Basic
#     df["temp_range"]       = df["temp_max"] - df["temp_min"]
#     df["temp_mean"]        = (df["temp_max"] + df["temp_min"]) / 2
#     df["heat_index"]       = compute_heat_index(df["temp_max"], df["humidity"])
#     df["humidex"]          = compute_humidex(df["temp_max"], df["humidity"])
#     df["feels_like_excess"]= df["heat_index"] - df["temp_max"]
#     df["wind_heat_ratio"]  = df["temp_max"] / (df["wind"].replace(0, 0.1))

#     # Rolling
#     for w in [3, 7]:
#         df[f"temp_max_roll{w}"]  = df["temp_max"].rolling(w, min_periods=1).mean()
#         df[f"humidity_roll{w}"]  = df["humidity"].rolling(w, min_periods=1).mean()
#         df[f"aqi_roll{w}"]       = df["aqi"].rolling(w, min_periods=1).mean()
#         df[f"rainfall_roll{w}"]  = df["rainfall"].rolling(w, min_periods=1).sum()

#     # Lags
#     for lag in [1, 2]:
#         df[f"temp_max_lag{lag}"] = df["temp_max"].shift(lag)
#         df[f"aqi_lag{lag}"]      = df["aqi"].shift(lag)
#         df[f"humidity_lag{lag}"] = df["humidity"].shift(lag)

#     # Departure
#     monthly_mean = df.groupby("month")["temp_max"].transform("mean")
#     monthly_std  = df.groupby("month")["temp_max"].transform("std").replace(0, 1e-6)
#     df["temp_departure"] = df["temp_max"] - monthly_mean
#     df["temp_zscore"]    = (df["temp_max"] - monthly_mean) / monthly_std
#     aqi_mm               = df.groupby("month")["aqi"].transform("mean")
#     df["aqi_departure"]  = df["aqi"] - aqi_mm

#     # Drought
#     df["drought_flag"]    = (df["rainfall"].rolling(7, min_periods=1).sum() < 2).astype(int)
#     hot_rain              = (df["rainfall"] < 1.0).astype(int)
#     df["dry_days_streak"] = hot_rain.groupby((hot_rain != hot_rain.shift()).cumsum()).cumcount()
#     df["spi_30"]          = compute_spi_30(df["rainfall"])

#     # AQI category
#     bins = [0, 50, 100, 200, 300, 400, 500]
#     lbl  = [1, 2, 3, 4, 5, 6]
#     df["aqi_category"] = pd.cut(df["aqi"], bins=bins, labels=lbl).astype(float)

#     # Consecutive hot days
#     hot = (df["temp_max"] >= 40.0).astype(int)
#     df["consec_hot_days"] = hot.groupby((hot != hot.shift()).cumsum()).cumcount() + hot

#     # Compound
#     df["compound_heat_aqi"]      = ((df["temp_max"] >= 38) & (df["aqi"] >= 200)).astype(int)
#     df["compound_heat_drought"]  = ((df["temp_max"] >= 38) & (df["drought_flag"] == 1)).astype(int)
#     df["compound_heat_humidity"] = ((df["temp_max"] >= 38) & (df["humidity"] >= 60)).astype(int)
#     df["triple_compound"]        = (
#         (df["temp_max"] >= 38) & (df["aqi"] >= 150) & (df["drought_flag"] == 1)
#     ).astype(int)

#     today_feats = df.iloc[[-1]][FEATURE_COLS].fillna(0)
#     return today_feats, df.iloc[-1]


# # ── PREDICTOR CLASS ───────────────────────────────────────────────────────
# class HeatwavePredictor:
#     def __init__(self, city: str = "all"):
#         model_path = os.path.join(MODEL_DIR, f"classifier_{city}.pkl")
#         reg_path   = os.path.join(MODEL_DIR, f"regressor_{city}.pkl")
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model not found: {model_path}. Run step4 first.")

#         with open(model_path, "rb") as f:
#             bundle = pickle.load(f)
#         self.clf       = bundle["model"]
#         self.meta      = bundle["meta"]
#         self.scaler    = bundle["meta"].get("scaler", None)
#         self.city      = city

#         self.reg = None
#         if os.path.exists(reg_path):
#             with open(reg_path, "rb") as f:
#                 self.reg = pickle.load(f)

#     def predict(self, today: dict, history_df: pd.DataFrame) -> dict:
#         """
#         today      : {'date': ..., 'temp_max': ..., 'temp_min': ...,
#                        'humidity': ..., 'wind': ..., 'rainfall': ..., 'aqi': ...}
#         history_df : last 30+ days of raw observations
#         Returns dict with full prediction report.
#         """
#         X_feat, enriched_row = build_features_from_history(history_df, today)
#         X = X_feat.values

#         # Classification
#         probas     = self.clf.predict_proba(X)[0]
#         risk_level = int(np.argmax(probas))
#         confidence = float(probas[risk_level])

#         # Regression
#         score_pred = float(self.reg.predict(X)[0]) if self.reg else None

#         # Active compound flags
#         active_compounds = [k for k in COMPOUND_ADVISORIES
#                              if enriched_row.get(k, 0) == 1]

#         # Advisory
#         advisory = ADVISORIES[risk_level]
#         if active_compounds:
#             advisory += "\n\nCompound Risks:\n"
#             for k in active_compounds:
#                 advisory += f"  • {COMPOUND_ADVISORIES[k]}\n"

#         # Pillar scores (re-compute for transparency)
#         from step2_risk_labeling import (temp_score, heat_index_score,
#                                           aqi_score, drought_score, compound_score)
#         single_row = pd.DataFrame([enriched_row])
#         pillars = {
#             "Temperature" : float(temp_score(single_row["temp_max"]).iloc[0]),
#             "Heat Index"  : float(heat_index_score(single_row["heat_index"]).iloc[0]),
#             "AQI"         : float(aqi_score(single_row["aqi"]).iloc[0]),
#             "Drought"     : float(drought_score(single_row["drought_flag"],
#                                                  single_row["dry_days_streak"],
#                                                  single_row["spi_30"]).iloc[0]),
#             "Compound"    : float(compound_score(single_row).iloc[0]),
#         }

#         return {
#             "date"           : str(today["date"]),
#             "city"           : self.city,
#             "risk_level"     : risk_level,
#             "risk_label"     : RISK_LABELS[risk_level],
#             "emoji"          : RISK_EMOJIS[risk_level],
#             "confidence"     : round(confidence * 100, 1),
#             "probabilities"  : {RISK_LABELS[i]: round(float(p)*100, 1) for i, p in enumerate(probas)},
#             "composite_score": round(score_pred, 1) if score_pred is not None else None,
#             "pillar_scores"  : pillars,
#             "active_compounds": active_compounds,
#             "advisory"       : advisory,
#             "heat_index"     : round(float(enriched_row["heat_index"]), 1),
#             "humidex"        : round(float(enriched_row["humidex"]), 1),
#             "consec_hot_days": int(enriched_row["consec_hot_days"]),
#         }

#     def print_report(self, result: dict):
#         print("\n" + "━"*55)
#         print(f"  HEATWAVE RISK REPORT  —  {result['date']}")
#         print(f"  City: {result['city'].upper()}")
#         print("━"*55)
#         print(f"  {result['emoji']}  Risk Level   : {result['risk_label']}")
#         print(f"  📊  Composite Score: {result['composite_score']}")
#         print(f"  🎯  Confidence     : {result['confidence']}%")
#         print(f"  🌡   Heat Index    : {result['heat_index']}°C")
#         print(f"  💧  Humidex        : {result['humidex']}°C")
#         print(f"  🔥  Consec Hot Days: {result['consec_hot_days']}")
#         print("\n  Pillar Scores (0–100):")
#         for pillar, score in result["pillar_scores"].items():
#             bar = "█" * int(score / 5)
#             print(f"    {pillar:14s}: {score:5.1f}  {bar}")
#         print("\n  Class Probabilities:")
#         for label, prob in result["probabilities"].items():
#             bar = "░" * int(prob / 3)
#             print(f"    {label:10s}: {prob:5.1f}%  {bar}")
#         if result["active_compounds"]:
#             print(f"\n  ⚠  Active Compound Risks:")
#             for k in result["active_compounds"]:
#                 print(f"    • {k}")
#         print(f"\n  Advisory:\n  {result['advisory']}")
#         print("━"*55 + "\n")


# # ── DEMO / TEST ───────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     print("\n" + "="*60)
#     print("  STEP 5 ─ DAILY PREDICTION ENGINE (Demo)")
#     print("="*60)

#     # Load 30 days of history from processed data (for demo)
#     hist_path = os.path.join(PROC_DIR, "labelled_all.csv")
#     if not os.path.exists(hist_path):
#         print("[!] Run steps 1–4 first to generate processed data and models.")
#         exit()

#     df_all = pd.read_csv(hist_path, parse_dates=["date"])

#     for city in ["Delhi", "Hyderabad", "Nagpur"]:
#         city_hist = df_all[df_all["city"] == city].tail(35).copy()

#         # Simulate "today" as the last known day
#         last_row = city_hist.iloc[-1]
#         today_obs = {
#             "date"    : str((last_row["date"] + timedelta(days=1)).date()),
#             "temp_max": last_row["temp_max"] + np.random.uniform(-1, 2),
#             "temp_min": last_row["temp_min"] + np.random.uniform(-1, 1),
#             "humidity": last_row["humidity"] + np.random.uniform(-3, 3),
#             "wind"    : last_row["wind"]     + np.random.uniform(-1, 1),
#             "rainfall": max(0, last_row["rainfall"] + np.random.uniform(-1, 1)),
#             "aqi"     : last_row["aqi"]     + np.random.uniform(-10, 10),
#         }
#         history = city_hist.iloc[:-1][["date","temp_max","temp_min",
#                                         "humidity","wind","rainfall","aqi"]]

#         model_file = os.path.join(MODEL_DIR, f"classifier_{city}.pkl")
#         if not os.path.exists(model_file):
#             model_file = os.path.join(MODEL_DIR, "classifier_all.pkl")
#         city_tag = city if os.path.exists(
#             os.path.join(MODEL_DIR, f"classifier_{city}.pkl")) else "all"

#         predictor = HeatwavePredictor(city=city_tag)
#         result    = predictor.predict(today_obs, history)
#         predictor.print_report(result)

#     print("[Done] Daily prediction complete.")



"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Step 5: Daily Prediction Inference Engine
========================================================================
Given today's raw observations (temp_max, temp_min, humidity,
wind, rainfall, aqi) + the last N days of history:

Outputs:
  • Predicted risk level (Low / Moderate / High / Severe)
  • Composite risk score (0–100)
  • Per-pillar score breakdown (Temperature, Heat Index, AQI, ...)
  • Compound risk flags active today
  • Prediction confidence (class probabilities)
  • Natural-language advisory message
  • Tomorrow's predicted risk (next-day forward prediction)
========================================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime, timedelta, date

MODEL_DIR = "models"
PROC_DIR  = "data/processed"

FEATURE_COLS = [
    "temp_max", "temp_min", "humidity", "wind", "rainfall", "aqi",
    "heat_index", "humidex", "temp_range", "temp_mean", "feels_like_excess",
    "wind_heat_ratio",
    "temp_max_roll3", "temp_max_roll7", "humidity_roll7", "aqi_roll7",
    "rainfall_roll7",
    "temp_max_lag1", "temp_max_lag2", "aqi_lag1", "humidity_lag1",
    "temp_departure", "temp_zscore", "aqi_departure",
    "dry_days_streak", "spi_30", "drought_flag",
    "month", "day_of_year", "is_summer",
    "compound_heat_aqi", "compound_heat_drought", "compound_heat_humidity",
    "triple_compound", "consec_hot_days", "aqi_category",
]

RISK_LABELS   = {0: "Low", 1: "Moderate", 2: "High", 3: "Severe"}
RISK_EMOJIS   = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}

# ── ADVISORY TEMPLATES ────────────────────────────────────────────────────
ADVISORIES = {
    0: "No significant heat risk. Normal outdoor activities are safe.",
    1: ("Moderate heat conditions. Stay hydrated, avoid prolonged outdoor "
        "activity during peak hours (12–4 PM). Monitor elderly and children."),
    2: ("HIGH HEAT RISK. Limit outdoor exertion. Drink water regularly. "
        "Avoid direct sun exposure. Seek air-conditioned spaces if possible. "
        "Watch for signs of heat exhaustion."),
    3: ("⚠ SEVERE HEAT WAVE CONDITIONS. AVOID all non-essential outdoor activity. "
        "Drink 2–3 litres of water daily. Stay in cool/shaded environments. "
        "Emergency services should be on alert for heat-related illness. "
        "Vulnerable populations (elderly, infants, outdoor workers) at extreme risk."),
}

COMPOUND_ADVISORIES = {
    "compound_heat_aqi"     : "High pollution compounds heat stress — avoid outdoor exercise.",
    "compound_heat_drought" : "Drought conditions increase heat island effect and fire risk.",
    "compound_heat_humidity": "High humidity reduces body's cooling ability — risk of heat stroke.",
    "triple_compound"       : "⚠ TRIPLE COMPOUND EVENT: Extreme combined health and environmental risk.",
}


# ── FEATURE BUILDER ───────────────────────────────────────────────────────
def build_features_from_history(history_df: pd.DataFrame,
                                 today: dict) -> pd.DataFrame:
    """
    Appends today's observation to history and engineers features.
    history_df : last 30 days of raw data (date, temp_max, temp_min,
                 humidity, wind, rainfall, aqi)
    today      : dict with same keys for the current day
    Returns a 1-row DataFrame with all feature columns.
    """
    from step1_preprocessing import (
        compute_heat_index, compute_humidex, compute_spi_30
    )
    from step2_risk_labeling import assign_risk_labels

    today_row = pd.DataFrame([today])
    df = pd.concat([history_df, today_row], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Calendar
    df["month"]       = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_summer"]   = df["month"].isin([4, 5, 6]).astype(int)

    # Basic
    df["temp_range"]       = df["temp_max"] - df["temp_min"]
    df["temp_mean"]        = (df["temp_max"] + df["temp_min"]) / 2
    df["heat_index"]       = compute_heat_index(df["temp_max"], df["humidity"])
    df["humidex"]          = compute_humidex(df["temp_max"], df["humidity"])
    df["feels_like_excess"]= df["heat_index"] - df["temp_max"]
    df["wind_heat_ratio"]  = df["temp_max"] / (df["wind"].replace(0, 0.1))

    # Rolling
    for w in [3, 7]:
        df[f"temp_max_roll{w}"]  = df["temp_max"].rolling(w, min_periods=1).mean()
        df[f"humidity_roll{w}"]  = df["humidity"].rolling(w, min_periods=1).mean()
        df[f"aqi_roll{w}"]       = df["aqi"].rolling(w, min_periods=1).mean()
        df[f"rainfall_roll{w}"]  = df["rainfall"].rolling(w, min_periods=1).sum()

    # Lags
    for lag in [1, 2]:
        df[f"temp_max_lag{lag}"] = df["temp_max"].shift(lag)
        df[f"aqi_lag{lag}"]      = df["aqi"].shift(lag)
        df[f"humidity_lag{lag}"] = df["humidity"].shift(lag)

    # Departure
    monthly_mean = df.groupby("month")["temp_max"].transform("mean")
    monthly_std  = df.groupby("month")["temp_max"].transform("std").replace(0, 1e-6)
    df["temp_departure"] = df["temp_max"] - monthly_mean
    df["temp_zscore"]    = (df["temp_max"] - monthly_mean) / monthly_std
    aqi_mm               = df.groupby("month")["aqi"].transform("mean")
    df["aqi_departure"]  = df["aqi"] - aqi_mm

    # Drought
    df["drought_flag"]    = (df["rainfall"].rolling(7, min_periods=1).sum() < 2).astype(int)
    hot_rain              = (df["rainfall"] < 1.0).astype(int)
    df["dry_days_streak"] = hot_rain.groupby((hot_rain != hot_rain.shift()).cumsum()).cumcount()
    df["spi_30"]          = compute_spi_30(df["rainfall"])

    # AQI category
    bins = [0, 50, 100, 200, 300, 400, 500]
    lbl  = [1, 2, 3, 4, 5, 6]
    df["aqi_category"] = pd.cut(df["aqi"], bins=bins, labels=lbl).astype(float)

    # Consecutive hot days
    hot = (df["temp_max"] >= 40.0).astype(int)
    df["consec_hot_days"] = hot.groupby((hot != hot.shift()).cumsum()).cumcount() + hot

    # Compound
    df["compound_heat_aqi"]      = ((df["temp_max"] >= 38) & (df["aqi"] >= 200)).astype(int)
    df["compound_heat_drought"]  = ((df["temp_max"] >= 38) & (df["drought_flag"] == 1)).astype(int)
    df["compound_heat_humidity"] = ((df["temp_max"] >= 38) & (df["humidity"] >= 60)).astype(int)
    df["triple_compound"]        = (
        (df["temp_max"] >= 38) & (df["aqi"] >= 150) & (df["drought_flag"] == 1)
    ).astype(int)

    today_feats = df.iloc[[-1]][FEATURE_COLS].fillna(0)
    return today_feats, df.iloc[-1]


# ── PREDICTOR CLASS ───────────────────────────────────────────────────────
class HeatwavePredictor:
    def __init__(self, city: str = "all"):
        model_path = os.path.join(MODEL_DIR, f"classifier_{city}.pkl")
        reg_path   = os.path.join(MODEL_DIR, f"regressor_{city}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Run step4 first.")

        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.clf       = bundle["model"]
        self.meta      = bundle["meta"]
        self.scaler    = bundle["meta"].get("scaler", None)
        self.city      = city

        self.reg = None
        if os.path.exists(reg_path):
            with open(reg_path, "rb") as f:
                self.reg = pickle.load(f)

    def predict(self, today: dict, history_df: pd.DataFrame) -> dict:
        """
        today      : {'date': ..., 'temp_max': ..., 'temp_min': ...,
                       'humidity': ..., 'wind': ..., 'rainfall': ..., 'aqi': ...}
        history_df : last 30+ days of raw observations
        Returns dict with full prediction report.
        """
        X_feat, enriched_row = build_features_from_history(history_df, today)
        X = X_feat.values

        # Classification
        probas     = self.clf.predict_proba(X)[0]
        risk_level = int(np.argmax(probas))
        confidence = float(probas[risk_level])

        # Regression
        score_pred = float(self.reg.predict(X)[0]) if self.reg else None

        # Active compound flags
        active_compounds = [k for k in COMPOUND_ADVISORIES
                             if enriched_row.get(k, 0) == 1]

        # Advisory
        advisory = ADVISORIES[risk_level]
        if active_compounds:
            advisory += "\n\nCompound Risks:\n"
            for k in active_compounds:
                advisory += f"  • {COMPOUND_ADVISORIES[k]}\n"

        # Pillar scores (re-compute for transparency)
        from step2_risk_labeling import (temp_score, heat_index_score,
                                          aqi_score, drought_score, compound_score)
        single_row = pd.DataFrame([enriched_row])
        pillars = {
            "Temperature" : float(temp_score(single_row["temp_max"]).iloc[0]),
            "Heat Index"  : float(heat_index_score(single_row["heat_index"]).iloc[0]),
            "AQI"         : float(aqi_score(single_row["aqi"]).iloc[0]),
            "Drought"     : float(drought_score(single_row["drought_flag"],
                                                 single_row["dry_days_streak"],
                                                 single_row["spi_30"]).iloc[0]),
            "Compound"    : float(compound_score(single_row).iloc[0]),
        }

        return {
            "date"           : str(today["date"]),
            "city"           : self.city,
            "risk_level"     : risk_level,
            "risk_label"     : RISK_LABELS[risk_level],
            "emoji"          : RISK_EMOJIS[risk_level],
            "confidence"     : round(confidence * 100, 1),
            "probabilities"  : {RISK_LABELS[i]: round(float(p)*100, 1) for i, p in enumerate(probas)},
            "composite_score": round(score_pred, 1) if score_pred is not None else None,
            "pillar_scores"  : pillars,
            "active_compounds": active_compounds,
            "advisory"       : advisory,
            "heat_index"     : round(float(enriched_row["heat_index"]), 1),
            "humidex"        : round(float(enriched_row["humidex"]), 1),
            "consec_hot_days": int(enriched_row["consec_hot_days"]),
        }

    def print_report(self, result: dict):
        print("\n" + "━"*55)
        print(f"  HEATWAVE RISK REPORT  —  {result['date']}")
        print(f"  City: {result['city'].upper()}")
        print("━"*55)
        print(f"  {result['emoji']}  Risk Level   : {result['risk_label']}")
        print(f"  📊  Composite Score: {result['composite_score']}")
        print(f"  🎯  Confidence     : {result['confidence']}%")
        print(f"  🌡   Heat Index    : {result['heat_index']}°C")
        print(f"  💧  Humidex        : {result['humidex']}°C")
        print(f"  🔥  Consec Hot Days: {result['consec_hot_days']}")
        print("\n  Pillar Scores (0–100):")
        for pillar, score in result["pillar_scores"].items():
            bar = "█" * int(score / 5)
            print(f"    {pillar:14s}: {score:5.1f}  {bar}")
        print("\n  Class Probabilities:")
        for label, prob in result["probabilities"].items():
            bar = "░" * int(prob / 3)
            print(f"    {label:10s}: {prob:5.1f}%  {bar}")
        if result["active_compounds"]:
            print(f"\n  ⚠  Active Compound Risks:")
            for k in result["active_compounds"]:
                print(f"    • {k}")
        print(f"\n  Advisory:\n  {result['advisory']}")
        print("━"*55 + "\n")


# ── HELPERS: same API functions as step7 ────────────────────────────────
def _fetch_weather(lat, lon, target_date):
    """Open-Meteo forecast — free, no key."""
    import requests as _req
    try:
        r = _req.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
            "forecast_days": 7,
            "daily": ["temperature_2m_max","temperature_2m_min",
                      "precipitation_sum","wind_speed_10m_max",
                      "relative_humidity_2m_max"],
        }, timeout=12)
        r.raise_for_status()
        daily = r.json()["daily"]
        dates = pd.to_datetime(daily["time"])
        target_ts = pd.Timestamp(target_date)
        if target_ts not in list(dates):
            return None
        i = list(dates).index(target_ts)
        return {
            "date"    : str(target_date),
            "temp_max": round(daily["temperature_2m_max"][i], 1),
            "temp_min": round(daily["temperature_2m_min"][i], 1),
            "humidity": round(daily["relative_humidity_2m_max"][i] or 50, 1),
            "wind"    : round(daily["wind_speed_10m_max"][i] or 0, 1),
            "rainfall": round(daily["precipitation_sum"][i] or 0, 1),
        }
    except Exception as e:
        print(f"  [!] Weather fetch failed: {e}")
        return None

def _fetch_aqi(lat, lon, target_date):
    """Open-Meteo AQ — free, no key. Returns India CPCB AQI."""
    import requests as _req
    CPCB_PM25 = [(0,30,0,50),(30.1,60,51,100),(60.1,90,101,200),
                 (90.1,120,201,300),(120.1,250,301,400),(250.1,500,401,500)]
    CPCB_PM10 = [(0,50,0,50),(51,100,51,100),(101,250,101,200),
                 (251,350,201,300),(351,430,301,400),(431,600,401,500)]
    def _c(v, bps):
        if not v or v<=0: return 0.0
        for lo,hi,ilo,ihi in bps:
            if lo<=v<=hi: return round(((ihi-ilo)/(hi-lo))*(v-lo)+ilo,1)
        return 500.0
    try:
        r = _req.get("https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude":lat,"longitude":lon,"timezone":"Asia/Kolkata",
                    "forecast_days":7,"hourly":["pm2_5","pm10"]}, timeout=12)
        r.raise_for_status()
        hourly = r.json()["hourly"]
        times  = pd.to_datetime(hourly["time"])
        pm25   = hourly["pm2_5"]
        pm10   = hourly["pm10"]
        peak   = pd.Timestamp(target_date).replace(hour=14)
        idxs   = [i for i,t in enumerate(times) if t.date()==target_date]
        if not idxs: return None
        best   = min(idxs, key=lambda i: abs((times[i]-peak).total_seconds()))
        return round(max(_c(pm25[best] or 0,CPCB_PM25),
                         _c(pm10[best] or 0,CPCB_PM10)), 0)
    except Exception as e:
        print(f"  [!] AQI fetch failed: {e}")
        return None

def _load_city_history(city: str) -> pd.DataFrame:
    """
    Loads rolling history from the per-city labelled/processed CSV.
    Same priority order as the dashboard — NOT labelled_all.csv.
    Uses tail(40) to match the dashboard window exactly.
    """
    raw_cols = ["date","temp_max","temp_min","humidity","wind","rainfall","aqi"]
    for fname in [f"labelled_{city}.csv", f"processed_{city}.csv"]:
        p = os.path.join(PROC_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["date"])
            available = [c for c in raw_cols if c in df.columns]
            return df[available].dropna().tail(40).copy()
    # Fallback to labelled_all.csv filtered by city
    p = os.path.join(PROC_DIR, "labelled_all.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, parse_dates=["date"])
        df = df[df["city"] == city]
        available = [c for c in raw_cols if c in df.columns]
        return df[available].dropna().tail(40).copy()
    raise FileNotFoundError(f"No history CSV found for {city}")

CITY_COORDS = {
    "Delhi"     : (28.6139, 77.2090),
    "Hyderabad" : (17.3850, 78.4867),
    "Nagpur"    : (21.1458, 79.0882),
}

# ── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tomorrow", action="store_true",
                        help="Predict for tomorrow instead of today")
    parser.add_argument("--city", default="all",
                        help="Delhi / Hyderabad / Nagpur / all")
    args = parser.parse_args()

    target = date.today() + timedelta(days=1) if args.tomorrow else date.today()
    cities = list(CITY_COORDS.keys()) if args.city == "all" else [args.city]

    print("\n" + "="*60)
    print("  STEP 5 — DAILY PREDICTION ENGINE")
    print(f"  Date   : {target}  {'(Tomorrow)' if args.tomorrow else '(Today)'}")
    print(f"  Cities : {', '.join(cities)}")
    print("="*60)
    print("  Data sources:")
    print("  • Weather    : Open-Meteo forecast API (free, no key)")
    print("  • AQI        : Open-Meteo AQ API → India CPCB scale")
    print("  • History    : per-city labelled CSV (tail 40 days)")
    print("  • Model      : best classifier + regressor from step4")
    print()

    for city in cities:
        lat, lon = CITY_COORDS[city]

        # ── 1. Fetch real data (same as dashboard) ────────────────────
        print(f"  [{city}] Fetching weather...", end=" ", flush=True)
        weather = _fetch_weather(lat, lon, target)
        if weather is None:
            # Fallback: use last row from history
            hist_fb = _load_city_history(city)
            last    = hist_fb.iloc[-1]
            weather = {
                "date"    : str(target),
                "temp_max": round(float(last["temp_max"]), 1),
                "temp_min": round(float(last["temp_min"]), 1),
                "humidity": round(float(last["humidity"]), 1),
                "wind"    : round(float(last["wind"]), 1),
                "rainfall": round(float(last["rainfall"]), 1),
            }
            print(f"API failed — using last historical row ({last['date'].date()})")
        else:
            print("OK")

        print(f"  [{city}] Fetching AQI...", end=" ", flush=True)
        aqi = _fetch_aqi(lat, lon, target)
        if aqi is None:
            hist_fb = _load_city_history(city)
            aqi = round(float(hist_fb["aqi"].tail(7).mean()), 0)
            print(f"API failed — using 7-day avg ({aqi})")
        else:
            print(f"OK  (CPCB AQI = {aqi})")

        today_obs = {**weather, "aqi": aqi}

        # ── 2. Load history — same source + same window as dashboard ──
        history = _load_city_history(city)
        history["date"] = pd.to_datetime(history["date"], errors="coerce")
        last_hist = history["date"].max().date()
        print(f"  [{city}] History: {len(history)} rows, "
              f"last date {last_hist}")
        if last_hist < target - timedelta(days=7):
            print(f"  [{city}] WARNING: history is stale (>{(target-last_hist).days} days old). "
                  f"Run step7b_backfill.py first.")

        # ── 3. Pick model (city-specific first, else combined) ────────
        city_tag = city if os.path.exists(
            os.path.join(MODEL_DIR, f"classifier_{city}.pkl")) else "all"

        # ── 4. Predict ────────────────────────────────────────────────
        predictor = HeatwavePredictor(city=city_tag)
        result    = predictor.predict(today_obs, history)
        predictor.print_report(result)

    print("[Done] Step5 prediction complete.")
    print("\nNOTE: These values should now exactly match the dashboard")
    print("      Live Prediction page for the same date and city.")