# """
# ========================================================================
# HEATWAVE RISK PREDICTION SYSTEM
# Step 7: Real-Time API Integration & Live Daily Prediction
# ========================================================================

# TWO FREE APIS USED (no paid plan needed):
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#   SOURCE 1 — Open-Meteo (Weather Forecast)
#     URL   : https://api.open-meteo.com  ← COMPLETELY FREE, NO KEY
#     Gives : temp_max, temp_min, humidity, wind_speed, precipitation
#             for today AND tomorrow (24-hour advance warning)
#     Docs  : https://open-meteo.com/en/docs

#   SOURCE 2 — OpenWeatherMap Air Pollution API  ← FREE, KEY NEEDED
#     URL   : http://api.openweathermap.org/data/2.5/air_pollution
#     Gives : PM2.5, PM10, NO2, O3, SO2 → converted to India CPCB AQI
#     Key   : Free at https://openweathermap.org/api (60 calls/min)

#   SOURCE 2 (FALLBACK) — AQICN World Air Quality Index  ← FREE, KEY NEEDED
#     URL   : https://api.waqi.info/feed/{city}/
#     Gives : AQI directly in local scale (closer to India CPCB)
#     Key   : Free at https://aqicn.org/data-platform/token/

# ⚠  IMPORTANT — AQI SCALE DIFFERENCE:
#     OpenWeatherMap uses a 1–5 European scale (1=Good, 5=Very Poor).
#     India uses CPCB scale: 0–500 (Good < 50, Severe > 400).
#     This module converts PM2.5 concentration (µg/m³) → India CPCB AQI
#     using the official CPCB breakpoint table. This is the correct way.

# WHAT THIS MODULE DOES:
#   1. Fetches today's + tomorrow's weather from Open-Meteo (no key)
#   2. Fetches AQI data from OWM or AQICN (free key)
#   3. Converts raw pollutant data → India CPCB AQI
#   4. Appends today's data to the rolling history CSV
#   5. Runs the ML prediction pipeline → risk report
#   6. Saves prediction log with timestamp
#   7. Can be run daily via cron / Task Scheduler

# HOW TO USE:
#   python step7_realtime_api.py --city Delhi
#   python step7_realtime_api.py --city Hyderabad --tomorrow
#   python step7_realtime_api.py --city all          ← runs all 3 cities
# ========================================================================
# """

# import os, json, argparse, requests, pickle
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta, date
# import warnings
# warnings.filterwarnings("ignore")

# # ── CONFIG ───────────────────────────────────────────────────────────────

# # !! SET YOUR KEYS HERE (or pass via environment variables) !!
# OWM_API_KEY   = os.environ.get("OWM_API_KEY",  "YOUR_OWM_KEY_HERE")
# AQICN_TOKEN   = os.environ.get("AQICN_TOKEN",  "YOUR_AQICN_TOKEN_HERE")

# PROC_DIR  = "data/processed"
# MODEL_DIR = "models"
# LOG_DIR   = "data/predictions"
# os.makedirs(LOG_DIR, exist_ok=True)

# # City coordinates (lat, lon) and AQICN city names
# CITY_CONFIG = {
#     "Delhi"     : {"lat": 28.6139,  "lon": 77.2090,  "aqicn": "delhi"},
#     "Hyderabad" : {"lat": 17.3850,  "lon": 78.4867,  "aqicn": "hyderabad"},
#     "Nagpur"    : {"lat": 21.1458,  "lon": 79.0882,  "aqicn": "nagpur"},
# }

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


# # ════════════════════════════════════════════════════════════════════════
# # SECTION A: AQI CONVERSION — OWM PM2.5 → India CPCB AQI
# # ════════════════════════════════════════════════════════════════════════

# # Official CPCB breakpoint table for PM2.5 (µg/m³) → AQI
# # Source: Central Pollution Control Board, India
# CPCB_PM25_BREAKPOINTS = [
#     # (C_low, C_high, I_low, I_high)
#     (0.0,   30.0,   0,   50),    # Good
#     (30.1,  60.0,   51,  100),   # Satisfactory
#     (60.1,  90.0,   101, 200),   # Moderately Polluted
#     (90.1,  120.0,  201, 300),   # Poor
#     (120.1, 250.0,  301, 400),   # Very Poor
#     (250.1, 500.0,  401, 500),   # Severe
# ]

# CPCB_PM10_BREAKPOINTS = [
#     (0,    50,    0,   50),
#     (51,   100,   51,  100),
#     (101,  250,   101, 200),
#     (251,  350,   201, 300),
#     (351,  430,   301, 400),
#     (431,  600,   401, 500),
# ]

# def concentration_to_cpcb_aqi(concentration: float,
#                                 breakpoints: list) -> float:
#     """
#     Linear interpolation within CPCB breakpoint table.
#     AQI = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
#     """
#     if concentration < 0:
#         return 0.0
#     for (C_lo, C_hi, I_lo, I_hi) in breakpoints:
#         if C_lo <= concentration <= C_hi:
#             aqi = ((I_hi - I_lo) / (C_hi - C_lo)) * (concentration - C_lo) + I_lo
#             return round(aqi, 1)
#     # Above highest breakpoint
#     return 500.0

# def owm_components_to_india_aqi(components: dict) -> float:
#     """
#     OWM returns raw pollutant concentrations (µg/m³).
#     Convert PM2.5 and PM10 to India CPCB AQI and return the higher value
#     (conservative: worst pollutant governs overall AQI per CPCB method).
#     """
#     pm25 = components.get("pm2_5", 0)
#     pm10 = components.get("pm10",  0)

#     aqi_pm25 = concentration_to_cpcb_aqi(pm25, CPCB_PM25_BREAKPOINTS)
#     aqi_pm10 = concentration_to_cpcb_aqi(pm10, CPCB_PM10_BREAKPOINTS)

#     # CPCB: final AQI = maximum of all sub-indices
#     final_aqi = max(aqi_pm25, aqi_pm10)
#     return final_aqi


# # ════════════════════════════════════════════════════════════════════════
# # SECTION B: API FETCHERS
# # ════════════════════════════════════════════════════════════════════════

# def fetch_open_meteo(lat: float, lon: float,
#                      target_date: date) -> dict | None:
#     """
#     Open-Meteo Forecast API — completely free, no key required.
#     Returns daily values for target_date.

#     Variables fetched:
#       temperature_2m_max, temperature_2m_min,
#       relative_humidity_2m_max (proxy), wind_speed_10m_max,
#       precipitation_sum
#     """
#     url = "https://api.open-meteo.com/v1/forecast"
#     params = {
#         "latitude"              : lat,
#         "longitude"             : lon,
#         "daily"                 : [
#             "temperature_2m_max",
#             "temperature_2m_min",
#             "precipitation_sum",
#             "wind_speed_10m_max",
#             "relative_humidity_2m_max",   # daily max humidity
#         ],
#         "timezone"              : "Asia/Kolkata",
#         "forecast_days"         : 7,
#     }
#     try:
#         r = requests.get(url, params=params, timeout=10)
#         r.raise_for_status()
#         data  = r.json()
#         daily = data["daily"]
#         dates = pd.to_datetime(daily["time"])

#         # Find index for target date
#         target_ts = pd.Timestamp(target_date)
#         if target_ts not in dates:
#             print(f"  [!] Open-Meteo: {target_date} not in forecast window.")
#             return None
#         idx = list(dates).index(target_ts)

#         return {
#             "date"    : str(target_date),
#             "temp_max": round(daily["temperature_2m_max"][idx], 1),
#             "temp_min": round(daily["temperature_2m_min"][idx], 1),
#             "humidity": round(daily["relative_humidity_2m_max"][idx], 1),
#             "wind"    : round(daily["wind_speed_10m_max"][idx], 1),
#             "rainfall": round(daily["precipitation_sum"][idx] or 0, 1),
#         }
#     except requests.exceptions.RequestException as e:
#         print(f"  [!] Open-Meteo request failed: {e}")
#         return None
#     except (KeyError, IndexError) as e:
#         print(f"  [!] Open-Meteo parse error: {e}")
#         return None


# def fetch_owm_aqi(lat: float, lon: float,
#                   target_date: date = None) -> float | None:
#     """
#     OpenWeatherMap Air Pollution API (free tier).
#     For today → uses /air_pollution (current).
#     For tomorrow → uses /air_pollution/forecast, picks closest hour.
#     Returns India CPCB AQI (converted from PM2.5/PM10).
#     """
#     if OWM_API_KEY == "YOUR_OWM_KEY_HERE":
#         print("  [!] OWM key not set — skipping OWM AQI fetch.")
#         return None

#     today = date.today()
#     is_forecast = (target_date is not None and target_date > today)

#     if is_forecast:
#         url    = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"
#         params = {"lat": lat, "lon": lon, "appid": OWM_API_KEY}
#         try:
#             r = requests.get(url, params=params, timeout=10)
#             r.raise_for_status()
#             items = r.json()["list"]
#             # Find entry closest to noon on target_date
#             target_noon = datetime.combine(target_date, datetime.min.time()) \
#                           .replace(hour=12)
#             best  = min(items, key=lambda x: abs(
#                 datetime.fromtimestamp(x["dt"]) - target_noon))
#             return owm_components_to_india_aqi(best["components"])
#         except Exception as e:
#             print(f"  [!] OWM forecast AQI failed: {e}")
#             return None
#     else:
#         url    = "http://api.openweathermap.org/data/2.5/air_pollution"
#         params = {"lat": lat, "lon": lon, "appid": OWM_API_KEY}
#         try:
#             r = requests.get(url, params=params, timeout=10)
#             r.raise_for_status()
#             components = r.json()["list"][0]["components"]
#             return owm_components_to_india_aqi(components)
#         except Exception as e:
#             print(f"  [!] OWM current AQI failed: {e}")
#             return None


# def fetch_aqicn_aqi(city_slug: str) -> float | None:
#     """
#     AQICN World Air Quality Index API (free, needs token).
#     Returns AQI in the station's local scale — for Indian cities this
#     is typically close to the India CPCB scale.
#     Note: AQICN only provides current (no forecast).
#     """
#     if AQICN_TOKEN == "YOUR_AQICN_TOKEN_HERE":
#         print("  [!] AQICN token not set — skipping AQICN fetch.")
#         return None

#     url = f"https://api.waqi.info/feed/{city_slug}/"
#     try:
#         r   = requests.get(url, params={"token": AQICN_TOKEN}, timeout=10)
#         r.raise_for_status()
#         data = r.json()
#         if data.get("status") == "ok":
#             aqi_val = data["data"]["aqi"]
#             return float(aqi_val) if aqi_val != "-" else None
#         else:
#             print(f"  [!] AQICN error: {data.get('data', 'unknown')}")
#             return None
#     except Exception as e:
#         print(f"  [!] AQICN request failed: {e}")
#         return None


# def get_aqi_for_city(lat: float, lon: float, city_slug: str,
#                      target_date: date) -> float:
#     """
#     AQI fetch strategy (priority order):
#       1. Try OWM (supports forecast for tomorrow)
#       2. Try AQICN (fallback, current only)
#       3. Last resort: use 7-day rolling average from history CSV
#     """
#     print(f"  Fetching AQI...")
#     aqi = fetch_owm_aqi(lat, lon, target_date)
#     if aqi is not None:
#         print(f"  [✓] OWM AQI (India CPCB): {aqi:.0f}")
#         return aqi

#     aqi = fetch_aqicn_aqi(city_slug)
#     if aqi is not None:
#         print(f"  [✓] AQICN AQI: {aqi:.0f}")
#         return aqi

#     print("  [!] Both AQI sources failed. Using historical rolling average.")
#     return None   # caller handles fallback from history


# # ════════════════════════════════════════════════════════════════════════
# # SECTION C: FEATURE ENGINEERING (mirrors step1, self-contained)
# # ════════════════════════════════════════════════════════════════════════

# def _heat_index(T, RH):
#     TF = T * 9/5 + 32
#     HI = (-42.379 + 2.04901523*TF + 10.14333127*RH
#           - 0.22475541*TF*RH - 0.00683783*TF**2
#           - 0.05481717*RH**2 + 0.00122874*TF**2*RH
#           + 0.00085282*TF*RH**2 - 0.00000199*TF**2*RH**2)
#     return (HI - 32) * 5/9

# def _humidex(T, RH):
#     Td = T - ((100 - RH) / 5)
#     e  = 6.105 * np.exp(25.22 * (Td - 273.16) / (Td + 1e-9))
#     return T + 0.5555 * (e - 10)

# def build_feature_row(history_df: pd.DataFrame,
#                        today_obs: dict) -> pd.Series:
#     """
#     Appends today's raw observation to history and engineers all features.
#     Returns the single-row feature Series for today.
#     """
#     today_row = pd.DataFrame([today_obs])
#     df = pd.concat([history_df, today_row], ignore_index=True)
#     df["date"] = pd.to_datetime(df["date"])
#     df.sort_values("date", inplace=True)
#     df.reset_index(drop=True, inplace=True)

#     df["month"]       = df["date"].dt.month
#     df["day_of_year"] = df["date"].dt.dayofyear
#     df["is_summer"]   = df["month"].isin([4,5,6]).astype(int)
#     df["temp_range"]  = df["temp_max"] - df["temp_min"]
#     df["temp_mean"]   = (df["temp_max"] + df["temp_min"]) / 2
#     df["heat_index"]  = df.apply(lambda r: _heat_index(r["temp_max"], r["humidity"]), axis=1)
#     df["humidex"]     = df.apply(lambda r: _humidex(r["temp_max"], r["humidity"]), axis=1)
#     df["feels_like_excess"] = df["heat_index"] - df["temp_max"]
#     df["wind_heat_ratio"]   = df["temp_max"] / (df["wind"].replace(0, 0.1))

#     for w in [3, 7]:
#         df[f"temp_max_roll{w}"]  = df["temp_max"].rolling(w, min_periods=1).mean()
#         df[f"humidity_roll{w}"]  = df["humidity"].rolling(w, min_periods=1).mean()
#         df[f"aqi_roll{w}"]       = df["aqi"].rolling(w, min_periods=1).mean()
#         df[f"rainfall_roll{w}"]  = df["rainfall"].rolling(w, min_periods=1).sum()

#     for lag in [1, 2]:
#         df[f"temp_max_lag{lag}"] = df["temp_max"].shift(lag)
#         df[f"aqi_lag{lag}"]      = df["aqi"].shift(lag)
#         df[f"humidity_lag{lag}"] = df["humidity"].shift(lag)

#     mm = df.groupby("month")["temp_max"].transform("mean")
#     ms = df.groupby("month")["temp_max"].transform("std").replace(0, 1e-6)
#     df["temp_departure"] = df["temp_max"] - mm
#     df["temp_zscore"]    = (df["temp_max"] - mm) / ms
#     df["aqi_departure"]  = df["aqi"] - df.groupby("month")["aqi"].transform("mean")

#     df["drought_flag"]    = (df["rainfall"].rolling(7, min_periods=1).sum() < 2).astype(int)
#     hot_rain              = (df["rainfall"] < 1.0).astype(int)
#     df["dry_days_streak"] = hot_rain.groupby((hot_rain != hot_rain.shift()).cumsum()).cumcount()

#     rolling_rain = df["rainfall"].rolling(30, min_periods=7)
#     spi_mean = rolling_rain.mean()
#     spi_std  = rolling_rain.std().replace(0, 1e-6)
#     df["spi_30"] = (df["rainfall"] - spi_mean) / spi_std

#     bins = [0, 50, 100, 200, 300, 400, 500]
#     df["aqi_category"] = pd.cut(df["aqi"], bins=bins,
#                                  labels=[1,2,3,4,5,6]).astype(float)

#     hot = (df["temp_max"] >= 40.0).astype(int)
#     df["consec_hot_days"] = hot.groupby((hot != hot.shift()).cumsum()).cumcount() + hot

#     df["compound_heat_aqi"]      = ((df["temp_max"]>=38) & (df["aqi"]>=200)).astype(int)
#     df["compound_heat_drought"]  = ((df["temp_max"]>=38) & (df["drought_flag"]==1)).astype(int)
#     df["compound_heat_humidity"] = ((df["temp_max"]>=38) & (df["humidity"]>=60)).astype(int)
#     df["triple_compound"]        = ((df["temp_max"]>=38) & (df["aqi"]>=150) &
#                                     (df["drought_flag"]==1)).astype(int)

#     return df.iloc[-1]


# # ════════════════════════════════════════════════════════════════════════
# # SECTION D: PREDICTION ENGINE
# # ════════════════════════════════════════════════════════════════════════

# RISK_LABELS  = {0:"Low", 1:"Moderate", 2:"High", 3:"Severe"}
# RISK_EMOJIS  = {0:"🟢", 1:"🟡",       2:"🟠",   3:"🔴"}
# ADVISORIES   = {
#     0: "No significant heat risk. Normal activities are safe.",
#     1: "Moderate heat. Stay hydrated. Avoid peak sun hours (12–4 PM).",
#     2: "HIGH HEAT RISK. Limit outdoor activity. Seek cool shelter.",
#     3: "⚠ SEVERE HEAT WAVE. Avoid all outdoor activity. Emergency alert.",
# }

# def load_model(city: str):
#     for tag in [city, "all"]:
#         p = os.path.join(MODEL_DIR, f"classifier_{tag}.pkl")
#         if os.path.exists(p):
#             with open(p, "rb") as f:
#                 bundle = pickle.load(f)
#             reg_p = os.path.join(MODEL_DIR, f"regressor_{tag}.pkl")
#             reg = pickle.load(open(reg_p,"rb")) if os.path.exists(reg_p) else None
#             return bundle["model"], reg
#     raise FileNotFoundError(f"No model found for {city}. Run step4 first.")

# def predict_from_row(enriched_row: pd.Series, clf, reg) -> dict:
#     X = enriched_row[FEATURE_COLS].fillna(0).values.reshape(1, -1)
#     probas     = clf.predict_proba(X)[0]
#     risk_level = int(np.argmax(probas))
#     score      = float(reg.predict(X)[0]) if reg else None

#     compound_flags = {
#         "Heat + AQI"     : int(enriched_row.get("compound_heat_aqi",      0)),
#         "Heat + Drought" : int(enriched_row.get("compound_heat_drought",  0)),
#         "Heat + Humidity": int(enriched_row.get("compound_heat_humidity", 0)),
#         "Triple Compound": int(enriched_row.get("triple_compound",        0)),
#     }
#     active = [k for k, v in compound_flags.items() if v == 1]

#     return {
#         "risk_level"     : risk_level,
#         "risk_label"     : RISK_LABELS[risk_level],
#         "emoji"          : RISK_EMOJIS[risk_level],
#         "confidence"     : round(float(probas[risk_level]) * 100, 1),
#         "probabilities"  : {RISK_LABELS[i]: round(float(p)*100,1)
#                              for i, p in enumerate(probas)},
#         "composite_score": round(score, 1) if score else None,
#         "heat_index"     : round(float(enriched_row["heat_index"]), 1),
#         "humidex"        : round(float(enriched_row["humidex"]), 1),
#         "consec_hot_days": int(enriched_row["consec_hot_days"]),
#         "active_compounds": active,
#         "advisory"       : ADVISORIES[risk_level],
#     }


# # ════════════════════════════════════════════════════════════════════════
# # SECTION E: HISTORY MANAGEMENT
# # ════════════════════════════════════════════════════════════════════════

# def load_history(city: str, n_days: int = 40) -> pd.DataFrame:
#     """
#     Loads the last n_days of raw observations for a city.
#     First tries the labelled CSV (from step2), then the processed CSV.
#     """
#     for fname in [f"labelled_{city}.csv", f"processed_{city}.csv"]:
#         p = os.path.join(PROC_DIR, fname)
#         if os.path.exists(p):
#             df = pd.read_csv(p, parse_dates=["date"])
#             raw_cols = ["date","temp_max","temp_min","humidity","wind","rainfall","aqi"]
#             available = [c for c in raw_cols if c in df.columns]
#             return df[available].tail(n_days).copy()
#     raise FileNotFoundError(f"No history found for {city}. Run steps 1–2 first.")


# def append_to_history(city: str, obs: dict):
#     """
#     Appends today's fetched observation to the processed CSV
#     so future runs have an up-to-date rolling window.
#     """
#     for fname in [f"labelled_{city}.csv", f"processed_{city}.csv"]:
#         p = os.path.join(PROC_DIR, fname)
#         if os.path.exists(p):
#             df = pd.read_csv(p, parse_dates=["date"])
#             # Avoid duplicate dates
#             if pd.Timestamp(obs["date"]) not in df["date"].values:
#                 new_row = pd.DataFrame([{**obs, "city": city}])
#                 df = pd.concat([df, new_row], ignore_index=True)
#                 df.to_csv(p, index=False)
#                 print(f"  [✓] Appended today's observation to {fname}")
#             else:
#                 print(f"  [i] {obs['date']} already in history — not re-appended.")
#             return


# # ════════════════════════════════════════════════════════════════════════
# # SECTION F: FULL CITY PREDICTION PIPELINE
# # ════════════════════════════════════════════════════════════════════════

# def run_city_prediction(city: str, target_date: date = None,
#                          save_log: bool = True) -> dict:
#     """
#     Full pipeline for one city:
#       fetch → build features → predict → print → log
#     """
#     if target_date is None:
#         target_date = date.today()

#     cfg = CITY_CONFIG[city]
#     is_tomorrow = (target_date > date.today())

#     print(f"\n{'━'*60}")
#     print(f"  {city.upper()}  —  {'TOMORROW FORECAST' if is_tomorrow else 'TODAY'}"
#           f"  ({target_date})")
#     print(f"{'━'*60}")

#     # ── 1. Fetch weather ───────────────────────────────────────────────
#     print("  Fetching weather from Open-Meteo...")
#     weather = fetch_open_meteo(cfg["lat"], cfg["lon"], target_date)
#     if weather is None:
#         print(f"  [✗] Weather fetch failed for {city} on {target_date}")
#         return {}
#     print(f"  [✓] Weather: Tmax={weather['temp_max']}°C  "
#           f"Tmin={weather['temp_min']}°C  "
#           f"Humidity={weather['humidity']}%  "
#           f"Wind={weather['wind']}km/h  "
#           f"Rain={weather['rainfall']}mm")

#     # ── 2. Fetch AQI ───────────────────────────────────────────────────
#     aqi = get_aqi_for_city(cfg["lat"], cfg["lon"], cfg["aqicn"], target_date)
#     if aqi is None:
#         # Fallback: rolling 7-day average from history
#         hist = load_history(city, 10)
#         aqi  = float(hist["aqi"].tail(7).mean())
#         print(f"  [i] AQI fallback (7-day avg from history): {aqi:.0f}")

#     today_obs = {**weather, "aqi": round(aqi, 0)}

#     # ── 3. Load rolling history & build features ───────────────────────
#     history = load_history(city, 40)
#     print("  Building feature vector...")
#     enriched = build_feature_row(history, today_obs)

#     # ── 4. Load model & predict ────────────────────────────────────────
#     clf, reg = load_model(city)
#     result   = predict_from_row(enriched, clf, reg)
#     result.update({
#         "city"       : city,
#         "date"       : str(target_date),
#         "is_forecast": is_tomorrow,
#         "raw_obs"    : today_obs,
#     })

#     # ── 5. Print report ────────────────────────────────────────────────
#     print(f"\n  {result['emoji']}  Risk Level   : {result['risk_label']}")
#     print(f"  📊  Composite Score: {result['composite_score']}")
#     print(f"  🎯  Confidence     : {result['confidence']}%")
#     print(f"  🌡   Heat Index    : {result['heat_index']}°C")
#     print(f"  💧  Humidex        : {result['humidex']}°C")
#     print(f"  🔥  Consec Hot Days: {result['consec_hot_days']}")
#     if result["active_compounds"]:
#         print(f"\n  ⚠  Compound Risks: {', '.join(result['active_compounds'])}")
#     print(f"\n  Advisory: {result['advisory']}")
#     print(f"\n  Class Probabilities:")
#     for lbl, p in result["probabilities"].items():
#         bar = "█" * int(p / 5)
#         print(f"    {lbl:10s}: {p:5.1f}%  {bar}")

#     # ── 6. Append to history (today only, not forecast) ────────────────
#     if not is_tomorrow:
#         append_to_history(city, today_obs)

#     # ── 7. Save prediction log ─────────────────────────────────────────
#     if save_log:
#         log_path = os.path.join(LOG_DIR,
#             f"prediction_{city}_{target_date}.json")
#         with open(log_path, "w") as f:
#             json.dump(result, f, indent=2, default=str)
#         print(f"\n  [✓] Prediction saved → {log_path}")

#     return result


# # ════════════════════════════════════════════════════════════════════════
# # SECTION G: MAIN ENTRY POINT
# # ════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Heatwave Risk — Real-Time Daily Prediction")
#     parser.add_argument("--city",     default="all",
#         help="City name or 'all'  (Delhi / Hyderabad / Nagpur / all)")
#     parser.add_argument("--tomorrow", action="store_true",
#         help="Predict for tomorrow instead of today")
#     parser.add_argument("--no-log",   action="store_true",
#         help="Don't save prediction JSON")
#     args = parser.parse_args()

#     target = date.today() + timedelta(days=1) if args.tomorrow else date.today()
#     cities = list(CITY_CONFIG.keys()) if args.city == "all" else [args.city]

#     print("\n" + "="*60)
#     print("  HEATWAVE RISK PREDICTION — REAL-TIME MODULE")
#     print(f"  Target date : {target}  {'(Tomorrow Forecast)' if args.tomorrow else '(Today)'}")
#     print(f"  Cities      : {', '.join(cities)}")
#     print("="*60)

#     all_results = {}
#     for city in cities:
#         if city not in CITY_CONFIG:
#             print(f"[!] Unknown city: {city}")
#             continue
#         result = run_city_prediction(city, target, save_log=not args.no_log)
#         all_results[city] = result

#     # ── Summary across cities ──────────────────────────────────────────
#     if len(all_results) > 1:
#         print("\n" + "="*60)
#         print("  MULTI-CITY SUMMARY")
#         print("="*60)
#         for city, res in all_results.items():
#             if res:
#                 print(f"  {res['emoji']}  {city:12s}: {res['risk_label']:10s} "
#                       f"(score {res['composite_score']}, "
#                       f"confidence {res['confidence']}%)")

#     print("\n[Done] Real-time prediction complete.\n")



"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Step 7: Real-Time API Integration & Live Daily Prediction
========================================================================

TWO FREE APIS USED (no paid plan needed):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  SOURCE 1 — Open-Meteo (Weather Forecast)
    URL   : https://api.open-meteo.com  ← COMPLETELY FREE, NO KEY
    Gives : temp_max, temp_min, humidity, wind_speed, precipitation
            for today AND tomorrow (24-hour advance warning)
    Docs  : https://open-meteo.com/en/docs

  SOURCE 2 — OpenWeatherMap Air Pollution API  ← FREE, KEY NEEDED
    URL   : http://api.openweathermap.org/data/2.5/air_pollution
    Gives : PM2.5, PM10, NO2, O3, SO2 → converted to India CPCB AQI
    Key   : Free at https://openweathermap.org/api (60 calls/min)

  SOURCE 2 (FALLBACK) — AQICN World Air Quality Index  ← FREE, KEY NEEDED
    URL   : https://api.waqi.info/feed/{city}/
    Gives : AQI directly in local scale (closer to India CPCB)
    Key   : Free at https://aqicn.org/data-platform/token/

⚠  IMPORTANT — AQI SCALE DIFFERENCE:
    OpenWeatherMap uses a 1–5 European scale (1=Good, 5=Very Poor).
    India uses CPCB scale: 0–500 (Good < 50, Severe > 400).
    This module converts PM2.5 concentration (µg/m³) → India CPCB AQI
    using the official CPCB breakpoint table. This is the correct way.

WHAT THIS MODULE DOES:
  1. Fetches today's + tomorrow's weather from Open-Meteo (no key)
  2. Fetches AQI data from OWM or AQICN (free key)
  3. Converts raw pollutant data → India CPCB AQI
  4. Appends today's data to the rolling history CSV
  5. Runs the ML prediction pipeline → risk report
  6. Saves prediction log with timestamp
  7. Can be run daily via cron / Task Scheduler

HOW TO USE:
  python step7_realtime_api.py --city Delhi
  python step7_realtime_api.py --city Hyderabad --tomorrow
  python step7_realtime_api.py --city all          ← runs all 3 cities
========================================================================
"""

import os, json, argparse, requests, pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────

# !! SET YOUR KEYS HERE (or pass via environment variables) !!
OWM_API_KEY   = os.environ.get("OWM_API_KEY",  "YOUR_OWM_KEY_HERE")
AQICN_TOKEN   = os.environ.get("AQICN_TOKEN",  "YOUR_AQICN_TOKEN_HERE")

PROC_DIR  = "data/processed"
MODEL_DIR = "models"
LOG_DIR   = "data/predictions"
os.makedirs(LOG_DIR, exist_ok=True)

# City coordinates (lat, lon) and AQICN city names
CITY_CONFIG = {
    "Delhi"     : {"lat": 28.6139,  "lon": 77.2090,  "aqicn": "delhi"},
    "Hyderabad" : {"lat": 17.3850,  "lon": 78.4867,  "aqicn": "hyderabad"},
    "Nagpur"    : {"lat": 21.1458,  "lon": 79.0882,  "aqicn": "nagpur"},
}

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


# ════════════════════════════════════════════════════════════════════════
# SECTION A: AQI CONVERSION — OWM PM2.5 → India CPCB AQI
# ════════════════════════════════════════════════════════════════════════

# Official CPCB breakpoint table for PM2.5 (µg/m³) → AQI
# Source: Central Pollution Control Board, India
CPCB_PM25_BREAKPOINTS = [
    # (C_low, C_high, I_low, I_high)
    (0.0,   30.0,   0,   50),    # Good
    (30.1,  60.0,   51,  100),   # Satisfactory
    (60.1,  90.0,   101, 200),   # Moderately Polluted
    (90.1,  120.0,  201, 300),   # Poor
    (120.1, 250.0,  301, 400),   # Very Poor
    (250.1, 500.0,  401, 500),   # Severe
]

CPCB_PM10_BREAKPOINTS = [
    (0,    50,    0,   50),
    (51,   100,   51,  100),
    (101,  250,   101, 200),
    (251,  350,   201, 300),
    (351,  430,   301, 400),
    (431,  600,   401, 500),
]

def concentration_to_cpcb_aqi(concentration: float,
                                breakpoints: list) -> float:
    """
    Linear interpolation within CPCB breakpoint table.
    AQI = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
    """
    if concentration < 0:
        return 0.0
    for (C_lo, C_hi, I_lo, I_hi) in breakpoints:
        if C_lo <= concentration <= C_hi:
            aqi = ((I_hi - I_lo) / (C_hi - C_lo)) * (concentration - C_lo) + I_lo
            return round(aqi, 1)
    # Above highest breakpoint
    return 500.0

def owm_components_to_india_aqi(components: dict) -> float:
    """
    OWM returns raw pollutant concentrations (µg/m³).
    Convert PM2.5 and PM10 to India CPCB AQI and return the higher value
    (conservative: worst pollutant governs overall AQI per CPCB method).
    """
    pm25 = components.get("pm2_5", 0)
    pm10 = components.get("pm10",  0)

    aqi_pm25 = concentration_to_cpcb_aqi(pm25, CPCB_PM25_BREAKPOINTS)
    aqi_pm10 = concentration_to_cpcb_aqi(pm10, CPCB_PM10_BREAKPOINTS)

    # CPCB: final AQI = maximum of all sub-indices
    final_aqi = max(aqi_pm25, aqi_pm10)
    return final_aqi


# ════════════════════════════════════════════════════════════════════════
# SECTION B: API FETCHERS
# ════════════════════════════════════════════════════════════════════════

def fetch_open_meteo(lat: float, lon: float,
                     target_date: date) -> dict | None:
    """
    Open-Meteo Forecast API — completely free, no key required.
    Returns daily values for target_date.

    Variables fetched:
      temperature_2m_max, temperature_2m_min,
      relative_humidity_2m_max (proxy), wind_speed_10m_max,
      precipitation_sum
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude"              : lat,
        "longitude"             : lon,
        "daily"                 : [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "relative_humidity_2m_max",   # daily max humidity
        ],
        "timezone"              : "Asia/Kolkata",
        "forecast_days"         : 7,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data  = r.json()
        daily = data["daily"]
        dates = pd.to_datetime(daily["time"])

        # Find index for target date
        target_ts = pd.Timestamp(target_date)
        if target_ts not in dates:
            print(f"  [!] Open-Meteo: {target_date} not in forecast window.")
            return None
        idx = list(dates).index(target_ts)

        return {
            "date"    : str(target_date),
            "temp_max": round(daily["temperature_2m_max"][idx], 1),
            "temp_min": round(daily["temperature_2m_min"][idx], 1),
            "humidity": round(daily["relative_humidity_2m_max"][idx], 1),
            "wind"    : round(daily["wind_speed_10m_max"][idx], 1),
            "rainfall": round(daily["precipitation_sum"][idx] or 0, 1),
        }
    except requests.exceptions.RequestException as e:
        print(f"  [!] Open-Meteo request failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"  [!] Open-Meteo parse error: {e}")
        return None


def fetch_open_meteo_aqi(lat: float, lon: float,
                         target_date: date) -> float | None:
    """
    Open-Meteo Air Quality API — completely FREE, NO KEY required.
    Same provider as the weather API. Provides hourly PM2.5 and PM10
    which are converted to India CPCB AQI.
    Supports both today and forecast (up to 5 days ahead).
    Docs: https://air-quality-api.open-meteo.com
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude"  : lat,
        "longitude" : lon,
        "hourly"    : ["pm2_5", "pm10"],
        "timezone"  : "Asia/Kolkata",
        "forecast_days": 7,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data    = r.json()
        hourly  = data["hourly"]
        times   = pd.to_datetime(hourly["time"])
        pm25_vals = hourly["pm2_5"]
        pm10_vals = hourly["pm10"]

        # Filter to target date, pick the reading closest to 2 PM (peak pollution)
        target_ts   = pd.Timestamp(target_date)
        target_peak = target_ts.replace(hour=14)
        mask = times.date == target_date
        idx_list = [i for i, t in enumerate(times) if t.date() == target_date]
        if not idx_list:
            print(f"  [!] Open-Meteo AQ: no data for {target_date}")
            return None

        # Pick reading closest to 2 PM
        best_idx = min(idx_list,
                       key=lambda i: abs((times[i] - target_peak).total_seconds()))

        pm25 = pm25_vals[best_idx] or 0
        pm10 = pm10_vals[best_idx] or 0
        aqi  = owm_components_to_india_aqi({"pm2_5": pm25, "pm10": pm10})
        return aqi

    except requests.exceptions.RequestException as e:
        print(f"  [!] Open-Meteo AQ request failed: {e}")
        return None
    except Exception as e:
        print(f"  [!] Open-Meteo AQ parse error: {e}")
        return None


def fetch_owm_aqi(lat: float, lon: float,
                  target_date: date = None) -> float | None:
    """
    OpenWeatherMap Air Pollution API (free tier).
    NOTE: New OWM keys take up to 2 hours to activate after registration.
    If you get a 401 error, wait and retry — or use Open-Meteo AQ instead.
    For today → uses /air_pollution (current).
    For tomorrow → uses /air_pollution/forecast, picks closest hour.
    Returns India CPCB AQI (converted from PM2.5/PM10).
    """
    if OWM_API_KEY == "YOUR_OWM_KEY_HERE":
        print("  [!] OWM key not set — skipping OWM AQI fetch.")
        return None

    today = date.today()
    is_forecast = (target_date is not None and target_date > today)

    if is_forecast:
        url    = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"
        params = {"lat": lat, "lon": lon, "appid": OWM_API_KEY}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            items = r.json()["list"]
            # Find entry closest to noon on target_date
            target_noon = datetime.combine(target_date, datetime.min.time()) \
                          .replace(hour=12)
            best  = min(items, key=lambda x: abs(
                datetime.fromtimestamp(x["dt"]) - target_noon))
            return owm_components_to_india_aqi(best["components"])
        except Exception as e:
            print(f"  [!] OWM forecast AQI failed: {e}")
            return None
    else:
        url    = "http://api.openweathermap.org/data/2.5/air_pollution"
        params = {"lat": lat, "lon": lon, "appid": OWM_API_KEY}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            components = r.json()["list"][0]["components"]
            return owm_components_to_india_aqi(components)
        except Exception as e:
            print(f"  [!] OWM current AQI failed: {e}")
            return None


def fetch_aqicn_aqi(city_slug: str) -> float | None:
    """
    AQICN World Air Quality Index API (free, needs token).
    Returns AQI in the station's local scale — for Indian cities this
    is typically close to the India CPCB scale.
    Note: AQICN only provides current (no forecast).
    """
    if AQICN_TOKEN == "YOUR_AQICN_TOKEN_HERE":
        print("  [!] AQICN token not set — skipping AQICN fetch.")
        return None

    url = f"https://api.waqi.info/feed/{city_slug}/"
    try:
        r   = requests.get(url, params={"token": AQICN_TOKEN}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "ok":
            aqi_val = data["data"]["aqi"]
            return float(aqi_val) if aqi_val != "-" else None
        else:
            print(f"  [!] AQICN error: {data.get('data', 'unknown')}")
            return None
    except Exception as e:
        print(f"  [!] AQICN request failed: {e}")
        return None


def get_aqi_for_city(lat: float, lon: float, city_slug: str,
                     target_date: date) -> float:
    """
    AQI fetch strategy (priority order):
      1. Open-Meteo Air Quality API  — FREE, no key, supports forecast
      2. OpenWeatherMap Air Pollution — free key needed; 401 = key not yet active
      3. AQICN                        — free token needed, current only
      4. Historical 7-day rolling avg — always available offline fallback
    """
    print(f"  Fetching AQI...")

    # ── Priority 1: Open-Meteo AQ (no key, best option) ──────────────
    aqi = fetch_open_meteo_aqi(lat, lon, target_date)
    if aqi is not None:
        print(f"  [✓] Open-Meteo AQ → India CPCB AQI: {aqi:.0f}")
        return aqi

    # ── Priority 2: OWM (needs active key) ───────────────────────────
    aqi = fetch_owm_aqi(lat, lon, target_date)
    if aqi is not None:
        print(f"  [✓] OWM AQI (India CPCB): {aqi:.0f}")
        return aqi

    # ── Priority 3: AQICN fallback ────────────────────────────────────
    aqi = fetch_aqicn_aqi(city_slug)
    if aqi is not None:
        print(f"  [✓] AQICN AQI: {aqi:.0f}")
        return aqi

    print("  [!] All AQI sources failed. Will use historical rolling average.")
    return None   # caller handles fallback from history


# ════════════════════════════════════════════════════════════════════════
# SECTION C: FEATURE ENGINEERING (mirrors step1, self-contained)
# ════════════════════════════════════════════════════════════════════════

def _heat_index(T, RH):
    TF = T * 9/5 + 32
    HI = (-42.379 + 2.04901523*TF + 10.14333127*RH
          - 0.22475541*TF*RH - 0.00683783*TF**2
          - 0.05481717*RH**2 + 0.00122874*TF**2*RH
          + 0.00085282*TF*RH**2 - 0.00000199*TF**2*RH**2)
    return (HI - 32) * 5/9

def _humidex(T, RH):
    Td = T - ((100 - RH) / 5)
    e  = 6.105 * np.exp(25.22 * (Td - 273.16) / (Td + 1e-9))
    return T + 0.5555 * (e - 10)

def build_feature_row(history_df: pd.DataFrame,
                       today_obs: dict) -> pd.Series:
    """
    Appends today's raw observation to history and engineers all features.
    Returns the single-row feature Series for today.
    """
    today_row = pd.DataFrame([today_obs])
    df = pd.concat([history_df, today_row], ignore_index=True)
    ###-------------------CHANGED-------------------###
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["month"]       = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_summer"]   = df["month"].isin([4,5,6]).astype(int)
    df["temp_range"]  = df["temp_max"] - df["temp_min"]
    df["temp_mean"]   = (df["temp_max"] + df["temp_min"]) / 2
    df["heat_index"]  = df.apply(lambda r: _heat_index(r["temp_max"], r["humidity"]), axis=1)
    df["humidex"]     = df.apply(lambda r: _humidex(r["temp_max"], r["humidity"]), axis=1)
    df["feels_like_excess"] = df["heat_index"] - df["temp_max"]
    df["wind_heat_ratio"]   = df["temp_max"] / (df["wind"].replace(0, 0.1))

    for w in [3, 7]:
        df[f"temp_max_roll{w}"]  = df["temp_max"].rolling(w, min_periods=1).mean()
        df[f"humidity_roll{w}"]  = df["humidity"].rolling(w, min_periods=1).mean()
        df[f"aqi_roll{w}"]       = df["aqi"].rolling(w, min_periods=1).mean()
        df[f"rainfall_roll{w}"]  = df["rainfall"].rolling(w, min_periods=1).sum()

    for lag in [1, 2]:
        df[f"temp_max_lag{lag}"] = df["temp_max"].shift(lag)
        df[f"aqi_lag{lag}"]      = df["aqi"].shift(lag)
        df[f"humidity_lag{lag}"] = df["humidity"].shift(lag)

    mm = df.groupby("month")["temp_max"].transform("mean")
    ms = df.groupby("month")["temp_max"].transform("std").replace(0, 1e-6)
    df["temp_departure"] = df["temp_max"] - mm
    df["temp_zscore"]    = (df["temp_max"] - mm) / ms
    df["aqi_departure"]  = df["aqi"] - df.groupby("month")["aqi"].transform("mean")

    df["drought_flag"]    = (df["rainfall"].rolling(7, min_periods=1).sum() < 2).astype(int)
    hot_rain              = (df["rainfall"] < 1.0).astype(int)
    df["dry_days_streak"] = hot_rain.groupby((hot_rain != hot_rain.shift()).cumsum()).cumcount()

    rolling_rain = df["rainfall"].rolling(30, min_periods=7)
    spi_mean = rolling_rain.mean()
    spi_std  = rolling_rain.std().replace(0, 1e-6)
    df["spi_30"] = (df["rainfall"] - spi_mean) / spi_std

    bins = [0, 50, 100, 200, 300, 400, 500]
    df["aqi_category"] = pd.cut(df["aqi"], bins=bins,
                                 labels=[1,2,3,4,5,6]).astype(float)

    hot = (df["temp_max"] >= 40.0).astype(int)
    df["consec_hot_days"] = hot.groupby((hot != hot.shift()).cumsum()).cumcount() + hot

    df["compound_heat_aqi"]      = ((df["temp_max"]>=38) & (df["aqi"]>=200)).astype(int)
    df["compound_heat_drought"]  = ((df["temp_max"]>=38) & (df["drought_flag"]==1)).astype(int)
    df["compound_heat_humidity"] = ((df["temp_max"]>=38) & (df["humidity"]>=60)).astype(int)
    df["triple_compound"]        = ((df["temp_max"]>=38) & (df["aqi"]>=150) &
                                    (df["drought_flag"]==1)).astype(int)

    return df.iloc[-1]


# ════════════════════════════════════════════════════════════════════════
# SECTION D: PREDICTION ENGINE
# ════════════════════════════════════════════════════════════════════════

RISK_LABELS  = {0:"Low", 1:"Moderate", 2:"High", 3:"Severe"}
RISK_EMOJIS  = {0:"🟢", 1:"🟡",       2:"🟠",   3:"🔴"}
ADVISORIES   = {
    0: "No significant heat risk. Normal activities are safe.",
    1: "Moderate heat. Stay hydrated. Avoid peak sun hours (12–4 PM).",
    2: "HIGH HEAT RISK. Limit outdoor activity. Seek cool shelter.",
    3: "⚠ SEVERE HEAT WAVE. Avoid all outdoor activity. Emergency alert.",
}

def load_model(city: str):
    for tag in [city, "all"]:
        p = os.path.join(MODEL_DIR, f"classifier_{tag}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                bundle = pickle.load(f)
            reg_p = os.path.join(MODEL_DIR, f"regressor_{tag}.pkl")
            reg = pickle.load(open(reg_p,"rb")) if os.path.exists(reg_p) else None
            return bundle["model"], reg
    raise FileNotFoundError(f"No model found for {city}. Run step4 first.")

def predict_from_row(enriched_row: pd.Series, clf, reg) -> dict:
    X = enriched_row[FEATURE_COLS].fillna(0).values.reshape(1, -1)
    probas     = clf.predict_proba(X)[0]
    risk_level = int(np.argmax(probas))
    score      = float(reg.predict(X)[0]) if reg else None

    compound_flags = {
        "Heat + AQI"     : int(enriched_row.get("compound_heat_aqi",      0)),
        "Heat + Drought" : int(enriched_row.get("compound_heat_drought",  0)),
        "Heat + Humidity": int(enriched_row.get("compound_heat_humidity", 0)),
        "Triple Compound": int(enriched_row.get("triple_compound",        0)),
    }
    active = [k for k, v in compound_flags.items() if v == 1]

    return {
        "risk_level"     : risk_level,
        "risk_label"     : RISK_LABELS[risk_level],
        "emoji"          : RISK_EMOJIS[risk_level],
        "confidence"     : round(float(probas[risk_level]) * 100, 1),
        "probabilities"  : {RISK_LABELS[i]: round(float(p)*100,1)
                             for i, p in enumerate(probas)},
        "composite_score": round(score, 1) if score else None,
        "heat_index"     : round(float(enriched_row["heat_index"]), 1),
        "humidex"        : round(float(enriched_row["humidex"]), 1),
        "consec_hot_days": int(enriched_row["consec_hot_days"]),
        "active_compounds": active,
        "advisory"       : ADVISORIES[risk_level],
    }


# ════════════════════════════════════════════════════════════════════════
# SECTION E: HISTORY MANAGEMENT
# ════════════════════════════════════════════════════════════════════════

def load_history(city: str, n_days: int = 40) -> pd.DataFrame:
    """
    Loads the last n_days of raw observations for a city.
    First tries the labelled CSV (from step2), then the processed CSV.
    """
    for fname in [f"labelled_{city}.csv", f"processed_{city}.csv"]:
        p = os.path.join(PROC_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["date"])
            raw_cols = ["date","temp_max","temp_min","humidity","wind","rainfall","aqi"]
            available = [c for c in raw_cols if c in df.columns]
            return df[available].tail(n_days).copy()
    raise FileNotFoundError(f"No history found for {city}. Run steps 1–2 first.")


def append_to_history(city: str, obs: dict):
    """
    Appends today's fetched observation to the processed CSV
    so future runs have an up-to-date rolling window.
    """
    for fname in [f"labelled_{city}.csv", f"processed_{city}.csv"]:
        p = os.path.join(PROC_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["date"])
            # Avoid duplicate dates
            if pd.Timestamp(obs["date"]) not in df["date"].values:
                new_row = pd.DataFrame([{**obs, "city": city}])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(p, index=False)
                print(f"  [✓] Appended today's observation to {fname}")
            else:
                print(f"  [i] {obs['date']} already in history — not re-appended.")
            return


# ════════════════════════════════════════════════════════════════════════
# SECTION F: FULL CITY PREDICTION PIPELINE
# ════════════════════════════════════════════════════════════════════════

def run_city_prediction(city: str, target_date: date = None,
                         save_log: bool = True) -> dict:
    """
    Full pipeline for one city:
      fetch → build features → predict → print → log
    """
    if target_date is None:
        target_date = date.today()

    cfg = CITY_CONFIG[city]
    is_tomorrow = (target_date > date.today())

    print(f"\n{'━'*60}")
    print(f"  {city.upper()}  —  {'TOMORROW FORECAST' if is_tomorrow else 'TODAY'}"
          f"  ({target_date})")
    print(f"{'━'*60}")

    # ── 1. Fetch weather ───────────────────────────────────────────────
    print("  Fetching weather from Open-Meteo...")
    weather = fetch_open_meteo(cfg["lat"], cfg["lon"], target_date)
    if weather is None:
        print(f"  [✗] Weather fetch failed for {city} on {target_date}")
        return {}
    print(f"  [✓] Weather: Tmax={weather['temp_max']}°C  "
          f"Tmin={weather['temp_min']}°C  "
          f"Humidity={weather['humidity']}%  "
          f"Wind={weather['wind']}km/h  "
          f"Rain={weather['rainfall']}mm")

    # ── 2. Fetch AQI ───────────────────────────────────────────────────
    aqi = get_aqi_for_city(cfg["lat"], cfg["lon"], cfg["aqicn"], target_date)
    if aqi is None:
        # Fallback: rolling 7-day average from history
        hist = load_history(city, 10)
        aqi  = float(hist["aqi"].tail(7).mean())
        print(f"  [i] AQI fallback (7-day avg from history): {aqi:.0f}")

    today_obs = {**weather, "aqi": round(aqi, 0)}

    # ── 3. Load rolling history & build features ───────────────────────
    history = load_history(city, 40)
    print("  Building feature vector...")
    enriched = build_feature_row(history, today_obs)

    # ── 4. Load model & predict ────────────────────────────────────────
    clf, reg = load_model(city)
    result   = predict_from_row(enriched, clf, reg)
    result.update({
        "city"       : city,
        "date"       : str(target_date),
        "is_forecast": is_tomorrow,
        "raw_obs"    : today_obs,
    })

    # ── 5. Print report ────────────────────────────────────────────────
    print(f"\n  {result['emoji']}  Risk Level   : {result['risk_label']}")
    print(f"  📊  Composite Score: {result['composite_score']}")
    print(f"  🎯  Confidence     : {result['confidence']}%")
    print(f"  🌡   Heat Index    : {result['heat_index']}°C")
    print(f"  💧  Humidex        : {result['humidex']}°C")
    print(f"  🔥  Consec Hot Days: {result['consec_hot_days']}")
    if result["active_compounds"]:
        print(f"\n  ⚠  Compound Risks: {', '.join(result['active_compounds'])}")
    print(f"\n  Advisory: {result['advisory']}")
    print(f"\n  Class Probabilities:")
    for lbl, p in result["probabilities"].items():
        bar = "█" * int(p / 5)
        print(f"    {lbl:10s}: {p:5.1f}%  {bar}")

    # ── 6. Append to history (today only, not forecast) ────────────────
    if not is_tomorrow:
        append_to_history(city, today_obs)

    # ── 7. Save prediction log ─────────────────────────────────────────
    if save_log:
        log_path = os.path.join(LOG_DIR,
            f"prediction_{city}_{target_date}.json")
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  [✓] Prediction saved → {log_path}")

    return result


# ════════════════════════════════════════════════════════════════════════
# SECTION G: MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Heatwave Risk — Real-Time Daily Prediction")
    parser.add_argument("--city",     default="all",
        help="City name or 'all'  (Delhi / Hyderabad / Nagpur / all)")
    parser.add_argument("--tomorrow", action="store_true",
        help="Predict for tomorrow instead of today")
    parser.add_argument("--no-log",   action="store_true",
        help="Don't save prediction JSON")
    args = parser.parse_args()

    target = date.today() + timedelta(days=1) if args.tomorrow else date.today()
    cities = list(CITY_CONFIG.keys()) if args.city == "all" else [args.city]

    print("\n" + "="*60)
    print("  HEATWAVE RISK PREDICTION — REAL-TIME MODULE")
    print(f"  Target date : {target}  {'(Tomorrow Forecast)' if args.tomorrow else '(Today)'}")
    print(f"  Cities      : {', '.join(cities)}")
    print("="*60)

    all_results = {}
    for city in cities:
        if city not in CITY_CONFIG:
            print(f"[!] Unknown city: {city}")
            continue
        result = run_city_prediction(city, target, save_log=not args.no_log)
        all_results[city] = result

    # ── Summary across cities ──────────────────────────────────────────
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("  MULTI-CITY SUMMARY")
        print("="*60)
        for city, res in all_results.items():
            if res:
                print(f"  {res['emoji']}  {city:12s}: {res['risk_label']:10s} "
                      f"(score {res['composite_score']}, "
                      f"confidence {res['confidence']}%)")

    print("\n[Done] Real-time prediction complete.\n")