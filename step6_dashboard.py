
# # ++++++++++++++++++++++++++ VERSION 4 (WITH SEASONAL FORECASTING) +++++++++++++++++++++++++++++++++

# """
# ========================================================================
# HEATWAVE RISK PREDICTION SYSTEM
# Final Integrated Dashboard  —  step6_dashboard_integrated.py
# ========================================================================
# Run with:  streamlit run step6_dashboard_integrated.py

# Combines historical analysis (step6) + real-time API prediction (step7)
# into one application.

# Pages:
#   🔴 Live Prediction   – Fetch today/tomorrow from APIs, run ML model
#   🏠 Overview          – Historical risk cards + period stats
#   📈 Trends            – Time-series, compound events, heatmaps
#   🔍 Manual Predict    – Slider-based custom prediction + radar chart
#   📊 Analytics         – Cross-city comparison, seasonality, violin plots
#   📋 Prediction Log    – History of all live predictions made

# APIs used (all free):
#   Open-Meteo Weather   : no key needed
#   Open-Meteo AQ        : no key needed  ← primary AQI source
#   OpenWeatherMap AQ    : free key needed (fallback, ~2h activation delay)
#   AQICN                : free token needed (fallback, current only)
# ========================================================================
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from datetime import datetime, timedelta, date
# import os, pickle, json, requests, warnings
# warnings.filterwarnings("ignore")

# # ── PAGE CONFIG ───────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title  = "Heatwave Risk System",
#     page_icon   = "🌡",
#     layout      = "wide",
#     initial_sidebar_state = "expanded",
# )

# # ── CONSTANTS ────────────────────────────────────────────────────────────
# CITIES      = ["Delhi", "Hyderabad", "Nagpur"]
# RISK_COLORS = {0:"#2ECC71", 1:"#F1C40F", 2:"#E67E22", 3:"#E74C3C"}
# RISK_LABELS = {0:"Low",     1:"Moderate", 2:"High",   3:"Severe"}
# RISK_EMOJIS = {0:"🟢",      1:"🟡",       2:"🟠",     3:"🔴"}
# CITY_COLORS = {"Delhi":"#3498DB", "Hyderabad":"#E74C3C", "Nagpur":"#2ECC71"}
# CITY_CONFIG = {
#     "Delhi"     : {"lat": 28.6139, "lon": 77.2090, "aqicn": "delhi"},
#     "Hyderabad" : {"lat": 17.3850, "lon": 78.4867, "aqicn": "hyderabad"},
#     "Nagpur"    : {"lat": 21.1458, "lon": 79.0882, "aqicn": "nagpur"},
# }
# PROC_DIR  = "data/processed"
# MODEL_DIR = "models"
# LOG_DIR   = "data/predictions"
# os.makedirs(LOG_DIR, exist_ok=True)

# FEATURE_COLS = [
#     "temp_max","temp_min","humidity","wind","rainfall","aqi",
#     "heat_index","humidex","temp_range","temp_mean","feels_like_excess",
#     "wind_heat_ratio",
#     "temp_max_roll3","temp_max_roll7","humidity_roll7","aqi_roll7","rainfall_roll7",
#     "temp_max_lag1","temp_max_lag2","aqi_lag1","humidity_lag1",
#     "temp_departure","temp_zscore","aqi_departure",
#     "dry_days_streak","spi_30","drought_flag",
#     "month","day_of_year","is_summer",
#     "compound_heat_aqi","compound_heat_drought","compound_heat_humidity",
#     "triple_compound","consec_hot_days","aqi_category",
# ]

# ADVISORIES = {
#     0: "No significant heat risk. Normal outdoor activities are safe.",
#     1: "Moderate heat. Stay hydrated. Avoid peak sun (12–4 PM). Monitor elderly & children.",
#     2: "HIGH HEAT RISK. Limit outdoor activity. Seek cool shelter. Watch for heat exhaustion.",
#     3: "⚠ SEVERE HEAT WAVE. Avoid ALL outdoor activity. Emergency health protocols active.",
# }

# # ── CSS ───────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
# .risk-card {
#   border-radius:14px; padding:20px; margin:4px;
#   text-align:center; font-family:Arial,sans-serif;
# }
# .live-badge {
#   display:inline-block; background:#E74C3C; color:#fff;
#   font-size:0.7rem; font-weight:700; border-radius:20px;
#   padding:2px 8px; margin-left:6px; vertical-align:middle;
#   animation: pulse 2s infinite;
# }
# @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
# .metric-big  { font-size:2.2rem; font-weight:800; }
# .metric-sub  { font-size:0.85rem; color:#666; }
# .risk-LOW      { background:#d5f5e3; border:2px solid #2ECC71; }
# .risk-MODERATE { background:#fef9e7; border:2px solid #F1C40F; }
# .risk-HIGH     { background:#fdebd0; border:2px solid #E67E22; }
# .risk-SEVERE   { background:#fadbd8; border:2px solid #E74C3C; }
# .source-tag {
#   font-size:0.7rem; background:#eee; border-radius:8px;
#   padding:2px 7px; color:#555;
# }
# .advisory-box {
#   background:#f8f9fa; border-left:4px solid #E67E22;
#   padding:10px 14px; border-radius:6px;
#   font-size:0.9rem; margin-top:8px;
# }
# </style>
# """, unsafe_allow_html=True)


# # ════════════════════════════════════════════════════════════════════════
# # SECTION A: AQI CONVERSION  (PM2.5/PM10 µg/m³ → India CPCB 0–500)
# # ════════════════════════════════════════════════════════════════════════
# CPCB_PM25 = [(0,30,0,50),(30.1,60,51,100),(60.1,90,101,200),
#              (90.1,120,201,300),(120.1,250,301,400),(250.1,500,401,500)]
# CPCB_PM10 = [(0,50,0,50),(51,100,51,100),(101,250,101,200),
#              (251,350,201,300),(351,430,301,400),(431,600,401,500)]

# def _cpcb(c, bps):
#     if c <= 0: return 0.0
#     for lo, hi, ilo, ihi in bps:
#         if lo <= c <= hi:
#             return round(((ihi-ilo)/(hi-lo))*(c-lo)+ilo, 1)
#     return 500.0

# def pm_to_india_aqi(pm25, pm10):
#     return max(_cpcb(pm25 or 0, CPCB_PM25), _cpcb(pm10 or 0, CPCB_PM10))


# # ════════════════════════════════════════════════════════════════════════
# # SECTION B: API FETCHERS
# # ════════════════════════════════════════════════════════════════════════

# def fetch_weather(lat, lon, target_date):
#     """Open-Meteo forecast — free, no key."""
#     try:
#         r = requests.get("https://api.open-meteo.com/v1/forecast", params={
#             "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
#             "forecast_days": 7,
#             "daily": ["temperature_2m_max","temperature_2m_min",
#                       "precipitation_sum","wind_speed_10m_max",
#                       "relative_humidity_2m_max"],
#         }, timeout=12)
#         r.raise_for_status()
#         daily  = r.json()["daily"]
#         dates  = pd.to_datetime(daily["time"])
#         target = pd.Timestamp(target_date)
#         if target not in list(dates): return None, "date not in window"
#         i = list(dates).index(target)
#         return {
#             "date"    : str(target_date),
#             "temp_max": round(daily["temperature_2m_max"][i], 1),
#             "temp_min": round(daily["temperature_2m_min"][i], 1),
#             "humidity": round(daily["relative_humidity_2m_max"][i] or 50, 1),
#             "wind"    : round(daily["wind_speed_10m_max"][i] or 0, 1),
#             "rainfall": round(daily["precipitation_sum"][i] or 0, 1),
#         }, None
#     except Exception as e:
#         return None, str(e)


# def fetch_aqi_open_meteo(lat, lon, target_date):
#     """Open-Meteo Air Quality — free, no key. Primary AQI source."""
#     try:
#         r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality",
#             params={"latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
#                     "forecast_days": 7, "hourly": ["pm2_5","pm10"]}, timeout=12)
#         r.raise_for_status()
#         hourly = r.json()["hourly"]
#         times  = pd.to_datetime(hourly["time"])
#         pm25   = hourly["pm2_5"]
#         pm10   = hourly["pm10"]
#         peak   = pd.Timestamp(target_date).replace(hour=14)
#         idxs   = [i for i, t in enumerate(times) if t.date() == target_date]
#         if not idxs: return None, "no data for date"
#         best   = min(idxs, key=lambda i: abs((times[i]-peak).total_seconds()))
#         aqi    = pm_to_india_aqi(pm25[best] or 0, pm10[best] or 0)
#         return round(aqi, 0), None
#     except Exception as e:
#         return None, str(e)


# def fetch_aqi_owm(lat, lon, target_date, api_key):
#     """OpenWeatherMap Air Pollution — free key needed (2h activation)."""
#     if not api_key or api_key == "YOUR_KEY":
#         return None, "key not set"
#     today    = date.today()
#     forecast = target_date > today
#     url      = ("http://api.openweathermap.org/data/2.5/air_pollution/forecast"
#                 if forecast else
#                 "http://api.openweathermap.org/data/2.5/air_pollution")
#     try:
#         r = requests.get(url, params={"lat":lat,"lon":lon,"appid":api_key}, timeout=12)
#         if r.status_code == 401:
#             return None, "401 — key not yet active (wait ~2h after signup)"
#         r.raise_for_status()
#         items = r.json()["list"]
#         if forecast:
#             peak = datetime.combine(target_date, datetime.min.time()).replace(hour=14)
#             item = min(items, key=lambda x: abs(datetime.fromtimestamp(x["dt"])-peak))
#         else:
#             item = items[0]
#         c   = item["components"]
#         aqi = pm_to_india_aqi(c.get("pm2_5",0), c.get("pm10",0))
#         return round(aqi, 0), None
#     except Exception as e:
#         return None, str(e)


# def fetch_aqi_aqicn(city_slug, token):
#     """AQICN — free token, current only."""
#     if not token or token == "YOUR_TOKEN":
#         return None, "token not set"
#     try:
#         r   = requests.get(f"https://api.waqi.info/feed/{city_slug}/",
#                            params={"token": token}, timeout=12)
#         r.raise_for_status()
#         d   = r.json()
#         if d.get("status") == "ok":
#             v = d["data"]["aqi"]
#             return (float(v), None) if v != "-" else (None, "no reading")
#         return None, d.get("data","unknown error")
#     except Exception as e:
#         return None, str(e)


# def get_aqi(lat, lon, city_slug, target_date, owm_key="", aqicn_token=""):
#     """
#     Priority: Open-Meteo AQ → OWM → AQICN → history fallback.
#     Returns (aqi_value, source_label).
#     """
#     aqi, err = fetch_aqi_open_meteo(lat, lon, target_date)
#     if aqi is not None:
#         return aqi, "Open-Meteo AQ (free)"

#     aqi, err = fetch_aqi_owm(lat, lon, target_date, owm_key)
#     if aqi is not None:
#         return aqi, "OpenWeatherMap"

#     aqi, err = fetch_aqi_aqicn(city_slug, aqicn_token)
#     if aqi is not None:
#         return aqi, "AQICN"

#     return None, "all sources failed"


# # ════════════════════════════════════════════════════════════════════════
# # SECTION C: FEATURE ENGINEERING  (self-contained, mirrors step1)
# # ════════════════════════════════════════════════════════════════════════

# def _heat_index(T, RH):
#     TF = T*9/5+32
#     HI = (-42.379+2.04901523*TF+10.14333127*RH-0.22475541*TF*RH
#           -0.00683783*TF**2-0.05481717*RH**2+0.00122874*TF**2*RH
#           +0.00085282*TF*RH**2-0.00000199*TF**2*RH**2)
#     return (HI-32)*5/9

# def _humidex(T, RH):
#     Td = T-((100-RH)/5)
#     e  = 6.105*np.exp(25.22*(Td-273.16)/(Td+1e-9))
#     return T+0.5555*(e-10)

# def build_features(history_df: pd.DataFrame, today_obs: dict) -> pd.Series:
#     today_row = pd.DataFrame([today_obs])
#     df = pd.concat([history_df, today_row], ignore_index=True)
#     df["date"] = pd.to_datetime(df["date"])
#     df.sort_values("date", inplace=True)
#     df.reset_index(drop=True, inplace=True)

#     df["month"]        = df["date"].dt.month
#     df["day_of_year"]  = df["date"].dt.dayofyear
#     df["is_summer"]    = df["month"].isin([4,5,6]).astype(int)
#     df["temp_range"]   = df["temp_max"]-df["temp_min"]
#     df["temp_mean"]    = (df["temp_max"]+df["temp_min"])/2
#     df["heat_index"]   = df.apply(lambda r: _heat_index(r["temp_max"],r["humidity"]),axis=1)
#     df["humidex"]      = df.apply(lambda r: _humidex(r["temp_max"],r["humidity"]),axis=1)
#     df["feels_like_excess"] = df["heat_index"]-df["temp_max"]
#     df["wind_heat_ratio"]   = df["temp_max"]/(df["wind"].replace(0,0.1))

#     for w in [3,7]:
#         df[f"temp_max_roll{w}"] = df["temp_max"].rolling(w,min_periods=1).mean()
#         df[f"humidity_roll{w}"] = df["humidity"].rolling(w,min_periods=1).mean()
#         df[f"aqi_roll{w}"]      = df["aqi"].rolling(w,min_periods=1).mean()
#         df[f"rainfall_roll{w}"] = df["rainfall"].rolling(w,min_periods=1).sum()

#     for lag in [1,2]:
#         df[f"temp_max_lag{lag}"] = df["temp_max"].shift(lag)
#         df[f"aqi_lag{lag}"]      = df["aqi"].shift(lag)
#         df[f"humidity_lag{lag}"] = df["humidity"].shift(lag)

#     mm = df.groupby("month")["temp_max"].transform("mean")
#     ms = df.groupby("month")["temp_max"].transform("std").replace(0,1e-6)
#     df["temp_departure"] = df["temp_max"]-mm
#     df["temp_zscore"]    = (df["temp_max"]-mm)/ms
#     df["aqi_departure"]  = df["aqi"]-df.groupby("month")["aqi"].transform("mean")

#     df["drought_flag"]    = (df["rainfall"].rolling(7,min_periods=1).sum()<2).astype(int)
#     hr = (df["rainfall"]<1.0).astype(int)
#     df["dry_days_streak"] = hr.groupby((hr!=hr.shift()).cumsum()).cumcount()

#     roll = df["rainfall"].rolling(30,min_periods=7)
#     df["spi_30"] = (df["rainfall"]-roll.mean())/(roll.std().replace(0,1e-6))

#     df["aqi_category"] = pd.cut(df["aqi"],bins=[0,50,100,200,300,400,500],
#                                  labels=[1,2,3,4,5,6]).astype(float)
#     hot = (df["temp_max"]>=40.0).astype(int)
#     df["consec_hot_days"] = hot.groupby((hot!=hot.shift()).cumsum()).cumcount()+hot

#     df["compound_heat_aqi"]           = ((df["temp_max"]>=38)&(df["aqi"]>=200)).astype(int)
#     df["compound_heat_humidity"]      = ((df["temp_max"]>=38)&(df["humidity"]>=60)).astype(int)
#     # Heat + Air Pollution: broader AQI>=150 (Moderately Polluted) threshold
#     df["compound_heat_air_pollution"] = ((df["temp_max"]>=38)&(df["aqi"]>=150)).astype(int)
#     # Triple: Heat + High AQI + High Humidity (all daily-observable, no drought)
#     df["triple_compound"]             = ((df["temp_max"]>=38)&(df["aqi"]>=150)&
#                                          (df["humidity"]>=60)).astype(int)
#     # Keep drought flag computed but only used as a feature for the model, not shown as compound
#     df["compound_heat_drought"]       = ((df["temp_max"]>=38)&(df["drought_flag"]==1)).astype(int)
#     return df.iloc[-1]


# # ════════════════════════════════════════════════════════════════════════
# # SECTION D: MODEL LOADING & PREDICTION
# # ════════════════════════════════════════════════════════════════════════

# # ── MODEL LOADING ────────────────────────────────────────────────────────
# # No @st.cache_resource here intentionally.
# # If you retrain the model (step4) and click Fetch again, the dashboard
# # picks up the new .pkl immediately without needing a restart.
# def load_model(city):
#     for tag in [city, "all"]:
#         p = os.path.join(MODEL_DIR, f"classifier_{tag}.pkl")
#         if os.path.exists(p):
#             with open(p,"rb") as f: bundle = pickle.load(f)
#             rp = os.path.join(MODEL_DIR, f"regressor_{tag}.pkl")
#             reg = pickle.load(open(rp,"rb")) if os.path.exists(rp) else None
#             return bundle["model"], reg
#     return None, None

# # ── HISTORY LOADING ───────────────────────────────────────────────────────
# # Reads from per-city labelled CSVs (updated by step7b + step7 daily).
# # NOT from labelled_all.csv which is only regenerated by step1/step2.
# # No @st.cache_data so every Fetch button click gets fresh data.
# def load_city_history(city: str) -> pd.DataFrame | None:
#     """
#     Reads the most current labelled/processed CSV for a single city.
#     Priority: labelled_{city}.csv > processed_{city}.csv > labelled_all.csv
#     """
#     for fname in [f"labelled_{city}.csv",
#                   f"processed_{city}.csv"]:
#         p = os.path.join(PROC_DIR, fname)
#         if os.path.exists(p):
#             return pd.read_csv(p, parse_dates=["date"])
#     # Last resort: filter labelled_all.csv
#     p = os.path.join(PROC_DIR, "labelled_all.csv")
#     if os.path.exists(p):
#         df = pd.read_csv(p, parse_dates=["date"])
#         return df[df["city"] == city].copy()
#     return None

# @st.cache_data(ttl=0)   # ttl=0 → never serve stale; always re-read from disk
# def load_history_df():
#     """Used only by historical analysis pages, not by live prediction."""
#     p = os.path.join(PROC_DIR, "labelled_all.csv")
#     if not os.path.exists(p):
#         return None
#     return pd.read_csv(p, parse_dates=["date"])

# def predict_for_city(city, today_obs, pred_date=None):
#     """
#     Loads rolling history BEFORE pred_date (or today), builds features,
#     runs the ML model, and returns a result dict.
#     pred_date: datetime.date or None (defaults to today)
#     """
#     city_df = load_city_history(city)
#     if city_df is None:
#         return None, "No history data found. Run steps 1–2 first."

#     raw_cols  = ["date","temp_max","temp_min","humidity","wind","rainfall","aqi"]
#     available = [c for c in raw_cols if c in city_df.columns]
#     city_df["date"] = pd.to_datetime(city_df["date"], errors="coerce")

#     # ── Use history strictly BEFORE the prediction date ──────────────────
#     if pred_date is None:
#         pred_date = date.today()
#     cutoff = pd.Timestamp(pred_date)
#     history_pool = city_df[city_df["date"] < cutoff]

#     if history_pool.empty:
#         # Fall back to all history if nothing before the date (e.g., very old date)
#         history_pool = city_df

#     history = history_pool[available].dropna().tail(40).copy()

#     # Warn if rolling window is stale relative to chosen date
#     last_hist_date = history["date"].max().date()
#     if last_hist_date < pred_date - timedelta(days=7):
#         st.warning(
#             f"⚠ **{city}**: Rolling history last date is **{last_hist_date}** "
#             f"(more than 7 days before {pred_date}). Rolling features may be approximate.",
#             icon="🕐"
#         )

#     clf, reg = load_model(city)
#     if clf is None:
#         return None, f"Model not found for {city}. Run step4 first."

#     enriched = build_features(history, today_obs)
#     X        = enriched[FEATURE_COLS].fillna(0).values.reshape(1,-1)
#     probas   = clf.predict_proba(X)[0]
#     lvl      = int(np.argmax(probas))
#     score    = float(reg.predict(X)[0]) if reg else None

#     compounds = {
#         "Heat + AQI (Severe)": int(enriched.get("compound_heat_aqi",0)),
#         "Heat + Air Pollution": int(enriched.get("compound_heat_air_pollution",0)),
#         "Heat + Humidity"    : int(enriched.get("compound_heat_humidity",0)),
#         "Triple Compound"    : int(enriched.get("triple_compound",0)),
#     }
#     return {
#         "city"           : city,
#         "date"           : today_obs["date"],
#         "risk_level"     : lvl,
#         "risk_label"     : RISK_LABELS[lvl],
#         "emoji"          : RISK_EMOJIS[lvl],
#         "composite_score": round(score,1) if score else None,
#         "confidence"     : round(float(probas[lvl])*100,1),
#         "probabilities"  : {RISK_LABELS[i]:round(float(p)*100,1) for i,p in enumerate(probas)},
#         "heat_index"     : round(float(enriched["heat_index"]),1),
#         "humidex"        : round(float(enriched["humidex"]),1),
#         "consec_hot_days": int(enriched["consec_hot_days"]),
#         "active_compounds": [k for k,v in compounds.items() if v==1],
#         "advisory"       : ADVISORIES[lvl],
#         "raw_obs"        : today_obs,
#     }, None


# def save_prediction_log(result: dict):
#     p = os.path.join(LOG_DIR, f"pred_{result['city']}_{result['date']}.json")
#     with open(p,"w") as f:
#         json.dump(result, f, indent=2, default=str)


# def append_to_history(city, obs):
#     """Appends today's fetched observation to the labelled CSV."""
#     for fname in [f"labelled_{city}.csv", f"processed_{city}.csv"]:
#         p = os.path.join(PROC_DIR, fname)
#         if os.path.exists(p):
#             df = pd.read_csv(p, parse_dates=["date"])
#             if pd.Timestamp(obs["date"]) not in df["date"].values:
#                 df = pd.concat([df, pd.DataFrame([{**obs,"city":city}])],
#                                ignore_index=True)
#                 df.to_csv(p, index=False)
#             return


# # ════════════════════════════════════════════════════════════════════════
# # SECTION E: DATA LOADING
# # ════════════════════════════════════════════════════════════════════════

# @st.cache_data(ttl=300)   # refresh every 5 minutes
# def load_all_data():
#     p = os.path.join(PROC_DIR, "labelled_all.csv")
#     if not os.path.exists(p): return None
#     return pd.read_csv(p, parse_dates=["date"])


# # ════════════════════════════════════════════════════════════════════════
# # SIDEBAR
# # ════════════════════════════════════════════════════════════════════════

# with st.sidebar:
#     st.markdown("## 🌡 Heatwave Risk System")
#     st.caption("Delhi · Hyderabad · Nagpur")
#     st.markdown("---")

#     page = st.radio("Navigation", [
#         "🔴 Live & 7-Day Forecast",
#         "📆 Monthly(Seasonal) Outlook",
#         "🏠 Historical Overview",
#         "📈 Trends & Analysis",
#         "🔍 Manual Prediction",
#         "📊 Cross-City Analytics",
#         "📋 Prediction Log",
#     ])

#     st.markdown("---")
#     st.markdown("**API Keys** *(optional — Open-Meteo works without)*")
#     owm_key    = st.text_input("OWM Key", type="password",
#                                placeholder="paste OpenWeatherMap key")
#     aqicn_tok  = st.text_input("AQICN Token", type="password",
#                                placeholder="paste AQICN token")
#     st.caption("New OWM keys take ~2h to activate.")

#     st.markdown("---")
#     st.markdown("**Filters** *(historical pages)*")
#     sel_city   = st.selectbox("City", ["All"]+CITIES)
#     try:
#         date_range = st.date_input("Date Range",
#             value=(datetime(2020,1,1), datetime(2025,12,31)))
#     except Exception:
#         date_range = (datetime(2020,1,1), datetime(2025,12,31))


# df_all = load_all_data()

# # Apply sidebar filters for historical pages
# if df_all is not None:
#     df_view = df_all[df_all["city"]==sel_city].copy() if sel_city!="All" else df_all.copy()
#     if isinstance(date_range, (list,tuple)) and len(date_range)==2:
#         df_view = df_view[
#             (df_view["date"]>=pd.Timestamp(date_range[0])) &
#             (df_view["date"]<=pd.Timestamp(date_range[1]))
#         ]
# else:
#     df_view = None


# # ════════════════════════════════════════════════════════════════════════
# # PAGE 1: LIVE PREDICTION
# # ════════════════════════════════════════════════════════════════════════
# if "Live" in page:
#     st.title("🔴 Live & 7-Day Forecas")
#     pred_mode = st.radio("Select Prediction Mode", ["Live Prediction (Today/Tomorrow)", "7-Day Forecast"], horizontal=True)
#     st.markdown("---")
#     if "Live Prediction" in pred_mode:
#         st.markdown("## 🔴 Live Prediction <span class='live-badge'>LIVE</span>",
#                     unsafe_allow_html=True)
#         st.caption("Fetches real weather & AQ data, runs ML model using your historical data as context.")

#         # Controls
#         c1, c2, c3 = st.columns([2,2,3])
#         with c1:
#             live_cities = st.multiselect("Cities to predict",
#                                          CITIES, default=CITIES)
#         with c2:
#             predict_day = st.radio("Predict for", ["Today","Tomorrow"],
#                                    horizontal=True)
#         with c3:
#             st.markdown("<br>",unsafe_allow_html=True)
#             fetch_btn = st.button("🌐 Fetch Live Data & Predict",
#                                   type="primary", use_container_width=True)

#         st.markdown("---")



#         # ── Run on button click ────────────────────────────────────────────
#         if fetch_btn:
#             target = (date.today()+timedelta(days=1)
#                       if predict_day=="Tomorrow" else date.today())

#             st.session_state["live_results"]   = {}
#             st.session_state["live_target"]    = str(target)
#             st.session_state["live_timestamp"] = datetime.now().strftime("%d %b %Y %H:%M")

#             prog = st.progress(0, text="Starting fetch...")
#             for idx, city in enumerate(live_cities):
#                 prog.progress((idx+0.1)/len(live_cities),
#                               text=f"Fetching weather for {city}...")
#                 cfg = CITY_CONFIG[city]

#                 # 1. Weather
#                 weather, werr = fetch_weather(cfg["lat"], cfg["lon"], target)
#                 if weather is None:
#                     st.session_state["live_results"][city] = {
#                         "error": f"Weather fetch failed: {werr}"}
#                     continue

#                 # 2. AQI
#                 prog.progress((idx+0.5)/len(live_cities),
#                               text=f"Fetching AQI for {city}...")
#                 aqi_val, aqi_src = get_aqi(cfg["lat"], cfg["lon"],
#                                             cfg["aqicn"], target,
#                                             owm_key, aqicn_tok)
#                 if aqi_val is None:
#                     # fallback: 7-day avg from history
#                     if df_all is not None:
#                         hist_c = df_all[df_all["city"]==city]
#                         aqi_val = float(hist_c["aqi"].tail(7).mean())
#                         aqi_src = "7-day historical avg"
#                     else:
#                         aqi_val = 120
#                         aqi_src = "default fallback"

#                 today_obs = {**weather, "aqi": round(float(aqi_val), 0)}

#                 # 3. Predict
#                 prog.progress((idx+0.8)/len(live_cities),
#                               text=f"Running prediction for {city}...")
#                 result, err = predict_for_city(city, today_obs)
#                 if err:
#                     st.session_state["live_results"][city] = {"error": err}
#                     continue

#                 result["aqi_source"] = aqi_src
#                 st.session_state["live_results"][city] = result
#                 save_prediction_log(result)
#                 if predict_day == "Today":
#                     append_to_history(city, today_obs)

#             prog.progress(1.0, text="Done!")
#             load_all_data.clear()   # refresh historical cache with new data

#         # ── Display results ────────────────────────────────────────────────
#         if "live_results" in st.session_state and st.session_state["live_results"]:
#             results = st.session_state["live_results"]
#             ts      = st.session_state.get("live_timestamp","")
#             target_str = st.session_state.get("live_target","")

#             st.caption(f"Last fetched: **{ts}**  |  Prediction for: **{target_str}**")

#             # ── Risk cards ───────────────────────────────────────────────
#             cols = st.columns(len(results))
#             for col, (city, res) in zip(cols, results.items()):
#                 if "error" in res:
#                     col.error(f"**{city}**: {res['error']}")
#                     continue
#                 risk  = res["risk_label"]
#                 emoji = res["emoji"]
#                 col.markdown(f"""
#                 <div class="risk-card risk-{risk.upper()}">
#                   <div style="font-size:1.3rem;font-weight:700">{city}</div>
#                   <div style="font-size:3rem">{emoji}</div>
#                   <div class="metric-big">{risk}</div>
#                   <div class="metric-sub">Score: {res['composite_score']} / 100</div>
#                   <div class="metric-sub">Confidence: {res['confidence']}%</div>
#                   <hr style="margin:8px 0">
#                   <div>🌡 Tmax: <b>{res['raw_obs']['temp_max']}°C</b></div>
#                   <div>🥵 Heat Index: <b>{res['heat_index']}°C</b></div>
#                   <div>💧 Humidity: <b>{res['raw_obs']['humidity']}%</b></div>
#                   <div>💨 AQI: <b>{res['raw_obs']['aqi']:.0f}</b></div>
#                   <div>🌧 Rain: <b>{res['raw_obs']['rainfall']} mm</b></div>
#                   <div>🔥 Consec Hot Days: <b>{res['consec_hot_days']}</b></div>
#                   <div style="margin-top:6px">
#                     <span class="source-tag">AQI: {res.get('aqi_source','API')}</span>
#                   </div>
#                 </div>""", unsafe_allow_html=True)

#             st.markdown("---")

#             # ── Detailed panels per city ─────────────────────────────────
#             valid = {c:r for c,r in results.items() if "error" not in r}
#             if not valid: st.stop()

#             tabs = st.tabs([f"{r['emoji']} {c}" for c,r in valid.items()])
#             for tab, (city, res) in zip(tabs, valid.items()):
#                 with tab:
#                     a1, a2 = st.columns([1,1])

#                     with a1:
#                         # Probability bar chart
#                         probs = res["probabilities"]
#                         fig = go.Figure(go.Bar(
#                             x=list(probs.values()),
#                             y=list(probs.keys()),
#                             orientation="h",
#                             marker_color=[RISK_COLORS[i] for i in range(4)],
#                             text=[f"{v:.1f}%" for v in probs.values()],
#                             textposition="outside",
#                         ))
#                         fig.update_layout(
#                             title="Class Probabilities",
#                             xaxis=dict(range=[0,105], title="%"),
#                             height=220, margin=dict(l=10,r=10,t=40,b=10),
#                             showlegend=False,
#                         )
#                         st.plotly_chart(fig, use_container_width=True)

#                     with a2:
#                         # Pillar radar
#                         obs = res["raw_obs"]
#                         def _s(x,bps):
#                             for lo,hi,ilo,ihi in bps:
#                                 if lo<=x<=hi: return round(((ihi-ilo)/(hi-lo))*(x-lo)+ilo,1)
#                             return 100.0
#                         hi_val = _heat_index(obs["temp_max"], obs["humidity"])
#                         st_t = 0 if obs["temp_max"]<35 else 25 if obs["temp_max"]<38 else 50 if obs["temp_max"]<40 else 75 if obs["temp_max"]<45 else 100
#                         st_h = _s(hi_val, [(0,27,0,0),(27,32,0,25),(32,41,25,50),(41,54,50,75),(54,100,75,100)])
#                         st_a = _s(obs["aqi"], [(0,50,0,0),(50,100,0,15),(100,200,15,40),(200,300,40,60),(300,400,60,80),(400,500,80,100)])
#                         # Heat+Air Pollution compound score (replaces drought)
#                         st_ap = min(100, int(obs["temp_max"]>=38 and obs["aqi"]>=150)*50 +
#                                          int(obs["temp_max"]>=38 and obs["aqi"]>=200)*50)
#                         st_c = min(100, (int(obs["temp_max"]>=38 and obs["aqi"]>=200)*30 +
#                                          int(obs["temp_max"]>=38 and obs["aqi"]>=150)*25 +
#                                          int(obs["temp_max"]>=38 and obs["humidity"]>=60)*20))
#                         rc   = RISK_COLORS[res["risk_level"]]
#                         fig  = go.Figure(go.Scatterpolar(
#                             r=[st_t,st_h,st_a,st_ap,st_c],
#                             theta=["Temperature","Heat Index","AQI","Air Pollution","Compound"],
#                             fill="toself",
#                             fillcolor='rgba(241, 196, 15, 0.27)',
#                             line_color=rc,
#                         ))
#                         fig.update_layout(
#                             title="Pillar Breakdown",
#                             polar=dict(radialaxis=dict(range=[0,100],tickfont_size=9)),
#                             height=220, margin=dict(l=10,r=10,t=40,b=10),
#                             showlegend=False,
#                         )
#                         st.plotly_chart(fig, use_container_width=True)

#                     # Advisory
#                     st.markdown(f"""<div class="advisory-box">
#                       <b>Advisory:</b> {res['advisory']}
#                     </div>""", unsafe_allow_html=True)

#                     # Compound flags
#                     if res["active_compounds"]:
#                         st.warning("⚠ Active Compound Risks: " +
#                                    " | ".join(res["active_compounds"]))

#                     # Context from history
#                     if df_all is not None:
#                         st.markdown("**Historical context — last 14 days:**")
#                         hist14 = (df_all[df_all["city"]==city]
#                                   .tail(14)[["date","temp_max","aqi","risk_label","composite_score"]]
#                                   .copy())
#                         hist14["date"] = hist14["date"].dt.strftime("%d %b")
#                         hist14.columns = ["Date","Tmax°C","AQI","Risk","Score"]
#                         st.dataframe(hist14.set_index("Date"), use_container_width=True)

#         else:
#             st.info("👆 Click **Fetch Live Data & Predict** to run today's or tomorrow's prediction.")



#     # ════════════════════════════════════════════════════════════════════════
#     # PAGE 2: HISTORICAL OVERVIEW
#     # ════════════════════════════════════════════════════════════════════════
#     else:
#         st.title("📅 7-Day Heatwave Forecast")
#         st.caption(
#             "Chained ML forecast using Open-Meteo 16-day weather + 5-day AQ. "
#             "Confidence decays each day forward — see reliability label.")

#         # ── Controls ──────────────────────────────────────────────────────
#         fc1, fc2, fc3 = st.columns([2, 2, 3])
#         with fc1:
#             fc_cities = st.multiselect("Cities", CITIES, default=CITIES, key="fc_cities")
#         with fc2:
#             n_days = st.slider("Days ahead", 1, 16, 7, key="fc_days")
#         with fc3:
#             st.markdown("<br>", unsafe_allow_html=True)
#             run_fc = st.button("🔮 Run Forecast", type="primary",
#                                use_container_width=True, key="fc_btn")

#         # Reliability legend
#         st.markdown(
#             "> **Reliability:** "
#             "🟦 **High** (days 1–2) &nbsp;|&nbsp; "
#             "🟨 **Medium** (days 3–4) &nbsp;|&nbsp; "
#             "🟧 **Low** (days 5–7) &nbsp;|&nbsp; "
#             "⬜ **Indicative** (days 8+)"
#         )
#         st.markdown("---")

#         if run_fc:
#             # ── import the forecast engine ─────────────────────────────
#             import importlib.util, sys
#             spec = importlib.util.spec_from_file_location(
#                 "step7c", os.path.join(os.path.dirname(__file__),
#                                         "step7c_multiday_forecast.py"))
#             if spec is None:
#                 st.error("step7c_multiday_forecast.py not found in the same folder.")
#                 st.stop()
#             mod = importlib.util.module_from_spec(spec)
#             sys.modules["step7c"] = mod
#             spec.loader.exec_module(mod)

#             all_fc = {}
#             prog   = st.progress(0, text="Starting forecast...")
#             for idx, city in enumerate(fc_cities):
#                 prog.progress((idx + 0.3) / len(fc_cities),
#                               text=f"Forecasting {city}...")
#                 fc = mod.forecast_city(city, n_days=n_days)
#                 all_fc[city] = fc
#                 prog.progress((idx + 1) / len(fc_cities),
#                               text=f"{city} done")

#             prog.progress(1.0, text="Complete!")
#             st.session_state["fc_results"] = all_fc
#             st.session_state["fc_n_days"]  = n_days

#         # ── Display results ─────────────────────────────────────────────
#         if "fc_results" in st.session_state and st.session_state["fc_results"]:
#             all_fc = st.session_state["fc_results"]
#             n      = st.session_state.get("fc_n_days", 7)

#             RELIABILITY_COLORS = {
#                 "High"       : "#2ECC71",
#                 "Medium"     : "#F1C40F",
#                 "Low"        : "#E67E22",
#                 "Indicative" : "#BDC3C7",
#             }

#             for city, fc_list in all_fc.items():
#                 if not fc_list:
#                     st.error(f"{city}: forecast failed.")
#                     continue

#                 st.subheader(f"📍 {city}")

#                 # ── Risk card strip ──────────────────────────────────
#                 card_cols = st.columns(min(len(fc_list), 8))
#                 for col, day in zip(card_cols, fc_list[:8]):
#                     risk  = day["risk_label"]
#                     bg    = {"Low":"#d5f5e3","Moderate":"#fef9e7",
#                               "High":"#fdebd0","Severe":"#fadbd8"}.get(risk,"#f0f0f0")
#                     rel_c = RELIABILITY_COLORS.get(day["reliability"], "#ccc")
#                     col.markdown(f"""
#                     <div style="background:{bg};border:2px solid {rel_c};
#                       border-radius:10px;padding:8px;text-align:center;font-size:0.8rem">
#                       <div style="font-size:0.7rem;color:#888">{day["date"][5:]}</div>
#                       <div style="font-size:1.6rem">{day["emoji"]}</div>
#                       <div style="font-weight:700;font-size:0.8rem">{risk}</div>
#                       <div style="font-size:0.7rem">{day["adj_confidence"]:.0f}%</div>
#                       <div style="font-size:0.7rem">🌡{day["temp_max"]}°C</div>
#                     </div>""", unsafe_allow_html=True)

#                 # ── Charts ───────────────────────────────────────────
#                 fc_df = pd.DataFrame(fc_list)
#                 fc_df["date"] = pd.to_datetime(fc_df["date"])

#                 ch1, ch2 = st.columns(2)

#                 with ch1:
#                     # Composite score + confidence band
#                     fig = go.Figure()
#                     fig.add_trace(go.Scatter(
#                         x=fc_df["date"], y=fc_df["composite_score"],
#                         name="Risk Score",
#                         line=dict(color=CITY_COLORS.get(city,"#3498DB"), width=2),
#                         mode="lines+markers",
#                     ))
#                     # Confidence band: score ± (1-decay)*score
#                     fc_df["upper"] = fc_df["composite_score"] * (
#                         1 + (1 - fc_df["decay_factor"]) * 0.5)
#                     fc_df["lower"] = fc_df["composite_score"] * (
#                         1 - (1 - fc_df["decay_factor"]) * 0.5)
#                     # fig.add_trace(go.Scatter(
#                     #     x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
#                     #     y=pd.concat([fc_df["upper"], fc_df["lower"][::-1]]),
#                     #     fill="toself",
#                     #     fillcolor=CITY_COLORS.get(city,"#3498DB") + "22",
#                     #     line=dict(color="rgba(0,0,0,0)"),
#                     #     name="Uncertainty band",
#                     # ))
#                     hex_color = CITY_COLORS.get(city, "#3498DB").lstrip("#")
#                     r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
#                     fill_col = f"rgba({r}, {g}, {b}, 0.13)"

#                     fig.add_trace(go.Scatter(
#                         x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
#                         y=pd.concat([fc_df["upper"], fc_df["lower"][::-1]]),
#                         fill="toself",
#                         fillcolor=fill_col,
#                         line=dict(color="rgba(0,0,0,0)"),
#                         name="Uncertainty band",
#                     ))

#                     fig.add_hline(y=50, line_dash="dot",
#                                   line_color="orange", annotation_text="High (50)")
#                     fig.add_hline(y=75, line_dash="dot",
#                                   line_color="red", annotation_text="Severe (75)")
#                     fig.update_layout(
#                         title="Composite Risk Score + Uncertainty",
#                         height=300, margin=dict(l=0,r=0,t=40,b=0),
#                         legend=dict(orientation="h", y=-0.3),
#                     )
#                     st.plotly_chart(fig, use_container_width=True)

#                 with ch2:
#                     # Temperature + AQI dual axis
#                     fig2 = make_subplots(specs=[[{"secondary_y": True}]])
#                     fig2.add_trace(go.Bar(
#                         x=fc_df["date"], y=fc_df["temp_max"],
#                         name="Tmax (°C)",
#                         marker_color=[RISK_COLORS[r] for r in fc_df["risk_level"]],
#                         opacity=0.8,
#                     ), secondary_y=False)
#                     fig2.add_trace(go.Scatter(
#                         x=fc_df["date"], y=fc_df["aqi"],
#                         name="AQI",
#                         line=dict(color="#8E44AD", width=2, dash="dot"),
#                         mode="lines+markers",
#                     ), secondary_y=True)
#                     fig2.update_yaxes(title_text="Tmax (°C)", secondary_y=False)
#                     fig2.update_yaxes(title_text="AQI",       secondary_y=True)
#                     fig2.update_layout(
#                         title="Temperature & AQI forecast",
#                         height=300, margin=dict(l=0,r=0,t=40,b=0),
#                         legend=dict(orientation="h", y=-0.3),
#                     )
#                     st.plotly_chart(fig2, use_container_width=True)

#                 # ── Detailed table ────────────────────────────────────
#                 with st.expander(f"📋 {city} — Full forecast table"):
#                     tbl = fc_df[[
#                         "date","risk_label","composite_score",
#                         "adj_confidence","reliability",
#                         "temp_max","heat_index","humidity","aqi","rainfall",
#                         "consec_hot_days",
#                     ]].copy()
#                     tbl.columns = [
#                         "Date","Risk","Score","Confidence%","Reliability",
#                         "Tmax°C","HeatIdx°C","Humidity%","AQI","Rain(mm)",
#                         "HotDayStreak",
#                     ]
#                     tbl["Date"] = tbl["Date"].dt.strftime("%d %b %Y (%a)")

#                     def colour_risk(val):
#                         c = {"Low":"#d5f5e3","Moderate":"#fef9e7",
#                              "High":"#fdebd0","Severe":"#fadbd8"}
#                         return f"background-color:{c.get(val,'')}"

#                     def colour_rel(val):
#                         c = {"High":"#d5f5e3","Medium":"#fef9e7",
#                              "Low":"#fdebd0","Indicative":"#f0f0f0"}
#                         return f"background-color:{c.get(val,'')}"

#                     styled = (tbl.style
#                               .map(colour_risk, subset=["Risk"])
#                               .map(colour_rel,  subset=["Reliability"]))
#                     st.dataframe(styled, use_container_width=True)

#                 # Compound event warning
#                 compound_days = [d for d in fc_list if d["active_compounds"]]
#                 if compound_days:
#                     st.warning(
#                         f"⚠ **{city}** — compound risk events in forecast: " +
#                         ", ".join([f"{d['date'][5:]} ({', '.join(d['active_compounds'])})"
#                                    for d in compound_days])
#                     )
#                 st.markdown("---")

#             # ── Download all forecasts ────────────────────────────────
#             all_rows = []
#             for city, fc_list in all_fc.items():
#                 for d in fc_list:
#                     all_rows.append({
#                         "City": city, **{k: d[k] for k in
#                             ["date","days_ahead","risk_label","composite_score",
#                              "adj_confidence","reliability","temp_max","heat_index",
#                              "aqi","rainfall","active_compounds"]}
#                     })
#             csv_out = pd.DataFrame(all_rows).to_csv(index=False)
#             st.download_button("⬇ Download full forecast CSV",
#                                csv_out, "forecast.csv", "text/csv")

#         else:
#             st.info("👆 Click **Run Forecast** to generate predictions.")

# elif "Monthly" in page:
#     st.title("📆 Monthly Outlook (Seasonal Prediction)")
#     st.caption(
#         "Seasonal prediction using historical climate normals + warming trend. "
#         "Months 1–6 are reliable; months 7–12 are indicative trend only.")

#     sc1, sc2, sc3 = st.columns([2, 2, 3])
#     with sc1:
#         s_cities  = st.multiselect("Cities", CITIES, default=CITIES, key="sc_cities")
#     with sc2:
#         s_months  = st.slider("Months ahead", 1, 12, 12, key="s_months")
#     with sc3:
#         st.markdown("<br>", unsafe_allow_html=True)
#         s_btn = st.button("📆 Generate Monthly Outlook",
#                           type="primary", use_container_width=True, key="s_btn")

#     st.markdown(
#         "> **Reliability:** "
#         "🟦 **High** (months 1–2) &nbsp;|&nbsp; "
#         "🟨 **Medium** (months 3–6) &nbsp;|&nbsp; "
#         "🟧 **Low** (months 7–9) &nbsp;|&nbsp; "
#         "⬜ **Indicative** (months 10–12)"
#     )
#     st.markdown("---")

#     if s_btn:
#         import importlib.util, sys as _sys
#         # load step10
#         spec10 = importlib.util.spec_from_file_location(
#             "step10", os.path.join(os.path.dirname(__file__),
#                                     "step10_seasonal_forecast.py"))
#         if spec10 is None:
#             st.error("step10_seasonal_forecast.py not found.")
#             st.stop()
#         mod10 = importlib.util.module_from_spec(spec10)
#         _sys.modules["step10"] = mod10
#         spec10.loader.exec_module(mod10)

#         s_results = {}
#         prog = st.progress(0, text="Generating outlook...")
#         for idx, city in enumerate(s_cities):
#             prog.progress((idx+0.3)/len(s_cities),
#                           text=f"Forecasting {city}...")
#             fc = mod10.forecast_city_outlook(city, n_months=s_months)
#             s_results[city] = fc
#             prog.progress((idx+1)/len(s_cities))
#         prog.progress(1.0, text="Done!")
#         st.session_state["s_results"] = s_results

#     if "s_results" in st.session_state and st.session_state["s_results"]:
#         s_all = st.session_state["s_results"]
#         MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
#                        "Jul","Aug","Sep","Oct","Nov","Dec"]
#         RELY_COLORS = {"High":"#2ECC71","Medium":"#F1C40F",
#                        "Low":"#E67E22","Indicative":"#BDC3C7"}

#         for city, fc_list in s_all.items():
#             if not fc_list: continue
#             st.subheader(f"📍 {city}")

#             # ── Month card strip ──────────────────────────────────
#             n_show = min(len(fc_list), 12)
#             card_cols = st.columns(n_show)
#             for col, m in zip(card_cols, fc_list[:n_show]):
#                 risk = m["risk_label"]
#                 bg   = {"Low":"#d5f5e3","Moderate":"#fef9e7",
#                         "High":"#fdebd0","Severe":"#fadbd8"}.get(risk,"#f0f0f0")
#                 rc   = RELY_COLORS.get(m["reliability"],"#ccc")
#                 col.markdown(f"""
#                 <div style="background:{bg};border:2px solid {rc};
#                   border-radius:10px;padding:6px;text-align:center;font-size:0.75rem">
#                   <div style="font-size:0.65rem;color:#888">{m['month_name']} {m['year']}</div>
#                   <div style="font-size:1.5rem">{m['emoji']}</div>
#                   <div style="font-weight:700;font-size:0.7rem">{risk}</div>
#                   <div style="font-size:0.65rem">{m['adj_confidence']:.0f}%</div>
#                   <div style="font-size:0.65rem">~{m['exp_tmax']}°C</div>
#                 </div>""", unsafe_allow_html=True)

#             fc_df = pd.DataFrame(fc_list)
#             fc_df["label"] = fc_df["month_name"] + " " + fc_df["year"].astype(str)

#             st.markdown("<br>", unsafe_allow_html=True)
#             m1, m2, m3, m4 = st.columns(4)
#             avg_tmax = fc_df["exp_tmax"].mean()
#             max_tmax = fc_df["exp_tmax"].max()
#             total_high = fc_df["n_high_risk_days"].sum() if "n_high_risk_days" in fc_df.columns else 0
#             trend_adj_max = fc_df["trend_adj_applied"].max() if "trend_adj_applied" in fc_df.columns else 0
            
#             m1.metric("Average Tmax (Year)", f"{avg_tmax:.1f}°C")
#             m2.metric("Peak Tmax Predicted", f"{max_tmax:.1f}°C")
#             m3.metric("Total High-Risk Days", f"{total_high:.0f}")
#             m4.metric("Climate Warming Adj.", f"+{trend_adj_max:.2f}°C")
#             st.markdown("<br>", unsafe_allow_html=True)

#             ch1, ch2 = st.columns(2)

#             with ch1:
#                 # Risk level timeline bar chart
#                 color_map = {0:"#2ECC71",1:"#F1C40F",2:"#E67E22",3:"#E74C3C"}
#                 fig = go.Figure(go.Bar(
#                     x=fc_df["label"],
#                     y=fc_df["adj_confidence"],
#                     marker_color=[color_map[r] for r in fc_df["risk_level"]],
#                     text=fc_df["risk_label"],
#                     textposition="inside",
#                     textfont=dict(size=9),
#                 ))
#                 fig.update_layout(
#                     title="Monthly Risk Level + Confidence",
#                     yaxis_title="Adjusted Confidence %",
#                     height=300, margin=dict(l=0,r=0,t=40,b=60),
#                     xaxis_tickangle=-45,
#                 )
#                 st.plotly_chart(fig, use_container_width=True)

#             with ch2:
#                 # Expected temperature + high-risk days
#                 fig2 = make_subplots(specs=[[{"secondary_y": True}]])
#                 fig2.add_trace(go.Scatter(
#                     x=fc_df["label"], y=fc_df["exp_tmax"],
#                     name="Expected Tmax", mode="lines+markers",
#                     line=dict(color="#E74C3C", width=2),
#                 ), secondary_y=False)
#                 if "n_high_risk_days" in fc_df.columns:
#                     fig2.add_trace(go.Bar(
#                         x=fc_df["label"],
#                         y=fc_df["n_high_risk_days"].fillna(0),
#                         name="Est. high-risk days",
#                         marker_color="#E67E22", opacity=0.4,
#                     ), secondary_y=True)
#                 if "exp_aqi" in fc_df.columns:
#                     fig2.add_trace(go.Scatter(
#                         x=fc_df["label"], y=fc_df["exp_aqi"],
#                         name="Expected AQI", mode="lines+markers",
#                         line=dict(color="#8E44AD", width=2, dash="dot"),
#                     ), secondary_y=True)
#                 fig2.update_yaxes(title_text="Tmax °C",    secondary_y=False)
#                 fig2.update_yaxes(title_text="Days / AQI", secondary_y=True)
#                 fig2.update_layout(
#                     title="Expected Temp, AQI & High-Risk Days",
#                     height=300, margin=dict(l=0,r=0,t=40,b=60),
#                     xaxis_tickangle=-45,
#                     legend=dict(orientation="h", y=-0.4),
#                 )
#                 st.plotly_chart(fig2, use_container_width=True)

#             # Severe + compound probability chart
#             if "p_severe_day" in fc_df.columns and fc_df["p_severe_day"].notna().any():
#                 fig3 = go.Figure()
#                 fig3.add_trace(go.Scatter(
#                     x=fc_df["label"],
#                     y=fc_df["p_severe_day"].fillna(0),
#                     name="P(Severe day)", fill="tozeroy",
#                     line=dict(color="#E74C3C"),
#                     fillcolor = 'rgba(241, 196, 15, 0.27)',
#                 ))
#                 if "p_compound_event" in fc_df.columns:
#                     fig3.add_trace(go.Scatter(
#                         x=fc_df["label"],
#                         y=fc_df["p_compound_event"].fillna(0),
#                         name="P(Compound event)", fill="tozeroy",
#                         line=dict(color="#8E44AD"),
#                         fillcolor = 'rgba(241, 196, 15, 0.27)',
#                     ))
#                 fig3.update_layout(
#                     title="Probability of Extreme Events Within Month",
#                     yaxis_title="Probability %",
#                     height=260, margin=dict(l=0,r=0,t=40,b=60),
#                     xaxis_tickangle=-45,
#                     legend=dict(orientation="h", y=-0.5),
#                 )
#                 st.plotly_chart(fig3, use_container_width=True)

#             # ── Seasonal Compound Risk Breakdown ──────────────────────────
#             st.markdown("**🔥 Seasonal Compound Risk Analysis**")
#             st.caption(
#                 "Estimated compound risks per month based on climate normals. "
#                 "Heat+Drought is included here — appropriate at the seasonal scale "
#                 "(slow-moving phenomenon). Daily prediction uses Heat+Air Pollution instead.")

#             _hum_normal = {3:45,4:40,5:45,6:65,7:80,8:78,9:72,
#                            10:55,11:50,12:50,1:55,2:50}
#             _dry_months  = {3,4,5,11,12}

#             comp_rows = []
#             for m_s in fc_list:
#                 month_num  = m_s["month"]
#                 month_lbl  = m_s["month_name"] + " " + str(m_s["year"])
#                 exp_t      = m_s["exp_tmax"]
#                 exp_aqi    = m_s["exp_aqi"]
#                 exp_hum    = _hum_normal.get(month_num, 50)

#                 heat_aqi_flag      = int(exp_t >= 38 and exp_aqi >= 200)
#                 heat_ap_flag       = int(exp_t >= 38 and exp_aqi >= 150)
#                 heat_hum_flag      = int(exp_t >= 38 and exp_hum >= 60)
#                 heat_drought_flag  = int(exp_t >= 38 and month_num in _dry_months)
#                 triple_flag        = int(exp_t >= 38 and exp_aqi >= 150 and exp_hum >= 60)
#                 quad_flag          = int(heat_drought_flag and triple_flag)

#                 comp_rows.append({
#                     "Month"           : month_lbl,
#                     "Exp Tmax°C"      : exp_t,
#                     "Exp AQI"         : int(exp_aqi),
#                     "Est Humidity%"   : exp_hum,
#                     "Heat+AQI Severe" : "✅" if heat_aqi_flag     else "—",
#                     "Heat+Air Poll."  : "✅" if heat_ap_flag      else "—",
#                     "Heat+Humidity"   : "✅" if heat_hum_flag     else "—",
#                     "Heat+Drought"    : "✅" if heat_drought_flag else "—",
#                     "Triple"          : "✅" if triple_flag       else "—",
#                     "Quadruple"       : "✅" if quad_flag         else "—",
#                 })

#             comp_df = pd.DataFrame(comp_rows)
#             compound_flag_cols = [
#                 "Heat+AQI Severe","Heat+Air Poll.",
#                 "Heat+Humidity","Heat+Drought","Triple","Quadruple"
#             ]
#             counts_s = comp_df[compound_flag_cols].apply(
#                 lambda col: (col == "✅").astype(int))
#             counts_s["Month"] = comp_df["Month"]
#             melted_s = counts_s.melt(id_vars="Month",
#                                      value_vars=compound_flag_cols,
#                                      var_name="Compound Type",
#                                      value_name="Active")
#             melted_s = melted_s[melted_s["Active"] == 1]

#             if not melted_s.empty:
#                 fig_cmp = px.bar(
#                     melted_s, x="Month", y="Active",
#                     color="Compound Type", barmode="stack",
#                     title=f"{city} — Seasonal Compound Risks per Month",
#                     color_discrete_sequence=[
#                         "#E74C3C","#E67E22","#3498DB",
#                         "#27AE60","#8E44AD","#2C3E50"
#                     ],
#                     labels={"Active": "Active Compound Types"},
#                     height=300,
#                 )
#                 fig_cmp.update_layout(
#                     margin=dict(l=0,r=0,t=40,b=80),
#                     xaxis_tickangle=-45,
#                     legend=dict(orientation="h", y=-0.65),
#                     yaxis=dict(dtick=1, title=""),
#                 )
#                 st.plotly_chart(fig_cmp, use_container_width=True)

#             with st.expander(f"📋 {city} — Seasonal compound details table"):
#                 st.dataframe(comp_df.set_index("Month"), use_container_width=True)

#             # Detailed table
#             with st.expander(f"📋 {city} — Full monthly table"):
#                 tbl = fc_df[[
#                     "label","risk_label","adj_confidence","reliability",
#                     "exp_tmax","exp_aqi","n_high_risk_days",
#                     "p_severe_day","p_compound_event","trend_adj_applied",
#                 ]].copy()
#                 tbl.columns = [
#                     "Month","Risk","Confidence%","Reliability",
#                     "Exp Tmax°C","Exp AQI","Est HighDays",
#                     "P(Severe%)","P(Compound%)","Trend adj°C",
#                 ]
#                 def cr(v):
#                     c = {"Low":"#d5f5e3","Moderate":"#fef9e7",
#                          "High":"#fdebd0","Severe":"#fadbd8"}
#                     return f"background-color:{c.get(v,'')}"
#                 st.dataframe(tbl.style.map(cr, subset=["Risk"]),
#                              use_container_width=True)

#             st.markdown("---")

#         # Download
#         all_rows = []
#         for city, fc_list in s_all.items():
#             for m in fc_list:
#                 all_rows.append({"City":city,**{k:m[k] for k in
#                     ["year","month","month_name","risk_label","adj_confidence",
#                      "reliability","exp_tmax","exp_aqi","n_high_risk_days",
#                      "p_severe_day","p_compound_event"]}})
#         csv_s = pd.DataFrame(all_rows).to_csv(index=False)
#         st.download_button("⬇ Download monthly outlook CSV",
#                            csv_s, "monthly_outlook.csv", "text/csv")

#     else:
#         st.info("👆 Click **Generate Monthly Outlook** to run the seasonal forecast.")
#     #     st.markdown("""
#     #     **How this works:**
#     #     - Historical monthly statistics (2020–2025+) are used to compute
#     #       climate normals for each month.
#     #     - A long-term warming trend (°C/year) is estimated from your data.
#     #     - For each forecast month: baseline = climate normal + trend adjustment.
#     #     - Lag features (last month, last year same month) use real observed values.
#     #     - The trained seasonal ML model converts these into risk level + probabilities.
#     #     """)

# elif "Overview" in page:
#     st.title("🏠 Historical Risk Overview")

#     if df_all is None:
#         st.error("No data found. Run steps 1–4 first.")
#         st.stop()

#     if df_view is None or df_view.empty:
#         st.warning("No data found in the selected date range.")
#         st.stop()

#     st.markdown("""
#     <style>
#     .historical-card {
#         border-radius: 12px;
#         padding: 16px;
#         margin-bottom: 16px;
#         text-align: center;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.05);
#         transition: transform 0.2s;
#     }
#     .historical-card:hover {
#         transform: translateY(-2px);
#     }
#     .month-title {
#         font-size: 1.4rem;
#         font-weight: 700;
#         margin-bottom: 12px;
#         color: #2C3E50;
#         text-transform: uppercase;
#         letter-spacing: 1px;
#     }
#     .main-temp {
#         font-size: 2rem;
#         font-weight: 800;
#         margin-bottom: 12px;
#     }
#     .metric-row {
#         display: flex;
#         justify-content: space-between;
#         font-size: 0.95rem;
#         padding: 4px 8px;
#         background: rgba(255,255,255,0.4);
#         border-radius: 6px;
#         margin-bottom: 4px;
#         color: #34495E;
#     }
#     .metric-val {
#         font-weight: 700;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     overview_mode = st.radio("Select View:", ["🏙️ Places", "📊 Analysis", "⚖️ Comparison"], horizontal=True)
#     st.markdown("---")
    
#     month_names = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June", 
#                    7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"}

#     if overview_mode == "🏙️ Places":
#         cities_to_display = sorted(df_view["city"].unique())

#         if len(cities_to_display) > 1:
#             tabs = st.tabs([f"📍 {c}" for c in cities_to_display])
#         else:
#             tabs = [st.container()]

#         for idx, city in enumerate(cities_to_display):
#             with tabs[idx]:
#                 if len(cities_to_display) == 1:
#                     st.markdown(f"### 📍 {city} — Average Monthly Climate Conditions (2017–2025)")
#                 else:
#                     st.markdown("### Average Monthly Climate Conditions (2017–2025)")
                
#                 st.caption(f"Average historical values for each month across all years in the dataset, specific to **{city}**.")
                
#                 # Filter and compute averages by month for the current city
#                 city_df = df_view[df_view["city"] == city]
#                 monthly_stats = city_df.groupby("month")[["temp_max", "aqi", "rainfall", "humidity"]].mean().reset_index()

#                 st.markdown("<br>", unsafe_allow_html=True)

#                 # Display in a 4x3 grid
#                 for row in range(3):
#                     cols = st.columns(4)
#                     for col_idx in range(4):
#                         month_id = row * 4 + col_idx + 1
#                         col = cols[col_idx]
                        
#                         stat = monthly_stats[monthly_stats["month"] == month_id]
#                         if not stat.empty:
#                             t = stat["temp_max"].values[0]
#                             a = stat["aqi"].values[0]
#                             r = stat["rainfall"].values[0]
#                             h = stat["humidity"].values[0]
                            
#                             # Dynamic styling based on heat
#                             if t >= 38:
#                                 bg = "linear-gradient(135deg, #fadbd8, #f5b7b1)"
#                                 border = "#E74C3C"
#                                 color_temp = "#C0392B"
#                             elif t >= 33:
#                                 bg = "linear-gradient(135deg, #fdebd0, #fad7a1)"
#                                 border = "#E67E22"
#                                 color_temp = "#D35400"
#                             elif t >= 25:
#                                 bg = "linear-gradient(135deg, #fef9e7, #fcf3cf)"
#                                 border = "#F1C40F"
#                                 color_temp = "#B7950B"
#                             else:
#                                 bg = "linear-gradient(135deg, #d5f5e3, #abebc6)"
#                                 border = "#2ECC71"
#                                 color_temp = "#1E8449"
                            
#                             col.markdown(f"""
#                             <div class="historical-card" style="background: {bg}; border-top: 4px solid {border};">
#                                 <div class="month-title">{month_names[month_id]}</div>
#                                 <div class="main-temp" style="color: {color_temp};">🌡️ {t:.1f}°C</div>
#                                 <div class="metric-row">
#                                     <span>💨 AQI</span>
#                                     <span class="metric-val">{a:.0f}</span>
#                                 </div>
#                                 <div class="metric-row">
#                                     <span>💧 Humidity</span>
#                                     <span class="metric-val">{h:.1f}%</span>
#                                 </div>
#                                 <div class="metric-row">
#                                     <span>🌧️ Rain</span>
#                                     <span class="metric-val">{r:.1f} mm</span>
#                                 </div>
#                             </div>
#                             """, unsafe_allow_html=True)
#                         else:
#                             col.markdown(f"""
#                             <div class="historical-card" style="background: #f8f9fa; border-top: 4px solid #ced4da;">
#                                 <div class="month-title" style="color: #6c757d;">{month_names[month_id]}</div>
#                                 <div style="font-size:1.2rem; color:#adb5bd; margin: 30px 0;">No Data</div>
#                             </div>
#                             """, unsafe_allow_html=True)
#                 st.markdown("---")

#     elif overview_mode == "📊 Analysis":
#         st.markdown("### Monthly Climate Trends (City-wise Analysis)")
#         st.caption("Trends of average Temp, AQI, Humidity, and Rainfall over the year for each city.")
        
#         cities_to_display = sorted(df_view["city"].unique())
        
#         if len(cities_to_display) > 1:
#             city_tabs = st.tabs([f"📍 {c}" for c in cities_to_display])
#         else:
#             city_tabs = [st.container()]
            
#         for idx, city in enumerate(cities_to_display):
#             with city_tabs[idx]:
#                 city_df = df_view[df_view["city"] == city]
#                 m_stats = city_df.groupby("month")[["temp_max", "aqi", "rainfall", "humidity"]].mean().reset_index()
#                 m_stats["Month"] = m_stats["month"].map(month_names)
                
#                 c1, c2 = st.columns(2)
#                 with c1:
#                     fig_t = px.line(m_stats, x="Month", y="temp_max", markers=True, 
#                                     title=f"Avg Max Temperature - {city}", 
#                                     labels={"temp_max": "Temp (°C)"}, color_discrete_sequence=["#E74C3C"])
#                     st.plotly_chart(fig_t, use_container_width=True)
                    
#                     fig_h = px.bar(m_stats, x="Month", y="humidity", 
#                                    title=f"Avg Humidity - {city}", 
#                                    labels={"humidity": "Humidity (%)"}, color_discrete_sequence=["#3498DB"])
#                     st.plotly_chart(fig_h, use_container_width=True)
#                 with c2:
#                     fig_a = px.line(m_stats, x="Month", y="aqi", markers=True, 
#                                     title=f"Avg AQI - {city}", 
#                                     labels={"aqi": "AQI"}, color_discrete_sequence=["#8E44AD"])
#                     st.plotly_chart(fig_a, use_container_width=True)
                    
#                     fig_r = px.bar(m_stats, x="Month", y="rainfall", 
#                                    title=f"Avg Rainfall - {city}", 
#                                    labels={"rainfall": "Rainfall (mm)"}, color_discrete_sequence=["#2ECC71"])
#                     st.plotly_chart(fig_r, use_container_width=True)

#     elif overview_mode == "⚖️ Comparison":
#         st.markdown("### City Comparison")
#         st.caption("Compare average climatic conditions across different cities.")
        
#         # Use df_all to guarantee we compare all cities regardless of the single city filter
#         m_stats_all = df_all.groupby(["city", "month"])[["temp_max", "aqi", "rainfall", "humidity"]].mean().reset_index()
#         m_stats_all["Month"] = m_stats_all["month"].map(month_names)
        
#         c1, c2 = st.columns(2)
#         with c1:
#             fig_tc = px.line(m_stats_all, x="Month", y="temp_max", color="city", markers=True,
#                              title="Temperature Comparison", labels={"temp_max": "Temp (°C)"},
#                              color_discrete_map=CITY_COLORS)
#             st.plotly_chart(fig_tc, use_container_width=True)
            
#             fig_hc = px.line(m_stats_all, x="Month", y="humidity", color="city", markers=True,
#                              title="Humidity Comparison", labels={"humidity": "Humidity (%)"},
#                              color_discrete_map=CITY_COLORS)
#             st.plotly_chart(fig_hc, use_container_width=True)
            
#         with c2:
#             fig_ac = px.line(m_stats_all, x="Month", y="aqi", color="city", markers=True,
#                              title="AQI Comparison", labels={"aqi": "AQI"},
#                              color_discrete_map=CITY_COLORS)
#             st.plotly_chart(fig_ac, use_container_width=True)
            
#             fig_rc = px.bar(m_stats_all, x="Month", y="rainfall", color="city", barmode="group",
#                             title="Rainfall Comparison", labels={"rainfall": "Rainfall (mm)"},
#                             color_discrete_map=CITY_COLORS)
#             st.plotly_chart(fig_rc, use_container_width=True)

#         st.markdown("---")
#         st.markdown("### Seasonal Patterns")
        
#         if "season" in df_all.columns:
#             seasonal = df_all.groupby(["city","season"])["composite_score"].mean().reset_index()
#             fig_season = px.bar(seasonal,x="season",y="composite_score",color="city",
#                          barmode="group",color_discrete_map=CITY_COLORS,
#                          title="Average Risk Score by Season",
#                          category_orders={"season":["Winter","Pre-Monsoon","Monsoon","Post-Monsoon"]})
#             st.plotly_chart(fig_season,use_container_width=True)

#         # Monthly trend lines for composite score
#         monthly_comp = df_all.groupby(["city","month"])["composite_score"].mean().reset_index()
#         fig_month = px.line(monthly_comp,x="month",y="composite_score",color="city",
#                        color_discrete_map=CITY_COLORS,markers=True,
#                        title="Average Risk Score by Month",
#                        labels={"month":"Month","composite_score":"Avg Score"})
#         st.plotly_chart(fig_month,use_container_width=True)


# # ════════════════════════════════════════════════════════════════════════
# # PAGE 3: TRENDS & ANALYSIS
# # ════════════════════════════════════════════════════════════════════════
# elif "Trends" in page:
#     st.title("📈 Trends & Analysis")

#     if df_view is None or len(df_view)==0:
#         st.warning("No data for selected filters.")
#         st.stop()

#     tab1,tab2,tab3,tab4 = st.tabs([
#         "Risk Timeline","Temperature & Heat Index",
#         "Compound Events","Monthly Heatmap",
#     ])

#     with tab1:
#         city_t = st.selectbox("City",CITIES,key="tt1")
#         sub    = df_all[df_all["city"]==city_t] if df_all is not None else df_view

#         fig = make_subplots(rows=3,cols=1,shared_xaxes=True,
#                             subplot_titles=("Risk Score (coloured by level)",
#                                             "Temp Max (°C)","AQI"),
#                             vertical_spacing=0.07)
#         for lvl,lbl in RISK_LABELS.items():
#             mask = sub["risk_level"]==lvl
#             fig.add_trace(go.Scatter(x=sub[mask]["date"],y=sub[mask]["composite_score"],
#                 mode="markers",marker=dict(color=RISK_COLORS[lvl],size=2.5),
#                 name=lbl),row=1,col=1)
#         fig.add_trace(go.Scatter(x=sub["date"],y=sub["temp_max"],
#             line=dict(color="#E74C3C",width=1),name="Tmax"),row=2,col=1)
#         fig.add_hline(y=40,line_dash="dot",line_color="darkred",row=2,col=1)
#         fig.add_trace(go.Scatter(x=sub["date"],y=sub["aqi"],
#             line=dict(color="#8E44AD",width=1),name="AQI"),row=3,col=1)
#         fig.add_hline(y=200,line_dash="dot",line_color="purple",row=3,col=1)
#         fig.update_layout(height=580,title=f"{city_t} — Full Timeline")
#         st.plotly_chart(fig,use_container_width=True)

#     with tab2:
#         fig = px.box(df_view,x="month",y="heat_index",color="city",
#                      color_discrete_map=CITY_COLORS,
#                      title="Monthly Heat Index Distribution")
#         st.plotly_chart(fig,use_container_width=True)

#         sample = df_view.sample(min(3000,len(df_view)))
#         fig2   = px.scatter(sample,x="temp_max",y="heat_index",
#                             color="risk_label",size="aqi",size_max=10,
#                             color_discrete_map={"Low":"#2ECC71","Moderate":"#F1C40F",
#                                                 "High":"#E67E22","Severe":"#E74C3C"},
#                             title="Heat Index vs Temp (bubble size = AQI)")
#         st.plotly_chart(fig2,use_container_width=True)

#     with tab3:
#         # Use available compound columns (daily-appropriate only)
#         avail_cols = [c for c in ["compound_heat_aqi","compound_heat_air_pollution",
#                                    "compound_heat_humidity","triple_compound"]
#                       if c in df_view.columns]
#         nice_map = {
#             "compound_heat_aqi"          : "Heat+AQI (Severe ≥200)",
#             "compound_heat_air_pollution": "Heat+Air Pollution (≥150)",
#             "compound_heat_humidity"     : "Heat+Humidity",
#             "triple_compound"            : "Triple (Heat+AQI+Humidity)",
#         }
#         nice = [nice_map[c] for c in avail_cols]
#         grp   = df_view.groupby(["city","month"])[avail_cols].sum().reset_index()
#         grp.columns = ["city","month"]+nice
#         melted = grp.melt(id_vars=["city","month"],value_vars=nice,
#                           var_name="Type",value_name="Days")
#         fig = px.bar(melted,x="month",y="Days",color="Type",facet_col="city",
#                      barmode="group",title="Monthly Compound Event Frequency (Daily Risks)",
#                      color_discrete_sequence=["#E74C3C","#E67E22","#3498DB","#8E44AD"])
#         st.plotly_chart(fig,use_container_width=True)

#         # Compound event timeline
#         tc_col = "triple_compound" if "triple_compound" in df_view.columns else avail_cols[-1]
#         fig2 = px.area(df_view,x="date",y=tc_col,color="city",
#                        color_discrete_map=CITY_COLORS,
#                        title="Triple Compound Events Over Time (Heat + AQI ≥150 + Humidity ≥60%)")
#         st.plotly_chart(fig2,use_container_width=True)

#     with tab4:
#         city_h = st.selectbox("City",CITIES,key="hmap")
#         sub_h  = (df_all[df_all["city"]==city_h].copy()
#                   if df_all is not None else df_view)
#         sub_h["week"] = sub_h["date"].dt.isocalendar().week.astype(int)
#         pivot = sub_h.pivot_table(index="month",columns="week",
#                                    values="composite_score",aggfunc="mean")
#         fig = px.imshow(pivot,color_continuous_scale="RdYlGn_r",
#                         title=f"{city_h} — Avg Risk Score (Month × Week of Year)",
#                         labels={"x":"Week","y":"Month","color":"Score"})
#         st.plotly_chart(fig,use_container_width=True)


# # ════════════════════════════════════════════════════════════════════════
# # PAGE 4: MANUAL PREDICTION
# # ════════════════════════════════════════════════════════════════════════
# elif "Manual" in page:
#     st.title("🔍 Manual Prediction")
#     st.caption("Choose how you want to supply the weather inputs — then click Predict.")

#     # ── City picker ───────────────────────────────────────────────────────
#     city_m = st.selectbox("City", CITIES, key="man_city")

#     st.markdown("---")

#     # ── Mode toggle ───────────────────────────────────────────────────────
#     mode = st.radio(
#         "**Input Mode**",
#         options=["📅 Pick a Date from History", "🎛️ Set Values Manually"],
#         horizontal=True,
#         key="man_mode",
#     )
#     st.markdown("")

#     # ─────────────────────────────────────────────────────────────────────
#     # MODE A: DATE LOOKUP  — loads values from processed CSV for that date
#     # ─────────────────────────────────────────────────────────────────────
#     if mode == "📅 Pick a Date from History":
#         st.markdown("##### Select a date — observed values are loaded automatically from the processed CSV.")
#         pred_date = st.date_input("Date", value=date.today(), key="man_date_a")

#         _city_hist = load_city_history(city_m)
#         _obs_from_csv = None
#         if _city_hist is not None:
#             _city_hist["date"] = pd.to_datetime(_city_hist["date"], errors="coerce")
#             _match = _city_hist[_city_hist["date"].dt.date == pred_date]
#             if not _match.empty:
#                 _row = _match.iloc[-1]
#                 _obs_from_csv = {
#                     "date"    : str(pred_date),
#                     "temp_max": float(_row.get("temp_max", 38)),
#                     "temp_min": float(_row.get("temp_min", 24)),
#                     "humidity": float(_row.get("humidity", 45)),
#                     "wind"    : float(_row.get("wind",    10)),
#                     "rainfall": float(_row.get("rainfall",  0)),
#                     "aqi"     : float(_row.get("aqi",     120)),
#                 }

#         if _obs_from_csv:
#             st.success(f"✅ **{pred_date}** found in processed data for **{city_m}**", icon="📂")
#             o = _obs_from_csv
#             # Display as clean read-only metrics — no sliders needed
#             mc = st.columns(6)
#             for col_ui, lbl, v in zip(mc,
#                 ["Tmax °C","Tmin °C","Humidity %","Wind km/h","Rainfall mm","AQI"],
#                 [o["temp_max"],o["temp_min"],o["humidity"],o["wind"],o["rainfall"],o["aqi"]]):
#                 col_ui.metric(lbl, f"{v:.1f}")
#             t_max    = o["temp_max"]
#             t_min    = o["temp_min"]
#             humidity = int(o["humidity"])
#             wind     = o["wind"]
#             rainfall = o["rainfall"]
#             aqi_m    = int(o["aqi"])
#             obs_ready = True
#         else:
#             if pred_date < date.today():
#                 st.warning(
#                     f"⚠️ **{pred_date}** not found in the processed CSV for **{city_m}**. "
#                     "Try another date, or switch to **Set Values Manually**.", icon="🗓️")
#             else:
#                 st.info("Today's / future dates are not in historical CSV. "
#                         "Switch to **Set Values Manually** mode.", icon="✍️")
#             obs_ready = False
#             t_max = t_min = humidity = wind = rainfall = aqi_m = None

#     # ─────────────────────────────────────────────────────────────────────
#     # MODE B: MANUAL SLIDERS — full control over every weather parameter
#     # ─────────────────────────────────────────────────────────────────────
#     else:
#         st.markdown("##### Adjust the sliders to any custom weather condition.")
#         pred_date = st.date_input("Date (for logging)", value=date.today(), key="man_date_b")
#         c1, c2 = st.columns(2)
#         with c1:
#             t_max    = st.slider("Temperature Max (°C)", 20.0, 50.0, 38.0, 0.5, key="man_tmax")
#             t_min    = st.slider("Temperature Min (°C)", 10.0, 35.0, 24.0, 0.5, key="man_tmin")
#             humidity = st.slider("Humidity (%)", 5, 100, 45, key="man_hum")
#         with c2:
#             wind     = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0, 0.5, key="man_wind")
#             rainfall = st.slider("Rainfall (mm)", 0.0, 100.0, 0.0, 0.5, key="man_rain")
#             aqi_m    = st.slider("AQI (India CPCB)", 0, 500, 120, key="man_aqi")
#         obs_ready = True

#     # ─────────────────────────────────────────────────────────────────────
#     # PREDICT — shared for both modes
#     # ─────────────────────────────────────────────────────────────────────
#     st.markdown("")
#     if st.button("🔮 Predict Risk", type="primary", key="man_predict_btn",
#                  disabled=not obs_ready):
#         obs = {"date": str(pred_date), "temp_max": t_max, "temp_min": t_min,
#                "humidity": humidity, "wind": wind, "rainfall": rainfall, "aqi": aqi_m}
#         result, err = predict_for_city(city_m, obs, pred_date=pred_date)

#         if err:
#             st.error(err)
#         else:
#             risk  = result["risk_label"]
#             emoji = result["emoji"]
#             rc    = RISK_COLORS[result["risk_level"]]
#             src_lbl = "from CSV" if mode.startswith("📅") else "manual input"

#             st.markdown(f"""
#             <div class="risk-card risk-{risk.upper()}" style="max-width:480px;margin:auto">
#               <div style="font-size:1.2rem;font-weight:700">{city_m} — {pred_date}
#                 <span style="font-size:0.75rem;opacity:0.7"> ({src_lbl})</span></div>
#               <div style="font-size:3.5rem">{emoji}</div>
#               <div class="metric-big">{risk}</div>
#               <div class="metric-sub">Score: {result['composite_score']} / 100
#                &nbsp;|&nbsp; Confidence: {result['confidence']}%</div>
#               <hr>
#               <div>🥵 Heat Index: <b>{result['heat_index']}°C</b></div>
#               <div>🔥 Consecutive hot days: <b>{result['consec_hot_days']}</b></div>
#             </div>""", unsafe_allow_html=True)

#             if result["active_compounds"]:
#                 st.warning("⚠ Compound risks: " + ", ".join(result["active_compounds"]))
#             st.info(result["advisory"])

#             r1, r2 = st.columns(2)
#             with r1:
#                 probs = result["probabilities"]
#                 fig = go.Figure(go.Bar(
#                     x=list(probs.values()), y=list(probs.keys()),
#                     orientation="h",
#                     marker_color=[RISK_COLORS[i] for i in range(4)],
#                     text=[f"{v:.1f}%" for v in probs.values()],
#                     textposition="outside",
#                 ))
#                 fig.update_layout(title="Class probabilities", xaxis_range=[0, 110],
#                                   height=250, margin=dict(l=0,r=0,t=40,b=0),
#                                   showlegend=False)
#                 st.plotly_chart(fig, use_container_width=True)

#             with r2:
#                 hi_v  = _heat_index(t_max, humidity)
#                 st_t  = 0 if t_max<35 else 25 if t_max<38 else 50 if t_max<40 else 75 if t_max<45 else 100
#                 st_h  = 0 if hi_v<27 else 25 if hi_v<32 else 50 if hi_v<41 else 75 if hi_v<54 else 100
#                 st_a  = 0 if aqi_m<=50 else 15 if aqi_m<=100 else 40 if aqi_m<=200 else 60 if aqi_m<=300 else 80 if aqi_m<=400 else 100
#                 st_ap = min(100, int(t_max>=38 and aqi_m>=150)*50 +
#                                  int(t_max>=38 and aqi_m>=200)*50)
#                 st_c  = min(100, int(t_max>=38 and aqi_m>=200)*30 +
#                                  int(t_max>=38 and aqi_m>=150)*25 +
#                                  int(t_max>=38 and humidity>=60)*20)
#                 fig = go.Figure(go.Scatterpolar(
#                     r=[st_t, st_h, st_a, st_ap, st_c],
#                     theta=["Temperature", "Heat Index", "AQI", "Air Pollution", "Compound"],
#                     fill="toself",
#                     fillcolor='rgba(241, 196, 15, 0.27)',
#                     line_color=rc,
#                 ))
#                 fig.update_layout(title="Pillar scores",
#                                   polar=dict(radialaxis=dict(range=[0, 100])),
#                                   height=250, margin=dict(l=0,r=0,t=40,b=0),
#                                   showlegend=False)
#                 st.plotly_chart(fig, use_container_width=True)


# # ════════════════════════════════════════════════════════════════════════
# # PAGE 5: CROSS-CITY ANALYTICS
# # ════════════════════════════════════════════════════════════════════════
# elif "Analytics" in page:
#     st.title("📊 Cross-City Analytics")

#     if df_all is None:
#         st.error("No data found.")
#         st.stop()

#     tab1,tab2,tab3 = st.tabs(["City Comparison","Seasonal Patterns","Distributions"])

#     with tab1:
#         stats = df_all.groupby("city").agg(
#             Avg_Score      =("composite_score","mean"),
#             Max_Score      =("composite_score","max"),
#             Severe_Days    =("risk_level",lambda x:(x==3).sum()),
#             High_Days      =("risk_level",lambda x:(x==2).sum()),
#             Avg_HeatIndex  =("heat_index","mean"),
#             Avg_AQI        =("aqi","mean"),
#             Triple_Events  =("triple_compound","sum"),
#         ).round(2).reset_index()
#         st.dataframe(stats,use_container_width=True)

#         fig = px.bar(stats,x="city",y=["Severe_Days","High_Days"],barmode="group",
#                      color_discrete_sequence=["#E74C3C","#E67E22"],
#                      title="Severe & High Risk Days by City")
#         st.plotly_chart(fig,use_container_width=True)

#         fig2 = px.scatter(stats,x="Avg_AQI",y="Avg_Score",size="Triple_Events",
#                           color="city",color_discrete_map=CITY_COLORS,
#                           text="city",title="Avg AQI vs Avg Score (bubble = triple events)")
#         st.plotly_chart(fig2,use_container_width=True)

#     with tab2:
#         seasonal = df_all.groupby(["city","season"])["composite_score"].mean().reset_index()
#         fig = px.bar(seasonal,x="season",y="composite_score",color="city",
#                      barmode="group",color_discrete_map=CITY_COLORS,
#                      title="Average Risk Score by Season",
#                      category_orders={"season":["Winter","Pre-Monsoon","Monsoon","Post-Monsoon"]})
#         st.plotly_chart(fig,use_container_width=True)

#         # Monthly trend lines
#         monthly = df_all.groupby(["city","month"])["composite_score"].mean().reset_index()
#         fig2 = px.line(monthly,x="month",y="composite_score",color="city",
#                        color_discrete_map=CITY_COLORS,markers=True,
#                        title="Average Risk Score by Month",
#                        labels={"month":"Month","composite_score":"Avg Score"})
#         st.plotly_chart(fig2,use_container_width=True)

#     with tab3:
#         metric = st.selectbox("Feature",
#             ["temp_max","heat_index","aqi","humidity","composite_score","consec_hot_days"])
#         fig = px.violin(df_all,x="city",y=metric,color="risk_label",box=True,
#                         color_discrete_map={"Low":"#2ECC71","Moderate":"#F1C40F",
#                                             "High":"#E67E22","Severe":"#E74C3C"},
#                         title=f"{metric} distribution by city & risk level")
#         st.plotly_chart(fig,use_container_width=True)


# # ════════════════════════════════════════════════════════════════════════
# # PAGE 6: PREDICTION LOG
# # ════════════════════════════════════════════════════════════════════════
# elif "Log" in page:
#     st.title("📋 Prediction Log")
#     st.caption("All live predictions made from the dashboard.")

#     log_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith(".json")],
#                        reverse=True)
#     if not log_files:
#         st.info("No predictions logged yet. Run a live prediction first.")
#         st.stop()

#     # Load all logs
#     records = []
#     for fname in log_files:
#         try:
#             with open(os.path.join(LOG_DIR,fname)) as f:
#                 r = json.load(f)
#             records.append({
#                 "Date"       : r.get("date",""),
#                 "City"       : r.get("city",""),
#                 "Risk"       : r.get("risk_label",""),
#                 "Score"      : r.get("composite_score",""),
#                 "Confidence" : f"{r.get('confidence',0):.1f}%",
#                 "Tmax"       : r.get("raw_obs",{}).get("temp_max",""),
#                 "AQI"        : r.get("raw_obs",{}).get("aqi",""),
#                 "Compounds"  : ", ".join(r.get("active_compounds",[]) or ["-"]),
#                 "AQI Source" : r.get("aqi_source",""),
#             })
#         except Exception:
#             continue

#     log_df = pd.DataFrame(records)

#     # Filter
#     fc1,fc2 = st.columns(2)
#     city_f = fc1.selectbox("Filter city",["All"]+CITIES,key="logcity")
#     risk_f = fc2.selectbox("Filter risk",["All","Low","Moderate","High","Severe"],key="logrisk")
#     if city_f != "All": log_df = log_df[log_df["City"]==city_f]
#     if risk_f != "All": log_df = log_df[log_df["Risk"]==risk_f]

#     # Colour rows
#     def colour_risk(val):
#         c = {"Low":"#d5f5e3","Moderate":"#fef9e7","High":"#fdebd0","Severe":"#fadbd8"}
#         return f"background-color:{c.get(val,'')}"

#     # styled = log_df.style.applymap(colour_risk, subset=["Risk"])
#     styled = log_df.style.map(colour_risk, subset=["Risk"])
#     st.dataframe(styled, use_container_width=True, height=420)

#     # Mini trend of logged predictions
#     if len(log_df) > 1:
#         log_df["Score_num"] = pd.to_numeric(log_df["Score"],errors="coerce")
#         fig = px.line(log_df.sort_values("Date"),x="Date",y="Score_num",
#                       color="City",color_discrete_map=CITY_COLORS,markers=True,
#                       title="Composite Score — Live Prediction History")
#         fig.add_hline(y=50,line_dash="dot",line_color="orange")
#         fig.add_hline(y=75,line_dash="dot",line_color="red")
#         st.plotly_chart(fig,use_container_width=True)

#     # Download
#     csv = log_df.to_csv(index=False)
#     st.download_button("⬇ Download log as CSV",csv,
#                        "prediction_log.csv","text/csv")







# ++++++++++++++++++++++++++ VERSION 4 (WITH SEASONAL FORECASTING) +++++++++++++++++++++++++++++++++

"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Final Integrated Dashboard  —  step6_dashboard_integrated.py
========================================================================
Run with:  streamlit run step6_dashboard_integrated.py

Combines historical analysis (step6) + real-time API prediction (step7)
into one application.

Pages:
  🔴 Live Prediction   – Fetch today/tomorrow from APIs, run ML model
  🏠 Overview          – Historical risk cards + period stats
  📈 Trends            – Time-series, compound events, heatmaps
  🔍 Manual Predict    – Slider-based custom prediction + radar chart
  📋 Prediction Log    – History of all live predictions made

APIs used (all free):
  Open-Meteo Weather   : no key needed
  Open-Meteo AQ        : no key needed  ← primary AQI source
  OpenWeatherMap AQ    : free key needed (fallback, ~2h activation delay)
  AQICN                : free token needed (fallback, current only)
========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import os, pickle, json, requests, warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Heatwave Risk System",
    page_icon   = "🌡",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── CONSTANTS ────────────────────────────────────────────────────────────
CITIES      = ["Delhi", "Hyderabad", "Nagpur"]
RISK_COLORS = {0:"#2ECC71", 1:"#F1C40F", 2:"#E67E22", 3:"#E74C3C"}
RISK_LABELS = {0:"Low",     1:"Moderate", 2:"High",   3:"Severe"}
RISK_EMOJIS = {0:"🟢",      1:"🟡",       2:"🟠",     3:"🔴"}
CITY_COLORS = {"Delhi":"#3498DB", "Hyderabad":"#E74C3C", "Nagpur":"#2ECC71"}
CITY_CONFIG = {
    "Delhi"     : {"lat": 28.6139, "lon": 77.2090, "aqicn": "delhi"},
    "Hyderabad" : {"lat": 17.3850, "lon": 78.4867, "aqicn": "hyderabad"},
    "Nagpur"    : {"lat": 21.1458, "lon": 79.0882, "aqicn": "nagpur"},
}
PROC_DIR  = "data/processed"
MODEL_DIR = "models"
LOG_DIR   = "data/predictions"
os.makedirs(LOG_DIR, exist_ok=True)

FEATURE_COLS = [
    "temp_max","temp_min","humidity","wind","rainfall","aqi",
    "heat_index","humidex","temp_range","temp_mean","feels_like_excess",
    "wind_heat_ratio",
    "temp_max_roll3","temp_max_roll7","humidity_roll7","aqi_roll7","rainfall_roll7",
    "temp_max_lag1","temp_max_lag2","aqi_lag1","humidity_lag1",
    "temp_departure","temp_zscore","aqi_departure",
    "dry_days_streak","spi_30","drought_flag",
    "month","day_of_year","is_summer",
    "compound_heat_aqi","compound_heat_drought","compound_heat_humidity",
    "triple_compound","consec_hot_days","aqi_category",
]

ADVISORIES = {
    0: "No significant heat risk. Normal outdoor activities are safe.",
    1: "Moderate heat. Stay hydrated. Avoid peak sun (12–4 PM). Monitor elderly & children.",
    2: "HIGH HEAT RISK. Limit outdoor activity. Seek cool shelter. Watch for heat exhaustion.",
    3: "⚠ SEVERE HEAT WAVE. Avoid ALL outdoor activity. Emergency health protocols active.",
}

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.risk-card {
  border-radius:14px; padding:20px; margin:4px;
  text-align:center; font-family:Arial,sans-serif;
}
.live-badge {
  display:inline-block; background:#E74C3C; color:#fff;
  font-size:0.7rem; font-weight:700; border-radius:20px;
  padding:2px 8px; margin-left:6px; vertical-align:middle;
  animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
.metric-big  { font-size:2.2rem; font-weight:800; }
.metric-sub  { font-size:0.85rem; color:#666; }
.risk-LOW      { background:#d5f5e3; border:2px solid #2ECC71; }
.risk-MODERATE { background:#fef9e7; border:2px solid #F1C40F; }
.risk-HIGH     { background:#fdebd0; border:2px solid #E67E22; }
.risk-SEVERE   { background:#fadbd8; border:2px solid #E74C3C; }
.source-tag {
  font-size:0.7rem; background:#eee; border-radius:8px;
  padding:2px 7px; color:#555;
}
.advisory-box {
  background:#f8f9fa; border-left:4px solid #E67E22;
  padding:10px 14px; border-radius:6px;
  font-size:0.9rem; margin-top:8px;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# SECTION A: AQI CONVERSION  (PM2.5/PM10 µg/m³ → India CPCB 0–500)
# ════════════════════════════════════════════════════════════════════════
CPCB_PM25 = [(0,30,0,50),(30.1,60,51,100),(60.1,90,101,200),
             (90.1,120,201,300),(120.1,250,301,400),(250.1,500,401,500)]
CPCB_PM10 = [(0,50,0,50),(51,100,51,100),(101,250,101,200),
             (251,350,201,300),(351,430,301,400),(431,600,401,500)]

def _cpcb(c, bps):
    if c <= 0: return 0.0
    for lo, hi, ilo, ihi in bps:
        if lo <= c <= hi:
            return round(((ihi-ilo)/(hi-lo))*(c-lo)+ilo, 1)
    return 500.0

def pm_to_india_aqi(pm25, pm10):
    return max(_cpcb(pm25 or 0, CPCB_PM25), _cpcb(pm10 or 0, CPCB_PM10))


# ════════════════════════════════════════════════════════════════════════
# SECTION B: API FETCHERS
# ════════════════════════════════════════════════════════════════════════

def fetch_weather(lat, lon, target_date):
    """Open-Meteo forecast — free, no key."""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
            "forecast_days": 7,
            "daily": ["temperature_2m_max","temperature_2m_min",
                      "precipitation_sum","wind_speed_10m_max",
                      "relative_humidity_2m_max"],
        }, timeout=12)
        r.raise_for_status()
        daily  = r.json()["daily"]
        dates  = pd.to_datetime(daily["time"])
        target = pd.Timestamp(target_date)
        if target not in list(dates): return None, "date not in window"
        i = list(dates).index(target)
        return {
            "date"    : str(target_date),
            "temp_max": round(daily["temperature_2m_max"][i], 1),
            "temp_min": round(daily["temperature_2m_min"][i], 1),
            "humidity": round(daily["relative_humidity_2m_max"][i] or 50, 1),
            "wind"    : round(daily["wind_speed_10m_max"][i] or 0, 1),
            "rainfall": round(daily["precipitation_sum"][i] or 0, 1),
        }, None
    except Exception as e:
        return None, str(e)


def fetch_aqi_open_meteo(lat, lon, target_date):
    """Open-Meteo Air Quality — free, no key. Primary AQI source."""
    try:
        r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
                    "forecast_days": 7, "hourly": ["pm2_5","pm10"]}, timeout=12)
        r.raise_for_status()
        hourly = r.json()["hourly"]
        times  = pd.to_datetime(hourly["time"])
        pm25   = hourly["pm2_5"]
        pm10   = hourly["pm10"]
        peak   = pd.Timestamp(target_date).replace(hour=14)
        idxs   = [i for i, t in enumerate(times) if t.date() == target_date]
        if not idxs: return None, "no data for date"
        best   = min(idxs, key=lambda i: abs((times[i]-peak).total_seconds()))
        aqi    = pm_to_india_aqi(pm25[best] or 0, pm10[best] or 0)
        return round(aqi, 0), None
    except Exception as e:
        return None, str(e)


def fetch_aqi_owm(lat, lon, target_date, api_key):
    """OpenWeatherMap Air Pollution — free key needed (2h activation)."""
    if not api_key or api_key == "YOUR_KEY":
        return None, "key not set"
    today    = date.today()
    forecast = target_date > today
    url      = ("http://api.openweathermap.org/data/2.5/air_pollution/forecast"
                if forecast else
                "http://api.openweathermap.org/data/2.5/air_pollution")
    try:
        r = requests.get(url, params={"lat":lat,"lon":lon,"appid":api_key}, timeout=12)
        if r.status_code == 401:
            return None, "401 — key not yet active (wait ~2h after signup)"
        r.raise_for_status()
        items = r.json()["list"]
        if forecast:
            peak = datetime.combine(target_date, datetime.min.time()).replace(hour=14)
            item = min(items, key=lambda x: abs(datetime.fromtimestamp(x["dt"])-peak))
        else:
            item = items[0]
        c   = item["components"]
        aqi = pm_to_india_aqi(c.get("pm2_5",0), c.get("pm10",0))
        return round(aqi, 0), None
    except Exception as e:
        return None, str(e)


def fetch_aqi_aqicn(city_slug, token):
    """AQICN — free token, current only."""
    if not token or token == "YOUR_TOKEN":
        return None, "token not set"
    try:
        r   = requests.get(f"https://api.waqi.info/feed/{city_slug}/",
                           params={"token": token}, timeout=12)
        r.raise_for_status()
        d   = r.json()
        if d.get("status") == "ok":
            v = d["data"]["aqi"]
            return (float(v), None) if v != "-" else (None, "no reading")
        return None, d.get("data","unknown error")
    except Exception as e:
        return None, str(e)


def get_aqi(lat, lon, city_slug, target_date, owm_key="", aqicn_token=""):
    """
    Priority: Open-Meteo AQ → OWM → AQICN → history fallback.
    Returns (aqi_value, source_label).
    """
    aqi, err = fetch_aqi_open_meteo(lat, lon, target_date)
    if aqi is not None:
        return aqi, "Open-Meteo AQ (free)"

    aqi, err = fetch_aqi_owm(lat, lon, target_date, owm_key)
    if aqi is not None:
        return aqi, "OpenWeatherMap"

    aqi, err = fetch_aqi_aqicn(city_slug, aqicn_token)
    if aqi is not None:
        return aqi, "AQICN"

    return None, "all sources failed"


# ════════════════════════════════════════════════════════════════════════
# SECTION C: FEATURE ENGINEERING  (self-contained, mirrors step1)
# ════════════════════════════════════════════════════════════════════════

def _heat_index(T, RH):
    TF = T*9/5+32
    HI = (-42.379+2.04901523*TF+10.14333127*RH-0.22475541*TF*RH
          -0.00683783*TF**2-0.05481717*RH**2+0.00122874*TF**2*RH
          +0.00085282*TF*RH**2-0.00000199*TF**2*RH**2)
    return (HI-32)*5/9

def _humidex(T, RH):
    Td = T-((100-RH)/5)
    e  = 6.105*np.exp(25.22*(Td-273.16)/(Td+1e-9))
    return T+0.5555*(e-10)

def build_features(history_df: pd.DataFrame, today_obs: dict) -> pd.Series:
    today_row = pd.DataFrame([today_obs])
    df = pd.concat([history_df, today_row], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["month"]        = df["date"].dt.month
    df["day_of_year"]  = df["date"].dt.dayofyear
    df["is_summer"]    = df["month"].isin([4,5,6]).astype(int)
    df["temp_range"]   = df["temp_max"]-df["temp_min"]
    df["temp_mean"]    = (df["temp_max"]+df["temp_min"])/2
    df["heat_index"]   = df.apply(lambda r: _heat_index(r["temp_max"],r["humidity"]),axis=1)
    df["humidex"]      = df.apply(lambda r: _humidex(r["temp_max"],r["humidity"]),axis=1)
    df["feels_like_excess"] = df["heat_index"]-df["temp_max"]
    df["wind_heat_ratio"]   = df["temp_max"]/(df["wind"].replace(0,0.1))

    for w in [3,7]:
        df[f"temp_max_roll{w}"] = df["temp_max"].rolling(w,min_periods=1).mean()
        df[f"humidity_roll{w}"] = df["humidity"].rolling(w,min_periods=1).mean()
        df[f"aqi_roll{w}"]      = df["aqi"].rolling(w,min_periods=1).mean()
        df[f"rainfall_roll{w}"] = df["rainfall"].rolling(w,min_periods=1).sum()

    for lag in [1,2]:
        df[f"temp_max_lag{lag}"] = df["temp_max"].shift(lag)
        df[f"aqi_lag{lag}"]      = df["aqi"].shift(lag)
        df[f"humidity_lag{lag}"] = df["humidity"].shift(lag)

    mm = df.groupby("month")["temp_max"].transform("mean")
    ms = df.groupby("month")["temp_max"].transform("std").replace(0,1e-6)
    df["temp_departure"] = df["temp_max"]-mm
    df["temp_zscore"]    = (df["temp_max"]-mm)/ms
    df["aqi_departure"]  = df["aqi"]-df.groupby("month")["aqi"].transform("mean")

    df["drought_flag"]    = (df["rainfall"].rolling(7,min_periods=1).sum()<2).astype(int)
    hr = (df["rainfall"]<1.0).astype(int)
    df["dry_days_streak"] = hr.groupby((hr!=hr.shift()).cumsum()).cumcount()

    roll = df["rainfall"].rolling(30,min_periods=7)
    df["spi_30"] = (df["rainfall"]-roll.mean())/(roll.std().replace(0,1e-6))

    df["aqi_category"] = pd.cut(df["aqi"],bins=[0,50,100,200,300,400,500],
                                 labels=[1,2,3,4,5,6]).astype(float)
    hot = (df["temp_max"]>=40.0).astype(int)
    df["consec_hot_days"] = hot.groupby((hot!=hot.shift()).cumsum()).cumcount()+hot

    df["compound_heat_aqi"]           = ((df["temp_max"]>=38)&(df["aqi"]>=200)).astype(int)
    df["compound_heat_humidity"]      = ((df["temp_max"]>=38)&(df["humidity"]>=60)).astype(int)
    # Heat + Air Pollution: broader AQI>=150 (Moderately Polluted) threshold
    df["compound_heat_air_pollution"] = ((df["temp_max"]>=38)&(df["aqi"]>=150)).astype(int)
    # Triple: Heat + High AQI + High Humidity (all daily-observable, no drought)
    df["triple_compound"]             = ((df["temp_max"]>=38)&(df["aqi"]>=150)&
                                         (df["humidity"]>=60)).astype(int)
    # Keep drought flag computed but only used as a feature for the model, not shown as compound
    df["compound_heat_drought"]       = ((df["temp_max"]>=38)&(df["drought_flag"]==1)).astype(int)
    return df.iloc[-1]


# ════════════════════════════════════════════════════════════════════════
# SECTION D: MODEL LOADING & PREDICTION
# ════════════════════════════════════════════════════════════════════════

# ── MODEL LOADING ────────────────────────────────────────────────────────
# No @st.cache_resource here intentionally.
# If you retrain the model (step4) and click Fetch again, the dashboard
# picks up the new .pkl immediately without needing a restart.
def load_model(city):
    for tag in [city, "all"]:
        p = os.path.join(MODEL_DIR, f"classifier_{tag}.pkl")
        if os.path.exists(p):
            with open(p,"rb") as f: bundle = pickle.load(f)
            rp = os.path.join(MODEL_DIR, f"regressor_{tag}.pkl")
            reg = pickle.load(open(rp,"rb")) if os.path.exists(rp) else None
            return bundle["model"], reg
    return None, None

# ── HISTORY LOADING ───────────────────────────────────────────────────────
# Reads from per-city labelled CSVs (updated by step7b + step7 daily).
# NOT from labelled_all.csv which is only regenerated by step1/step2.
# No @st.cache_data so every Fetch button click gets fresh data.
def load_city_history(city: str) -> pd.DataFrame | None:
    """
    Reads the most current labelled/processed CSV for a single city.
    Priority: labelled_{city}.csv > processed_{city}.csv > labelled_all.csv
    """
    for fname in [f"labelled_{city}.csv",
                  f"processed_{city}.csv"]:
        p = os.path.join(PROC_DIR, fname)
        if os.path.exists(p):
            return pd.read_csv(p, parse_dates=["date"])
    # Last resort: filter labelled_all.csv
    p = os.path.join(PROC_DIR, "labelled_all.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, parse_dates=["date"])
        return df[df["city"] == city].copy()
    return None

@st.cache_data(ttl=0)   # ttl=0 → never serve stale; always re-read from disk
def load_history_df():
    """Used only by historical analysis pages, not by live prediction."""
    p = os.path.join(PROC_DIR, "labelled_all.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, parse_dates=["date"])

def predict_for_city(city, today_obs, pred_date=None):
    """
    Loads rolling history BEFORE pred_date (or today), builds features,
    runs the ML model, and returns a result dict.
    pred_date: datetime.date or None (defaults to today)
    """
    city_df = load_city_history(city)
    if city_df is None:
        return None, "No history data found. Run steps 1–2 first."

    raw_cols  = ["date","temp_max","temp_min","humidity","wind","rainfall","aqi"]
    available = [c for c in raw_cols if c in city_df.columns]
    city_df["date"] = pd.to_datetime(city_df["date"], errors="coerce")

    # ── Use history strictly BEFORE the prediction date ──────────────────
    if pred_date is None:
        pred_date = date.today()
    cutoff = pd.Timestamp(pred_date)
    history_pool = city_df[city_df["date"] < cutoff]

    if history_pool.empty:
        # Fall back to all history if nothing before the date (e.g., very old date)
        history_pool = city_df

    history = history_pool[available].dropna().tail(40).copy()

    # Warn if rolling window is stale relative to chosen date
    last_hist_date = history["date"].max().date()
    if last_hist_date < pred_date - timedelta(days=7):
        st.warning(
            f"⚠ **{city}**: Rolling history last date is **{last_hist_date}** "
            f"(more than 7 days before {pred_date}). Rolling features may be approximate.",
            icon="🕐"
        )

    clf, reg = load_model(city)
    if clf is None:
        return None, f"Model not found for {city}. Run step4 first."

    enriched = build_features(history, today_obs)
    X        = enriched[FEATURE_COLS].fillna(0).values.reshape(1,-1)
    probas   = clf.predict_proba(X)[0]
    lvl      = int(np.argmax(probas))
    score    = float(reg.predict(X)[0]) if reg else None

    compounds = {
        "Heat + AQI (Severe)": int(enriched.get("compound_heat_aqi",0)),
        "Heat + Air Pollution": int(enriched.get("compound_heat_air_pollution",0)),
        "Heat + Humidity"    : int(enriched.get("compound_heat_humidity",0)),
        "Triple Compound"    : int(enriched.get("triple_compound",0)),
    }
    return {
        "city"           : city,
        "date"           : today_obs["date"],
        "risk_level"     : lvl,
        "risk_label"     : RISK_LABELS[lvl],
        "emoji"          : RISK_EMOJIS[lvl],
        "composite_score": round(score,1) if score else None,
        "confidence"     : round(float(probas[lvl])*100,1),
        "probabilities"  : {RISK_LABELS[i]:round(float(p)*100,1) for i,p in enumerate(probas)},
        "heat_index"     : round(float(enriched["heat_index"]),1),
        "humidex"        : round(float(enriched["humidex"]),1),
        "consec_hot_days": int(enriched["consec_hot_days"]),
        "active_compounds": [k for k,v in compounds.items() if v==1],
        "advisory"       : ADVISORIES[lvl],
        "raw_obs"        : today_obs,
    }, None


def save_prediction_log(result: dict):
    p = os.path.join(LOG_DIR, f"pred_{result['city']}_{result['date']}.json")
    with open(p,"w") as f:
        json.dump(result, f, indent=2, default=str)


def append_to_history(city, obs):
    """Appends today's fetched observation to the labelled CSV."""
    for fname in [f"labelled_{city}.csv", f"processed_{city}.csv"]:
        p = os.path.join(PROC_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["date"])
            if pd.Timestamp(obs["date"]) not in df["date"].values:
                df = pd.concat([df, pd.DataFrame([{**obs,"city":city}])],
                               ignore_index=True)
                df.to_csv(p, index=False)
            return


# ════════════════════════════════════════════════════════════════════════
# SECTION E: DATA LOADING
# ════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_all_data():
    p = os.path.join(PROC_DIR, "labelled_all.csv")
    if not os.path.exists(p): return None
    return pd.read_csv(p, parse_dates=["date"])


# ════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌡 Heatwave Risk System")
    st.caption("Delhi · Hyderabad · Nagpur")
    st.markdown("---")

    page = st.radio("Navigation", [
        "🔴 Live & 7-Day Forecast",
        "📆 Monthly(Seasonal) Outlook",
        "🏠 Historical Overview",
        "📈 Trends & Analysis",
        "🔍 Manual Prediction",
        "📋 Prediction Log",
    ])

    st.markdown("---")
    st.markdown("**API Keys** *(optional — Open-Meteo works without)*")
    owm_key    = st.text_input("OWM Key", type="password",
                               placeholder="paste OpenWeatherMap key")
    aqicn_tok  = st.text_input("AQICN Token", type="password",
                               placeholder="paste AQICN token")
    st.caption("New OWM keys take ~2h to activate.")

    st.markdown("---")
    st.markdown("**Filters** *(historical pages)*")
    sel_city   = st.selectbox("City", ["All"]+CITIES)
    try:
        date_range = st.date_input("Date Range",
            value=(datetime(2020,1,1), datetime(2025,12,31)))
    except Exception:
        date_range = (datetime(2020,1,1), datetime(2025,12,31))


df_all = load_all_data()

# Apply sidebar filters for historical pages
if df_all is not None:
    df_view = df_all[df_all["city"]==sel_city].copy() if sel_city!="All" else df_all.copy()
    if isinstance(date_range, (list,tuple)) and len(date_range)==2:
        df_view = df_view[
            (df_view["date"]>=pd.Timestamp(date_range[0])) &
            (df_view["date"]<=pd.Timestamp(date_range[1]))
        ]
else:
    df_view = None


# ════════════════════════════════════════════════════════════════════════
# PAGE 1: LIVE PREDICTION
# ════════════════════════════════════════════════════════════════════════
if "Live" in page:
    st.title("🔴 Live & 7-Day Forecas")
    pred_mode = st.radio("Select Prediction Mode", ["Live Prediction (Today/Tomorrow)", "7-Day Forecast"], horizontal=True)
    st.markdown("---")
    if "Live Prediction" in pred_mode:
        st.markdown("## 🔴 Live Prediction <span class='live-badge'>LIVE</span>",
                    unsafe_allow_html=True)
        st.caption("Fetches real weather & AQ data, runs ML model using your historical data as context.")

        # Controls
        c1, c2, c3 = st.columns([2,2,3])
        with c1:
            live_cities = st.multiselect("Cities to predict",
                                         CITIES, default=CITIES)
        with c2:
            predict_day = st.radio("Predict for", ["Today","Tomorrow"],
                                   horizontal=True)
        with c3:
            st.markdown("<br>",unsafe_allow_html=True)
            fetch_btn = st.button("🌐 Fetch Live Data & Predict",
                                  type="primary", use_container_width=True)

        st.markdown("---")



        # ── Run on button click ────────────────────────────────────────────
        if fetch_btn:
            target = (date.today()+timedelta(days=1)
                      if predict_day=="Tomorrow" else date.today())

            st.session_state["live_results"]   = {}
            st.session_state["live_target"]    = str(target)
            st.session_state["live_timestamp"] = datetime.now().strftime("%d %b %Y %H:%M")

            prog = st.progress(0, text="Starting fetch...")
            for idx, city in enumerate(live_cities):
                prog.progress((idx+0.1)/len(live_cities),
                              text=f"Fetching weather for {city}...")
                cfg = CITY_CONFIG[city]

                # 1. Weather
                weather, werr = fetch_weather(cfg["lat"], cfg["lon"], target)
                if weather is None:
                    st.session_state["live_results"][city] = {
                        "error": f"Weather fetch failed: {werr}"}
                    continue

                # 2. AQI
                prog.progress((idx+0.5)/len(live_cities),
                              text=f"Fetching AQI for {city}...")
                aqi_val, aqi_src = get_aqi(cfg["lat"], cfg["lon"],
                                            cfg["aqicn"], target,
                                            owm_key, aqicn_tok)
                if aqi_val is None:
                    # fallback: 7-day avg from history
                    if df_all is not None:
                        hist_c = df_all[df_all["city"]==city]
                        aqi_val = float(hist_c["aqi"].tail(7).mean())
                        aqi_src = "7-day historical avg"
                    else:
                        aqi_val = 120
                        aqi_src = "default fallback"

                today_obs = {**weather, "aqi": round(float(aqi_val), 0)}

                # 3. Predict
                prog.progress((idx+0.8)/len(live_cities),
                              text=f"Running prediction for {city}...")
                result, err = predict_for_city(city, today_obs)
                if err:
                    st.session_state["live_results"][city] = {"error": err}
                    continue

                result["aqi_source"] = aqi_src
                st.session_state["live_results"][city] = result
                save_prediction_log(result)
                if predict_day == "Today":
                    append_to_history(city, today_obs)

            prog.progress(1.0, text="Done!")
            load_all_data.clear()   # refresh historical cache with new data

        # ── Display results ────────────────────────────────────────────────
        if "live_results" in st.session_state and st.session_state["live_results"]:
            results = st.session_state["live_results"]
            ts      = st.session_state.get("live_timestamp","")
            target_str = st.session_state.get("live_target","")

            st.caption(f"Last fetched: **{ts}**  |  Prediction for: **{target_str}**")

            # ── Risk cards ───────────────────────────────────────────────
            cols = st.columns(len(results))
            for col, (city, res) in zip(cols, results.items()):
                if "error" in res:
                    col.error(f"**{city}**: {res['error']}")
                    continue
                risk  = res["risk_label"]
                emoji = res["emoji"]
                col.markdown(f"""
                <div class="risk-card risk-{risk.upper()}">
                  <div style="font-size:1.3rem;font-weight:700">{city}</div>
                  <div style="font-size:3rem">{emoji}</div>
                  <div class="metric-big">{risk}</div>
                  <div class="metric-sub">Score: {res['composite_score']} / 100</div>
                  <div class="metric-sub">Confidence: {res['confidence']}%</div>
                  <hr style="margin:8px 0">
                  <div>🌡 Tmax: <b>{res['raw_obs']['temp_max']}°C</b></div>
                  <div>🥵 Heat Index: <b>{res['heat_index']}°C</b></div>
                  <div>💧 Humidity: <b>{res['raw_obs']['humidity']}%</b></div>
                  <div>💨 AQI: <b>{res['raw_obs']['aqi']:.0f}</b></div>
                  <div>🌧 Rain: <b>{res['raw_obs']['rainfall']} mm</b></div>
                  <div>🔥 Consec Hot Days: <b>{res['consec_hot_days']}</b></div>
                  <div style="margin-top:6px">
                    <span class="source-tag">AQI: {res.get('aqi_source','API')}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Detailed panels per city ─────────────────────────────────
            valid = {c:r for c,r in results.items() if "error" not in r}
            if not valid: st.stop()

            tabs = st.tabs([f"{r['emoji']} {c}" for c,r in valid.items()])
            for tab, (city, res) in zip(tabs, valid.items()):
                with tab:
                    a1, a2 = st.columns([1,1])

                    with a1:
                        # Probability bar chart
                        probs = res["probabilities"]
                        fig = go.Figure(go.Bar(
                            x=list(probs.values()),
                            y=list(probs.keys()),
                            orientation="h",
                            marker_color=[RISK_COLORS[i] for i in range(4)],
                            text=[f"{v:.1f}%" for v in probs.values()],
                            textposition="outside",
                        ))
                        fig.update_layout(
                            title="Class Probabilities",
                            xaxis=dict(range=[0,105], title="%"),
                            height=220, margin=dict(l=10,r=10,t=40,b=10),
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with a2:
                        # Pillar radar
                        obs = res["raw_obs"]
                        def _s(x,bps):
                            for lo,hi,ilo,ihi in bps:
                                if lo<=x<=hi: return round(((ihi-ilo)/(hi-lo))*(x-lo)+ilo,1)
                            return 100.0
                        hi_val = _heat_index(obs["temp_max"], obs["humidity"])
                        st_t = 0 if obs["temp_max"]<35 else 25 if obs["temp_max"]<38 else 50 if obs["temp_max"]<40 else 75 if obs["temp_max"]<45 else 100
                        st_h = _s(hi_val, [(0,27,0,0),(27,32,0,25),(32,41,25,50),(41,54,50,75),(54,100,75,100)])
                        st_a = _s(obs["aqi"], [(0,50,0,0),(50,100,0,15),(100,200,15,40),(200,300,40,60),(300,400,60,80),(400,500,80,100)])
                        # Heat+Air Pollution compound score (replaces drought)
                        st_ap = min(100, int(obs["temp_max"]>=38 and obs["aqi"]>=150)*50 +
                                         int(obs["temp_max"]>=38 and obs["aqi"]>=200)*50)
                        st_c = min(100, (int(obs["temp_max"]>=38 and obs["aqi"]>=200)*30 +
                                         int(obs["temp_max"]>=38 and obs["aqi"]>=150)*25 +
                                         int(obs["temp_max"]>=38 and obs["humidity"]>=60)*20))
                        rc   = RISK_COLORS[res["risk_level"]]
                        fig  = go.Figure(go.Scatterpolar(
                            r=[st_t,st_h,st_a,st_ap,st_c],
                            theta=["Temperature","Heat Index","AQI","Air Pollution","Compound"],
                            fill="toself",
                            fillcolor='rgba(241, 196, 15, 0.27)',
                            line_color=rc,
                        ))
                        fig.update_layout(
                            title="Pillar Breakdown",
                            polar=dict(radialaxis=dict(range=[0,100],tickfont_size=9)),
                            height=220, margin=dict(l=10,r=10,t=40,b=10),
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Advisory
                    st.markdown(f"""<div class="advisory-box">
                      <b>Advisory:</b> {res['advisory']}
                    </div>""", unsafe_allow_html=True)

                    # Compound flags
                    if res["active_compounds"]:
                        st.warning("⚠ Active Compound Risks: " +
                                   " | ".join(res["active_compounds"]))

                    # Context from history
                    if df_all is not None:
                        st.markdown("**Historical context — last 14 days:**")
                        hist14 = (df_all[df_all["city"]==city]
                                  .tail(14)[["date","temp_max","aqi","risk_label","composite_score"]]
                                  .copy())
                        hist14["date"] = hist14["date"].dt.strftime("%d %b")
                        hist14.columns = ["Date","Tmax°C","AQI","Risk","Score"]
                        st.dataframe(hist14.set_index("Date"), use_container_width=True)

        else:
            st.info("👆 Click **Fetch Live Data & Predict** to run today's or tomorrow's prediction.")



    # ════════════════════════════════════════════════════════════════════════
    # PAGE 2: HISTORICAL OVERVIEW
    # ════════════════════════════════════════════════════════════════════════
    else:
        st.title("📅 7-Day Heatwave Forecast")
        st.caption(
            "Chained ML forecast using Open-Meteo 16-day weather + 5-day AQ. "
            "Confidence decays each day forward — see reliability label.")

        # ── Controls ──────────────────────────────────────────────────────
        fc1, fc2, fc3 = st.columns([2, 2, 3])
        with fc1:
            fc_cities = st.multiselect("Cities", CITIES, default=CITIES, key="fc_cities")
        with fc2:
            n_days = st.slider("Days ahead", 1, 16, 7, key="fc_days")
        with fc3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_fc = st.button("🔮 Run Forecast", type="primary",
                               use_container_width=True, key="fc_btn")

        # Reliability legend
        st.markdown(
            "> **Reliability:** "
            "🟦 **High** (days 1–2) &nbsp;|&nbsp; "
            "🟨 **Medium** (days 3–4) &nbsp;|&nbsp; "
            "🟧 **Low** (days 5–7) &nbsp;|&nbsp; "
            "⬜ **Indicative** (days 8+)"
        )
        st.markdown("---")

        if run_fc:
            # ── import the forecast engine ─────────────────────────────
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location(
                "step7c", os.path.join(os.path.dirname(__file__),
                                        "step7c_multiday_forecast.py"))
            if spec is None:
                st.error("step7c_multiday_forecast.py not found in the same folder.")
                st.stop()
            mod = importlib.util.module_from_spec(spec)
            sys.modules["step7c"] = mod
            spec.loader.exec_module(mod)

            all_fc = {}
            prog   = st.progress(0, text="Starting forecast...")
            for idx, city in enumerate(fc_cities):
                prog.progress((idx + 0.3) / len(fc_cities),
                              text=f"Forecasting {city}...")
                fc = mod.forecast_city(city, n_days=n_days)
                all_fc[city] = fc
                prog.progress((idx + 1) / len(fc_cities),
                              text=f"{city} done")

            prog.progress(1.0, text="Complete!")
            st.session_state["fc_results"] = all_fc
            st.session_state["fc_n_days"]  = n_days

        # ── Display results ─────────────────────────────────────────────
        if "fc_results" in st.session_state and st.session_state["fc_results"]:
            all_fc = st.session_state["fc_results"]
            n      = st.session_state.get("fc_n_days", 7)

            RELIABILITY_COLORS = {
                "High"       : "#2ECC71",
                "Medium"     : "#F1C40F",
                "Low"        : "#E67E22",
                "Indicative" : "#BDC3C7",
            }

            for city, fc_list in all_fc.items():
                if not fc_list:
                    st.error(f"{city}: forecast failed.")
                    continue

                st.subheader(f"📍 {city}")

                # ── Risk card strip ──────────────────────────────────
                card_cols = st.columns(min(len(fc_list), 8))
                for col, day in zip(card_cols, fc_list[:8]):
                    risk  = day["risk_label"]
                    bg    = {"Low":"#d5f5e3","Moderate":"#fef9e7",
                              "High":"#fdebd0","Severe":"#fadbd8"}.get(risk,"#f0f0f0")
                    rel_c = RELIABILITY_COLORS.get(day["reliability"], "#ccc")
                    col.markdown(f"""
                    <div style="background:{bg};border:2px solid {rel_c};
                      border-radius:10px;padding:8px;text-align:center;font-size:0.8rem">
                      <div style="font-size:0.7rem;color:#888">{day["date"][5:]}</div>
                      <div style="font-size:1.6rem">{day["emoji"]}</div>
                      <div style="font-weight:700;font-size:0.8rem">{risk}</div>
                      <div style="font-size:0.7rem">{day["adj_confidence"]:.0f}%</div>
                      <div style="font-size:0.7rem">🌡{day["temp_max"]}°C</div>
                    </div>""", unsafe_allow_html=True)

                # ── Charts ───────────────────────────────────────────
                fc_df = pd.DataFrame(fc_list)
                fc_df["date"] = pd.to_datetime(fc_df["date"])

                ch1, ch2 = st.columns(2)

                with ch1:
                    # Composite score + confidence band
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fc_df["date"], y=fc_df["composite_score"],
                        name="Risk Score",
                        line=dict(color=CITY_COLORS.get(city,"#3498DB"), width=2),
                        mode="lines+markers",
                    ))
                    # Confidence band: score ± (1-decay)*score
                    fc_df["upper"] = fc_df["composite_score"] * (
                        1 + (1 - fc_df["decay_factor"]) * 0.5)
                    fc_df["lower"] = fc_df["composite_score"] * (
                        1 - (1 - fc_df["decay_factor"]) * 0.5)
                    # fig.add_trace(go.Scatter(
                    #     x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
                    #     y=pd.concat([fc_df["upper"], fc_df["lower"][::-1]]),
                    #     fill="toself",
                    #     fillcolor=CITY_COLORS.get(city,"#3498DB") + "22",
                    #     line=dict(color="rgba(0,0,0,0)"),
                    #     name="Uncertainty band",
                    # ))
                    hex_color = CITY_COLORS.get(city, "#3498DB").lstrip("#")
                    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    fill_col = f"rgba({r}, {g}, {b}, 0.13)"

                    fig.add_trace(go.Scatter(
                        x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
                        y=pd.concat([fc_df["upper"], fc_df["lower"][::-1]]),
                        fill="toself",
                        fillcolor=fill_col,
                        line=dict(color="rgba(0,0,0,0)"),
                        name="Uncertainty band",
                    ))

                    fig.add_hline(y=50, line_dash="dot",
                                  line_color="orange", annotation_text="High (50)")
                    fig.add_hline(y=75, line_dash="dot",
                                  line_color="red", annotation_text="Severe (75)")
                    fig.update_layout(
                        title="Composite Risk Score + Uncertainty",
                        height=300, margin=dict(l=0,r=0,t=40,b=0),
                        legend=dict(orientation="h", y=-0.3),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with ch2:
                    # Temperature + AQI dual axis
                    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                    fig2.add_trace(go.Bar(
                        x=fc_df["date"], y=fc_df["temp_max"],
                        name="Tmax (°C)",
                        marker_color=[RISK_COLORS[r] for r in fc_df["risk_level"]],
                        opacity=0.8,
                    ), secondary_y=False)
                    fig2.add_trace(go.Scatter(
                        x=fc_df["date"], y=fc_df["aqi"],
                        name="AQI",
                        line=dict(color="#8E44AD", width=2, dash="dot"),
                        mode="lines+markers",
                    ), secondary_y=True)
                    fig2.update_yaxes(title_text="Tmax (°C)", secondary_y=False)
                    fig2.update_yaxes(title_text="AQI",       secondary_y=True)
                    fig2.update_layout(
                        title="Temperature & AQI forecast",
                        height=300, margin=dict(l=0,r=0,t=40,b=0),
                        legend=dict(orientation="h", y=-0.3),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # ── Detailed table ────────────────────────────────────
                with st.expander(f"📋 {city} — Full forecast table"):
                    tbl = fc_df[[
                        "date","risk_label","composite_score",
                        "adj_confidence","reliability",
                        "temp_max","heat_index","humidity","aqi","rainfall",
                        "consec_hot_days",
                    ]].copy()
                    tbl.columns = [
                        "Date","Risk","Score","Confidence%","Reliability",
                        "Tmax°C","HeatIdx°C","Humidity%","AQI","Rain(mm)",
                        "HotDayStreak",
                    ]
                    tbl["Date"] = tbl["Date"].dt.strftime("%d %b %Y (%a)")

                    def colour_risk(val):
                        c = {"Low":"#d5f5e3","Moderate":"#fef9e7",
                             "High":"#fdebd0","Severe":"#fadbd8"}
                        return f"background-color:{c.get(val,'')}"

                    def colour_rel(val):
                        c = {"High":"#d5f5e3","Medium":"#fef9e7",
                             "Low":"#fdebd0","Indicative":"#f0f0f0"}
                        return f"background-color:{c.get(val,'')}"

                    styled = (tbl.style
                              .map(colour_risk, subset=["Risk"])
                              .map(colour_rel,  subset=["Reliability"]))
                    st.dataframe(styled, use_container_width=True)

                # Compound event warning
                compound_days = [d for d in fc_list if d["active_compounds"]]
                if compound_days:
                    st.warning(
                        f"⚠ **{city}** — compound risk events in forecast: " +
                        ", ".join([f"{d['date'][5:]} ({', '.join(d['active_compounds'])})"
                                   for d in compound_days])
                    )
                st.markdown("---")

            # ── Download all forecasts ────────────────────────────────
            all_rows = []
            for city, fc_list in all_fc.items():
                for d in fc_list:
                    all_rows.append({
                        "City": city, **{k: d[k] for k in
                            ["date","days_ahead","risk_label","composite_score",
                             "adj_confidence","reliability","temp_max","heat_index",
                             "aqi","rainfall","active_compounds"]}
                    })
            csv_out = pd.DataFrame(all_rows).to_csv(index=False)
            st.download_button("⬇ Download full forecast CSV",
                               csv_out, "forecast.csv", "text/csv")

        else:
            st.info("👆 Click **Run Forecast** to generate predictions.")

elif "Monthly" in page:
    st.title("📆 Monthly Outlook (Seasonal Prediction)")
    st.caption(
        "Seasonal prediction using historical climate normals + warming trend. "
        "Months 1–6 are reliable; months 7–12 are indicative trend only.")

    sc1, sc2, sc3 = st.columns([2, 2, 3])
    with sc1:
        s_cities  = st.multiselect("Cities", CITIES, default=CITIES, key="sc_cities")
    with sc2:
        s_months  = st.slider("Months ahead", 1, 12, 12, key="s_months")
    with sc3:
        st.markdown("<br>", unsafe_allow_html=True)
        s_btn = st.button("📆 Generate Monthly Outlook",
                          type="primary", use_container_width=True, key="s_btn")

    st.markdown(
        "> **Reliability:** "
        "🟦 **High** (months 1–2) &nbsp;|&nbsp; "
        "🟨 **Medium** (months 3–6) &nbsp;|&nbsp; "
        "🟧 **Low** (months 7–9) &nbsp;|&nbsp; "
        "⬜ **Indicative** (months 10–12)"
    )
    st.markdown("---")

    if s_btn:
        import importlib.util, sys as _sys
        # load step10
        spec10 = importlib.util.spec_from_file_location(
            "step10", os.path.join(os.path.dirname(__file__),
                                    "step10_seasonal_forecast.py"))
        if spec10 is None:
            st.error("step10_seasonal_forecast.py not found.")
            st.stop()
        mod10 = importlib.util.module_from_spec(spec10)
        _sys.modules["step10"] = mod10
        spec10.loader.exec_module(mod10)

        s_results = {}
        prog = st.progress(0, text="Generating outlook...")
        for idx, city in enumerate(s_cities):
            prog.progress((idx+0.3)/len(s_cities),
                          text=f"Forecasting {city}...")
            fc = mod10.forecast_city_outlook(city, n_months=s_months)
            s_results[city] = fc
            prog.progress((idx+1)/len(s_cities))
        prog.progress(1.0, text="Done!")
        st.session_state["s_results"] = s_results

    if "s_results" in st.session_state and st.session_state["s_results"]:
        s_all = st.session_state["s_results"]
        MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        RELY_COLORS = {"High":"#2ECC71","Medium":"#F1C40F",
                       "Low":"#E67E22","Indicative":"#BDC3C7"}

        for city, fc_list in s_all.items():
            if not fc_list: continue
            st.subheader(f"📍 {city}")

            # ── Month card strip ──────────────────────────────────
            n_show = min(len(fc_list), 12)
            card_cols = st.columns(n_show)
            for col, m in zip(card_cols, fc_list[:n_show]):
                risk = m["risk_label"]
                bg   = {"Low":"#d5f5e3","Moderate":"#fef9e7",
                        "High":"#fdebd0","Severe":"#fadbd8"}.get(risk,"#f0f0f0")
                rc   = RELY_COLORS.get(m["reliability"],"#ccc")
                col.markdown(f"""
                <div style="background:{bg};border:2px solid {rc};
                  border-radius:10px;padding:6px;text-align:center;font-size:0.75rem">
                  <div style="font-size:0.65rem;color:#888">{m['month_name']} {m['year']}</div>
                  <div style="font-size:1.5rem">{m['emoji']}</div>
                  <div style="font-weight:700;font-size:0.7rem">{risk}</div>
                  <div style="font-size:0.65rem">{m['adj_confidence']:.0f}%</div>
                  <div style="font-size:0.65rem">~{m['exp_tmax']}°C</div>
                </div>""", unsafe_allow_html=True)

            fc_df = pd.DataFrame(fc_list)
            fc_df["label"] = fc_df["month_name"] + " " + fc_df["year"].astype(str)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            avg_tmax = fc_df["exp_tmax"].mean()
            max_tmax = fc_df["exp_tmax"].max()
            total_high = fc_df["n_high_risk_days"].sum() if "n_high_risk_days" in fc_df.columns else 0
            trend_adj_max = fc_df["trend_adj_applied"].max() if "trend_adj_applied" in fc_df.columns else 0
            
            m1.metric("Average Tmax (Year)", f"{avg_tmax:.1f}°C")
            m2.metric("Peak Tmax Predicted", f"{max_tmax:.1f}°C")
            m3.metric("Total High-Risk Days", f"{total_high:.0f}")
            m4.metric("Climate Warming Adj.", f"+{trend_adj_max:.2f}°C")
            st.markdown("<br>", unsafe_allow_html=True)

            ch1, ch2 = st.columns(2)

            with ch1:
                # Risk level timeline bar chart
                color_map = {0:"#2ECC71",1:"#F1C40F",2:"#E67E22",3:"#E74C3C"}
                fig = go.Figure(go.Bar(
                    x=fc_df["label"],
                    y=fc_df["adj_confidence"],
                    marker_color=[color_map[r] for r in fc_df["risk_level"]],
                    text=fc_df["risk_label"],
                    textposition="inside",
                    textfont=dict(size=9),
                ))
                fig.update_layout(
                    title="Monthly Risk Level + Confidence",
                    yaxis_title="Adjusted Confidence %",
                    height=300, margin=dict(l=0,r=0,t=40,b=60),
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig, use_container_width=True)

            with ch2:
                # Expected temperature + high-risk days
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(
                    x=fc_df["label"], y=fc_df["exp_tmax"],
                    name="Expected Tmax", mode="lines+markers",
                    line=dict(color="#E74C3C", width=2),
                ), secondary_y=False)
                if "n_high_risk_days" in fc_df.columns:
                    fig2.add_trace(go.Bar(
                        x=fc_df["label"],
                        y=fc_df["n_high_risk_days"].fillna(0),
                        name="Est. high-risk days",
                        marker_color="#E67E22", opacity=0.4,
                    ), secondary_y=True)
                if "exp_aqi" in fc_df.columns:
                    fig2.add_trace(go.Scatter(
                        x=fc_df["label"], y=fc_df["exp_aqi"],
                        name="Expected AQI", mode="lines+markers",
                        line=dict(color="#8E44AD", width=2, dash="dot"),
                    ), secondary_y=True)
                fig2.update_yaxes(title_text="Tmax °C",    secondary_y=False)
                fig2.update_yaxes(title_text="Days / AQI", secondary_y=True)
                fig2.update_layout(
                    title="Expected Temp, AQI & High-Risk Days",
                    height=300, margin=dict(l=0,r=0,t=40,b=60),
                    xaxis_tickangle=-45,
                    legend=dict(orientation="h", y=-0.4),
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Severe + compound probability chart
            if "p_severe_day" in fc_df.columns and fc_df["p_severe_day"].notna().any():
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=fc_df["label"],
                    y=fc_df["p_severe_day"].fillna(0),
                    name="P(Severe day)", fill="tozeroy",
                    line=dict(color="#E74C3C"),
                    fillcolor = 'rgba(241, 196, 15, 0.27)',
                ))
                if "p_compound_event" in fc_df.columns:
                    fig3.add_trace(go.Scatter(
                        x=fc_df["label"],
                        y=fc_df["p_compound_event"].fillna(0),
                        name="P(Compound event)", fill="tozeroy",
                        line=dict(color="#8E44AD"),
                        fillcolor = 'rgba(241, 196, 15, 0.27)',
                    ))
                fig3.update_layout(
                    title="Probability of Extreme Events Within Month",
                    yaxis_title="Probability %",
                    height=260, margin=dict(l=0,r=0,t=40,b=60),
                    xaxis_tickangle=-45,
                    legend=dict(orientation="h", y=-0.5),
                )
                st.plotly_chart(fig3, use_container_width=True)

            # ── Seasonal Compound Risk Breakdown ──────────────────────────
            st.markdown("**🔥 Seasonal Compound Risk Analysis**")
            st.caption(
                "Estimated compound risks per month based on climate normals. "
                "Heat+Drought is included here — appropriate at the seasonal scale "
                "(slow-moving phenomenon). Daily prediction uses Heat+Air Pollution instead.")

            _hum_normal = {3:45,4:40,5:45,6:65,7:80,8:78,9:72,
                           10:55,11:50,12:50,1:55,2:50}
            _dry_months  = {3,4,5,11,12}

            comp_rows = []
            for m_s in fc_list:
                month_num  = m_s["month"]
                month_lbl  = m_s["month_name"] + " " + str(m_s["year"])
                exp_t      = m_s["exp_tmax"]
                exp_aqi    = m_s["exp_aqi"]
                exp_hum    = _hum_normal.get(month_num, 50)

                heat_aqi_flag      = int(exp_t >= 38 and exp_aqi >= 200)
                heat_ap_flag       = int(exp_t >= 38 and exp_aqi >= 150)
                heat_hum_flag      = int(exp_t >= 38 and exp_hum >= 60)
                heat_drought_flag  = int(exp_t >= 38 and month_num in _dry_months)
                triple_flag        = int(exp_t >= 38 and exp_aqi >= 150 and exp_hum >= 60)
                quad_flag          = int(heat_drought_flag and triple_flag)

                comp_rows.append({
                    "Month"           : month_lbl,
                    "Exp Tmax°C"      : exp_t,
                    "Exp AQI"         : int(exp_aqi),
                    "Est Humidity%"   : exp_hum,
                    "Heat+AQI Severe" : "✅" if heat_aqi_flag     else "—",
                    "Heat+Air Poll."  : "✅" if heat_ap_flag      else "—",
                    "Heat+Humidity"   : "✅" if heat_hum_flag     else "—",
                    "Heat+Drought"    : "✅" if heat_drought_flag else "—",
                    "Triple"          : "✅" if triple_flag       else "—",
                    "Quadruple"       : "✅" if quad_flag         else "—",
                })

            comp_df = pd.DataFrame(comp_rows)
            compound_flag_cols = [
                "Heat+AQI Severe","Heat+Air Poll.",
                "Heat+Humidity","Heat+Drought","Triple","Quadruple"
            ]
            counts_s = comp_df[compound_flag_cols].apply(
                lambda col: (col == "✅").astype(int))
            counts_s["Month"] = comp_df["Month"]
            melted_s = counts_s.melt(id_vars="Month",
                                     value_vars=compound_flag_cols,
                                     var_name="Compound Type",
                                     value_name="Active")
            melted_s = melted_s[melted_s["Active"] == 1]

            if not melted_s.empty:
                fig_cmp = px.bar(
                    melted_s, x="Month", y="Active",
                    color="Compound Type", barmode="stack",
                    title=f"{city} — Seasonal Compound Risks per Month",
                    color_discrete_sequence=[
                        "#E74C3C","#E67E22","#3498DB",
                        "#27AE60","#8E44AD","#2C3E50"
                    ],
                    labels={"Active": "Active Compound Types"},
                    height=300,
                )
                fig_cmp.update_layout(
                    margin=dict(l=0,r=0,t=40,b=80),
                    xaxis_tickangle=-45,
                    legend=dict(orientation="h", y=-0.65),
                    yaxis=dict(dtick=1, title=""),
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

            with st.expander(f"📋 {city} — Seasonal compound details table"):
                st.dataframe(comp_df.set_index("Month"), use_container_width=True)

            # Detailed table
            with st.expander(f"📋 {city} — Full monthly table"):
                tbl = fc_df[[
                    "label","risk_label","adj_confidence","reliability",
                    "exp_tmax","exp_aqi","n_high_risk_days",
                    "p_severe_day","p_compound_event","trend_adj_applied",
                ]].copy()
                tbl.columns = [
                    "Month","Risk","Confidence%","Reliability",
                    "Exp Tmax°C","Exp AQI","Est HighDays",
                    "P(Severe%)","P(Compound%)","Trend adj°C",
                ]
                def cr(v):
                    c = {"Low":"#d5f5e3","Moderate":"#fef9e7",
                         "High":"#fdebd0","Severe":"#fadbd8"}
                    return f"background-color:{c.get(v,'')}"
                st.dataframe(tbl.style.map(cr, subset=["Risk"]),
                             use_container_width=True)

            st.markdown("---")

        # Download
        all_rows = []
        for city, fc_list in s_all.items():
            for m in fc_list:
                all_rows.append({"City":city,**{k:m[k] for k in
                    ["year","month","month_name","risk_label","adj_confidence",
                     "reliability","exp_tmax","exp_aqi","n_high_risk_days",
                     "p_severe_day","p_compound_event"]}})
        csv_s = pd.DataFrame(all_rows).to_csv(index=False)
        st.download_button("⬇ Download monthly outlook CSV",
                           csv_s, "monthly_outlook.csv", "text/csv")

    else:
        st.info("👆 Click **Generate Monthly Outlook** to run the seasonal forecast.")
    #     st.markdown("""
    #     **How this works:**
    #     - Historical monthly statistics (2020–2025+) are used to compute
    #       climate normals for each month.
    #     - A long-term warming trend (°C/year) is estimated from your data.
    #     - For each forecast month: baseline = climate normal + trend adjustment.
    #     - Lag features (last month, last year same month) use real observed values.
    #     - The trained seasonal ML model converts these into risk level + probabilities.
    #     """)

elif "Overview" in page:
    st.title("🏠 Historical Risk Overview")

    if df_all is None:
        st.error("No data found. Run steps 1–4 first.")
        st.stop()

    if df_view is None or df_view.empty:
        st.warning("No data found in the selected date range.")
        st.stop()

    st.markdown("""
    <style>
    .historical-card {
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .historical-card:hover {
        transform: translateY(-2px);
    }
    .month-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 12px;
        color: #2C3E50;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .main-temp {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 12px;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.95rem;
        padding: 4px 8px;
        background: rgba(255,255,255,0.4);
        border-radius: 6px;
        margin-bottom: 4px;
        color: #34495E;
    }
    .metric-val {
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)
    
    overview_mode = st.radio("Select View:", ["🏙️ Places", "📊 Analysis", "⚖️ Comparison"], horizontal=True)
    st.markdown("---")
    
    month_names = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June", 
                   7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"}

    if overview_mode == "🏙️ Places":
        cities_to_display = sorted(df_view["city"].unique())

        if len(cities_to_display) > 1:
            tabs = st.tabs([f"📍 {c}" for c in cities_to_display])
        else:
            tabs = [st.container()]

        for idx, city in enumerate(cities_to_display):
            with tabs[idx]:
                if len(cities_to_display) == 1:
                    st.markdown(f"### 📍 {city} — Average Monthly Climate Conditions (2017–2025)")
                else:
                    st.markdown("### Average Monthly Climate Conditions (2017–2025)")
                
                st.caption(f"Average historical values for each month across all years in the dataset, specific to **{city}**.")
                
                # Filter and compute averages by month for the current city
                city_df = df_view[df_view["city"] == city]
                monthly_stats = city_df.groupby("month")[["temp_max", "aqi", "rainfall", "humidity"]].mean().reset_index()

                st.markdown("<br>", unsafe_allow_html=True)

                # Display in a 4x3 grid
                for row in range(3):
                    cols = st.columns(4)
                    for col_idx in range(4):
                        month_id = row * 4 + col_idx + 1
                        col = cols[col_idx]
                        
                        stat = monthly_stats[monthly_stats["month"] == month_id]
                        if not stat.empty:
                            t = stat["temp_max"].values[0]
                            a = stat["aqi"].values[0]
                            r = stat["rainfall"].values[0]
                            h = stat["humidity"].values[0]
                            
                            # Dynamic styling based on heat
                            if t >= 38:
                                bg = "linear-gradient(135deg, #fadbd8, #f5b7b1)"
                                border = "#E74C3C"
                                color_temp = "#C0392B"
                            elif t >= 33:
                                bg = "linear-gradient(135deg, #fdebd0, #fad7a1)"
                                border = "#E67E22"
                                color_temp = "#D35400"
                            elif t >= 25:
                                bg = "linear-gradient(135deg, #fef9e7, #fcf3cf)"
                                border = "#F1C40F"
                                color_temp = "#B7950B"
                            else:
                                bg = "linear-gradient(135deg, #d5f5e3, #abebc6)"
                                border = "#2ECC71"
                                color_temp = "#1E8449"
                            
                            col.markdown(f"""
                            <div class="historical-card" style="background: {bg}; border-top: 4px solid {border};">
                                <div class="month-title">{month_names[month_id]}</div>
                                <div class="main-temp" style="color: {color_temp};">🌡️ {t:.1f}°C</div>
                                <div class="metric-row">
                                    <span>💨 AQI</span>
                                    <span class="metric-val">{a:.0f}</span>
                                </div>
                                <div class="metric-row">
                                    <span>💧 Humidity</span>
                                    <span class="metric-val">{h:.1f}%</span>
                                </div>
                                <div class="metric-row">
                                    <span>🌧️ Rain</span>
                                    <span class="metric-val">{r:.1f} mm</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            col.markdown(f"""
                            <div class="historical-card" style="background: #f8f9fa; border-top: 4px solid #ced4da;">
                                <div class="month-title" style="color: #6c757d;">{month_names[month_id]}</div>
                                <div style="font-size:1.2rem; color:#adb5bd; margin: 30px 0;">No Data</div>
                            </div>
                            """, unsafe_allow_html=True)
                st.markdown("---")

    elif overview_mode == "📊 Analysis":
        st.markdown("### Monthly Climate Trends (City-wise Analysis)")
        st.caption("Trends of average Temp, AQI, Humidity, and Rainfall over the year for each city.")
        
        cities_to_display = sorted(df_view["city"].unique())
        
        if len(cities_to_display) > 1:
            city_tabs = st.tabs([f"📍 {c}" for c in cities_to_display])
        else:
            city_tabs = [st.container()]
            
        for idx, city in enumerate(cities_to_display):
            with city_tabs[idx]:
                city_df = df_view[df_view["city"] == city]
                m_stats = city_df.groupby("month")[["temp_max", "aqi", "rainfall", "humidity"]].mean().reset_index()
                m_stats["Month"] = m_stats["month"].map(month_names)
                
                c1, c2 = st.columns(2)
                with c1:
                    fig_t = px.line(m_stats, x="Month", y="temp_max", markers=True, 
                                    title=f"Avg Max Temperature - {city}", 
                                    labels={"temp_max": "Temp (°C)"}, color_discrete_sequence=["#E74C3C"])
                    st.plotly_chart(fig_t, use_container_width=True)
                    
                    fig_h = px.bar(m_stats, x="Month", y="humidity", 
                                   title=f"Avg Humidity - {city}", 
                                   labels={"humidity": "Humidity (%)"}, color_discrete_sequence=["#3498DB"])
                    st.plotly_chart(fig_h, use_container_width=True)
                with c2:
                    fig_a = px.line(m_stats, x="Month", y="aqi", markers=True, 
                                    title=f"Avg AQI - {city}", 
                                    labels={"aqi": "AQI"}, color_discrete_sequence=["#8E44AD"])
                    st.plotly_chart(fig_a, use_container_width=True)
                    
                    fig_r = px.bar(m_stats, x="Month", y="rainfall", 
                                   title=f"Avg Rainfall - {city}", 
                                   labels={"rainfall": "Rainfall (mm)"}, color_discrete_sequence=["#2ECC71"])
                    st.plotly_chart(fig_r, use_container_width=True)

    elif overview_mode == "⚖️ Comparison":
        st.markdown("### City Comparison")
        st.caption("Compare average climatic conditions across different cities.")
        
        # Use df_all to guarantee we compare all cities regardless of the single city filter
        m_stats_all = df_all.groupby(["city", "month"])[["temp_max", "aqi", "rainfall", "humidity"]].mean().reset_index()
        m_stats_all["Month"] = m_stats_all["month"].map(month_names)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_tc = px.line(m_stats_all, x="Month", y="temp_max", color="city", markers=True,
                             title="Temperature Comparison", labels={"temp_max": "Temp (°C)"},
                             color_discrete_map=CITY_COLORS)
            st.plotly_chart(fig_tc, use_container_width=True)
            
            fig_hc = px.line(m_stats_all, x="Month", y="humidity", color="city", markers=True,
                             title="Humidity Comparison", labels={"humidity": "Humidity (%)"},
                             color_discrete_map=CITY_COLORS)
            st.plotly_chart(fig_hc, use_container_width=True)
            
        with c2:
            fig_ac = px.line(m_stats_all, x="Month", y="aqi", color="city", markers=True,
                             title="AQI Comparison", labels={"aqi": "AQI"},
                             color_discrete_map=CITY_COLORS)
            st.plotly_chart(fig_ac, use_container_width=True)
            
            fig_rc = px.bar(m_stats_all, x="Month", y="rainfall", color="city", barmode="group",
                            title="Rainfall Comparison", labels={"rainfall": "Rainfall (mm)"},
                            color_discrete_map=CITY_COLORS)
            st.plotly_chart(fig_rc, use_container_width=True)

        st.markdown("---")
        st.markdown("### Seasonal Patterns")
        
        if "season" in df_all.columns:
            seasonal = df_all.groupby(["city","season"])["composite_score"].mean().reset_index()
            fig_season = px.bar(seasonal,x="season",y="composite_score",color="city",
                         barmode="group",color_discrete_map=CITY_COLORS,
                         title="Average Risk Score by Season",
                         category_orders={"season":["Winter","Pre-Monsoon","Monsoon","Post-Monsoon"]})
            st.plotly_chart(fig_season,use_container_width=True)

        # Monthly trend lines for composite score
        monthly_comp = df_all.groupby(["city","month"])["composite_score"].mean().reset_index()
        fig_month = px.line(monthly_comp,x="month",y="composite_score",color="city",
                       color_discrete_map=CITY_COLORS,markers=True,
                       title="Average Risk Score by Month",
                       labels={"month":"Month","composite_score":"Avg Score"})
        st.plotly_chart(fig_month,use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# PAGE 3: TRENDS & ANALYSIS
# ════════════════════════════════════════════════════════════════════════
elif "Trends" in page:
    st.title("📈 Trends & Analysis")

    if df_view is None or len(df_view)==0:
        st.warning("No data for selected filters.")
        st.stop()

    tab1,tab2,tab3,tab4 = st.tabs([
        "Risk Timeline","Temperature & Heat Index",
        "Compound Events","Monthly Heatmap",
    ])

    with tab1:
        city_t = st.selectbox("City",CITIES,key="tt1")
        sub    = df_all[df_all["city"]==city_t] if df_all is not None else df_view

        fig = make_subplots(rows=3,cols=1,shared_xaxes=True,
                            subplot_titles=("Risk Score (coloured by level)",
                                            "Temp Max (°C)","AQI"),
                            vertical_spacing=0.07)
        for lvl,lbl in RISK_LABELS.items():
            mask = sub["risk_level"]==lvl
            fig.add_trace(go.Scatter(x=sub[mask]["date"],y=sub[mask]["composite_score"],
                mode="markers",marker=dict(color=RISK_COLORS[lvl],size=2.5),
                name=lbl),row=1,col=1)
        fig.add_trace(go.Scatter(x=sub["date"],y=sub["temp_max"],
            line=dict(color="#E74C3C",width=1),name="Tmax"),row=2,col=1)
        fig.add_hline(y=40,line_dash="dot",line_color="darkred",row=2,col=1)
        fig.add_trace(go.Scatter(x=sub["date"],y=sub["aqi"],
            line=dict(color="#8E44AD",width=1),name="AQI"),row=3,col=1)
        fig.add_hline(y=200,line_dash="dot",line_color="purple",row=3,col=1)
        fig.update_layout(height=580,title=f"{city_t} — Full Timeline")
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        fig = px.box(df_view,x="month",y="heat_index",color="city",
                     color_discrete_map=CITY_COLORS,
                     title="Monthly Heat Index Distribution")
        st.plotly_chart(fig,use_container_width=True)

        sample = df_view.sample(min(3000,len(df_view)))
        fig2   = px.scatter(sample,x="temp_max",y="heat_index",
                            color="risk_label",size="aqi",size_max=10,
                            color_discrete_map={"Low":"#2ECC71","Moderate":"#F1C40F",
                                                "High":"#E67E22","Severe":"#E74C3C"},
                            title="Heat Index vs Temp (bubble size = AQI)")
        st.plotly_chart(fig2,use_container_width=True)

    with tab3:
        # Use available compound columns (daily-appropriate only)
        avail_cols = [c for c in ["compound_heat_aqi","compound_heat_air_pollution",
                                   "compound_heat_humidity","triple_compound"]
                      if c in df_view.columns]
        nice_map = {
            "compound_heat_aqi"          : "Heat+AQI (Severe ≥200)",
            "compound_heat_air_pollution": "Heat+Air Pollution (≥150)",
            "compound_heat_humidity"     : "Heat+Humidity",
            "triple_compound"            : "Triple (Heat+AQI+Humidity)",
        }
        nice = [nice_map[c] for c in avail_cols]
        grp   = df_view.groupby(["city","month"])[avail_cols].sum().reset_index()
        grp.columns = ["city","month"]+nice
        melted = grp.melt(id_vars=["city","month"],value_vars=nice,
                          var_name="Type",value_name="Days")
        fig = px.bar(melted,x="month",y="Days",color="Type",facet_col="city",
                     barmode="group",title="Monthly Compound Event Frequency (Daily Risks)",
                     color_discrete_sequence=["#E74C3C","#E67E22","#3498DB","#8E44AD"])
        st.plotly_chart(fig,use_container_width=True)

        # Compound event timeline
        tc_col = "triple_compound" if "triple_compound" in df_view.columns else avail_cols[-1]
        fig2 = px.area(df_view,x="date",y=tc_col,color="city",
                       color_discrete_map=CITY_COLORS,
                       title="Triple Compound Events Over Time (Heat + AQI ≥150 + Humidity ≥60%)")
        st.plotly_chart(fig2,use_container_width=True)

    with tab4:
        city_h = st.selectbox("City",CITIES,key="hmap")
        sub_h  = (df_all[df_all["city"]==city_h].copy()
                  if df_all is not None else df_view)
        sub_h["week"] = sub_h["date"].dt.isocalendar().week.astype(int)
        pivot = sub_h.pivot_table(index="month",columns="week",
                                   values="composite_score",aggfunc="mean")
        fig = px.imshow(pivot,color_continuous_scale="RdYlGn_r",
                        title=f"{city_h} — Avg Risk Score (Month × Week of Year)",
                        labels={"x":"Week","y":"Month","color":"Score"})
        st.plotly_chart(fig,use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# PAGE 4: MANUAL PREDICTION
# ════════════════════════════════════════════════════════════════════════
elif "Manual" in page:
    st.title("🔍 Manual Prediction")
    st.caption("Choose how you want to supply the weather inputs — then click Predict.")

    # ── City picker ───────────────────────────────────────────────────────
    city_m = st.selectbox("City", CITIES, key="man_city")

    st.markdown("---")

    # ── Mode toggle ───────────────────────────────────────────────────────
    mode = st.radio(
        "**Input Mode**",
        options=["📅 Pick a Date from History", "🎛️ Set Values Manually"],
        horizontal=True,
        key="man_mode",
    )
    st.markdown("")

    # ─────────────────────────────────────────────────────────────────────
    # MODE A: DATE LOOKUP  — loads values from processed CSV for that date
    # ─────────────────────────────────────────────────────────────────────
    if mode == "📅 Pick a Date from History":
        st.markdown("##### Select a date — observed values are loaded automatically from the processed CSV.")
        pred_date = st.date_input("Date", value=date.today(), key="man_date_a")

        _city_hist = load_city_history(city_m)
        _obs_from_csv = None
        if _city_hist is not None:
            _city_hist["date"] = pd.to_datetime(_city_hist["date"], errors="coerce")
            _match = _city_hist[_city_hist["date"].dt.date == pred_date]
            if not _match.empty:
                _row = _match.iloc[-1]
                _obs_from_csv = {
                    "date"    : str(pred_date),
                    "temp_max": float(_row.get("temp_max", 38)),
                    "temp_min": float(_row.get("temp_min", 24)),
                    "humidity": float(_row.get("humidity", 45)),
                    "wind"    : float(_row.get("wind",    10)),
                    "rainfall": float(_row.get("rainfall",  0)),
                    "aqi"     : float(_row.get("aqi",     120)),
                }

        if _obs_from_csv:
            st.success(f"✅ **{pred_date}** found in processed data for **{city_m}**", icon="📂")
            o = _obs_from_csv
            # Display as clean read-only metrics — no sliders needed
            mc = st.columns(6)
            for col_ui, lbl, v in zip(mc,
                ["Tmax °C","Tmin °C","Humidity %","Wind km/h","Rainfall mm","AQI"],
                [o["temp_max"],o["temp_min"],o["humidity"],o["wind"],o["rainfall"],o["aqi"]]):
                col_ui.metric(lbl, f"{v:.1f}")
            t_max    = o["temp_max"]
            t_min    = o["temp_min"]
            humidity = int(o["humidity"])
            wind     = o["wind"]
            rainfall = o["rainfall"]
            aqi_m    = int(o["aqi"])
            obs_ready = True
        else:
            if pred_date < date.today():
                st.warning(
                    f"⚠️ **{pred_date}** not found in the processed CSV for **{city_m}**. "
                    "Try another date, or switch to **Set Values Manually**.", icon="🗓️")
            else:
                st.info("Today's / future dates are not in historical CSV. "
                        "Switch to **Set Values Manually** mode.", icon="✍️")
            obs_ready = False
            t_max = t_min = humidity = wind = rainfall = aqi_m = None

    # ─────────────────────────────────────────────────────────────────────
    # MODE B: MANUAL SLIDERS — full control over every weather parameter
    # ─────────────────────────────────────────────────────────────────────
    else:
        st.markdown("##### Adjust the sliders to any custom weather condition.")
        pred_date = st.date_input("Date (for logging)", value=date.today(), key="man_date_b")
        c1, c2 = st.columns(2)
        with c1:
            t_max    = st.slider("Temperature Max (°C)", 20.0, 50.0, 38.0, 0.5, key="man_tmax")
            t_min    = st.slider("Temperature Min (°C)", 10.0, 35.0, 24.0, 0.5, key="man_tmin")
            humidity = st.slider("Humidity (%)", 5, 100, 45, key="man_hum")
        with c2:
            wind     = st.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0, 0.5, key="man_wind")
            rainfall = st.slider("Rainfall (mm)", 0.0, 100.0, 0.0, 0.5, key="man_rain")
            aqi_m    = st.slider("AQI (India CPCB)", 0, 500, 120, key="man_aqi")
        obs_ready = True

    # ─────────────────────────────────────────────────────────────────────
    # PREDICT — shared for both modes
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("")
    if st.button("🔮 Predict Risk", type="primary", key="man_predict_btn",
                 disabled=not obs_ready):
        obs = {"date": str(pred_date), "temp_max": t_max, "temp_min": t_min,
               "humidity": humidity, "wind": wind, "rainfall": rainfall, "aqi": aqi_m}
        result, err = predict_for_city(city_m, obs, pred_date=pred_date)

        if err:
            st.error(err)
        else:
            risk  = result["risk_label"]
            emoji = result["emoji"]
            rc    = RISK_COLORS[result["risk_level"]]
            src_lbl = "from CSV" if mode.startswith("📅") else "manual input"

            st.markdown(f"""
            <div class="risk-card risk-{risk.upper()}" style="max-width:480px;margin:auto">
              <div style="font-size:1.2rem;font-weight:700">{city_m} — {pred_date}
                <span style="font-size:0.75rem;opacity:0.7"> ({src_lbl})</span></div>
              <div style="font-size:3.5rem">{emoji}</div>
              <div class="metric-big">{risk}</div>
              <div class="metric-sub">Score: {result['composite_score']} / 100
               &nbsp;|&nbsp; Confidence: {result['confidence']}%</div>
              <hr>
              <div>🥵 Heat Index: <b>{result['heat_index']}°C</b></div>
              <div>🔥 Consecutive hot days: <b>{result['consec_hot_days']}</b></div>
            </div>""", unsafe_allow_html=True)

            if result["active_compounds"]:
                st.warning("⚠ Compound risks: " + ", ".join(result["active_compounds"]))
            st.info(result["advisory"])

            r1, r2 = st.columns(2)
            with r1:
                probs = result["probabilities"]
                fig = go.Figure(go.Bar(
                    x=list(probs.values()), y=list(probs.keys()),
                    orientation="h",
                    marker_color=[RISK_COLORS[i] for i in range(4)],
                    text=[f"{v:.1f}%" for v in probs.values()],
                    textposition="outside",
                ))
                fig.update_layout(title="Class probabilities", xaxis_range=[0, 110],
                                  height=250, margin=dict(l=0,r=0,t=40,b=0),
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with r2:
                hi_v  = _heat_index(t_max, humidity)
                st_t  = 0 if t_max<35 else 25 if t_max<38 else 50 if t_max<40 else 75 if t_max<45 else 100
                st_h  = 0 if hi_v<27 else 25 if hi_v<32 else 50 if hi_v<41 else 75 if hi_v<54 else 100
                st_a  = 0 if aqi_m<=50 else 15 if aqi_m<=100 else 40 if aqi_m<=200 else 60 if aqi_m<=300 else 80 if aqi_m<=400 else 100
                st_ap = min(100, int(t_max>=38 and aqi_m>=150)*50 +
                                 int(t_max>=38 and aqi_m>=200)*50)
                st_c  = min(100, int(t_max>=38 and aqi_m>=200)*30 +
                                 int(t_max>=38 and aqi_m>=150)*25 +
                                 int(t_max>=38 and humidity>=60)*20)
                fig = go.Figure(go.Scatterpolar(
                    r=[st_t, st_h, st_a, st_ap, st_c],
                    theta=["Temperature", "Heat Index", "AQI", "Air Pollution", "Compound"],
                    fill="toself",
                    fillcolor='rgba(241, 196, 15, 0.27)',
                    line_color=rc,
                ))
                fig.update_layout(title="Pillar scores",
                                  polar=dict(radialaxis=dict(range=[0, 100])),
                                  height=250, margin=dict(l=0,r=0,t=40,b=0),
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# PAGE 6: PREDICTION LOG
# ════════════════════════════════════════════════════════════════════════
elif "Log" in page:
    st.title("📋 Prediction Log")
    st.caption("All live predictions made from the dashboard.")

    log_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith(".json")],
                       reverse=True)
    if not log_files:
        st.info("No predictions logged yet. Run a live prediction first.")
        st.stop()

    # Load all logs
    records = []
    for fname in log_files:
        try:
            with open(os.path.join(LOG_DIR,fname)) as f:
                r = json.load(f)
            records.append({
                "Date"       : r.get("date",""),
                "City"       : r.get("city",""),
                "Risk"       : r.get("risk_label",""),
                "Score"      : r.get("composite_score",""),
                "Confidence" : f"{r.get('confidence',0):.1f}%",
                "Tmax"       : r.get("raw_obs",{}).get("temp_max",""),
                "AQI"        : r.get("raw_obs",{}).get("aqi",""),
                "Compounds"  : ", ".join(r.get("active_compounds",[]) or ["-"]),
                "AQI Source" : r.get("aqi_source",""),
            })
        except Exception:
            continue

    log_df = pd.DataFrame(records)

    # Filter
    fc1,fc2 = st.columns(2)
    city_f = fc1.selectbox("Filter city",["All"]+CITIES,key="logcity")
    risk_f = fc2.selectbox("Filter risk",["All","Low","Moderate","High","Severe"],key="logrisk")
    if city_f != "All": log_df = log_df[log_df["City"]==city_f]
    if risk_f != "All": log_df = log_df[log_df["Risk"]==risk_f]

    # Colour rows
    def colour_risk(val):
        c = {"Low":"#d5f5e3","Moderate":"#fef9e7","High":"#fdebd0","Severe":"#fadbd8"}
        return f"background-color:{c.get(val,'')}"

    # styled = log_df.style.applymap(colour_risk, subset=["Risk"])
    styled = log_df.style.map(colour_risk, subset=["Risk"])
    st.dataframe(styled, use_container_width=True, height=420)

    # Mini trend of logged predictions
    if len(log_df) > 1:
        log_df["Score_num"] = pd.to_numeric(log_df["Score"],errors="coerce")
        fig = px.line(log_df.sort_values("Date"),x="Date",y="Score_num",
                      color="City",color_discrete_map=CITY_COLORS,markers=True,
                      title="Composite Score — Live Prediction History")
        fig.add_hline(y=50,line_dash="dot",line_color="orange")
        fig.add_hline(y=75,line_dash="dot",line_color="red")
        st.plotly_chart(fig,use_container_width=True)

    # Download
    csv = log_df.to_csv(index=False)
    st.download_button("⬇ Download log as CSV",csv,
                       "prediction_log.csv","text/csv")