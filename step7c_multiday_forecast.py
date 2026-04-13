"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Step 7c: Multi-Day Forecast (1–16 days ahead)
========================================================================
HOW IT WORKS:
─────────────────────────────────────────────────────────────────────
  Day 1 (tomorrow):
    Features built from real historical CSV (tail 40) + API forecast.
    Highest accuracy — lag features are real observed values.

  Day 2–5:
    Features built from real history + previous FORECAST days chained.
    E.g. lag1 for Day 3 = Day 2's forecasted temp_max.
    Moderate accuracy — API forecast degrades beyond 3 days.

  Day 6–16:
    Same chain, but weather forecast uncertainty is high.
    AQI forecast not available beyond Day 5 → uses seasonal average.
    Lower confidence — used for trend indication only.

CONFIDENCE DEGRADATION MODEL:
  confidence_displayed = model_confidence × temporal_decay_factor
  decay = 1.0 → 0.95 → 0.88 → 0.80 → 0.72 → 0.64 (Day 1→6)
  Beyond Day 6: capped at 50% (trend only)

APIs USED:
  Open-Meteo Forecast    : 16-day weather — free, no key
  Open-Meteo AQ Forecast : 5-day AQ      — free, no key
  Fallback AQI           : 7-day seasonal rolling average from CSV

OUTPUTS (per city, per day):
  date, risk_label, composite_score, confidence,
  temp_max, heat_index, aqi, active_compounds,
  forecast_reliability ("High" / "Medium" / "Low" / "Indicative")
========================================================================
"""

import os, json, pickle, requests, argparse
import pandas as pd
import numpy as np
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

PROC_DIR  = "data/processed"
MODEL_DIR = "models"
LOG_DIR   = "data/predictions"
os.makedirs(LOG_DIR, exist_ok=True)

CITIES = ["Delhi", "Hyderabad", "Nagpur"]
CITY_CONFIG = {
    "Delhi"     : {"lat": 28.6139, "lon": 77.2090},
    "Hyderabad" : {"lat": 17.3850, "lon": 78.4867},
    "Nagpur"    : {"lat": 21.1458, "lon": 79.0882},
}

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

RISK_LABELS  = {0:"Low", 1:"Moderate", 2:"High", 3:"Severe"}
RISK_EMOJIS  = {0:"🟢",  1:"🟡",       2:"🟠",   3:"🔴"}

# How much we trust the prediction N days ahead
# Based on NWS/ECMWF published skill scores for temperature forecasts
TEMPORAL_DECAY = {
    1: 1.00,   # Tomorrow      — highly reliable
    2: 0.95,   # Day after     — very reliable
    3: 0.88,   # Day 3         — reliable
    4: 0.80,   # Day 4         — moderately reliable
    5: 0.72,   # Day 5         — fair
    6: 0.64,   # Day 6         — use with caution
    7: 0.56,   # Day 7         — indicative only
}

def get_decay(days_ahead: int) -> float:
    return TEMPORAL_DECAY.get(days_ahead, 0.50)

def reliability_label(days_ahead: int) -> str:
    if days_ahead <= 2:  return "High"
    if days_ahead <= 4:  return "Medium"
    if days_ahead <= 7:  return "Low"
    return "Indicative"


# ════════════════════════════════════════════════════════════════════════
# SECTION A: BULK API FETCH (all forecast days in one call)
# ════════════════════════════════════════════════════════════════════════

def fetch_weather_forecast(lat: float, lon: float,
                            n_days: int = 16) -> pd.DataFrame | None:
    """
    Fetches up to 16 days of daily weather forecast from Open-Meteo.
    Returns a DataFrame indexed by date.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude"    : lat,
        "longitude"   : lon,
        "timezone"    : "Asia/Kolkata",
        "forecast_days": min(n_days, 16),
        "daily"       : [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "relative_humidity_2m_max",
        ],
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        d = r.json()["daily"]
        df = pd.DataFrame({
            "date"    : pd.to_datetime(d["time"]),
            "temp_max": d["temperature_2m_max"],
            "temp_min": d["temperature_2m_min"],
            "humidity": d["relative_humidity_2m_max"],
            "wind"    : d["wind_speed_10m_max"],
            "rainfall": d["precipitation_sum"],
        })
        df["rainfall"] = df["rainfall"].fillna(0)
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)
        df.set_index("date", inplace=True)
        return df
    except Exception as e:
        print(f"  [!] Weather forecast fetch failed: {e}")
        return None


def fetch_aqi_forecast(lat: float, lon: float,
                        n_days: int = 5) -> pd.Series | None:
    """
    Fetches up to 5 days of hourly AQ forecast from Open-Meteo AQ API.
    Returns a daily Series (India CPCB AQI) indexed by date.
    """
    CPCB_PM25 = [(0,30,0,50),(30.1,60,51,100),(60.1,90,101,200),
                 (90.1,120,201,300),(120.1,250,301,400),(250.1,500,401,500)]
    CPCB_PM10 = [(0,50,0,50),(51,100,51,100),(101,250,101,200),
                 (251,350,201,300),(351,430,301,400),(431,600,401,500)]
    def _c(v, bps):
        if not v or v <= 0: return 0.0
        for lo, hi, ilo, ihi in bps:
            if lo <= v <= hi:
                return round(((ihi-ilo)/(hi-lo))*(v-lo)+ilo, 1)
        return 500.0

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude"    : lat,
        "longitude"   : lon,
        "timezone"    : "Asia/Kolkata",
        "forecast_days": min(n_days, 7),
        "hourly"      : ["pm2_5", "pm10"],
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        hourly = r.json()["hourly"]
        times  = pd.to_datetime(hourly["time"])
        pm25   = hourly["pm2_5"]
        pm10   = hourly["pm10"]
        aqi_h  = pd.Series(
            [max(_c(p25 or 0, CPCB_PM25), _c(p10 or 0, CPCB_PM10))
             for p25, p10 in zip(pm25, pm10)],
            index=times
        )
        # Daily: pick afternoon reading (14:00)
        daily = {}
        for d, grp in aqi_h.groupby(aqi_h.index.date):
            peak = pd.Timestamp(d).replace(hour=14)
            best = min(grp.index, key=lambda t: abs((t-peak).total_seconds()))
            daily[d] = grp[best]
        series = pd.Series(daily)
        series.index = pd.to_datetime(series.index)
        return series
    except Exception as e:
        print(f"  [!] AQI forecast fetch failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════
# SECTION B: FEATURE ENGINEERING  (mirrors step1, self-contained)
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

def engineer_features(window_df: pd.DataFrame) -> pd.Series:
    """
    Given a DataFrame whose last row is the target forecast day
    (all prior rows are real or previously-chained forecast days),
    returns the engineered feature Series for the LAST row only.
    """
    df = window_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["month"]        = df["date"].dt.month
    df["day_of_year"]  = df["date"].dt.dayofyear
    df["is_summer"]    = df["month"].isin([4,5,6]).astype(int)
    df["temp_range"]   = df["temp_max"] - df["temp_min"]
    df["temp_mean"]    = (df["temp_max"] + df["temp_min"]) / 2
    df["heat_index"]   = df.apply(
        lambda r: _heat_index(r["temp_max"], r["humidity"]), axis=1)
    df["humidex"]      = df.apply(
        lambda r: _humidex(r["temp_max"], r["humidity"]), axis=1)
    df["feels_like_excess"] = df["heat_index"] - df["temp_max"]
    df["wind_heat_ratio"]   = df["temp_max"] / (df["wind"].replace(0, 0.1))

    for w in [3, 7]:
        df[f"temp_max_roll{w}"] = df["temp_max"].rolling(w, min_periods=1).mean()
        df[f"humidity_roll{w}"] = df["humidity"].rolling(w, min_periods=1).mean()
        df[f"aqi_roll{w}"]      = df["aqi"].rolling(w, min_periods=1).mean()
        df[f"rainfall_roll{w}"] = df["rainfall"].rolling(w, min_periods=1).sum()

    for lag in [1, 2]:
        df[f"temp_max_lag{lag}"] = df["temp_max"].shift(lag)
        df[f"aqi_lag{lag}"]      = df["aqi"].shift(lag)
        df[f"humidity_lag{lag}"] = df["humidity"].shift(lag)

    mm = df.groupby("month")["temp_max"].transform("mean")
    ms = df.groupby("month")["temp_max"].transform("std").replace(0, 1e-6)
    df["temp_departure"] = df["temp_max"] - mm
    df["temp_zscore"]    = (df["temp_max"] - mm) / ms
    df["aqi_departure"]  = df["aqi"] - df.groupby("month")["aqi"].transform("mean")

    df["drought_flag"]   = (df["rainfall"].rolling(7,min_periods=1).sum()<2).astype(int)
    hr = (df["rainfall"] < 1.0).astype(int)
    df["dry_days_streak"]= hr.groupby((hr != hr.shift()).cumsum()).cumcount()

    roll = df["rainfall"].rolling(30, min_periods=7)
    df["spi_30"] = (df["rainfall"]-roll.mean())/(roll.std().replace(0,1e-6))

    df["aqi_category"] = pd.cut(df["aqi"],
        bins=[0,50,100,200,300,400,500], labels=[1,2,3,4,5,6]).astype(float)

    hot = (df["temp_max"] >= 40.0).astype(int)
    df["consec_hot_days"] = hot.groupby((hot != hot.shift()).cumsum()).cumcount() + hot

    df["compound_heat_aqi"]      = ((df["temp_max"]>=38)&(df["aqi"]>=200)).astype(int)
    df["compound_heat_drought"]  = ((df["temp_max"]>=38)&(df["drought_flag"]==1)).astype(int)
    df["compound_heat_humidity"] = ((df["temp_max"]>=38)&(df["humidity"]>=60)).astype(int)
    df["triple_compound"]        = ((df["temp_max"]>=38)&(df["aqi"]>=150)&
                                    (df["drought_flag"]==1)).astype(int)
    return df.iloc[-1]


# ════════════════════════════════════════════════════════════════════════
# SECTION C: MODEL
# ════════════════════════════════════════════════════════════════════════

def load_model(city: str):
    for tag in [city, "all"]:
        cp = os.path.join(MODEL_DIR, f"classifier_{tag}.pkl")
        if os.path.exists(cp):
            with open(cp,"rb") as f: bundle = pickle.load(f)
            rp  = os.path.join(MODEL_DIR, f"regressor_{tag}.pkl")
            reg = pickle.load(open(rp,"rb")) if os.path.exists(rp) else None
            return bundle["model"], reg
    raise FileNotFoundError(f"No model for {city}. Run step4 first.")

def load_city_history(city: str, tail_n: int = 40) -> pd.DataFrame:
    raw_cols = ["date","temp_max","temp_min","humidity","wind","rainfall","aqi"]
    for fname in [f"labelled_{city}.csv", f"processed_{city}.csv"]:
        p = os.path.join(PROC_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["date"])
            avail = [c for c in raw_cols if c in df.columns]
            return df[avail].dropna().tail(tail_n).copy()
    raise FileNotFoundError(f"No history CSV for {city}. Run steps 1-2 first.")

def predict_single(window: pd.DataFrame,
                   clf, reg, days_ahead: int) -> dict:
    """
    Runs model on the last row of window.
    Applies temporal decay to reported confidence.
    """
    feat_row = engineer_features(window)
    X        = feat_row[FEATURE_COLS].fillna(0).values.reshape(1,-1)
    probas   = clf.predict_proba(X)[0]
    risk_lvl = int(np.argmax(probas))
    raw_conf = float(probas[risk_lvl])
    decay    = get_decay(days_ahead)
    adj_conf = round(raw_conf * decay * 100, 1)
    score    = float(reg.predict(X)[0]) if reg else None

    compounds = {
        "Heat+AQI"     : int(feat_row.get("compound_heat_aqi",0)),
        "Heat+Drought" : int(feat_row.get("compound_heat_drought",0)),
        "Heat+Humidity": int(feat_row.get("compound_heat_humidity",0)),
        "Triple"       : int(feat_row.get("triple_compound",0)),
    }

    return {
        "days_ahead"         : days_ahead,
        "date"               : str(window.iloc[-1]["date"].date()
                                   if hasattr(window.iloc[-1]["date"], "date")
                                   else window.iloc[-1]["date"]),
        "risk_level"         : risk_lvl,
        "risk_label"         : RISK_LABELS[risk_lvl],
        "emoji"              : RISK_EMOJIS[risk_lvl],
        "composite_score"    : round(score, 1) if score else None,
        "raw_confidence"     : round(raw_conf * 100, 1),
        "adj_confidence"     : adj_conf,
        "decay_factor"       : decay,
        "reliability"        : reliability_label(days_ahead),
        "probabilities"      : {RISK_LABELS[i]: round(float(p)*100,1)
                                 for i,p in enumerate(probas)},
        "temp_max"           : round(float(window.iloc[-1]["temp_max"]),1),
        "humidity"           : round(float(window.iloc[-1]["humidity"]),1),
        "aqi"                : round(float(window.iloc[-1]["aqi"]),0),
        "rainfall"           : round(float(window.iloc[-1]["rainfall"]),1),
        "heat_index"         : round(float(feat_row["heat_index"]),1),
        "consec_hot_days"    : int(feat_row["consec_hot_days"]),
        "active_compounds"   : [k for k,v in compounds.items() if v==1],
        "aqi_is_forecast"    : days_ahead <= 5,
    }


# ════════════════════════════════════════════════════════════════════════
# SECTION D: CHAINED MULTI-DAY FORECAST
# ════════════════════════════════════════════════════════════════════════

def forecast_city(city: str, n_days: int = 7) -> list[dict]:
    """
    Generates a chained forecast for a city for n_days ahead.

    Key design decision — the rolling window:
      The window always contains real historical rows PLUS any
      previously-forecasted days. This way lag and rolling features
      for Day 3 correctly use Day 2's forecasted values, not stale
      2025 values.

    Returns list of n_days dicts, one per day.
    """
    cfg = CITY_CONFIG[city]
    clf, reg = load_model(city)

    # ── Load real history (tail 40) ───────────────────────────────────
    history = load_city_history(city, tail_n=40)
    last_hist_date = pd.Timestamp(history["date"].max()).date()
    today          = date.today()

    print(f"\n  {city}: history tail = {last_hist_date}, "
          f"forecast starts = {today + timedelta(days=1)}")

    if last_hist_date < today - timedelta(days=7):
        print(f"  ⚠  History is {(today-last_hist_date).days} days stale. "
              f"Run step7b_backfill.py first for accurate results.")

    # ── Fetch all forecast days in one API call ───────────────────────
    print(f"  Fetching {n_days}-day weather forecast...", end=" ", flush=True)
    wx_forecast = fetch_weather_forecast(cfg["lat"], cfg["lon"], n_days+1)
    if wx_forecast is None:
        print("FAILED")
        return []
    print("OK")

    print(f"  Fetching AQI forecast (up to 5 days)...", end=" ", flush=True)
    aqi_forecast = fetch_aqi_forecast(cfg["lat"], cfg["lon"], n_days)
    if aqi_forecast is None:
        print("unavailable — will use seasonal avg")
    else:
        print("OK")

    # AQI fallback: 7-day seasonal rolling average from history
    aqi_fallback = float(history["aqi"].tail(7).mean())

    # ── Build rolling window — starts as real history ─────────────────
    # This is the key structure: raw_cols only — features computed fresh
    RAW_COLS = ["date","temp_max","temp_min","humidity","wind","rainfall","aqi"]
    window = history[[c for c in RAW_COLS if c in history.columns]].copy()
    # window["date"] = pd.to_datetime(window["date"])
    # ------------------------ ADDED --------------------
    window["date"] = pd.to_datetime(window["date"], format="mixed", errors="coerce")

    forecasts = []

    for day_idx in range(1, n_days + 1):
        target = today + timedelta(days=day_idx)
        ts     = pd.Timestamp(target)

        # ── Get weather for this day ──────────────────────────────────
        if ts in wx_forecast.index:
            wx = wx_forecast.loc[ts]
        else:
            # API didn't cover this far → extrapolate from last available
            wx = wx_forecast.iloc[-1]
            print(f"  [i] Day {day_idx} weather: using last available forecast row")

        # ── Get AQI for this day ──────────────────────────────────────
        if aqi_forecast is not None and ts in aqi_forecast.index:
            aqi_val = float(aqi_forecast.loc[ts])
            aqi_source = "forecast"
        else:
            # Beyond 5-day AQ forecast: use same-month historical average
            month = target.month
            hist_month_aqi = history.copy()
            # hist_month_aqi["month"] = pd.to_datetime(hist_month_aqi["date"]).dt.month
            # ------------------------ ADDED ---------------------------
            hist_month_aqi["month"] = pd.to_datetime(
                hist_month_aqi["date"], format="mixed", errors="coerce"
            ).dt.month
            month_avg = hist_month_aqi[
                hist_month_aqi["month"] == month]["aqi"].mean()
            aqi_val   = float(month_avg) if not np.isnan(month_avg) else aqi_fallback
            aqi_source = "seasonal_avg"

        # ── Append this day to the rolling window ────────────────────
        new_row = pd.DataFrame([{
            "date"    : ts,
            "temp_max": round(float(wx["temp_max"]), 1),
            "temp_min": round(float(wx["temp_min"]), 1),
            "humidity": round(float(wx["humidity"] or 50), 1),
            "wind"    : round(float(wx["wind"] or 0), 1),
            "rainfall": round(float(wx["rainfall"] or 0), 1),
            "aqi"     : round(aqi_val, 0),
        }])
        window = pd.concat([window, new_row], ignore_index=True)

        # ── Predict using the window (last row = today's target) ──────
        result = predict_single(window, clf, reg, days_ahead=day_idx)
        result["city"]       = city
        result["aqi_source"] = aqi_source
        forecasts.append(result)

        print(f"  Day {day_idx:2d} ({target}): "
              f"{result['emoji']} {result['risk_label']:10s} "
              f"score={result['composite_score']:5.1f}  "
              f"conf={result['adj_confidence']:5.1f}%  "
              f"[{result['reliability']}]  "
              f"AQI={result['aqi']:.0f}({'fc' if aqi_source=='forecast' else 'avg'})")

    return forecasts


# ════════════════════════════════════════════════════════════════════════
# SECTION E: SAVE & PRINT
# ════════════════════════════════════════════════════════════════════════

def print_forecast_table(forecasts: list[dict], city: str):
    print(f"\n  {'━'*70}")
    print(f"  {city.upper()} — {len(forecasts)}-Day Forecast")
    print(f"  {'━'*70}")
    print(f"  {'Date':12s} {'Risk':10s} {'Score':>6s} {'Conf':>7s} "
          f"{'Tmax':>6s} {'HI':>6s} {'AQI':>5s} {'Reliability':>12s}")
    print(f"  {'-'*68}")
    for f in forecasts:
        compounds = " ⚠" if f["active_compounds"] else ""
        print(f"  {f['date']:12s} "
              f"{f['emoji']} {f['risk_label']:8s} "
              f"{f['composite_score']:6.1f} "
              f"{f['adj_confidence']:6.1f}% "
              f"{f['temp_max']:6.1f} "
              f"{f['heat_index']:6.1f} "
              f"{f['aqi']:5.0f} "
              f"{'['+f['reliability']+']':>12s}"
              f"{compounds}")
    print(f"  {'━'*70}")

def save_forecast(forecasts: list[dict], city: str):
    p = os.path.join(LOG_DIR,
        f"forecast_{city}_{date.today()}.json")
    with open(p, "w") as f:
        json.dump(forecasts, f, indent=2, default=str)
    print(f"  [✓] Saved → {p}")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Heatwave Risk — Multi-Day Forecast")
    parser.add_argument("--city",   default="all",
        help="Delhi / Hyderabad / Nagpur / all")
    parser.add_argument("--days",   type=int, default=7,
        help="Number of days to forecast (1–16, default 7)")
    parser.add_argument("--no-save",action="store_true",
        help="Don't save JSON output")
    args = parser.parse_args()

    n   = max(1, min(16, args.days))
    cities = list(CITY_CONFIG.keys()) if args.city == "all" else [args.city]

    print("\n" + "="*60)
    print("  STEP 7c — MULTI-DAY HEATWAVE FORECAST")
    print(f"  Forecast days : {n}")
    print(f"  Cities        : {', '.join(cities)}")
    print(f"  Run date      : {date.today()}")
    print("="*60)
    print("""
  Reliability guide:
    [High]       Days 1–2 : weather + AQ forecast, real lag features
    [Medium]     Days 3–4 : chained forecast, good skill
    [Low]        Days 5–7 : chained, AQ uses seasonal avg
    [Indicative] Days 8+  : trend only, high uncertainty
    """)

    all_forecasts = {}
    for city in cities:
        if city not in CITY_CONFIG:
            print(f"[!] Unknown city: {city}")
            continue
        fc = forecast_city(city, n_days=n)
        if fc:
            print_forecast_table(fc, city)
            all_forecasts[city] = fc
            if not args.no_save:
                save_forecast(fc, city)

    print("\n[Done] Multi-day forecast complete.\n")