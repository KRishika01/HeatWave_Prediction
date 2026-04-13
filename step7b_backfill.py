"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Step 7b: Historical Gap Backfill
========================================================================
PROBLEM:
  Your CSV ends at ~April/May 2025.
  Today is April 2026.
  The rolling window features (lag1, lag2, roll7, consec_hot_days,
  spi_30, dry_days_streak ...) need the LAST 30-40 REAL DAYS to be
  accurate — not 30-40 days from a year ago.

SOLUTION:
  Open-Meteo provides a completely FREE historical weather archive
  going back to 1940 via a separate endpoint:
    https://archive-api.open-meteo.com/v1/archive

  For AQI, Open-Meteo's historical air quality archive is also free:
    https://air-quality-api.open-meteo.com/v1/air-quality
    (with start_date / end_date params for past data)

  This script:
    1. Reads each city's CSV and finds the last recorded date
    2. Fetches every missing day up to yesterday from Open-Meteo archive
    3. Converts PM2.5/PM10 → India CPCB AQI for each day
    4. Appends all missing rows to the CSV in chronological order
    5. Prints a clear summary of what was filled

RUN THIS ONCE before using step 7 for live prediction.
After this, step 7 appends daily automatically — no more gaps.

Usage:
  python step7b_backfill.py                  # fills all 3 cities
  python step7b_backfill.py --city Delhi     # fills one city only
  python step7b_backfill.py --dry-run        # shows what WOULD be filled
========================================================================
"""

import os, argparse, requests, time
import pandas as pd
import numpy as np
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

PROC_DIR = "data/processed"
CITIES   = ["Delhi", "Hyderabad", "Nagpur"]

CITY_CONFIG = {
    "Delhi"     : {"lat": 28.6139, "lon": 77.2090},
    "Hyderabad" : {"lat": 17.3850, "lon": 78.4867},
    "Nagpur"    : {"lat": 21.1458, "lon": 79.0882},
}

# ── CPCB AQI CONVERSION ───────────────────────────────────────────────────
CPCB_PM25 = [(0,30,0,50),(30.1,60,51,100),(60.1,90,101,200),
             (90.1,120,201,300),(120.1,250,301,400),(250.1,500,401,500)]
CPCB_PM10 = [(0,50,0,50),(51,100,51,100),(101,250,101,200),
             (251,350,201,300),(351,430,301,400),(431,600,401,500)]

def _cpcb(c, bps):
    if not c or c <= 0: return 0.0
    for lo, hi, ilo, ihi in bps:
        if lo <= c <= hi:
            return round(((ihi - ilo) / (hi - lo)) * (c - lo) + ilo, 1)
    return 500.0

def pm_to_india_aqi(pm25, pm10):
    return max(_cpcb(pm25 or 0, CPCB_PM25), _cpcb(pm10 or 0, CPCB_PM10))


# ════════════════════════════════════════════════════════════════════════
# API FETCHERS  (archive endpoints — different URLs from forecast)
# ════════════════════════════════════════════════════════════════════════

def fetch_weather_archive(lat: float, lon: float,
                           start: date, end: date) -> pd.DataFrame | None:
    """
    Open-Meteo Historical Weather Archive — free, no key.
    Returns a DataFrame with one row per day covering start..end inclusive.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude"   : lat,
        "longitude"  : lon,
        "start_date" : str(start),
        "end_date"   : str(end),
        "daily"      : [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "relative_humidity_2m_max",
        ],
        "timezone"   : "Asia/Kolkata",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        daily = r.json()["daily"]
        df = pd.DataFrame({
            "date"    : pd.to_datetime(daily["time"]),
            "temp_max": daily["temperature_2m_max"],
            "temp_min": daily["temperature_2m_min"],
            "humidity": daily["relative_humidity_2m_max"],
            "wind"    : daily["wind_speed_10m_max"],
            "rainfall": daily["precipitation_sum"],
        })
        # Fill any NaN in rainfall with 0
        df["rainfall"] = df["rainfall"].fillna(0)
        # Fill other NaNs with forward fill
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        print(f"    [!] Weather archive fetch failed: {e}")
        return None
    except Exception as e:
        print(f"    [!] Weather archive parse error: {e}")
        return None


def fetch_aqi_archive(lat: float, lon: float,
                       start: date, end: date) -> pd.Series | None:
    """
    Open-Meteo Air Quality API with date range — free, no key.
    Fetches hourly PM2.5 and PM10, aggregates to daily CPCB AQI
    using the afternoon peak reading (14:00 IST).

    NOTE: Historical AQ data availability varies by location.
    Falls back to linear interpolation from nearest available if missing.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude"   : lat,
        "longitude"  : lon,
        "start_date" : str(start),
        "end_date"   : str(end),
        "hourly"     : ["pm2_5", "pm10"],
        "timezone"   : "Asia/Kolkata",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        hourly = r.json()["hourly"]
        times  = pd.to_datetime(hourly["time"])
        pm25   = pd.Series(hourly["pm2_5"], index=times)
        pm10   = pd.Series(hourly["pm10"],  index=times)

        # Compute India CPCB AQI hourly, then pick daily afternoon value
        aqi_hourly = pd.Series(
            [pm_to_india_aqi(p25 or 0, p10 or 0)
             for p25, p10 in zip(pm25, pm10)],
            index=times
        )

        # Resample: for each day take the value closest to 14:00
        daily_aqi = {}
        for day_ts, group in aqi_hourly.groupby(aqi_hourly.index.date):
            if len(group) == 0:
                continue
            # Pick hour closest to 14:00
            afternoon = pd.Timestamp(day_ts).replace(hour=14)
            best_idx  = (group.index - afternoon).abs().argmin()
            daily_aqi[day_ts] = group.iloc[best_idx]

        aqi_series = pd.Series(daily_aqi)
        aqi_series.index = pd.to_datetime(aqi_series.index)
        # Fill zeros (genuinely clean air, not missing) keep as-is;
        # fill NaN with rolling mean
        aqi_series = aqi_series.interpolate(method="time").fillna(
            aqi_series.mean() if aqi_series.notna().any() else 100
        )
        return aqi_series

    except requests.exceptions.RequestException as e:
        print(f"    [!] AQ archive fetch failed: {e}")
        return None
    except Exception as e:
        print(f"    [!] AQ archive parse error: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════
# GAP DETECTION
# ════════════════════════════════════════════════════════════════════════

def find_csv_for_city(city: str) -> str | None:
    """Returns path to the most complete CSV for a city."""
    for fname in [f"labelled_{city}.csv", f"processed_{city}.csv"]:
        p = os.path.join(PROC_DIR, fname)
        if os.path.exists(p):
            return p
    return None


def get_last_date(city: str) -> date | None:
    """Returns the last date recorded in the city's CSV."""
    p = find_csv_for_city(city)
    if p is None:
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    if df.empty:
        return None
    return df["date"].max().date()


def get_missing_range(city: str) -> tuple[date, date] | None:
    """
    Returns (start_missing, end_missing) = day after last CSV date → yesterday.
    Returns None if already up to date.
    """
    last = get_last_date(city)
    if last is None:
        return None
    yesterday    = date.today() - timedelta(days=1)
    start_missing = last + timedelta(days=1)
    if start_missing > yesterday:
        return None   # already up to date
    return start_missing, yesterday


# ════════════════════════════════════════════════════════════════════════
# BACKFILL CORE
# ════════════════════════════════════════════════════════════════════════

def backfill_city(city: str, dry_run: bool = False) -> dict:
    """
    Fetches and appends all missing days for one city.
    Returns a summary dict.
    """
    print(f"\n  ── {city.upper()} {'─'*40}")

    # ── Check gap ─────────────────────────────────────────────────────
    csv_path = find_csv_for_city(city)
    if csv_path is None:
        print(f"  [!] No CSV found for {city}. Run steps 1–2 first.")
        return {"city": city, "status": "no_csv", "days_filled": 0}

    gap = get_missing_range(city)
    if gap is None:
        last = get_last_date(city)
        print(f"  [✓] Already up to date. Last date: {last}")
        return {"city": city, "status": "up_to_date",
                "last_date": str(last), "days_filled": 0}

    start_fill, end_fill = gap
    n_days = (end_fill - start_fill).days + 1

    print(f"  CSV last date  : {start_fill - timedelta(days=1)}")
    print(f"  Today          : {date.today()}")
    print(f"  Gap to fill    : {start_fill} → {end_fill}  ({n_days} days)")

    if dry_run:
        print(f"  [DRY RUN] Would fetch {n_days} days — skipping actual fetch.")
        return {"city": city, "status": "dry_run", "days_filled": n_days}

    cfg = CITY_CONFIG[city]

    # ── Fetch weather archive ─────────────────────────────────────────
    print(f"  Fetching weather archive from Open-Meteo...")
    weather_df = fetch_weather_archive(cfg["lat"], cfg["lon"],
                                        start_fill, end_fill)
    if weather_df is None or weather_df.empty:
        print(f"  [✗] Weather archive fetch failed for {city}.")
        return {"city": city, "status": "weather_failed", "days_filled": 0}
    print(f"  [✓] Weather: {len(weather_df)} days fetched")
    time.sleep(0.5)   # be polite to free API

    # ── Fetch AQI archive ─────────────────────────────────────────────
    print(f"  Fetching AQ archive from Open-Meteo...")
    aqi_series = fetch_aqi_archive(cfg["lat"], cfg["lon"],
                                    start_fill, end_fill)
    if aqi_series is None or aqi_series.empty:
        # AQ archive may not reach all the way back — use historical avg
        existing = pd.read_csv(csv_path, parse_dates=["date"])
        fallback_aqi = existing["aqi"].dropna().tail(30).mean()
        aqi_series = pd.Series(
            [fallback_aqi] * len(weather_df),
            index=weather_df["date"]
        )
        print(f"  [i] AQ archive unavailable — using historical avg AQI: {fallback_aqi:.0f}")
    else:
        print(f"  [✓] AQ: {len(aqi_series)} days fetched, "
              f"avg CPCB AQI = {aqi_series.mean():.0f}")
    time.sleep(0.5)

    # ── Merge weather + AQI ───────────────────────────────────────────
    weather_df = weather_df.set_index("date")
    aqi_aligned = aqi_series.reindex(weather_df.index).fillna(
        aqi_series.mean() if len(aqi_series) > 0 else 100)

    new_rows = weather_df.copy()
    new_rows["aqi"]  = aqi_aligned.values
    new_rows["city"] = city
    new_rows.reset_index(inplace=True)

    # Round sensibly
    new_rows["temp_max"]  = new_rows["temp_max"].round(1)
    new_rows["temp_min"]  = new_rows["temp_min"].round(1)
    new_rows["humidity"]  = new_rows["humidity"].round(1)
    new_rows["wind"]      = new_rows["wind"].round(1)
    new_rows["rainfall"]  = new_rows["rainfall"].round(1)
    new_rows["aqi"]       = new_rows["aqi"].round(0)

    # ══════════════════════════════════════════════════════════════════
    # WRITE TO BOTH CSVs — this is the critical fix.
    #
    # Root cause of the overwrite bug:
    #   step1 reads  data/raw/Hyderabad.csv        (unchanged → ends 2025)
    #   step1 writes data/processed/Hyderabad.csv  (overwrites with 2025 data)
    #   step2 reads  data/processed/Hyderabad.csv  (overwrites labelled file)
    #
    # Fix: also append new rows to data/raw/Hyderabad.csv so step1
    #      sees the full 2025-2026 dataset when it regenerates features.
    # ══════════════════════════════════════════════════════════════════

    # ── 1. UPDATE RAW CSV (what step1 reads) ─────────────────────────
    raw_path = os.path.join("data", "raw", f"{city}.csv")

    if os.path.exists(raw_path):
        raw_existing = pd.read_csv(raw_path)

        # Detect whether raw CSV uses uppercase or lowercase column names
        # cols_upper = "DATE" in raw_existing.columns

        # if cols_upper:
        #     raw_new = pd.DataFrame({
        #         "DATE"           : new_rows["date"].dt.strftime("%Y-%m-%d"),
        #         "TEMPERATURE_MAX": new_rows["temp_max"].values,
        #         "TEMPERATURE_MIN": new_rows["temp_min"].values,
        #         "HUMIDITY"       : new_rows["humidity"].values,
        #         "WIND"           : new_rows["wind"].values,
        #         "RAINFALL"       : new_rows["rainfall"].values,
        #         "AQI"            : new_rows["aqi"].values,
        #     })
        #     date_col = "DATE"
        # else:
        #     raw_new = pd.DataFrame({
        #         "date"    : new_rows["date"].dt.strftime("%Y-%m-%d"),
        #         "temp_max": new_rows["temp_max"].values,
        #         "temp_min": new_rows["temp_min"].values,
        #         "humidity": new_rows["humidity"].values,
        #         "wind"    : new_rows["wind"].values,
        #         "rainfall": new_rows["rainfall"].values,
        #         "aqi"     : new_rows["aqi"].values,
        #     })
        #     date_col = "date"
        ###--------------------------------ADDED-----------------------------------###
        # Get actual column names from raw CSV
        cols = raw_existing.columns.tolist()

        # Identify date column dynamically
        date_col = [c for c in cols if c.lower() == "date"][0]

        # Build new row using SAME column names as existing CSV
        raw_new = pd.DataFrame()

        for col in cols:
            if col.lower() == "date":
                raw_new[col] = new_rows["date"].dt.strftime("%Y-%m-%d")
            elif col.lower() in ["temp_max", "temperature_max"]:
                raw_new[col] = new_rows["temp_max"].values
            elif col.lower() in ["temp_min", "temperature_min"]:
                raw_new[col] = new_rows["temp_min"].values
            elif col.lower() == "humidity":
                raw_new[col] = new_rows["humidity"].values
            elif col.lower() == "wind":
                raw_new[col] = new_rows["wind"].values
            elif col.lower() == "rainfall":
                raw_new[col] = new_rows["rainfall"].values
            elif col.lower() == "aqi":
                raw_new[col] = new_rows["aqi"].values

        # Remove overlapping dates then append
        raw_existing[date_col] = pd.to_datetime(raw_existing[date_col])
        raw_new[date_col]      = pd.to_datetime(raw_new[date_col])
        raw_existing = raw_existing[
            ~raw_existing[date_col].isin(raw_new[date_col])]
        raw_updated = pd.concat([raw_existing, raw_new], ignore_index=True)
        raw_updated.sort_values(date_col, inplace=True)
        raw_updated.reset_index(drop=True, inplace=True)
        raw_updated.to_csv(raw_path, index=False)
        print(f"  [✓] Raw CSV updated      → {raw_path}")
        print(f"      Covers: {raw_updated[date_col].iloc[0].date()} "
              f"→ {raw_updated[date_col].iloc[-1].date()}")
        print(f"      Now safe to re-run steps 1–2 without losing backfill.")
    else:
        print(f"  [!] Raw CSV not found at {raw_path}")
        print(f"      Steps 1-2 will overwrite the labelled file with old data.")
        print(f"      Place your original raw CSV at {raw_path} and re-run.")

    # ── 2. ALSO UPDATE PROCESSED/LABELLED CSV (for immediate use) ────
    # Lets step7 live prediction work right now without needing
    # to re-run steps 1-2 first.
    existing = pd.read_csv(csv_path, parse_dates=["date"])
    existing = existing[~existing["date"].isin(new_rows["date"])]
    updated  = pd.concat([existing, new_rows], ignore_index=True)
    updated.sort_values("date", inplace=True)
    updated.reset_index(drop=True, inplace=True)
    updated.to_csv(csv_path, index=False)
    print(f"  [✓] Labelled CSV updated → {csv_path}")
    print(f"      Covers: {updated['date'].min().date()} → "
          f"{updated['date'].max().date()}  ({len(updated)} total rows)")

    return {
        "city"         : city,
        "status"       : "filled",
        "days_filled"  : len(new_rows),
        "date_range"   : f"{start_fill} → {end_fill}",
        "new_last_date": str(updated["date"].max().date()),
    }


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill historical gap in city CSVs using Open-Meteo archive")
    parser.add_argument("--city",    default="all",
        help="City to backfill: Delhi / Hyderabad / Nagpur / all")
    parser.add_argument("--dry-run", action="store_true",
        help="Show what would be filled without fetching")
    args = parser.parse_args()

    cities = list(CITY_CONFIG.keys()) if args.city == "all" else [args.city]

    print("\n" + "="*60)
    print("  STEP 7b — HISTORICAL GAP BACKFILL")
    print("="*60)
    print(f"""
  What this does:
    Finds the gap between your CSV end date and yesterday,
    fetches every missing day from Open-Meteo's free archive,
    and appends them so your rolling window features are accurate.

  APIs used  : Open-Meteo Archive (weather) — free, no key
               Open-Meteo AQ Archive       — free, no key
  Run once   : After this, step 7 keeps the CSV current daily.
  Today      : {date.today()}
    """)

    results = []
    for city in cities:
        if city not in CITY_CONFIG:
            print(f"[!] Unknown city: {city}")
            continue
        r = backfill_city(city, dry_run=args.dry_run)
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    total_filled = 0
    for r in results:
        status = r["status"]
        city   = r["city"]
        if status == "filled":
            print(f"  ✓  {city:12s}: {r['days_filled']:4d} days filled  "
                  f"({r['date_range']})  →  now current to {r['new_last_date']}")
            total_filled += r["days_filled"]
        elif status == "up_to_date":
            print(f"  –  {city:12s}: already up to date (last: {r['last_date']})")
        elif status == "dry_run":
            print(f"  ?  {city:12s}: {r['days_filled']} days WOULD be filled (dry run)")
            total_filled += r["days_filled"]
        else:
            print(f"  ✗  {city:12s}: {status}")

    print(f"\n  Total days filled: {total_filled}")
    if not args.dry_run and total_filled > 0:
        print("""
  NEXT STEPS:
    1. Re-run steps 1–2 to recompute engineered features on the
       extended dataset:
         python step1_preprocessing.py
         python step2_risk_labeling.py

    2. (Optional) Re-run step 4 to retrain the model on the
       full updated dataset including the new 2025–2026 rows:
         python step4_modeling.py

    3. Run step 7 daily (or set up a cron job):
         python step7_realtime_api.py --city all
         # or use the dashboard Live Prediction button
        """)
    print("[Done]\n")