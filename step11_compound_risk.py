# """
# ========================================================================
# HEATWAVE RISK PREDICTION SYSTEM
# Step 11b: Compound Risk Analysis — Streamlit Page
# ========================================================================
# Imported by step6_dashboard.py.

# Entry point:
#     from step11_compound_risk import render_compound_risk_page
#     render_compound_risk_page(df_all)

# Requires:
#     data/processed/compound_weights.json  (run step11_compound_risk_optimizer.py)
#     data/processed/labelled_all.csv       (run steps 1–2)
# ========================================================================
# """

# import os
# import json
# import numpy as np
# import pandas as pd
# import streamlit as st
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots

# PROC_DIR     = "data/processed"
# WEIGHTS_PATH = os.path.join(PROC_DIR, "compound_weights.json")

# # ── RISK SCORE BANDS (same as Step 2) ────────────────────────────────────────
# RISK_THRESHOLDS = [
#     (0,   25,  "Low",      "#2ECC71", "#d5f5e3"),
#     (25,  50,  "Moderate", "#F1C40F", "#fef9e7"),
#     (50,  75,  "High",     "#E67E22", "#fdebd0"),
#     (75,  101, "Severe",   "#E74C3C", "#fadbd8"),
# ]

# # ── PILLAR SCORE HELPERS (mirrors Step 2) ────────────────────────────────────

# def _temp_score(t):
#     if t < 35: return 0.0
#     if t < 38: return 25.0
#     if t < 40: return 50.0
#     if t < 45: return 75.0
#     return 100.0

# def _heat_index(T, RH):
#     TF = T * 9 / 5 + 32
#     HI = (-42.379 + 2.04901523 * TF + 10.14333127 * RH
#           - 0.22475541 * TF * RH - 0.00683783 * TF ** 2
#           - 0.05481717 * RH ** 2 + 0.00122874 * TF ** 2 * RH
#           + 0.00085282 * TF * RH ** 2 - 0.00000199 * TF ** 2 * RH ** 2)
#     return (HI - 32) * 5 / 9

# def _hi_score(hi):
#     if hi < 27: return 0.0
#     if hi < 32: return 25.0
#     if hi < 41: return 50.0
#     if hi < 54: return 75.0
#     return 100.0

# def _aqi_score(aqi):
#     if aqi <= 50:  return 0.0
#     if aqi <= 100: return 15.0
#     if aqi <= 200: return 40.0
#     if aqi <= 300: return 60.0
#     if aqi <= 400: return 80.0
#     return 100.0

# def get_risk_label(score):
#     for lo, hi, label, color, bg in RISK_THRESHOLDS:
#         if lo <= score < hi:
#             return label, color, bg
#     return "Severe", "#E74C3C", "#fadbd8"


# # ── LOAD WEIGHTS ──────────────────────────────────────────────────────────────

# @st.cache_data(ttl=0)
# def load_compound_weights():
#     """
#     Load optimized weights from compound_weights.json.
#     Falls back to equal weights derived from Step 2 weights if file missing.
#     """
#     if os.path.exists(WEIGHTS_PATH):
#         with open(WEIGHTS_PATH) as f:
#             return json.load(f), True   # (data, is_optimized)

#     # Fallback: equal weights (Step 2 normalized)
#     fallback = {
#         "temp_humidity": {
#             "label": "Temperature + Humidity",
#             "features": ["w_temp", "w_hi"],
#             "best_model": "Fallback (equal weights)",
#             "best_weights": {"w_temp": 0.5, "w_hi": 0.5},
#         },
#         "temp_aqi": {
#             "label": "Temperature + AQI",
#             "features": ["w_temp", "w_aqi"],
#             "best_model": "Fallback (equal weights)",
#             "best_weights": {"w_temp": 0.5, "w_aqi": 0.5},
#         },
#         "temp_aqi_humidity": {
#             "label": "Temperature + AQI + Humidity",
#             "features": ["w_temp", "w_hi", "w_aqi"],
#             "best_model": "Fallback (equal weights)",
#             "best_weights": {"w_temp": 0.34, "w_hi": 0.33, "w_aqi": 0.33},
#         },
#     }
#     return fallback, False   # not optimized


# # ── INTENSITY COMPUTATION ────────────────────────────────────────────────────

# def compute_compound_intensities(temp_max, humidity, aqi, weights_data):
#     """
#     Returns a dict with four intensity scores (0–100):
#       - temp_only        : temperature pillar only
#       - temp_humidity    : Temp + Humidity (via Heat Index)
#       - temp_aqi         : Temp + AQI
#       - temp_aqi_humidity: Triple compound
#     Scores are clipped to [0, 100].
#     """
#     s_temp = _temp_score(temp_max)
#     hi_val  = _heat_index(temp_max, humidity)
#     s_hi   = _hi_score(hi_val)
#     s_aqi  = _aqi_score(aqi)

#     def _blend(weights_dict, scores_dict):
#         total = sum(weights_dict.values())
#         if total == 0:
#             return 0.0
#         return float(np.clip(
#             sum(weights_dict[k] * scores_dict[k] for k in weights_dict) / total,
#             0, 100
#         ))

#     # Temp-only: just the temperature pillar score directly
#     score_temp_only = s_temp

#     # Temp + Humidity
#     w_th = weights_data["temp_humidity"]["best_weights"]
#     score_th = _blend(w_th, {"w_temp": s_temp, "w_hi": s_hi})

#     # Temp + AQI
#     w_ta = weights_data["temp_aqi"]["best_weights"]
#     score_ta = _blend(w_ta, {"w_temp": s_temp, "w_aqi": s_aqi})

#     # Temp + AQI + Humidity
#     w_tah = weights_data["temp_aqi_humidity"]["best_weights"]
#     score_tah = _blend(w_tah, {"w_temp": s_temp, "w_hi": s_hi, "w_aqi": s_aqi})

#     return {
#         "temp_only"        : round(score_temp_only, 1),
#         "temp_humidity"    : round(score_th, 1),
#         "temp_aqi"         : round(score_ta, 1),
#         "temp_aqi_humidity": round(score_tah, 1),
#         # Raw pillar scores for radar chart
#         "_s_temp": s_temp,
#         "_s_hi"  : s_hi,
#         "_s_aqi" : s_aqi,
#         "_hi_val": round(hi_val, 1),
#     }


# # ── MAIN PAGE ────────────────────────────────────────────────────────────────

# def render_compound_risk_page(df_all):
#     st.title("🔥 Compound Risk Analysis")
#     st.caption(
#         "Compound risk occurs when two or more hazardous events happen simultaneously, "
#         "amplifying the total danger beyond what any single event would cause."
#     )

#     weights_data, is_optimized = load_compound_weights()

#     # ── Optimized Weights Panel ───────────────────────────────────────────────
#     opt_tag = "✅ ML-Optimized" if is_optimized else "⚠️ Fallback (equal weights)"
#     with st.expander(f"📊 Optimized Weights — {opt_tag}", expanded=False):
#         if not is_optimized:
#             st.warning(
#                 "Optimized weights not found. "
#                 "Run `python step11_compound_risk_optimizer.py` to generate them.",
#                 icon="⚠️"
#             )
#         else:
#             st.success("Weights optimized from historical data via 5-model comparison.")

#         for sk, sd in weights_data.items():
#             st.markdown(f"**{sd['label']}**  —  Best model: `{sd['best_model']}`")
#             wdf = pd.DataFrame([
#                 {"Pillar": k.replace("w_", "").capitalize(),
#                  "Weight": f"{v:.4f}",
#                  "Weight %": f"{v*100:.1f}%"}
#                 for k, v in sd["best_weights"].items()
#             ])
#             st.dataframe(wdf.set_index("Pillar"), use_container_width=False)

#             if is_optimized and "all_models" in sd:
#                 with st.expander(f"  All models — {sd['label']}", expanded=False):
#                     rows = []
#                     for mname, minfo in sd["all_models"].items():
#                         row = {"Model": mname, "R²": minfo["r2"], "MAE": minfo["mae"]}
#                         for fn, w in minfo["weights"].items():
#                             row[fn] = f"{w:.4f}"
#                         row["Best"] = "✅" if mname == sd["best_model"] else ""
#                         rows.append(row)
#                     st.dataframe(pd.DataFrame(rows).set_index("Model"),
#                                  use_container_width=True)
#             st.markdown("---")

#     st.markdown("")

#     # ════════════════════════════════════════════════════════════════════════
#     # SECTION 1 — INTERACTIVE CALCULATOR
#     # ════════════════════════════════════════════════════════════════════════
#     st.subheader("🧮 Interactive Compound Risk Calculator")
#     st.caption("Adjust weather parameters to see how intensity escalates as stressors compound.")

#     c1, c2, c3 = st.columns([1, 1, 1])
#     with c1:
#         temp_max = st.slider("🌡️ Temperature Max (°C)", 20.0, 50.0, 40.0, 0.5, key="cr_temp")
#     with c2:
#         humidity = st.slider("💧 Humidity (%)", 5, 100, 60, key="cr_hum")
#     with c3:
#         aqi_val  = st.slider("🌫️ AQI (India CPCB)", 0, 500, 200, key="cr_aqi")

#     scores = compute_compound_intensities(temp_max, humidity, aqi_val, weights_data)

#     scenario_info = [
#         ("🌡️ Temperature Only",       "temp_only",         "Baseline — only temperature is considered"),
#         ("💧 Temp + Humidity",         "temp_humidity",     "Heat + moisture stress (via Heat Index)"),
#         ("🌫️ Temp + AQI",             "temp_aqi",          "Heat + respiratory air quality stress"),
#         ("⚡ Temp + AQI + Humidity",   "temp_aqi_humidity", "Full triple compound — maximum intensity"),
#     ]

#     # ── Score Cards ──────────────────────────────────────────────────────────
#     cols = st.columns(4)
#     for col, (title, key, desc) in zip(cols, scenario_info):
#         s = scores[key]
#         label, color, bg = get_risk_label(s)
#         is_baseline = key == "temp_only"
#         amp = ""
#         if not is_baseline and scores["temp_only"] > 0:
#             amp_factor = s / scores["temp_only"]
#             amp = f"<div style='font-size:0.75rem;color:{color};font-weight:700;margin-top:4px'>+{((amp_factor-1)*100):.0f}% vs baseline</div>"
#         col.markdown(f"""
#         <div style="
#             background:{bg};border:2px solid {color};border-radius:14px;
#             padding:16px 10px;text-align:center;height:100%;
#         ">
#           <div style="font-size:0.78rem;font-weight:700;color:#444;margin-bottom:8px">{title}</div>
#           <div style="font-size:2.4rem;font-weight:900;color:{color}">{s:.0f}</div>
#           <div style="font-size:0.7rem;color:#666">/ 100</div>
#           <div style="font-size:1rem;font-weight:700;color:{color};margin-top:4px">{label}</div>
#           <div style="font-size:0.65rem;color:#888;margin-top:4px">{desc}</div>
#           {amp}
#         </div>""", unsafe_allow_html=True)

#     # Heat Index info
#     hi = scores["_hi_val"]
#     st.markdown(f"""
#     <div style="background:#f0f4ff;border-left:4px solid #3498DB;padding:8px 14px;
#                 border-radius:6px;margin-top:16px;font-size:0.85rem">
#       🥵 <b>Computed Heat Index:</b> {hi}°C &nbsp;|&nbsp;
#       🌡️ Temp score: <b>{scores['_s_temp']:.0f}</b> &nbsp;|&nbsp;
#       🥵 Heat Index score: <b>{scores['_s_hi']:.0f}</b> &nbsp;|&nbsp;
#       🌫️ AQI score: <b>{scores['_s_aqi']:.0f}</b>
#     </div>""", unsafe_allow_html=True)

#     st.markdown("")

#     # ── Comparison Charts ─────────────────────────────────────────────────────
#     ch1, ch2 = st.columns(2)

#     with ch1:
#         labels  = [s[0] for s in scenario_info]
#         vals    = [scores[s[1]] for s in scenario_info]
#         colors  = [get_risk_label(v)[1] for v in vals]

#         fig = go.Figure(go.Bar(
#             x=labels, y=vals,
#             marker_color=colors,
#             text=[f"{v:.0f}" for v in vals],
#             textposition="outside",
#             textfont=dict(size=13, color="#333"),
#         ))
#         fig.update_layout(
#             title="Intensity by Scenario",
#             yaxis=dict(range=[0, 115], title="Intensity Score (0–100)"),
#             height=360, margin=dict(l=0, r=0, t=40, b=60),
#             showlegend=False,
#             xaxis_tickfont_size=11,
#         )
#         fig.add_hline(y=25, line_dash="dot", line_color="#F1C40F",
#                       annotation_text="Moderate", annotation_position="right")
#         fig.add_hline(y=50, line_dash="dot", line_color="#E67E22",
#                       annotation_text="High",     annotation_position="right")
#         fig.add_hline(y=75, line_dash="dot", line_color="#E74C3C",
#                       annotation_text="Severe",   annotation_position="right")
#         st.plotly_chart(fig, use_container_width=True)

#     with ch2:
#         fig2 = go.Figure(go.Scatterpolar(
#             r=[scores["_s_temp"], scores["_s_hi"], scores["_s_aqi"],
#                scores["temp_aqi_humidity"], scores["_s_temp"]],
#             theta=["Temperature", "Heat Index", "AQI",
#                    "Triple\nCompound", "Temperature"],
#             fill="toself",
#             fillcolor="rgba(231, 76, 60, 0.18)",
#             line=dict(color="#E74C3C", width=2),
#             name="Pillar Scores",
#         ))
#         fig2.add_trace(go.Scatterpolar(
#             r=[scores["temp_only"]] * 5,
#             theta=["Temperature", "Heat Index", "AQI",
#                    "Triple\nCompound", "Temperature"],
#             line=dict(color="#2ECC71", width=1.5, dash="dot"),
#             name="Temp-only baseline",
#         ))
#         fig2.update_layout(
#             title="Pillar Score Radar",
#             polar=dict(radialaxis=dict(range=[0, 100], tickfont_size=9)),
#             height=360, margin=dict(l=20, r=20, t=50, b=20),
#             legend=dict(orientation="h", y=-0.15),
#         )
#         st.plotly_chart(fig2, use_container_width=True)

#     # ── Amplification Explanation ─────────────────────────────────────────────
#     base = scores["temp_only"]
#     triple = scores["temp_aqi_humidity"]
#     if base > 0:
#         amp_pct = ((triple - base) / base) * 100
#         amp_txt = (
#             f"At these conditions, compound effects increase heatwave intensity by "
#             f"**{amp_pct:.0f}%** compared to temperature alone. "
#         )
#         if triple >= 75:
#             amp_txt += "🔴 **SEVERE compound risk — multiple simultaneous stressors.**"
#         elif triple >= 50:
#             amp_txt += "🟠 **HIGH compound risk — significant amplification.**"
#         elif triple >= 25:
#             amp_txt += "🟡 **Moderate compound risk.**"
#         else:
#             amp_txt += "🟢 **Low compound risk under current conditions.**"
#         st.info(amp_txt)

#     # ════════════════════════════════════════════════════════════════════════
#     # SECTION 2 — HISTORICAL COMPOUND EVENTS
#     # ════════════════════════════════════════════════════════════════════════
#     st.markdown("---")
#     st.subheader("📈 Historical Compound Risk Analysis")
#     st.caption("Amplification factor = Triple compound score ÷ Temperature-only score")

#     if df_all is None:
#         st.error("No historical data. Run steps 1–2 first.")
#         return

#     # Compute scores on historical data
#     needed = ["score_temp", "score_hi", "score_aqi", "composite_score",
#               "city", "date", "temp_max", "humidity", "aqi"]
#     missing_hist = [c for c in needed if c not in df_all.columns]
#     if missing_hist:
#         st.error(f"Historical data missing columns: {missing_hist}. Run step 2 first.")
#         return

#     hdf = df_all[needed].dropna().copy()

#     # Compute weighted compound scores for each row
#     def _row_intensities(row):
#         s_temp = row["score_temp"]
#         s_hi   = row["score_hi"]
#         s_aqi  = row["score_aqi"]

#         w_th  = weights_data["temp_humidity"]["best_weights"]
#         w_ta  = weights_data["temp_aqi"]["best_weights"]
#         w_tah = weights_data["temp_aqi_humidity"]["best_weights"]

#         def blend(w, s):
#             t = sum(w.values())
#             return np.clip(sum(w[k] * s[k] for k in w) / t, 0, 100) if t else 0.0

#         sc_th  = blend(w_th,  {"w_temp": s_temp, "w_hi": s_hi})
#         sc_ta  = blend(w_ta,  {"w_temp": s_temp, "w_aqi": s_aqi})
#         sc_tah = blend(w_tah, {"w_temp": s_temp, "w_hi": s_hi, "w_aqi": s_aqi})
#         amp    = sc_tah / max(s_temp, 1.0)
#         return pd.Series({
#             "score_temp_only" : s_temp,
#             "score_t_humidity": round(sc_th,  1),
#             "score_t_aqi"     : round(sc_ta,  1),
#             "score_triple"    : round(sc_tah, 1),
#             "amplification"   : round(amp,    3),
#         })

#     with st.spinner("Computing historical compound intensities…"):
#         extra = hdf.apply(_row_intensities, axis=1)
#         hdf = pd.concat([hdf, extra], axis=1)

#     # City filter
#     CITIES = sorted(hdf["city"].unique())
#     city_sel = st.selectbox("City", ["All"] + CITIES, key="cr_city_hist")
#     hdf_f = hdf if city_sel == "All" else hdf[hdf["city"] == city_sel]

#     # ── Timeline: Temp-only vs Triple ─────────────────────────────────────────
#     if city_sel != "All":
#         fig3 = go.Figure()
#         sample = hdf_f.sort_values("date").tail(365)  # last year
#         fig3.add_trace(go.Scatter(
#             x=sample["date"], y=sample["score_temp_only"],
#             name="Temperature Only",
#             line=dict(color="#3498DB", width=1.5, dash="dot"),
#             mode="lines",
#         ))
#         fig3.add_trace(go.Scatter(
#             x=sample["date"], y=sample["score_triple"],
#             name="Triple Compound",
#             line=dict(color="#E74C3C", width=2),
#             fill="tonexty",
#             fillcolor="rgba(231, 76, 60, 0.12)",
#             mode="lines",
#         ))
#         fig3.add_hline(y=50, line_dash="dot", line_color="#E67E22",
#                        annotation_text="High (50)")
#         fig3.add_hline(y=75, line_dash="dot", line_color="#E74C3C",
#                        annotation_text="Severe (75)")
#         fig3.update_layout(
#             title=f"{city_sel} — Temperature-Only vs Triple Compound Intensity (Last 365 days)",
#             yaxis_title="Intensity Score", height=350,
#             margin=dict(l=0, r=0, t=50, b=0),
#             legend=dict(orientation="h", y=-0.25),
#             xaxis=dict(rangeslider=dict(visible=True)),
#         )
#         st.plotly_chart(fig3, use_container_width=True)

#     # ── Scenario Comparison Bar by City ──────────────────────────────────────
#     city_avg = hdf_f.groupby("city").agg(
#         avg_temp_only = ("score_temp_only", "mean"),
#         avg_t_humidity= ("score_t_humidity","mean"),
#         avg_t_aqi     = ("score_t_aqi",     "mean"),
#         avg_triple    = ("score_triple",    "mean"),
#     ).reset_index()

#     fig4 = go.Figure()
#     scenario_cols = [
#         ("avg_temp_only",  "Temperature Only", "#3498DB"),
#         ("avg_t_humidity", "Temp + Humidity",  "#27AE60"),
#         ("avg_t_aqi",      "Temp + AQI",       "#E67E22"),
#         ("avg_triple",     "Triple Compound",  "#E74C3C"),
#     ]
#     for col, name, clr in scenario_cols:
#         fig4.add_trace(go.Bar(
#             x=city_avg["city"], y=city_avg[col].round(1),
#             name=name, marker_color=clr,
#         ))
#     fig4.update_layout(
#         title="Average Intensity per Scenario — City Comparison",
#         barmode="group",
#         yaxis_title="Avg Intensity Score",
#         height=350, margin=dict(l=0, r=0, t=50, b=0),
#         legend=dict(orientation="h", y=-0.3),
#     )
#     st.plotly_chart(fig4, use_container_width=True)

#     # ── Top-10 Most Amplified Days ────────────────────────────────────────────
#     st.markdown("**🔝 Top-10 Most Amplified Days (by city)**")
#     top10 = (hdf_f.sort_values("amplification", ascending=False)
#              .head(10)[["date","city","temp_max","humidity","aqi",
#                          "score_temp_only","score_triple","amplification"]]
#              .copy())
#     top10["date"] = pd.to_datetime(top10["date"]).dt.strftime("%d %b %Y")
#     top10.columns = ["Date","City","Tmax°C","Humidity%","AQI",
#                      "Temp-Only Score","Triple Score","Amplification×"]
#     top10["Amplification×"] = top10["Amplification×"].round(2)

#     def _style_amp(val):
#         if val >= 3:   return "background-color:#fadbd8;font-weight:bold"
#         if val >= 2:   return "background-color:#fdebd0"
#         if val >= 1.5: return "background-color:#fef9e7"
#         return ""

#     styled_top = top10.style.map(_style_amp, subset=["Amplification×"])
#     st.dataframe(styled_top, use_container_width=True, height=380)

#     # ── Download ──────────────────────────────────────────────────────────────
#     csv = hdf_f[["date","city","temp_max","humidity","aqi",
#                   "score_temp_only","score_t_humidity","score_t_aqi",
#                   "score_triple","amplification"]].to_csv(index=False)
#     st.download_button(
#         "⬇️ Download compound risk dataset",
#         csv, "compound_risk_analysis.csv", "text/csv"
#     )


"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Step 11: Compound Risk Intensity Analysis
========================================================================
PURPOSE:
  The existing system flags compound events as binary (0/1).
  This module goes further — it QUANTIFIES how much each additional
  stressor (humidity, AQI) amplifies the base heatwave intensity.

FOUR INTENSITY METRICS (all 0–100 scale):

  I1 — Temperature-only intensity
       Pure thermodynamic heat based on Tmax alone.
       Uses IMD threshold bands. This is the BASELINE.

  I2 — Temperature + Humidity intensity
       Adds the physiological amplification from humidity.
       High humidity prevents sweat evaporation → body feels hotter.
       Derived from the Heat Index excess (heat_index − Tmax).

  I3 — Temperature + AQI intensity
       Adds the respiratory stress amplification from air pollution.
       Poor AQI during heat reduces the body's tolerance threshold.
       Mechanism: PM2.5 causes airway inflammation, reducing ability
       to breathe fast during heat — body can't cool through respiration.

  I4 — Temperature + AQI + Humidity (Full Compound)
       All three stressors active simultaneously.
       Includes a SYNERGY BONUS: when all three are high, the combined
       effect is greater than the sum of parts (non-linear compounding).
       Example: humidity stops sweating + AQI causes inflammation
       + heat = body has NO effective cooling mechanism.

AMPLIFICATION (Δ):
  Δ_humidity  = I2 − I1  (how much humidity worsens it)
  Δ_aqi       = I3 − I1  (how much AQI worsens it)
  Δ_compound  = I4 − I1  (total compound amplification)

KEY OUTPUT:
  If I4's risk tier > I1's risk tier → COMPOUND UPGRADE
  e.g. Tmax alone = Moderate, but with AQI+Humidity = High
  This is the core message: "A moderate temperature day becomes
  a High-risk day because of compound stressors."
========================================================================
"""

import pandas as pd
import numpy as np
import os, json
import warnings
warnings.filterwarnings("ignore")

PROC_DIR = "data/processed"
LOG_DIR  = "data/predictions"
CITIES   = ["Delhi", "Hyderabad", "Nagpur"]
os.makedirs(LOG_DIR, exist_ok=True)

RISK_LABELS = {0: "Low", 1: "Moderate", 2: "High", 3: "Severe"}
RISK_COLORS = {0: "#2ECC71", 1: "#F1C40F", 2: "#E67E22", 3: "#E74C3C"}


# ════════════════════════════════════════════════════════════════════════
# SECTION A: INDIVIDUAL INTENSITY CALCULATORS
# ════════════════════════════════════════════════════════════════════════

def intensity_temperature_only(temp_max: float) -> float:
    """
    I1 — Temperature-only intensity (0–100).

    Uses IMD heatwave threshold bands with linear interpolation
    within each band for smooth progression.

    Bands:
      < 30°C  → 0   (no heat stress)
      30–35°C → 0–15   (warm, minor stress)
      35–38°C → 15–35  (hot)
      38–40°C → 35–55  (very hot, approaching heat wave)
      40–45°C → 55–80  (IMD heat wave)
      45–50°C → 80–100 (IMD severe heat wave)
      > 50°C  → 100
    """
    bands = [
        (30.0, 35.0,  0.0, 15.0),
        (35.0, 38.0, 15.0, 35.0),
        (38.0, 40.0, 35.0, 55.0),
        (40.0, 45.0, 55.0, 80.0),
        (45.0, 50.0, 80.0, 100.0),
    ]
    if temp_max < 30.0:
        return 0.0
    if temp_max >= 50.0:
        return 100.0
    for t_lo, t_hi, i_lo, i_hi in bands:
        if t_lo <= temp_max < t_hi:
            ratio = (temp_max - t_lo) / (t_hi - t_lo)
            return round(i_lo + ratio * (i_hi - i_lo), 2)
    return 0.0


def humidity_amplifier(temp_max: float, humidity: float) -> float:
    """
    Humidity amplification (Δ_humidity, 0–30 pts).

    Mechanism: High humidity reduces evaporative cooling efficiency.
    The amplifier is based on the Heat Index excess over air temperature.

    heat_index_excess = heat_index(T, RH) − T
    This excess (in °C) is scaled to a 0–30 point amplification.

    Amplification logic:
      - Below 40% RH: no amplification (dry heat, body can sweat freely)
      - 40–60% RH: partial amplification
      - 60–80% RH: strong amplification
      - > 80% RH: maximum amplification (sweating almost useless)

    Also gated by temperature: humidity amplification only matters
    when it's already hot (Tmax ≥ 30°C). Cool humid days are not dangerous.
    """
    if temp_max < 30.0:
        return 0.0

    # Heat index excess
    TF = temp_max * 9/5 + 32
    RH = humidity
    HI = (-42.379 + 2.04901523*TF + 10.14333127*RH
          - 0.22475541*TF*RH - 0.00683783*TF**2
          - 0.05481717*RH**2 + 0.00122874*TF**2*RH
          + 0.00085282*TF*RH**2 - 0.00000199*TF**2*RH**2)
    hi_celsius = (HI - 32) * 5/9
    excess = max(0.0, hi_celsius - temp_max)

    # Scale excess to 0–30 pts
    # Maximum realistic excess is ~20°C (38°C at 90% → HI ~58°C)
    amp = min(30.0, (excess / 20.0) * 30.0)

    # Gate: amplification only meaningful when hot
    heat_gate = min(1.0, max(0.0, (temp_max - 30.0) / 15.0))
    return round(amp * heat_gate, 2)


def aqi_amplifier(temp_max: float, aqi: float) -> float:
    """
    AQI amplification (Δ_aqi, 0–25 pts).

    Mechanism: PM2.5 and PM10 cause airway inflammation, reducing
    respiratory capacity. This lowers the body's ability to dissipate
    heat through rapid breathing and increases cardiovascular strain.

    The amplifier is:
      1. AQI band score (0–20 pts based on CPCB category)
      2. Heat multiplier (amplification only matters when it's hot)

    AQI band scores (India CPCB):
      0–50   (Good)           → 0 pts
      51–100 (Satisfactory)   → 2 pts
      101–200 (Moderate)      → 6 pts
      201–300 (Poor)          → 12 pts
      301–400 (Very Poor)     → 18 pts
      400+   (Severe)         → 22 pts (capped at 25 with heat multiplier)

    Heat multiplier: amplification scales with temperature.
    At 30°C: 0.4×,  at 38°C: 0.8×,  at 43°C+: 1.0×
    """
    if aqi <= 0:
        return 0.0

    # AQI band base score
    if   aqi <= 50:   base = 0.0
    elif aqi <= 100:  base = 2.0
    elif aqi <= 200:  base = 6.0
    elif aqi <= 300:  base = 12.0
    elif aqi <= 400:  base = 18.0
    else:             base = 22.0

    # Heat multiplier
    heat_mult = min(1.0, max(0.0, (temp_max - 30.0) / 13.0))

    # Final AQI amplifier (max 25 pts)
    amp = min(25.0, base * (0.4 + 0.6 * heat_mult))
    return round(amp, 2)


def synergy_bonus(temp_max: float, humidity: float, aqi: float) -> float:
    """
    Synergy bonus when all three stressors are simultaneously high (0–10 pts).

    Rationale: When humidity stops sweating AND AQI causes inflammation
    AND heat is extreme, the body has NO effective cooling mechanism.
    This non-linear interaction is worse than the sum of parts.

    Bonus activates when:
      Tmax ≥ 38°C  AND  Humidity ≥ 60%  AND  AQI ≥ 150

    Scales linearly with how far each exceeds its trigger threshold.
    Maximum: 10 pts (when Tmax ≥ 44°C, Humidity ≥ 80%, AQI ≥ 300)
    """
    if temp_max < 38.0 or humidity < 60.0 or aqi < 150.0:
        return 0.0

    t_factor = min(1.0, (temp_max - 38.0) / 6.0)     # 0 at 38°C, 1 at 44°C
    h_factor = min(1.0, (humidity - 60.0) / 20.0)    # 0 at 60%, 1 at 80%
    a_factor = min(1.0, (aqi - 150.0) / 150.0)       # 0 at 150, 1 at 300

    bonus = 10.0 * t_factor * h_factor * a_factor
    return round(bonus, 2)


# ════════════════════════════════════════════════════════════════════════
# SECTION B: COMPOUND INTENSITY ENGINE
# ════════════════════════════════════════════════════════════════════════

def compute_compound_intensity(temp_max: float,
                                humidity: float,
                                aqi: float) -> dict:
    """
    Master function. Given three environmental values, returns
    all four intensity scores plus deltas and tier classifications.
    """
    # Core components
    I1_score   = intensity_temperature_only(temp_max)
    delta_hum  = humidity_amplifier(temp_max, humidity)
    delta_aqi  = aqi_amplifier(temp_max, aqi)
    delta_syn  = synergy_bonus(temp_max, humidity, aqi)

    # Four compound intensities (capped at 100)
    I1 = round(min(100.0, I1_score), 2)
    I2 = round(min(100.0, I1_score + delta_hum), 2)
    I3 = round(min(100.0, I1_score + delta_aqi), 2)
    I4 = round(min(100.0, I1_score + delta_hum + delta_aqi + delta_syn), 2)

    def to_tier(score):
        if score < 25: return 0
        if score < 50: return 1
        if score < 75: return 2
        return 3

    tier1, tier2, tier3, tier4 = to_tier(I1), to_tier(I2), to_tier(I3), to_tier(I4)

    # Compound upgrade flags
    compound_upgrade_hum  = tier2 > tier1
    compound_upgrade_aqi  = tier3 > tier1
    compound_upgrade_full = tier4 > tier1
    tiers_gained          = tier4 - tier1  # 0, 1, or 2 levels above baseline

    # Dominant stressor
    deltas = {"Humidity": delta_hum, "AQI": delta_aqi, "Synergy": delta_syn}
    dominant = max(deltas, key=deltas.get) if any(v > 0 for v in deltas.values()) else "None"

    # Generate advisory statement
    advisory = _generate_advisory(
        temp_max, I1, I4, tier1, tier4,
        delta_hum, delta_aqi, delta_syn,
        compound_upgrade_full)

    return {
        # Raw inputs
        "temp_max"              : round(temp_max, 1),
        "humidity"              : round(humidity, 1),
        "aqi"                   : round(aqi, 0),

        # Four intensity scores
        "I1_temp_only"          : I1,
        "I2_temp_humidity"      : I2,
        "I3_temp_aqi"           : I3,
        "I4_full_compound"      : I4,

        # Risk tiers for each
        "tier_I1"               : tier1,
        "tier_I2"               : tier2,
        "tier_I3"               : tier3,
        "tier_I4"               : tier4,

        "label_I1"              : RISK_LABELS[tier1],
        "label_I2"              : RISK_LABELS[tier2],
        "label_I3"              : RISK_LABELS[tier3],
        "label_I4"              : RISK_LABELS[tier4],

        # Amplification deltas
        "delta_humidity"        : round(delta_hum, 2),
        "delta_aqi"             : round(delta_aqi, 2),
        "delta_synergy"         : round(delta_syn, 2),
        "delta_total"           : round(I4 - I1, 2),

        # Compound upgrade flags
        "upgrade_by_humidity"   : compound_upgrade_hum,
        "upgrade_by_aqi"        : compound_upgrade_aqi,
        "upgrade_by_compound"   : compound_upgrade_full,
        "tiers_gained"          : tiers_gained,
        "dominant_stressor"     : dominant,

        # Advisory
        "advisory"              : advisory,
    }


def _generate_advisory(temp_max, I1, I4, tier1, tier4,
                        d_hum, d_aqi, d_syn, upgraded) -> str:
    """Natural-language advisory based on compound analysis."""
    base = f"Temperature alone: {RISK_LABELS[tier1]} (score {I1:.0f})."

    if I4 == I1:
        return base + " No compound amplification active today."

    parts = []
    if d_hum >= 3:
        parts.append(
            f"humidity adding {d_hum:.1f} pts "
            f"(sweating {_eff(d_hum)} effective)")
    if d_aqi >= 3:
        parts.append(
            f"AQI adding {d_aqi:.1f} pts "
            f"(respiratory stress {_eff(d_aqi)} severe)")
    if d_syn > 0:
        parts.append(
            f"synergy bonus {d_syn:.1f} pts "
            f"(no cooling mechanism remains)")

    compound_str = " + ".join(parts) if parts else "minor amplification"
    result = f"{base} Compound effect: {compound_str}."

    if upgraded:
        result += (f" ⚠ COMPOUND UPGRADE: Temperature alone is {RISK_LABELS[tier1]}, "
                   f"but compound conditions elevate this to "
                   f"{RISK_LABELS[tier4]} (score {I4:.0f}).")
    else:
        result += (f" Full compound score: {I4:.0f} "
                   f"(+{I4-I1:.1f} above temperature baseline).")
    return result


def _eff(delta):
    if delta < 5:  return "slightly less"
    if delta < 12: return "significantly less"
    return "barely"


# ════════════════════════════════════════════════════════════════════════
# SECTION C: APPLY TO FULL HISTORICAL DATASET
# ════════════════════════════════════════════════════════════════════════

def compute_compound_for_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies compound intensity to every row of the labelled dataset.
    Adds columns: I1, I2, I3, I4, delta_humidity, delta_aqi,
                  delta_synergy, delta_total, upgrade_by_compound,
                  tiers_gained, dominant_stressor.
    """
    results = df.apply(
        lambda r: compute_compound_intensity(
            r.get("temp_max", 0),
            r.get("humidity", 0),
            r.get("aqi", 0)
        ),
        axis=1
    )
    result_df = pd.DataFrame(results.tolist())

    # Add columns to original df
    add_cols = ["I1_temp_only","I2_temp_humidity","I3_temp_aqi",
                "I4_full_compound",
                "delta_humidity","delta_aqi","delta_synergy","delta_total",
                "tier_I1","tier_I2","tier_I3","tier_I4",
                "label_I1","label_I2","label_I3","label_I4",
                "upgrade_by_humidity","upgrade_by_aqi",
                "upgrade_by_compound","tiers_gained","dominant_stressor"]

    for col in add_cols:
        df[col] = result_df[col].values

    return df


# ════════════════════════════════════════════════════════════════════════
# SECTION D: SUMMARY STATISTICS PER CITY
# ════════════════════════════════════════════════════════════════════════

def compound_summary(df: pd.DataFrame, city: str) -> dict:
    """
    Produces summary statistics about compound amplification for a city.
    """
    sub = df[df["city"] == city].copy()
    if sub.empty:
        return {}

    total = len(sub)
    upgraded = sub["upgrade_by_compound"].sum()

    return {
        "city"                  : city,
        "total_days"            : total,
        # Days where compound upgrades the tier
        "days_compound_upgrade" : int(upgraded),
        "pct_compound_upgrade"  : round(upgraded / total * 100, 1),
        # Days upgraded by each stressor
        "days_upgrade_humidity" : int(sub["upgrade_by_humidity"].sum()),
        "days_upgrade_aqi"      : int(sub["upgrade_by_aqi"].sum()),
        # Average amplification on hot days (Tmax ≥ 38°C)
        "hot_days"              : int((sub["temp_max"] >= 38).sum()),
        "avg_delta_humidity_hot": round(
            sub.loc[sub["temp_max"] >= 38, "delta_humidity"].mean(), 2),
        "avg_delta_aqi_hot"     : round(
            sub.loc[sub["temp_max"] >= 38, "delta_aqi"].mean(), 2),
        "avg_delta_total_hot"   : round(
            sub.loc[sub["temp_max"] >= 38, "delta_total"].mean(), 2),
        # Dominant stressor breakdown
        "dominant_humidity_pct" : round(
            (sub["dominant_stressor"] == "Humidity").sum() / total * 100, 1),
        "dominant_aqi_pct"      : round(
            (sub["dominant_stressor"] == "AQI").sum() / total * 100, 1),
        # Worst compound events
        "max_I4"                : round(sub["I4_full_compound"].max(), 1),
        "max_delta_total"       : round(sub["delta_total"].max(), 1),
        "avg_I1"                : round(sub["I1_temp_only"].mean(), 1),
        "avg_I4"                : round(sub["I4_full_compound"].mean(), 1),
    }


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STEP 11 — COMPOUND RISK INTENSITY ANALYSIS")
    print("="*60)

    # ── 1. Demo: single day calculation ───────────────────────────────
    print("\n  DEMO — Four intensity scores for a single observation:")
    print(f"  {'─'*58}")
    examples = [
        ("Normal hot day",        40.0, 35.0, 100),
        ("Hot + humid",           40.0, 72.0, 100),
        ("Hot + polluted",        40.0, 35.0, 280),
        ("Full compound (worst)", 42.0, 68.0, 260),
    ]
    for label, T, H, A in examples:
        r = compute_compound_intensity(T, H, A)
        upgrade = " ⬆ COMPOUND UPGRADE" if r["upgrade_by_compound"] else ""
        print(f"\n  [{label}]  Tmax={T}°C  Humidity={H}%  AQI={A}")
        print(f"    I1 (temp only)        : {r['I1_temp_only']:5.1f}  → {r['label_I1']}")
        print(f"    I2 (+ humidity)       : {r['I2_temp_humidity']:5.1f}  → {r['label_I2']}"
              f"  Δhum={r['delta_humidity']:+.1f}")
        print(f"    I3 (+ AQI)            : {r['I3_temp_aqi']:5.1f}  → {r['label_I3']}"
              f"  Δaqi={r['delta_aqi']:+.1f}")
        print(f"    I4 (full compound)    : {r['I4_full_compound']:5.1f}  → {r['label_I4']}"
              f"  Δtot={r['delta_total']:+.1f}{upgrade}")
        print(f"    Advisory: {r['advisory'][:90]}...")

    # ── 2. Apply to full dataset ───────────────────────────────────────
    print(f"\n  {'─'*58}")
    print("  Applying to full historical dataset...")
    path = os.path.join(PROC_DIR, "labelled_all.csv")
    if not os.path.exists(path):
        print("  [!] labelled_all.csv not found. Run steps 1–2 first.")
        exit(1)

    df = pd.read_csv(path, parse_dates=["date"])
    df = compute_compound_for_dataset(df)

    # Save enriched dataset
    out_path = os.path.join(PROC_DIR, "compound_intensity_all.csv")
    df.to_csv(out_path, index=False)
    print(f"  [✓] Saved → {out_path}  ({df.shape[0]:,} rows)")

    # Per-city CSV
    for city in CITIES:
        city_df = df[df["city"] == city]
        if city_df.empty: continue
        cp = os.path.join(PROC_DIR, f"compound_intensity_{city}.csv")
        city_df.to_csv(cp, index=False)

    # ── 3. Summary stats ───────────────────────────────────────────────
    print(f"\n  {'─'*58}")
    print("  COMPOUND AMPLIFICATION SUMMARY BY CITY")
    print(f"  {'─'*58}")
    for city in CITIES:
        s = compound_summary(df, city)
        if not s: continue
        print(f"\n  {city}:")
        print(f"    Days where compound upgrades risk tier : "
              f"{s['days_compound_upgrade']} ({s['pct_compound_upgrade']}%)")
        print(f"    Avg amplification on hot days (Tmax≥38°C):")
        print(f"      Humidity: +{s['avg_delta_humidity_hot']:.1f} pts  "
              f"AQI: +{s['avg_delta_aqi_hot']:.1f} pts  "
              f"Total: +{s['avg_delta_total_hot']:.1f} pts")
        print(f"    Avg I1 (temp only): {s['avg_I1']:.1f}  →  "
              f"Avg I4 (full compound): {s['avg_I4']:.1f}")
        print(f"    Max single-day compound amplification: "
              f"+{s['max_delta_total']:.1f} pts")
        print(f"    Dominant stressor: "
              f"Humidity {s['dominant_humidity_pct']}%  "
              f"AQI {s['dominant_aqi_pct']}%")

    print("\n[Done] Step 11 complete. Run dashboard to view compound risk page.\n")