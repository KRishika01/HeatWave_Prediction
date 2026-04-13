"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Step 3: Exploratory Data Analysis (EDA)
========================================================================
Produces the following visualisation outputs (saved to plots/):
  1. Temperature trends & extremes per city
  2. Correlation heatmap of all features
  3. Monthly distribution boxplots (temp, AQI, humidity)
  4. Compound event frequency calendar-style heatmap
  5. Risk level distribution (pie + bar)
  6. Heat Index vs Temperature scatter (coloured by risk)
  7. AQI vs Temperature (compound risk view)
  8. Rolling 7-day temp trend with heat-wave threshold bands
  9. Composite score time-series
  10. Cross-city comparison radar chart
========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

PROC_DIR  = "data/processed"
PLOT_DIR  = "plots"
CITIES    = ["Delhi", "Hyderabad", "Nagpur"]
os.makedirs(PLOT_DIR, exist_ok=True)

RISK_COLORS  = {0: "#2ECC71", 1: "#F1C40F", 2: "#E67E22", 3: "#E74C3C"}
RISK_LABELS  = {0: "Low", 1: "Moderate", 2: "High", 3: "Severe"}
CITY_COLORS  = {"Delhi": "#3498DB", "Hyderabad": "#E74C3C", "Nagpur": "#2ECC71"}

plt.rcParams.update({
    "figure.dpi"    : 150,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "font.family"   : "DejaVu Sans",
})


def load_all() -> pd.DataFrame:
    path = os.path.join(PROC_DIR, "labelled_all.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run step1 and step2 first.")
    return pd.read_csv(path, parse_dates=["date"])


# ── PLOT 1: Temperature Trends ───────────────────────────────────────────
def plot_temperature_trends(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle("Daily Temperature Trends by City", fontsize=14, fontweight="bold")

    for ax, city in zip(axes, CITIES):
        sub = df[df["city"] == city].copy()
        ax.plot(sub["date"], sub["temp_max"], color=CITY_COLORS[city],
                lw=0.8, label="Temp Max")
        ax.plot(sub["date"], sub["temp_min"], color=CITY_COLORS[city],
                lw=0.5, alpha=0.5, linestyle="--", label="Temp Min")
        ax.fill_between(sub["date"], sub["temp_min"], sub["temp_max"],
                        alpha=0.15, color=CITY_COLORS[city])
        ax.axhline(40, color="red", linestyle=":", lw=1, label="Heat Wave (40°C)")
        ax.axhline(45, color="darkred", linestyle=":", lw=1, label="Severe (45°C)")
        ax.set_ylabel("°C", fontsize=9)
        ax.set_title(city, fontsize=10)
        ax.legend(fontsize=7, ncol=4)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/01_temperature_trends.png", bbox_inches="tight")
    plt.close()
    print("[✓] Plot 1: Temperature Trends")


# ── PLOT 2: Correlation Heatmap ──────────────────────────────────────────
def plot_correlation(df: pd.DataFrame):
    feature_cols = ["temp_max", "temp_min", "humidity", "wind", "rainfall", "aqi",
                    "heat_index", "humidex", "temp_range", "dry_days_streak",
                    "consec_hot_days", "aqi_category", "composite_score",
                    "compound_heat_aqi", "compound_heat_drought"]
    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=0, square=True, linewidths=0.5, annot_kws={"size": 7},
                ax=ax)
    ax.set_title("Feature Correlation Matrix (All Cities)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/02_correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    print("[✓] Plot 2: Correlation Heatmap")


# ── PLOT 3: Monthly Distributions ───────────────────────────────────────
def plot_monthly_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    targets = [("temp_max", "Max Temperature (°C)"),
               ("humidity",  "Humidity (%)"),
               ("aqi",       "AQI")]

    for ax, (col, title) in zip(axes, targets):
        sns.boxplot(data=df, x="month", y=col, hue="city",
                    palette=CITY_COLORS, ax=ax, fliersize=2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Month")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle("Monthly Distribution by City", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/03_monthly_distributions.png", bbox_inches="tight")
    plt.close()
    print("[✓] Plot 3: Monthly Distributions")


# ── PLOT 4: Risk Level Distribution ─────────────────────────────────────
def plot_risk_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart per city
    risk_counts = df.groupby(["city", "risk_label"]).size().unstack(fill_value=0)
    risk_counts_pct = risk_counts.div(risk_counts.sum(axis=1), axis=0) * 100
    colors = [RISK_COLORS[k] for k in sorted(RISK_COLORS)]
    risk_counts_pct[["Low","Moderate","High","Severe"]].plot(
        kind="bar", stacked=True, color=colors, ax=axes[0], edgecolor="white")
    axes[0].set_title("Risk Level Distribution by City (%)", fontweight="bold")
    axes[0].set_ylabel("Percentage of Days")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="Risk Level", bbox_to_anchor=(1, 1))

    # Overall pie
    overall = df["risk_label"].value_counts().reindex(["Low","Moderate","High","Severe"])
    axes[1].pie(overall, labels=overall.index, autopct="%1.1f%%",
                colors=colors, startangle=140,
                wedgeprops={"edgecolor":"white", "linewidth":1.5})
    axes[1].set_title("Overall Risk Distribution (All Cities)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/04_risk_distribution.png", bbox_inches="tight")
    plt.close()
    print("[✓] Plot 4: Risk Distribution")


# ── PLOT 5: Heat Index vs Temperature ────────────────────────────────────
def plot_heat_index_scatter(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Heat Index vs. Temperature (coloured by Risk Level)",
                 fontsize=13, fontweight="bold")

    for ax, city in zip(axes, CITIES):
        sub = df[df["city"] == city]
        colors_pt = sub["risk_level"].map(RISK_COLORS)
        ax.scatter(sub["temp_max"], sub["heat_index"], c=colors_pt,
                   alpha=0.5, s=10, edgecolors="none")
        ax.plot([sub["temp_max"].min(), sub["temp_max"].max()],
                [sub["temp_max"].min(), sub["temp_max"].max()],
                "k--", lw=0.7, label="1:1 line")
        ax.set_xlabel("Air Temp Max (°C)")
        ax.set_ylabel("Heat Index (°C)" if city == CITIES[0] else "")
        ax.set_title(city)
        ax.grid(alpha=0.2)

    handles = [mpatches.Patch(color=RISK_COLORS[k], label=RISK_LABELS[k])
               for k in sorted(RISK_COLORS)]
    fig.legend(handles=handles, title="Risk Level", loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/05_heat_index_scatter.png", bbox_inches="tight")
    plt.close()
    print("[✓] Plot 5: Heat Index Scatter")


# ── PLOT 6: AQI vs Temperature (Compound Risk) ───────────────────────────
def plot_compound_aqi_temp(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Temperature vs AQI — Compound Risk View",
                 fontsize=13, fontweight="bold")

    for ax, city in zip(axes, CITIES):
        sub = df[df["city"] == city]
        colors_pt = sub["risk_level"].map(RISK_COLORS)
        ax.scatter(sub["temp_max"], sub["aqi"], c=colors_pt,
                   alpha=0.5, s=10, edgecolors="none")
        ax.axvline(38, color="orange", linestyle="--", lw=0.8, label="38°C")
        ax.axhline(200, color="purple", linestyle="--", lw=0.8, label="AQI 200")
        ax.fill_betweenx([200, sub["aqi"].max()], 38, sub["temp_max"].max(),
                          alpha=0.05, color="red", label="Compound Zone")
        ax.set_xlabel("Temp Max (°C)")
        ax.set_ylabel("AQI" if city == CITIES[0] else "")
        ax.set_title(city)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/06_compound_aqi_temp.png", bbox_inches="tight")
    plt.close()
    print("[✓] Plot 6: AQI-Temp Compound Risk")


# ── PLOT 7: Composite Score Time-Series ──────────────────────────────────
def plot_composite_score_ts(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    fig.suptitle("Composite Risk Score Over Time", fontsize=13, fontweight="bold")

    for ax, city in zip(axes, CITIES):
        sub = df[df["city"] == city].copy()
        ax.fill_between(sub["date"], sub["composite_score"],
                        where=(sub["composite_score"] < 25),
                        color="#2ECC71", alpha=0.6, label="Low")
        ax.fill_between(sub["date"], sub["composite_score"],
                        where=((sub["composite_score"] >= 25) & (sub["composite_score"] < 50)),
                        color="#F1C40F", alpha=0.6, label="Moderate")
        ax.fill_between(sub["date"], sub["composite_score"],
                        where=((sub["composite_score"] >= 50) & (sub["composite_score"] < 75)),
                        color="#E67E22", alpha=0.6, label="High")
        ax.fill_between(sub["date"], sub["composite_score"],
                        where=(sub["composite_score"] >= 75),
                        color="#E74C3C", alpha=0.8, label="Severe")
        ax.set_ylim(0, 100)
        ax.set_title(city, fontsize=10)
        ax.set_ylabel("Score")
        ax.legend(fontsize=7, ncol=4)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/07_composite_score_ts.png", bbox_inches="tight")
    plt.close()
    print("[✓] Plot 7: Composite Score Time-Series")


# ── PLOT 8: Compound Event Frequency per Month ───────────────────────────
def plot_compound_frequency(df: pd.DataFrame):
    compound_cols = ["compound_heat_aqi", "compound_heat_drought",
                     "compound_heat_humidity", "triple_compound"]
    nice_names = ["Heat+AQI", "Heat+Drought", "Heat+Humidity", "Triple Compound"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Monthly Compound Event Frequency by City",
                 fontsize=13, fontweight="bold")

    for ax, city in zip(axes, CITIES):
        sub = df[df["city"] == city].groupby("month")[compound_cols].sum()
        sub.columns = nice_names
        sub.plot(kind="bar", stacked=False, ax=ax, colormap="Set2",
                 edgecolor="white")
        ax.set_title(city)
        ax.set_xlabel("Month")
        ax.set_ylabel("Days" if city == CITIES[0] else "")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/08_compound_frequency.png", bbox_inches="tight")
    plt.close()
    print("[✓] Plot 8: Compound Event Frequency")


# ── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STEP 3 ─ EXPLORATORY DATA ANALYSIS")
    print("="*60)

    df = load_all()
    print(f"\n  Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Cities : {df['city'].unique()}")
    print(f"  Date   : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\n  Generating plots...\n")

    plot_temperature_trends(df)
    plot_correlation(df)
    plot_monthly_distributions(df)
    plot_risk_distribution(df)
    plot_heat_index_scatter(df)
    plot_compound_aqi_temp(df)
    plot_composite_score_ts(df)
    plot_compound_frequency(df)

    print(f"\n  Summary Statistics:")
    print(df.groupby("city")[["temp_max","heat_index","aqi","composite_score"]].describe().round(1).to_string())

    print(f"\n[Done] All EDA plots saved to ./{PLOT_DIR}/\n")