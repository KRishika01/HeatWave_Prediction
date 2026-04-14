"""
========================================================================
HEATWAVE RISK PREDICTION SYSTEM
Step 2b: Data-Driven Pillar Weight Optimization
========================================================================
Replaces the manual pillar weights in step2 with statistically
learned/optimized weights. Four methods are available:

  METHOD 1 — Mutual Information Weights
    Measures how much each pillar score alone reduces uncertainty
    about a proxy target (heat-wave days, validated by IMD threshold).
    No model needed. Pure information theory.

  METHOD 2 — Logistic Regression Coefficient Weights
    Trains a multinomial logistic regression where the 5 pillar scores
    are the only features predicting IMD-validated risk labels.
    The learned coefficients (averaged across classes) become weights.

  METHOD 3 — scipy.optimize — Maximize Label Consistency
    Treats weights as 5 free parameters (summing to 1) and runs a
    constrained numerical optimizer to find the combination that
    maximises the Spearman correlation between the composite score and
    the IMD binary heat-wave flag (objective: higher score on IMD days,
    lower on non-IMD days). No labels needed — only the physics-based
    binary flag.

  METHOD 4 — Grid Search F1 Maximization
    Discretizes the weight space and exhaustively (or randomly) searches
    for the weight combination that maximises macro F1 between the
    resulting risk_level labels and the IMD-derived ground truth.
    Most direct: directly optimizes the metric that matters.

After all methods run, a COMPARISON TABLE is printed and the best
method's weights are saved to data/processed/optimal_weights.json.
step2_risk_labeling.py is then patched to use the learned weights.
========================================================================
"""

import pandas as pd
import numpy as np
import os, json, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model       import LogisticRegression
from sklearn.preprocessing      import StandardScaler
from sklearn.feature_selection  import mutual_info_classif
from sklearn.metrics            import f1_score, accuracy_score
from sklearn.model_selection    import cross_val_score
from scipy.optimize             import minimize
from scipy.stats                import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROC_DIR  = "data/processed"
PLOT_DIR  = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

PILLAR_NAMES = ["Temperature", "Heat Index", "AQI", "Drought", "Compound"]

# ── REFERENCE WEIGHTS (from step2 manual design) ─────────────────────────
MANUAL_WEIGHTS = np.array([0.35, 0.25, 0.20, 0.10, 0.10])

# ════════════════════════════════════════════════════════════════════════
# HELPER: Compute IMD ground-truth labels (physics-based, no model)
# ════════════════════════════════════════════════════════════════════════

def get_imd_ground_truth(df: pd.DataFrame) -> pd.Series:
    """
    IMD rules (no subjectivity):
      0 = Normal   : Tmax < 40°C
      1 = Hot      : 37 ≤ Tmax < 40°C
      2 = Heat Wave: 40 ≤ Tmax < 45°C  OR departure ≥ 4.5°C
      3 = Severe HW: Tmax ≥ 45°C       OR departure ≥ 6.4°C

    departure = Tmax − monthly_climatological_mean
    """
    monthly_mean = df.groupby("month")["temp_max"].transform("mean")
    departure    = df["temp_max"] - monthly_mean

    label = pd.Series(0, index=df.index)
    label[df["temp_max"] >= 37]                             = 1
    label[(df["temp_max"] >= 40) | (departure >= 4.5)]     = 2
    label[(df["temp_max"] >= 45) | (departure >= 6.4)]     = 3
    return label


def build_pillar_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with the 5 raw pillar scores (0–100 each).
    Same formulas as step2 — kept here so this module is self-contained.
    """
    def _temp_score(s):
        return pd.cut(s, bins=[-np.inf,35,38,40,45,np.inf],
                      labels=[0,25,50,75,100]).astype(float).fillna(0)

    def _hi_score(s):
        return pd.cut(s, bins=[-np.inf,27,32,41,54,np.inf],
                      labels=[0,25,50,75,100]).astype(float).fillna(0)

    def _aqi_score(s):
        return pd.cut(s, bins=[-np.inf,50,100,200,300,400,np.inf],
                      labels=[0,15,40,60,80,100]).astype(float).fillna(0)

    def _drought_score(df_):
        score  = df_["drought_flag"].fillna(0) * 40
        score += (df_["dry_days_streak"].clip(0,15) / 15) * 30
        score += (-df_["spi_30"].clip(-3,0) / 3) * 30
        return np.clip(score.fillna(0), 0, 100)

    def _compound_score(df_):
        score  = df_["compound_heat_aqi"].fillna(0)      * 30
        score += df_["compound_heat_drought"].fillna(0)  * 25
        score += df_["compound_heat_humidity"].fillna(0) * 20
        score += df_["triple_compound"].fillna(0)        * 25
        score += (df_["consec_hot_days"].clip(0,7) / 7)  * 20
        return np.clip(score.fillna(0), 0, 100)

    pillars = pd.DataFrame({
        "Temperature" : _temp_score(df["temp_max"]),
        "Heat Index"  : _hi_score(df["heat_index"]),
        "AQI"         : _aqi_score(df["aqi"]),
        "Drought"     : _drought_score(df),
        "Compound"    : _compound_score(df),
    })
    return pillars


def score_to_risk(composite: np.ndarray) -> np.ndarray:
    composite = np.clip(composite, 0, 100)
    levels = np.zeros(len(composite), dtype=int)
    levels[composite >= 25] = 1
    levels[composite >= 50] = 2
    levels[composite >= 75] = 3
    return levels


# ════════════════════════════════════════════════════════════════════════
# METHOD 1: MUTUAL INFORMATION
# ════════════════════════════════════════════════════════════════════════

def method_mutual_information(pillars: pd.DataFrame,
                               imd_labels: pd.Series) -> np.ndarray:
    """
    Computes MI between each pillar score and the IMD label.
    Normalises to sum to 1 → these are the weights.
    Interpretation: pillars that share more information with the
    ground-truth label receive higher weight.
    """
    mi = mutual_info_classif(pillars.values, imd_labels.values,
                              discrete_features=False, random_state=42)
    mi_weights = mi / mi.sum()
    return mi_weights


# ════════════════════════════════════════════════════════════════════════
# METHOD 2: LOGISTIC REGRESSION COEFFICIENTS
# ════════════════════════════════════════════════════════════════════════

def method_logistic_regression(pillars: pd.DataFrame,
                                imd_labels: pd.Series) -> np.ndarray:
    """
    Trains a multinomial logistic regression:
        risk_level ~ Temperature + HeatIndex + AQI + Drought + Compound
    Extracts the absolute mean coefficient per feature across all classes,
    normalises → weights.
    Cross-val F1 is also printed as a validity check.
    """
    scaler   = StandardScaler()
    X        = scaler.fit_transform(pillars.values)
    y        = imd_labels.values

    lr = LogisticRegression(multi_class="multinomial", max_iter=1000,
                             C=1.0, random_state=42)
    lr.fit(X, y)

    # coef_ shape: (n_classes, n_features)
    # Take mean absolute coefficient across classes
    mean_abs_coef = np.abs(lr.coef_).mean(axis=0)
    lr_weights    = mean_abs_coef / mean_abs_coef.sum()

    # Cross-val check
    cv_f1 = cross_val_score(lr, X, y, cv=5, scoring="f1_macro").mean()
    print(f"    [LR] 5-fold macro F1 with pillar-only features: {cv_f1:.3f}")

    return lr_weights


# ════════════════════════════════════════════════════════════════════════
# METHOD 3: SCIPY OPTIMIZE — MAXIMIZE SPEARMAN CORRELATION
# ════════════════════════════════════════════════════════════════════════

def method_scipy_optimize(pillars: pd.DataFrame,
                           imd_labels: pd.Series) -> np.ndarray:
    """
    Finds weights w* = argmax Spearman(w·P, imd_labels)
    subject to: sum(w) = 1, all w >= 0.05 (no pillar gets zeroed out).

    Objective function (negated, since scipy minimizes):
        -Spearman(composite_score, imd_labels)

    Uses SLSQP (Sequential Least Squares Programming).
    """
    P   = pillars.values.astype(float)
    y   = imd_labels.values.astype(float)

    def neg_spearman(w):
        composite = P @ w
        rho, _    = spearmanr(composite, y)
        return -rho if not np.isnan(rho) else 0.0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds      = [(0.05, 0.70)] * 5     # each weight ∈ [0.05, 0.70]
    w0          = MANUAL_WEIGHTS.copy()  # start from manual weights

    result = minimize(neg_spearman, w0,
                      method="SLSQP",
                      bounds=bounds,
                      constraints=constraints,
                      options={"ftol": 1e-9, "maxiter": 500})

    opt_w = result.x / result.x.sum()   # re-normalise for numerical safety
    print(f"    [OPT] Spearman rho achieved: {-result.fun:.4f}  (converged={result.success})")
    return opt_w


# ════════════════════════════════════════════════════════════════════════
# METHOD 4: RANDOM GRID SEARCH — MAXIMIZE MACRO F1
# ════════════════════════════════════════════════════════════════════════

def method_grid_search_f1(pillars: pd.DataFrame,
                           imd_labels: pd.Series,
                           n_trials: int = 8000) -> np.ndarray:
    """
    Randomly samples n_trials weight vectors from the simplex
    (sum=1, each w >= 0.04) and evaluates macro F1 against IMD labels.
    Returns the weight vector that achieved the highest macro F1.

    This is a stochastic search — increases n_trials for finer granularity.
    """
    P = pillars.values.astype(float)
    y = imd_labels.values.astype(int)

    best_f1 = -1.0
    best_w  = MANUAL_WEIGHTS.copy()
    MIN_W   = 0.04

    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        # Sample from Dirichlet (uniform over simplex), clip to MIN_W
        w = rng.dirichlet(np.ones(5))
        w = np.clip(w, MIN_W, None)
        w = w / w.sum()

        composite = P @ w
        pred      = score_to_risk(composite)
        try:
            f1 = f1_score(y, pred, average="macro", zero_division=0)
        except Exception:
            continue

        if f1 > best_f1:
            best_f1 = f1
            best_w  = w.copy()

    print(f"    [GRID] Best macro F1: {best_f1:.4f}  (over {n_trials} trials)")
    return best_w


# ════════════════════════════════════════════════════════════════════════
# EVALUATION HELPER
# ════════════════════════════════════════════════════════════════════════

def evaluate_weights(w: np.ndarray, pillars: pd.DataFrame,
                     imd_labels: pd.Series) -> dict:
    P         = pillars.values.astype(float)
    composite = P @ w
    pred      = score_to_risk(composite)
    y         = imd_labels.values.astype(int)

    rho, _    = spearmanr(composite, y.astype(float))
    f1_mac    = f1_score(y, pred, average="macro",    zero_division=0)
    f1_wt     = f1_score(y, pred, average="weighted", zero_division=0)
    acc       = accuracy_score(y, pred)

    return {"Spearman_rho": round(rho, 4),
            "Macro_F1"    : round(f1_mac, 4),
            "Weighted_F1" : round(f1_wt, 4),
            "Accuracy"    : round(acc, 4)}


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ════════════════════════════════════════════════════════════════════════

def plot_weight_comparison(all_weights: dict):
    """
    Grouped bar chart: each method's pillar weights side by side.
    """
    methods = list(all_weights.keys())
    x       = np.arange(len(PILLAR_NAMES))
    width   = 0.15
    colors  = ["#2C3E50","#2980B9","#27AE60","#E67E22","#E74C3C"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (method, w) in enumerate(all_weights.items()):
        ax.bar(x + i*width, w, width, label=method, color=colors[i],
               edgecolor="white", alpha=0.88)

    ax.set_xticks(x + width*2)
    ax.set_xticklabels(PILLAR_NAMES, fontsize=10)
    ax.set_ylabel("Weight")
    ax.set_title("Pillar Weight Comparison — Manual vs. Data-Driven Methods",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0.2, linestyle=":", color="gray", lw=0.8, label="Uniform (0.20)")
    ax.set_ylim(0, 0.65)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/weight_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"\n  [✓] Weight comparison chart → {PLOT_DIR}/weight_comparison.png")


def plot_score_distributions(all_weights: dict, pillars: pd.DataFrame,
                              imd_labels: pd.Series):
    """
    KDE of composite scores under each method, split by IMD risk level.
    A good weight set should maximally separate the distributions.
    """
    P      = pillars.values.astype(float)
    colors = {"Manual":"#2C3E50","MI":"#2980B9","LR":"#27AE60",
              "Optimize":"#E67E22","GridSearch":"#E74C3C"}
    risk_colors = {0:"#2ECC71",1:"#F1C40F",2:"#E67E22",3:"#E74C3C"}
    risk_names  = {0:"Low",1:"Moderate",2:"High",3:"Severe"}

    fig, axes = plt.subplots(1, len(all_weights), figsize=(4*len(all_weights), 4),
                              sharey=True)
    if len(all_weights) == 1:
        axes = [axes]

    for ax, (method, w) in zip(axes, all_weights.items()):
        composite = P @ w
        for lvl in [0,1,2,3]:
            vals = composite[imd_labels.values == lvl]
            if len(vals) < 5:
                continue
            ax.hist(vals, bins=25, alpha=0.55,
                    color=risk_colors[lvl], label=risk_names[lvl], density=True)
        ax.set_title(method, fontsize=9, fontweight="bold")
        ax.set_xlabel("Composite Score")
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Density")
    handles = [mpatches.Patch(color=risk_colors[k], label=risk_names[k]) for k in range(4)]
    fig.legend(handles=handles, loc="upper right", fontsize=8, title="IMD Risk")
    fig.suptitle("Composite Score Distribution by IMD Risk Level",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/score_distribution_by_method.png", bbox_inches="tight")
    plt.close()
    print(f"  [✓] Score distribution plot → {PLOT_DIR}/score_distribution_by_method.png")


# ════════════════════════════════════════════════════════════════════════
# PATCH step2 WITH OPTIMAL WEIGHTS
# ════════════════════════════════════════════════════════════════════════

def patch_step2_weights(optimal_weights: np.ndarray, method_name: str,
                         step2_path: str = "step2_risk_labeling.py"):
    """
    Overwrites the W_* constants in step2_risk_labeling.py with
    the learned optimal weights.
    """
    if not os.path.exists(step2_path):
        print(f"  [!] {step2_path} not found — weights not patched.")
        return

    with open(step2_path, "r") as f:
        src = f.read()

    w = optimal_weights
    new_block = (
        f"# ── PILLAR WEIGHTS (auto-optimised by step2b — method: {method_name}) ─\n"
        f"W_TEMP    = {w[0]:.4f}\n"
        f"W_HI      = {w[1]:.4f}\n"
        f"W_AQI     = {w[2]:.4f}\n"
        f"W_DROUGHT = {w[3]:.4f}\n"
        f"W_COMPOUND= {w[4]:.4f}\n"
    )

    # Replace the manual weight block
    import re
    pattern = (r"# ── PILLAR WEIGHTS.*?\n"
               r"W_TEMP\s*=.*?\n"
               r"W_HI\s*=.*?\n"
               r"W_AQI\s*=.*?\n"
               r"W_DROUGHT\s*=.*?\n"
               r"W_COMPOUND\s*=.*?\n")
    new_src, n = re.subn(pattern, new_block, src, flags=re.DOTALL)

    if n == 0:
        print("  [!] Could not locate weight block in step2. Manual patch needed.")
        return

    with open(step2_path, "w") as f:
        f.write(new_src)

    print(f"  [✓] step2_risk_labeling.py patched with {method_name} weights.")


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*65)
    print("  STEP 2b ─ DATA-DRIVEN PILLAR WEIGHT OPTIMIZATION")
    print("="*65)

    # ── Load processed data ───────────────────────────────────────────
    path = os.path.join(PROC_DIR, "all_cities.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run step1_preprocessing.py first.")

    df = pd.read_csv(path, parse_dates=["date"])
    df.dropna(subset=["temp_max","heat_index","aqi",
                       "drought_flag","dry_days_streak","spi_30",
                       "compound_heat_aqi","compound_heat_drought",
                       "compound_heat_humidity","triple_compound",
                       "consec_hot_days"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"\n  Dataset: {len(df):,} rows across {df['city'].nunique()} cities")

    # ── Build pillar matrix & IMD ground truth ────────────────────────
    pillars    = build_pillar_matrix(df)
    imd_labels = get_imd_ground_truth(df)

    print(f"\n  IMD Label distribution:")
    for k,v in imd_labels.value_counts().sort_index().items():
        print(f"    {k} ({['Normal','Hot','HeatWave','Severe'][k]}): {v:,} days ({v/len(df)*100:.1f}%)")

    # ── RUN ALL 4 METHODS ────────────────────────────────────────────
    print("\n" + "-"*65)
    print("  Running weight optimization methods...")
    print("-"*65)

    all_weights = {"Manual": MANUAL_WEIGHTS}

    print("\n  Method 1: Mutual Information")
    mi_w = method_mutual_information(pillars, imd_labels)
    all_weights["MI"] = mi_w

    print("\n  Method 2: Logistic Regression Coefficients")
    lr_w = method_logistic_regression(pillars, imd_labels)
    all_weights["LR"] = lr_w

    print("\n  Method 3: scipy.optimize (Spearman maximization)")
    opt_w = method_scipy_optimize(pillars, imd_labels)
    all_weights["Optimize"] = opt_w

    print("\n  Method 4: Random Grid Search (F1 maximization)")
    gs_w = method_grid_search_f1(pillars, imd_labels, n_trials=8000)
    all_weights["GridSearch"] = gs_w

    # ── COMPARISON TABLE ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("  RESULTS COMPARISON")
    print("="*65)

    header = f"  {'Method':14s}  {'Spearman':>10s}  {'Macro F1':>10s}  {'Weighted F1':>12s}  {'Accuracy':>10s}"
    print(header)
    print("  " + "-"*62)

    evaluations = {}
    for method, w in all_weights.items():
        ev = evaluate_weights(w, pillars, imd_labels)
        evaluations[method] = ev
        print(f"  {method:14s}  {ev['Spearman_rho']:>10.4f}  {ev['Macro_F1']:>10.4f}"
              f"  {ev['Weighted_F1']:>12.4f}  {ev['Accuracy']:>10.4f}")

    print("\n  Per-method weights:")
    print(f"  {'Pillar':14s}", end="")
    for m in all_weights:
        print(f"  {m:>11s}", end="")
    print()
    print("  " + "-"*76)
    for i, pname in enumerate(PILLAR_NAMES):
        print(f"  {pname:14s}", end="")
        for w in all_weights.values():
            print(f"  {w[i]:>11.4f}", end="")
        print()

    # ── SELECT BEST BY MACRO F1 ───────────────────────────────────────
    best_method = max(evaluations, key=lambda k: evaluations[k]["Macro_F1"])
    best_w      = all_weights[best_method]
    print(f"\n  ★ Best method by Macro F1: {best_method}")
    print(f"    Weights: { {n: round(w,4) for n,w in zip(PILLAR_NAMES, best_w)} }")

    # ── INTERVAL OPTIMIZATION ─────────────────────────────────────────
    from sklearn.tree import DecisionTreeClassifier
    print("\n" + "="*65)
    print("  INTERVAL / THRESHOLD OPTIMIZATION")
    print("="*65)
    
    best_composite_scores = (pillars.values.astype(float) @ best_w)
    y_true = imd_labels.values.astype(int)
    
    def eval_thresholds(t1, t2, t3):
        pred = np.zeros(len(best_composite_scores), dtype=int)
        pred[best_composite_scores >= t1] = 1
        pred[best_composite_scores >= t2] = 2
        pred[best_composite_scores >= t3] = 3
        return f1_score(y_true, pred, average="macro", zero_division=0)

    f1_default = eval_thresholds(25.0, 50.0, 75.0)
    
    # Grid Search
    best_gs_f1 = -1
    best_gs_thresh = (25.0, 50.0, 75.0)
    for t1 in np.arange(15, 40, 1.0):
        for t2 in np.arange(t1 + 5, 65, 1.0):
            for t3 in np.arange(t2 + 5, 90, 1.0):
                f1 = eval_thresholds(t1, t2, t3)
                if f1 > best_gs_f1:
                    best_gs_f1 = f1
                    best_gs_thresh = (t1, t2, t3)
                    
    gs_t1, gs_t2, gs_t3 = best_gs_thresh
    
    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=4, random_state=42)
    dt.fit(best_composite_scores.reshape(-1, 1), y_true)
    thresholds = np.sort(dt.tree_.threshold[dt.tree_.threshold != -2.0])
    
    if len(thresholds) >= 3:
        dt_t1, dt_t2, dt_t3 = thresholds[:3]
    else:
        dt_t1, dt_t2, dt_t3 = 25.0, 50.0, 75.0
        
    f1_dt = eval_thresholds(dt_t1, dt_t2, dt_t3)
    
    print(f"  [Default]       Thresholds: 25.0, 50.0, 75.0      → Macro F1: {f1_default:.4f}")
    if len(thresholds) >= 3:
        print(f"  [Decision Tree] Thresholds: {dt_t1:.1f}, {dt_t2:.1f}, {dt_t3:.1f}      → Macro F1: {f1_dt:.4f}")
    else:
        print(f"  [Decision Tree] Did not find 3 valid splits.")
    print(f"  [Grid Search]   Thresholds: {gs_t1:.1f}, {gs_t2:.1f}, {gs_t3:.1f}      → Macro F1: {best_gs_f1:.4f}")

    best_thresh = [25.0, 50.0, 75.0]
    best_interval_method = "Default"
    best_interval_f1 = f1_default
    
    if len(thresholds) >= 3 and f1_dt > best_interval_f1:
        best_interval_f1 = f1_dt
        best_thresh = [float(dt_t1), float(dt_t2), float(dt_t3)]
        best_interval_method = "Decision Tree"
        
    if best_gs_f1 > best_interval_f1:
        best_interval_f1 = best_gs_f1
        best_thresh = [float(gs_t1), float(gs_t2), float(gs_t3)]
        best_interval_method = "Grid Search"
        
    print(f"\n  ★ Optimal Thresholds ({best_interval_method}): {best_thresh[0]:.1f}, {best_thresh[1]:.1f}, {best_thresh[2]:.1f}")

    # ── SAVE WEIGHTS & INTERVALS ──────────────────────────────────────
    weights_out = {
        "best_method"  : best_method,
        "pillar_names" : PILLAR_NAMES,
        "all_weights"  : {m: w.tolist() for m, w in all_weights.items()},
        "evaluations"  : evaluations,
        "best_weights" : best_w.tolist(),
        "optimal_intervals": best_thresh,
        "interval_method": best_interval_method
    }
    out_path = os.path.join(PROC_DIR, "optimal_weights.json")
    with open(out_path, "w") as f:
        json.dump(weights_out, f, indent=2)
    print(f"\n  [✓] Weights saved → {out_path}")

    # ── PATCH step2 ───────────────────────────────────────────────────
    patch_step2_weights(best_w, best_method,
                         step2_path="step2_risk_labeling.py")

    # ── VISUALISATIONS ────────────────────────────────────────────────
    plot_weight_comparison(all_weights)
    plot_score_distributions(all_weights, pillars, imd_labels)

    print("\n[Done] Pillar weight optimization complete.")
    print(f"       Re-run step2_risk_labeling.py to regenerate labels with optimal weights.\n")