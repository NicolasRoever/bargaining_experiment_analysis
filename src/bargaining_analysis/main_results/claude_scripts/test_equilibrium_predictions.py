"""
Tests of equilibrium payoff predictions for treatment T4.

Tests included:
1. Global and region-specific residual mean-zero t-tests
2. OLS regression of observed on predicted payoff (joint test: intercept=0, slope=1)
3. Structural break test: estimated breakpoints vs. theoretical (b†=7.72, b*=21.40)
4. Discontinuity test: termination rates just below vs. just above cutoffs

conda run -n bargaining_analysis python -m src.bargaining_analysis.main_results.test_equilibrium_predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
import statsmodels.formula.api as smf
import statsmodels.api as sm
from src.bargaining_analysis.config import BLD, OVERLEAF_FIGURES, COLOR_SCHEME
from src.bargaining_analysis.helper import set_plot_theme, finalize_plot


# ─── Equilibrium payoff (copied from main_results.py) ────────────────────────

B_DAGGER = 7.72   # lower cutoff: below this, buyer gets 0
B_STAR   = 21.40  # upper cutoff: above this, buyer gets b/2

def equilibrium_payoff(x, c=0.05, r=0.01):
    if x <= B_DAGGER:
        return 0
    elif x >= B_STAR:
        return x - 10.7
    else:
        b = x
        return ((r * b + 2 * c) / (r * B_STAR + 2 * c)) * (b / 2 + c / r) - c / r


# ─── Helper: print section headers ───────────────────────────────────────────

def _header(title):
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _subheader(title):
    print(f"\n--- {title} ---")


# ─── Test 1: Residual mean-zero t-tests ──────────────────────────────────────

def test_residuals(df_buyers):
    """
    Compute residuals (observed payoff − predicted payoff) and run one-sample
    t-tests for mean = 0, globally and within each equilibrium region.
    """
    _header("TEST 1: Residual Mean-Zero t-Tests")

    df = df_buyers.copy()
    df["predicted_payoff"] = df["valuation"].apply(equilibrium_payoff)
    df["residual"] = df["payoff"] - df["predicted_payoff"]

    def _run_ttest(label, residuals):
        n = len(residuals)
        mean_r = residuals.mean()
        std_r  = residuals.std(ddof=1)
        t_stat, p_val = stats.ttest_1samp(residuals, popmean=0)
        print(
            f"  {label:<35}  n={n:>4}  "
            f"mean residual={mean_r:+.4f}  "
            f"std={std_r:.4f}  "
            f"t={t_stat:+.3f}  p={p_val:.4f}"
        )

    _subheader("Global")
    _run_ttest("All T4 buyers", df["residual"])

    _subheader("By equilibrium region")
    regions = {
        f"Region 1  (val ≤ {B_DAGGER})":
            df[df["valuation"] <= B_DAGGER]["residual"],
        f"Region 2  ({B_DAGGER} < val < {B_STAR})":
            df[(df["valuation"] > B_DAGGER) & (df["valuation"] < B_STAR)]["residual"],
        f"Region 3  (val ≥ {B_STAR})":
            df[df["valuation"] >= B_STAR]["residual"],
    }
    for label, res in regions.items():
        if len(res) >= 2:
            _run_ttest(label, res)
        else:
            print(f"  {label:<35}  not enough observations (n={len(res)})")

    return df  # return enriched df for downstream tests


# ─── Test 2: OLS observed ~ predicted (intercept=0, slope=1) ─────────────────

def test_ols_prediction(df_with_residuals):
    """
    Regress observed payoff on predicted payoff and run a joint Wald test
    for H0: intercept = 0 AND slope = 1.
    """
    _header("TEST 2: OLS Observed ~ Predicted  (H0: intercept=0, slope=1)")

    df = df_with_residuals.dropna(subset=["payoff", "predicted_payoff"])

    model = smf.ols("payoff ~ predicted_payoff", data=df).fit()

    print(model.summary().tables[1])  # coefficient table

    # Joint Wald test: alpha=0 and beta=1
    # Equivalent to testing residual of the restricted model
    # Use hypothesis matrix R * params = q
    # R = [[1, 0], [0, 1]],  q = [0, 1]
    hypotheses = "Intercept = 0, predicted_payoff = 1"
    wald = model.f_test(["Intercept = 0", "predicted_payoff = 1"])

    print(f"\n  Joint Wald test (intercept=0 AND slope=1):")
    print(f"    F-statistic : {wald.fvalue:.4f}")
    print(f"    p-value     : {wald.pvalue:.4f}")
    if wald.pvalue < 0.05:
        print("    → REJECT H0 at 5%: model level or sensitivity is off.")
    else:
        print("    → Fail to reject H0 at 5%: predictions are well-calibrated.")

    print(f"\n  R²  = {model.rsquared:.4f}   (adj. R² = {model.rsquared_adj:.4f})")
    print(f"  RMSE = {np.sqrt(model.mse_resid):.4f}")


# ─── Test 3: Structural break – estimated breakpoints vs. theoretical ─────────

def _piecewise_two_breaks(x, bp1, bp2):
    """Two-breakpoint piecewise linear design matrix columns (beside constant)."""
    return np.column_stack([
        x,
        np.maximum(x - bp1, 0),
        np.maximum(x - bp2, 0),
    ])


def _ssr(breakpoints, x, y):
    bp1, bp2 = sorted(breakpoints)
    X = np.column_stack([np.ones(len(x)), _piecewise_two_breaks(x, bp1, bp2)])
    try:
        beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return res[0] if len(res) else np.sum((y - X @ beta) ** 2)
    except Exception:
        return np.inf


def test_structural_breaks(df_buyers):
    """
    Grid-search over candidate breakpoints to find the pair (bp1, bp2) that
    minimises SSR in a two-segment piecewise linear model.  Compare to the
    theoretical values (7.72, 21.40).
    """
    _header("TEST 3: Structural Break – Estimated Breakpoints vs. Theoretical")

    df = df_buyers.dropna(subset=["valuation", "payoff"]).sort_values("valuation")
    x = df["valuation"].values
    y = df["payoff"].values

    # Grid search: 5th–95th percentile range to avoid boundary effects
    grid = np.linspace(np.percentile(x, 5), np.percentile(x, 95), 80)

    best_ssr   = np.inf
    best_bp    = (np.nan, np.nan)

    for i, bp1 in enumerate(grid):
        for bp2 in grid[i + 1:]:
            s = _ssr([bp1, bp2], x, y)
            if s < best_ssr:
                best_ssr = s
                best_bp  = (bp1, bp2)

    est_bp1, est_bp2 = best_bp
    print(f"\n  Theoretical breakpoints : b† = {B_DAGGER:.2f},  b* = {B_STAR:.2f}")
    print(f"  Estimated breakpoints   : b† = {est_bp1:.2f},  b* = {est_bp2:.2f}")
    print(f"  Δ lower cutoff          : {est_bp1 - B_DAGGER:+.2f}")
    print(f"  Δ upper cutoff          : {est_bp2 - B_STAR:+.2f}")

    # Also run single-break Chow-style F-test at theoretical breakpoints
    _subheader("Chow test at theoretical breakpoints")
    for label, bp in [(f"b† = {B_DAGGER}", B_DAGGER), (f"b* = {B_STAR}", B_STAR)]:
        below = df[df["valuation"] <  bp]
        above = df[df["valuation"] >= bp]
        if len(below) < 5 or len(above) < 5:
            print(f"  {label}: not enough obs in one segment")
            continue

        def _ols_ssr(d):
            if len(d) < 3:
                return 0.0
            res = smf.ols("payoff ~ valuation", data=d).fit()
            return res.ssr

        ssr_pooled    = _ols_ssr(df)
        ssr_below     = _ols_ssr(below)
        ssr_above     = _ols_ssr(above)
        ssr_unrestr   = ssr_below + ssr_above
        k = 2   # number of parameters (intercept + slope)
        n = len(df)
        chow_f = ((ssr_pooled - ssr_unrestr) / k) / (ssr_unrestr / (n - 2 * k))
        p_chow = 1 - stats.f.cdf(chow_f, k, n - 2 * k)
        print(f"  Chow F at {label:<12}: F={chow_f:.3f}  p={p_chow:.4f}  "
              f"(n_below={len(below)}, n_above={len(above)})")


# ─── Test 4: Termination-rate discontinuity at cutoffs ───────────────────────

def test_termination_discontinuity(df_buyers, bandwidth=3.0):
    """
    Compare player-termination rates just below vs. just above each theoretical
    cutoff (b† and b*) using a two-sample proportion test (z-test).

    bandwidth : valuation window on each side of the cutoff (default ±3)
    """
    _header("TEST 4: Termination-Rate Discontinuity at Cutoffs")

    # Player termination indicator (1 = player terminated, 0 = computer or acceptance)
    df = df_buyers.copy()
    df["player_terminated"] = (df["bargaining_outcome"] == "Player").astype(int)

    print(f"\n  Bandwidth = ±{bandwidth} around each cutoff")

    for label, bp in [(f"b† = {B_DAGGER}", B_DAGGER), (f"b* = {B_STAR}", B_STAR)]:
        below = df[(df["valuation"] >= bp - bandwidth) & (df["valuation"] <  bp)]
        above = df[(df["valuation"] >= bp)             & (df["valuation"] <  bp + bandwidth)]

        n_below = len(below)
        n_above = len(above)
        if n_below == 0 or n_above == 0:
            print(f"  {label}: no observations in one window")
            continue

        p_below = below["player_terminated"].mean()
        p_above = above["player_terminated"].mean()

        # Two-proportion z-test
        count = np.array([below["player_terminated"].sum(), above["player_terminated"].sum()])
        nobs  = np.array([n_below, n_above])
        from statsmodels.stats.proportion import proportions_ztest
        z_stat, p_val = proportions_ztest(count, nobs)

        print(
            f"\n  Cutoff {label}:"
            f"\n    Below [{bp - bandwidth:.2f}, {bp:.2f})  "
            f"term. rate = {p_below:.3f}  (n={n_below})"
            f"\n    Above [{bp:.2f}, {bp + bandwidth:.2f})  "
            f"term. rate = {p_above:.3f}  (n={n_above})"
            f"\n    z = {z_stat:+.3f}   p = {p_val:.4f}"
        )
        if p_val < 0.05:
            print(f"    → Significant discontinuity at 5%.")
        else:
            print(f"    → No significant discontinuity at 5%.")


# ─── Plot: observed data + theoretical curve + estimated piecewise fit ────────

def _fit_piecewise(x, y, bp1, bp2):
    """Return fitted values from a 2-breakpoint piecewise linear OLS."""
    X = np.column_stack([
        np.ones(len(x)),
        x,
        np.maximum(x - bp1, 0),
        np.maximum(x - bp2, 0),
    ])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return X @ beta, beta


def plot_breakpoint_comparison(df_buyers, figsize=(10, 6)):
    """
    Scatter of observed buyer payoff vs. valuation overlaid with:
    - Theoretical equilibrium payoff curve (b†=7.72, b*=21.40)
    - Grid-search piecewise linear fit at the estimated breakpoints

    Returns the figure.
    """
    df = df_buyers.dropna(subset=["valuation", "payoff"]).sort_values("valuation")
    x = df["valuation"].values
    y = df["payoff"].values

    # ── re-run grid search to get estimated breakpoints ──────────────────────
    grid = np.linspace(np.percentile(x, 5), np.percentile(x, 95), 80)
    best_ssr, best_bp = np.inf, (np.nan, np.nan)
    for i, bp1 in enumerate(grid):
        for bp2 in grid[i + 1:]:
            X = np.column_stack([
                np.ones(len(x)),
                x,
                np.maximum(x - bp1, 0),
                np.maximum(x - bp2, 0),
            ])
            beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
            ssr = res[0] if len(res) else np.sum((y - X @ beta) ** 2)
            if ssr < best_ssr:
                best_ssr = ssr
                best_bp = (bp1, bp2)

    est_bp1, est_bp2 = best_bp

    # ── smooth x-grid for curves ──────────────────────────────────────────────
    x_grid = np.linspace(0, 30, 500)

    # Theoretical curve
    y_theory = np.array([equilibrium_payoff(v) for v in x_grid])

    # Estimated piecewise linear fit evaluated on x_grid
    _, beta_est = _fit_piecewise(x, y, est_bp1, est_bp2)
    X_grid = np.column_stack([
        np.ones(len(x_grid)),
        x_grid,
        np.maximum(x_grid - est_bp1, 0),
        np.maximum(x_grid - est_bp2, 0),
    ])
    y_est = X_grid @ beta_est

    # ── plot ──────────────────────────────────────────────────────────────────
    set_plot_theme()
    fig, ax = plt.subplots(figsize=figsize)

    # scatter — colour by bargaining outcome
    outcome_color_map = {
        "Player":             COLOR_SCHEME[0],
        "Random_Termination": COLOR_SCHEME[1],
        "acceptance":         COLOR_SCHEME[2],
    }
    colors = df["bargaining_outcome"].map(outcome_color_map).fillna("grey")
    ax.scatter(df["valuation"], df["payoff"], c=colors, alpha=0.35, s=18, zorder=1)

    # theoretical curve
    ax.plot(x_grid, y_theory, color=COLOR_SCHEME[3], linewidth=2.2,
            label=rf"Theoretical (b$^\dagger$={B_DAGGER}, b*={B_STAR})", zorder=3)

    # vertical lines at theoretical cutoffs
    for bp, ls in [(B_DAGGER, "--"), (B_STAR, "--")]:
        ax.axvline(bp, color=COLOR_SCHEME[3], linestyle=ls, linewidth=1.2, alpha=0.7)

    # estimated piecewise fit
    ax.plot(x_grid, y_est, color=COLOR_SCHEME[4], linewidth=2.2, linestyle="-.",
            label=rf"Estimated piecewise (b$^\dagger$={est_bp1:.2f}, b*={est_bp2:.2f})",
            zorder=3)

    # vertical lines at estimated cutoffs
    for bp, ls in [(est_bp1, "-."), (est_bp2, "-.")]:
        ax.axvline(bp, color=COLOR_SCHEME[4], linestyle=ls, linewidth=1.2, alpha=0.7)

    # legend for scatter outcomes
    legend_scatter = [
        mlines.Line2D([], [], color=COLOR_SCHEME[0], marker="o", linestyle="None",
                      markersize=6, label="Player termination"),
        mlines.Line2D([], [], color=COLOR_SCHEME[1], marker="o", linestyle="None",
                      markersize=6, label="Computer termination"),
        mlines.Line2D([], [], color=COLOR_SCHEME[2], marker="o", linestyle="None",
                      markersize=6, label="Acceptance"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_scatter, fontsize=9, frameon=True,
              facecolor="white", edgecolor="black")

    ax.set_xlabel("Buyer Valuation")
    ax.set_ylabel("Buyer Payoff")

    finalize_plot(ax)
    return fig


# ─── Test 5: Split gains from trade vs. valuation (trade occurs only) ─────────

def test_split_gains_valuation(df_buyers):
    """
    OLS regression of split_gains_from_trade on valuation for T4 buyers
    where trade occurs (bargaining_outcome == 'acceptance').

    Tests whether buyers with higher valuations systematically receive a
    different share of the gains from trade.
    """
    _header("TEST 5: Split Gains from Trade ~ Valuation  (trade observations only)")

    df = df_buyers[df_buyers["bargaining_outcome"] == "acceptance"].dropna(
        subset=["split_gains_from_trade", "valuation"]
    ).copy()

    print(f"\n  Subsample: T4 buyers who traded  (n = {len(df)})")
    print(f"  split_gains_from_trade  mean={df['split_gains_from_trade'].mean():.4f}  "
          f"std={df['split_gains_from_trade'].std():.4f}")

    model = smf.ols("split_gains_from_trade ~ valuation", data=df).fit(cov_type="HC3")
    print(model.summary().tables[1])
    print(f"\n  R²  = {model.rsquared:.4f}   (adj. R² = {model.rsquared_adj:.4f})")

    # t-test: slope = 0
    slope_t   = model.tvalues["valuation"]
    slope_p   = model.pvalues["valuation"]
    slope_ci  = model.conf_int().loc["valuation"]
    print(f"\n  Slope on valuation: {model.params['valuation']:+.4f}  "
          f"t={slope_t:+.3f}  p={slope_p:.4f}  "
          f"95% CI [{slope_ci[0]:+.4f}, {slope_ci[1]:+.4f}]")
    if slope_p < 0.05:
        print("  → Significant relationship at 5%: split varies systematically with valuation.")
    else:
        print("  → No significant relationship at 5%: split does not vary with valuation.")

    return model


def plot_split_gains_valuation(df_buyers, figsize=(9, 5)):
    """
    Scatter plot of split_gains_from_trade vs. buyer valuation for T4 trade
    observations, overlaid with an OLS regression line and 95% CI band.

    Returns the figure.
    """
    df = df_buyers[df_buyers["bargaining_outcome"] == "acceptance"].dropna(
        subset=["split_gains_from_trade", "valuation"]
    ).copy()

    model = smf.ols("split_gains_from_trade ~ valuation", data=df).fit(cov_type="HC3")

    x_grid = np.linspace(df["valuation"].min(), df["valuation"].max(), 300)
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
    pred = model.get_prediction({"valuation": x_grid})
    pred_df = pred.summary_frame(alpha=0.05)

    set_plot_theme()
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        df["valuation"], df["split_gains_from_trade"],
        color=COLOR_SCHEME[0], alpha=0.45, s=22, zorder=2,
        label="Observed (trade)"
    )
    ax.plot(x_grid, pred_df["mean"], color=COLOR_SCHEME[1], linewidth=2.0,
            label=f"OLS fit (slope={model.params['valuation']:+.3f}, "
                  f"p={model.pvalues['valuation']:.3f})", zorder=3)
    ax.fill_between(
        x_grid, pred_df["mean_ci_lower"], pred_df["mean_ci_upper"],
        color=COLOR_SCHEME[1], alpha=0.15, zorder=1, label="95\\% CI"
    )

    ax.set_xlabel("Buyer Valuation")
    ax.set_ylabel("Split of Gains from Trade (Buyer Share)")
    ax.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="black")

    finalize_plot(ax)
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_all_tests(df):
    """
    Run all equilibrium prediction tests on treatment T4 buyers.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned dataset (BLD / "data" / "merged_data_full_excluded.csv").
    """
    df_buyers = df[
        (df["treatment"] == "T4") &
        (df["participant_role"] == "Buyer")
    ].copy()

    print(f"\nSample: T4 buyers  (n = {len(df_buyers)})")
    print(f"  Valuation range: [{df_buyers['valuation'].min():.2f}, "
          f"{df_buyers['valuation'].max():.2f}]")

    df_with_residuals = test_residuals(df_buyers)
    test_ols_prediction(df_with_residuals)
    test_structural_breaks(df_buyers)
    test_termination_discontinuity(df_buyers)
    test_split_gains_valuation(df_buyers)

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)

    return df_buyers


if __name__ == "__main__":
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")
    df_buyers = run_all_tests(df)

    fig = plot_breakpoint_comparison(df_buyers)
    out = OVERLEAF_FIGURES / "test3_breakpoint_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\n  Figure saved to: {out}")

    fig5 = plot_split_gains_valuation(df_buyers)
    out5 = OVERLEAF_FIGURES / "test5_split_gains_valuation.pdf"
    fig5.savefig(out5, bbox_inches="tight")
    print(f"\n  Figure saved to: {out5}")
