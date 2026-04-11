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

from src.bargaining_analysis.main_results.main_results import equilibrium_payoff
from src.bargaining_analysis.helper import set_plot_theme, finalize_plot

def test_residuals_model_actual(df):
    """
    Compute residuals (observed payoff − predicted payoff) and run one-sample
    t-tests for mean = 0, globally and within each equilibrium region.
    """
    df_buyers = df[
        (df["treatment"] == "T4") &
        (df["participant_role"] == "Buyer")
    ].copy()

    df = df_buyers.copy()
    df["predicted_payoff"] = df["valuation"].apply(equilibrium_payoff)
    df["residual"] = df["payoff"] - df["predicted_payoff"]
    average_surplus = (df["valuation"]).mean()

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
        results = {
            "label": label,
            "n_pred_obs": n,
            "mean_residual_pred_obs": mean_r,
            "std_residual": std_r,
            "t_stat": t_stat,
            "p_value_res_pred_obs": p_val
        }
        return results


    results = _run_ttest("All T4 buyers", df["residual"])

    results.update(residuals_fraction_surplus=round(df["residual"].mean() / average_surplus * 100 * (-1),0))

    return results



def test_ols_prediction_actual(df):
    """
    Regress observed payoff on predicted payoff and run a joint Wald test
    for H0: intercept = 0 AND slope = 1.
    """

    results = {}

    df_buyers = df[
        (df["treatment"] == "T4") &
        (df["participant_role"] == "Buyer")
    ].copy()


    df_buyers["predicted_payoff"] = df_buyers["valuation"].apply(equilibrium_payoff)


    df = df_buyers.dropna(subset=["payoff", "predicted_payoff"])

    model = smf.ols("payoff ~ predicted_payoff", data=df).fit()

    results.update(
        ols_r_squared_pred_empirical=round(model.rsquared * 100, 0),
        slope_pred_empirical=round(model.params["predicted_payoff"], 2),
        slope_pred_empirical_lb=round(model.conf_int().loc["predicted_payoff", 0], 2),
        slope_pred_empirical_ub=round(model.conf_int().loc["predicted_payoff", 1], 2)
    )

    return results


def plot_split_gains_valuation_t4_middle_high(
    df, figsize=(9, 5), jitter_x_sd=0.08, jitter_y_sd=0.01, jitter_seed=42
):
    """
    Scatter plot of split_gains_from_trade vs. buyer valuation for T4 trade
    observations, overlaid with an OLS regression line and 95% CI band.

    Returns the figure.
    """
    df_buyers = df[
        (df["treatment"] == "T4") &
        (df["participant_role"] == "Buyer") &
        (df["valuation"] > 7.72) 
    ].copy()

    df = df_buyers[df_buyers["bargaining_outcome"] == "acceptance"].dropna(
        subset=["split_gains_from_trade", "valuation"]
    ).copy()

    model = smf.ols("split_gains_from_trade ~ valuation", data=df).fit(cov_type="HC3")

    x_grid = np.linspace(df["valuation"].min(), df["valuation"].max(), 300)
    pred = model.get_prediction({"valuation": x_grid})
    pred_df = pred.summary_frame(alpha=0.05)

    # Jitter is used only for plotting to reduce overlap of observations.
    rng = np.random.default_rng(jitter_seed)
    x_plot = df["valuation"] + rng.normal(0, jitter_x_sd, size=len(df))
    y_plot = np.clip(
        df["split_gains_from_trade"] + rng.normal(0, jitter_y_sd, size=len(df)),
        0,
        1,
    )

    set_plot_theme()
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        x_plot, y_plot,
        color=COLOR_SCHEME[0], alpha=0.45, s=22, zorder=2,
        label="Observed (trade, jittered)"
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




    