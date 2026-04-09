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

    results.update(residuals_fraction_surplus=round(df["residual"].mean() / average_surplus * 100,0))

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
        ols_r_squared_pred_empirical=round(model.rsquared, 2),
        slope_pred_empirical=round(model.params["predicted_payoff"], 2),
        slope_pred_empirical_lb=round(model.conf_int().loc["predicted_payoff", 0], 2),
        slope_pred_empirical_ub=round(model.conf_int().loc["predicted_payoff", 1], 2)
    )

    return results




    