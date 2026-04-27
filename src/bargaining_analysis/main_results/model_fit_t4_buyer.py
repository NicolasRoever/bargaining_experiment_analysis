"""
Model-fit comparison table for T4 buyer payoffs.

For each model, computes three metrics on the T4 buyer sub-sample:
  - R² (generalised: 1 − SS_res / SS_tot, can be negative for non-OLS models)
  - Average residual  (mean of actual − predicted)
  - Slope of actual ~ predicted  (OLS, clustered by participant_code, with significance stars)

Models compared
---------------
1. Equilibrium payoff (``equilibrium_payoff`` from main_results.py)
2. Simple linear regression (payoff ~ valuation)
3. Equal split  (buyer gets 50 % of gains_from_trade)
4. Coase conjecture (buyer gets 100 % of gains_from_trade)
5. TILO – seller optimum  (max(valuation − 15, 0))
 conda run -n bargaining_analysis python -m src.bargaining_analysis.main_results.model_fit_t4_buyer
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.bargaining_analysis.config import BLD, OVERLEAF_TABLES
from src.bargaining_analysis.main_results.main_results import equilibrium_payoff


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""



# ---------------------------------------------------------------------------
# main computation
# ---------------------------------------------------------------------------

def compute_model_fit_metrics(df: pd.DataFrame) -> dict:
    """
    Filter T4 buyers, build predictions for each model, and return a dict of
    dicts with keys ``r2``, ``avg_residual``, ``slope``, ``slope_pval``.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned dataset (merged_data_full_excluded.csv).

    Returns
    -------
    dict[str, dict]
        {model_label: {r2, avg_residual, slope, slope_pval}}
    """
    df_t4 = (
        df[
            (df["treatment"] == "T4") &
            (df["participant_role"] == "Buyer")
        ]
        .dropna(subset=["payoff", "valuation", "gains_from_trade"])
        .copy()
    )

    y = df_t4["payoff"].values

    # --- predictions ---
    df_t4["pred_my_model"] = df_t4["valuation"].apply(equilibrium_payoff)

    ols_fit = smf.ols("payoff ~ valuation", data=df_t4).fit()
    df_t4["pred_ols"] = ols_fit.fittedvalues

    df_t4["pred_equal_split"] = 0.5 * df_t4["gains_from_trade"]
    df_t4["pred_coase"] = df_t4["gains_from_trade"]
    df_t4["pred_tilo"] = np.maximum(df_t4["valuation"] - 15.0, 0.0)

    models = {
        "My Model": "pred_my_model",
        "Linear Regression": "pred_ols",
        "Equal Split": "pred_equal_split",
        "Coase Conjecture": "pred_coase",
        "TILO": "pred_tilo",
    }

    results = {}
    for label, pred_col in models.items():
        preds = df_t4[pred_col].values

        avg_res = float(np.mean(y - preds))

        # slope + R²: actual ~ predicted, clustered SEs
        df_fit = df_t4[["payoff", pred_col, "participant_code"]].dropna().copy()
        df_fit = df_fit.rename(columns={pred_col: "predicted"})
        slope_mod = smf.ols("payoff ~ predicted", data=df_fit).fit(
            cov_type="cluster",
            cov_kwds={"groups": df_fit["participant_code"]},
        )
        slope = float(slope_mod.params["predicted"])
        slope_pval = float(slope_mod.pvalues["predicted"])
        r2 = float(slope_mod.rsquared)

        results[label] = {
            "r2": r2,
            "avg_residual": avg_res,
            "slope": slope,
            "slope_pval": slope_pval,
        }

    return results


# ---------------------------------------------------------------------------
# LaTeX table builder
# ---------------------------------------------------------------------------

def build_latex_table(metrics: dict) -> str:
    """
    Build a booktabs LaTeX table.

    Columns: one per model.
    Rows: R², Avg. Residual, Slope (with significance stars).

    Parameters
    ----------
    metrics : dict
        Output of ``compute_model_fit_metrics``.

    Returns
    -------
    str
        Complete LaTeX tabular environment (ready for \\input{}).
    """
    model_labels = list(metrics.keys())
    n = len(model_labels)

    col_spec = "l" + "c" * n

    # Column headers – use \shortstack for two-line labels
    short_labels = {
        "My Model":          r"\shortstack{Our\\Model}",
        "Linear Regression": r"\shortstack{Linear\\Regression}",
        "Equal Split":       r"\shortstack{Equal\\Split}",
        "Coase Conjecture":  r"\shortstack{Coase\\Conjecture}",
        "TILO":              r"\shortstack{TILO}",
    }
    header_cells = [short_labels.get(lbl, lbl) for lbl in model_labels]
    header_row = " & ".join([""] + header_cells) + r" \\"

    def row(label: str, values: list) -> str:
        return label + " & " + " & ".join(values) + r" \\"

    r2_vals = [f"{metrics[m]['r2']:.2f}" for m in model_labels]
    avg_res_vals = [f"{metrics[m]['avg_residual']:.2f}" for m in model_labels]
    slope_vals = [
        f"{metrics[m]['slope']:.2f}{_stars(metrics[m]['slope_pval'])}"
        for m in model_labels
    ]

    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header_row,
        r"\midrule",
        row(r"$R^2$", r2_vals),
        row("Avg.~Residual", avg_res_vals),
        row("Slope", slope_vals),
        r"\bottomrule",
        r"\end{tabular}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")
    metrics = compute_model_fit_metrics(df=df)
    latex_table = build_latex_table(metrics=metrics)

    output_path = OVERLEAF_TABLES / "model_fit_t4_buyer.tex"
    output_path.write_text(latex_table, encoding="utf-8")
    print(f"Table saved to {output_path}")
    print("\n--- Metrics ---")
    for model, vals in metrics.items():
        print(
            f"  {model:<22}  R²={vals['r2']:+.3f}  "
            f"AvgRes={vals['avg_residual']:+.3f}  "
            f"Slope={vals['slope']:.3f}{_stars(vals['slope_pval'])}"
        )
