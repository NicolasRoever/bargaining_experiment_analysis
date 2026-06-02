"""Regression table for buyer surplus share.

This script filters the sample to trades with positive gains from trade,
estimates clustered OLS specifications separately for no-TA-cost and
TA-cost subsamples, writes a LaTeX table, and prints the same table to
the console.

Run with:
    python -m src.bargaining_analysis.main_results.surplus_share_regression_table

"""

from pathlib import Path
from typing import Any, cast

import pandas as pd
import statsmodels.formula.api as smf
from pystout import pystout

from src.bargaining_analysis.config import BLD, OVERLEAF_TABLES, VARLABELS_REGRESSION
from src.bargaining_analysis.helper import fix_pandas_append_error
from src.bargaining_analysis.main_results.surplus_share_analysis import _prepare


def _fit_clustered_ols(formula: str, data: pd.DataFrame, cluster_col: str):
    """Fit clustered OLS after dropping missing values on all model inputs."""
    model_data = data.dropna(
        subset=[
            "buyer_share_realised",
            "information_asymmetry",
            "gains_from_trade",
            cluster_col,
        ]
    ).copy()

    if model_data[cluster_col].nunique() < 2:
        return smf.ols(formula, data=model_data).fit()

    return smf.ols(formula, data=model_data).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_data[cluster_col]},
    )


def _estimate_models(df: pd.DataFrame):
    """Estimate the four requested models split by TA-cost condition."""
    _, trades = _prepare(df)

    reg_df = trades[trades["gains_from_trade"] > 0].copy()

    no_cost = reg_df[reg_df["TA_costs"] == 0.0].copy()
    with_cost = reg_df[reg_df["TA_costs"] == 0.05].copy()

    model_no_cost = _fit_clustered_ols(
        "buyer_share_realised ~ C(information_asymmetry)",
        no_cost,
        "participant_code",
    )
    model_no_cost_gft = _fit_clustered_ols(
        "buyer_share_realised ~ C(information_asymmetry) + gains_from_trade",
        no_cost,
        "participant_code",
    )
    model_with_cost = _fit_clustered_ols(
        "buyer_share_realised ~ C(information_asymmetry)",
        with_cost,
        "participant_code",
    )
    model_with_cost_gft = _fit_clustered_ols(
        "buyer_share_realised ~ C(information_asymmetry) + gains_from_trade",
        with_cost,
        "participant_code",
    )

    return model_no_cost, model_no_cost_gft, model_with_cost, model_with_cost_gft


def build_surplus_share_regression_table(df: pd.DataFrame, output_path: Path) -> str:
    """Estimate the requested models and write the LaTeX table to disk."""
    model_no_cost, model_no_cost_gft, model_with_cost, model_with_cost_gft = _estimate_models(df)

    labels = {
        **VARLABELS_REGRESSION,
        "C(information_asymmetry)[T.two-sided]": "Two-sided Private Info",
        "gains_from_trade": "Gains from trade",
        "Intercept": "Constant",
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fix_pandas_append_error()
    pystout(
        models=[model_no_cost, model_no_cost_gft, model_with_cost, model_with_cost_gft],
         mgroups={"No TA costs": [1, 2], "TA costs": [3, 4]},
        file=str(output_path),
        digits=3,
        stars={0.1: "*", 0.05: "**", 0.01: "***"},
        varlabels=labels,
        exogvars=["C(information_asymmetry)[T.two-sided]", "gains_from_trade"],
        addrows={
            "GFT control": ["No", "Yes", "No", "Yes"],
        },
        modstat={"nobs": "Obs", "rsquared_adj": r"Adj. $R^2$"},
    )

    return output_path.read_text()


def main() -> None:
    """Run the regression table workflow and print the LaTeX to stdout."""
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")
    output_path = OVERLEAF_TABLES / "surplus_share_regression_table.tex"
    tex = build_surplus_share_regression_table(df, output_path)
    print(tex)


if __name__ == "__main__":
    main()