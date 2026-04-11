"""Run with
conda run -n bargaining_analysis python -m src.bargaining_analysis.main_results.seller_termination_punishment

Tests whether sellers are more likely to terminate after a buyer rejection under
one-sided uncertainty (T3/T4 — seller knows the buyer's valuation) compared to
symmetric uncertainty (T1/T2 — seller does not know the buyer's valuation).

Identification logic:
- "Buyer rejected" = buyer made at least one counter-offer (number_of_offers >= 2).
- Sample restricted to gains_from_trade > 0 so that the seller in T3/T4 can infer
  the buyer has positive gains and any rejection is strategic (not genuine uncertainty).
- Unit of analysis: one observation per negotiation (seller row).
"""

import re

import pandas as pd
import statsmodels.formula.api as smf
from pystout import pystout

from src.bargaining_analysis.config import BLD, OVERLEAF_ROOT, OVERLEAF_TABLES, VARLABELS_REGRESSION
from src.bargaining_analysis.helper import fix_pandas_append_error, inject_values


def run_seller_termination_after_rejection(
    df: pd.DataFrame,
    varlabels_regression: dict,
    output_path,
    verbose: bool = True,
) -> dict:
    """
    OLS regression of seller_terminated on one_sided, controlling for
    gains_from_trade and TA_costs. Sample: negotiations with at least one
    buyer counter-offer (number_of_offers >= 2) and gains_from_trade > 0.

    Returns a dict of key values for injection into main.tex.
    """
    # Work at negotiation level (one row per negotiation) using seller rows only
    seller_df = df[df["participant_role"] == "Seller"].copy()

    # Restrict to positive gains from trade (seller in T3/T4 can infer buyer has surplus)
    seller_df = seller_df[seller_df["gains_from_trade"] > 0].copy()

    # Filter to negotiations with at least one buyer counter-offer
    seller_df = seller_df[seller_df["number_of_offers"] >= 2].copy()

    # Binary indicator: 1 = one-sided uncertainty (T3/T4), 0 = symmetric (T1/T2)
    seller_df["one_sided"] = (seller_df["information_asymmetry"] == "one-sided").astype(int)

    # Session FE control
    seller_df["group_session"] = (
        seller_df["group_id_in_session"].astype(str) + "_" + seller_df["session_id"].astype(str)
    )

    n_total = len(seller_df)
    n_terminated = int(seller_df["player_terminated"].sum())

    # Raw termination rates by information asymmetry
    rates = seller_df.groupby("information_asymmetry")["player_terminated"].mean()

    if verbose:
        print("\n=== Seller Termination After Buyer Rejection ===")
        print(f"Sample: {n_total} negotiations (gains_from_trade > 0, buyer counter-offered)")
        print(f"Seller terminated: {n_terminated} ({n_terminated / n_total * 100:.1f}%)")
        print("\nRaw rates by information asymmetry:")
        for info_type, rate in rates.items():
            n = seller_df[seller_df["information_asymmetry"] == info_type].shape[0]
            print(f"  {info_type}: {rate * 100:.1f}%  (n={n})")

    def _fit(formula, data):
        data_clean = data.dropna(subset=_vars_from_formula(formula))
        return smf.ols(formula=formula, data=data_clean).fit(
            cov_type="cluster",
            cov_kwds={"groups": data_clean["participant_label"]},
        )

    # Model 1: one_sided + gains_from_trade + TA_costs
    model_main = _fit(
        "player_terminated ~ one_sided + gains_from_trade + TA_costs",
        seller_df,
    )

    # Model 2: add session FE as robustness
    model_session_fe = _fit(
        "player_terminated ~ one_sided + gains_from_trade + TA_costs + C(group_session)",
        seller_df,
    )

    if verbose:
        print("\n=== Main regression: player_terminated ~ one_sided + gains_from_trade + TA_costs ===")
        print(model_main.summary2().tables[1].to_string(float_format=lambda x: f"{x:0.4f}"))
        print(f"n = {int(model_main.nobs)}, adj. R² = {model_main.rsquared_adj:0.4f}")
        print("\n=== Robustness: + Session FE ===")
        print(model_session_fe.summary2().tables[1].to_string(float_format=lambda x: f"{x:0.4f}"))
        print(f"n = {int(model_session_fe.nobs)}, adj. R² = {model_session_fe.rsquared_adj:0.4f}")

    extra_labels = {
        "one_sided": "One-Sided Uncertainty",
        "TA_costs": "TA Costs",
    }
    combined_labels = {**varlabels_regression, **extra_labels}

    fix_pandas_append_error()

    pystout(
        models=[model_main, model_session_fe],
        endog_names=False,
        mgroups={"Seller Terminated": [1, 2]},
        file=output_path,
        digits=3,
        stars={0.1: "*", 0.05: "**", 0.01: "***"},
        varlabels=combined_labels,
        exogvars=["one_sided", "gains_from_trade", "TA_costs"],
        addrows={
            "Session FE": ["", "\\checkmark"],
        },
        modstat={"nobs": "Obs", "rsquared_adj": r"Adj. R\sym{2}"},
    )

    if verbose:
        print(f"\nSaved table to {output_path}")

    coef_one_sided = model_main.params["one_sided"]
    pval_one_sided = model_main.pvalues["one_sided"]
    rate_onesided = rates.get("one-sided", float("nan"))
    rate_twosided = rates.get("two-sided", float("nan"))

    return {
        "seller_term_coef_onesided": coef_one_sided,
        "seller_term_pval_onesided": pval_one_sided,
        "seller_term_rate_onesided": rate_onesided,
        "seller_term_rate_twosided": rate_twosided,
    }


def _vars_from_formula(formula: str) -> list[str]:
    """Extract plain variable names from a patsy formula for dropna."""
    lhs, rhs = formula.split("~")
    dep_var = lhs.strip()
    raw_vars = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", rhs)
    exclude = {"C", "treatment", "I", "np", "log", "exp"}
    indep_vars = [v for v in raw_vars if v not in exclude]
    return [dep_var] + indep_vars


if __name__ == "__main__":
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")
    values = run_seller_termination_after_rejection(
        df=df,
        varlabels_regression=VARLABELS_REGRESSION,
        output_path=OVERLEAF_TABLES / "seller_termination_after_rejection.tex",
    )
    inject_values(
        OVERLEAF_ROOT / "main.tex",
        **{k: round(v, 3) if isinstance(v, float) else v for k, v in values.items()},
    )
