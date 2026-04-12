"""Run with
conda run -n bargaining_analysis python -m src.bargaining_analysis.main_results.behavioral_factors_regressions
"""

import pandas as pd
import statsmodels.formula.api as smf
from pystout import pystout

from src.bargaining_analysis.config import BLD, OVERLEAF_TABLES, VARLABELS_REGRESSION
from src.bargaining_analysis.helper import fix_pandas_append_error


def run_behavioral_factors_regressions(
    df: pd.DataFrame,
    varlabels_regression: dict,
    output_path,
    verbose: bool = True,
) -> None:
    """
    Run regressions of split_gains_from_trade (and payoff as robustness) on
    behavioral variables (ultimatum offer, risk preferences, time preferences)
    and first-offer (first mover advantage), always controlling for treatment
    and total gains from trade. Restricted to gains_from_trade > 0.
    """
    base_df = df[(df["gains_from_trade"] > 0)].copy()
    base_df["group_session"] = base_df["group_id_in_session"].astype(str) + "_" + base_df["session_id"].astype(str)

    if verbose:
        print(
            "Running behavioral-factor regressions on "
            f"{len(base_df)} observations (gains_from_trade > 0)."
        )

    def _print_regression_output(title: str, model) -> None:
        print(f"\n=== {title} ===")
        print(model.summary2().tables[1].to_string(float_format=lambda x: f"{x:0.4f}"))
        print(f"n = {int(model.nobs)}, adj. R^2 = {model.rsquared_adj:0.4f}")

    def _fit(formula, data):
        data_clean = data.dropna(subset=_vars_from_formula(formula))
        return smf.ols(formula=formula, data=data_clean).fit(
            cov_type="cluster",
            cov_kwds={"groups": data_clean["participant_label"]},
        )

    controls = "C(treatment) + gains_from_trade + C(group_session)"

    # --- Main table: split_gains_from_trade as dependent variable ---
    model_ultimatum = _fit(
        f"split_gains_from_trade ~ {controls} + ultimatum_offer", base_df
    )
    model_risk = _fit(
        f"split_gains_from_trade ~ {controls} + risk_elicitation_choice", base_df
    )
    model_time = _fit(
        f"split_gains_from_trade ~ {controls} + time_preference_switching_points", base_df
    )
    model_first_offer = _fit(
        f"split_gains_from_trade ~ {controls} + first_offer", base_df
    )

    model_first_offer_fe = _fit(
        f"split_gains_from_trade ~ {controls} + first_offer + C(participant_label)", base_df)

    if verbose:
        _print_regression_output(
            "Main: split_gains_from_trade ~ controls + ultimatum_offer", model_ultimatum
        )
        _print_regression_output(
            "Main: split_gains_from_trade ~ controls + risk_elicitation_choice", model_risk
        )
        _print_regression_output(
            "Main: split_gains_from_trade ~ controls + time_preference_switching_points", model_time
        )
        _print_regression_output(
            "Main: split_gains_from_trade ~ controls + first_offer", model_first_offer
        )
        _print_regression_output(
            "Main: split_gains_from_trade ~ controls + first_offer + participant FE", model_first_offer_fe
        )
    # --- Robustness models: payoff as dependent variable ---
    model_ultimatum_r = _fit(
        f"payoff ~ {controls} + ultimatum_offer", base_df
    )
    model_risk_r = _fit(
        f"payoff ~ {controls} + risk_elicitation_choice", base_df
    )
    model_time_r = _fit(
        f"payoff ~ {controls} + time_preference_switching_points", base_df
    )
    model_first_offer_r = _fit(
        f"payoff ~ {controls} + first_offer", base_df
    )
    model_first_offer_r_fe = _fit(
        f"payoff ~ {controls} + first_offer + C(participant_label)", base_df
    )

    if verbose:
        _print_regression_output(
            "Robustness: payoff ~ controls + ultimatum_offer", model_ultimatum_r
        )
        _print_regression_output(
            "Robustness: payoff ~ controls + risk_elicitation_choice", model_risk_r
        )
        _print_regression_output(
            "Robustness: payoff ~ controls + time_preference_switching_points", model_time_r
        )
        _print_regression_output(
            "Robustness: payoff ~ controls + first_offer", model_first_offer_r
        )

    fix_pandas_append_error()

    pystout(
        models=[
            model_ultimatum, model_risk, model_time, model_first_offer, model_first_offer_fe,
            model_ultimatum_r, model_risk_r, model_time_r, model_first_offer_r, model_first_offer_r_fe,
        ],
        endog_names=False,
        mgroups={"Split Gains from Trade": [1, 5], "Payoff": [6, 10]},
        file=output_path,
        digits=2,
        stars={0.1: "*", 0.05: "**", 0.01: "***"},
        varlabels=varlabels_regression,
        exogvars=[
            "ultimatum_offer",
            "risk_elicitation_choice",
            "time_preference_switching_points",
            "first_offer",
        ],
        addrows={
            "Individual FE": ["", "", "", "", "\\checkmark", "", "", "", "", "\\checkmark"],
            "Baseline Controls": ["\\checkmark"] * 10,
        },
        modstat={"nobs": "Obs", "rsquared_adj": r"Adj. R\sym{2}"},
    )

    if verbose:
        print(f"Saved combined table to {output_path}")


def run_risk_aversion_acceptance_regression(
    df: pd.DataFrame,
    varlabels_regression: dict,
    output_path,
    verbose: bool = True,
) -> None:
    """
    Test whether more risk-averse individuals are more likely to accept the
    other player's offer. Uses a linear probability model (OLS) with
    accepted_other_offer as the dependent variable.

    accepted_other_offer = 1 if the participant accepted the other's offer
    (bargaining_outcome == 'acceptance' AND own_offer_accepted == 0), else 0.
    """
    base_df = df.copy()
    base_df["accepted_other_offer"] = (
        (base_df["bargaining_outcome"] == "acceptance") & (base_df["own_offer_accepted"] == 0)
    ).astype(int)
    base_df["group_session"] = (
        base_df["group_id_in_session"].astype(str) + "_" + base_df["session_id"].astype(str)
    )

    if verbose:
        print(
            f"Running risk-aversion acceptance regression on {len(base_df)} observations.\n"
            f"  accepted_other_offer == 1: {base_df['accepted_other_offer'].sum()}\n"
            f"  accepted_other_offer == 0: {(base_df['accepted_other_offer'] == 0).sum()}"
        )

    controls = "C(treatment) + gains_from_trade + C(group_session)"

    # Exclude the first four sessions for timing analyses (technical issues)
    timing_df = base_df[base_df["exclude_for_time_analysis"] != 1].copy()

    def _fit(formula, data):
        data_clean = data.dropna(subset=_vars_from_formula(formula))
        return smf.ols(formula=formula, data=data_clean).fit(
            cov_type="cluster",
            cov_kwds={"groups": data_clean["participant_label"]},
        )

    model_base = _fit(
        f"accepted_other_offer ~ {controls} + risk_elicitation_choice", base_df
    )
    model_fe = _fit(
        f"accepted_other_offer ~ {controls} + risk_elicitation_choice + C(participant_label)",
        base_df,
    )
    model_time = _fit(
        f"bargaining_time_full_sec ~ {controls} + risk_elicitation_choice", timing_df
    )

    if verbose:
        print(f"\n=== Risk aversion → acceptance (baseline controls) ===")
        print(model_base.summary2().tables[1].to_string(float_format=lambda x: f"{x:0.4f}"))
        print(f"n = {int(model_base.nobs)}, adj. R^2 = {model_base.rsquared_adj:0.4f}")
        print(f"\n=== Risk aversion → acceptance (individual FE) ===")
        print(model_fe.summary2().tables[1].to_string(float_format=lambda x: f"{x:0.4f}"))
        print(f"n = {int(model_fe.nobs)}, adj. R^2 = {model_fe.rsquared_adj:0.4f}")
        print(f"\n=== Risk aversion → bargaining time (excl. first 4 sessions) ===")
        print(model_time.summary2().tables[1].to_string(float_format=lambda x: f"{x:0.4f}"))
        print(f"n = {int(model_time.nobs)}, adj. R^2 = {model_time.rsquared_adj:0.4f}")

    fix_pandas_append_error()

    pystout(
        models=[model_base, model_fe, model_time],
        endog_names=False,
        mgroups={"Accepted Other's Offer": [1, 2], "Bargaining Time (sec)": [3, 3]},
        file=output_path,
        digits=3,
        stars={0.1: "*", 0.05: "**", 0.01: "***"},
        varlabels=varlabels_regression,
        exogvars=["risk_elicitation_choice"],
        addrows={
            "Individual FE": ["", "\\checkmark", ""],
            "Baseline Controls": ["\\checkmark", "\\checkmark", "\\checkmark"],
        },
        modstat={"nobs": "Obs", "rsquared_adj": r"Adj. R\sym{2}"},
    )

    if verbose:
        print(f"\nSaved table to {output_path}")


def _vars_from_formula(formula: str) -> list[str]:
    """Extract variable names needed for dropna from a formula string."""
    lhs, rhs = formula.split("~")
    dep_var = lhs.strip()
    # Extract simple variable names (not patsy expressions like C(...))
    import re
    raw_vars = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", rhs)
    # Filter out patsy keywords and treatment levels
    exclude = {"C", "treatment", "I", "np", "log", "exp"}
    indep_vars = [v for v in raw_vars if v not in exclude]
    return [dep_var] + indep_vars


if __name__ == "__main__":
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")
    run_behavioral_factors_regressions(
        df=df,
        varlabels_regression=VARLABELS_REGRESSION,
        output_path=OVERLEAF_TABLES / "behavioral_factors_regressions_combined.tex",
    )
    run_risk_aversion_acceptance_regression(
        df=df,
        varlabels_regression=VARLABELS_REGRESSION,
        output_path=OVERLEAF_TABLES / "risk_aversion_acceptance.tex",
    )
