import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
import statsmodels.formula.api as smf
import statsmodels.api as sm
from src.bargaining_analysis.config import BLD, OVERLEAF_FIGURES, OVERLEAF_TABLES, COLOR_SCHEME
from src.bargaining_analysis.main_results.main_results import equilibrium_payoff

from src.bargaining_analysis.helper import set_plot_theme, finalize_plot, _clustered_mean_test, _ols_row

SELLER_ID = 1
BUYER_ID  = 2


def _prepare(df):
    buyers  = df[df["participant_role"] == "Buyer"].copy()
    sellers = df[df["participant_role"] == "Seller"][
        ["negotiation_id", "payoff", "first_offer", "number_of_offers", "offer_1",
         "offer_time_1", "own_offer_accepted", "participant_code",
         "last_offer", "last_offer_time"]
    ].rename(columns={
        "payoff":             "seller_payoff",
        "first_offer":        "first_offer_seller",
        "number_of_offers":   "number_of_offers_seller",
        "offer_1":            "offer_1_seller",
        "offer_time_1":       "offer_time_1_seller",
        "own_offer_accepted": "seller_offer_accepted",
        "participant_code":   "seller_code",
        "last_offer":         "last_offer_seller",
        "last_offer_time":    "last_offer_time_seller",
    })
    m = buyers.merge(sellers, on="negotiation_id", how="left")
    m.rename(columns={
        "first_offer":      "first_offer_buyer",
        "number_of_offers": "number_of_offers_buyer",
        "offer_1":          "offer_1_buyer",
        "offer_time_1":     "offer_time_1_buyer",
        "last_offer":       "last_offer_buyer",
    }, inplace=True)

    m["total_offers"]         = m["number_of_offers_buyer"] + m["number_of_offers_seller"]
    m["two_sided"]            = (m["information_asymmetry"] == "two-sided").astype(int)
    m["has_cost"]             = (m["TA_costs"] == 0.05).astype(int)
    m["time_to_first_offer"]  = m[["offer_time_1_buyer", "offer_time_1_seller"]].min(axis=1)
    m["total_realised"]       = m["payoff"] + m["seller_payoff"]
    m["buyer_share_realised"] = np.where(
        m["total_realised"] > 0, m["payoff"] / m["total_realised"], np.nan
    )

    keep = (m["treatment"].isin(["T3", "T4"])) | (
        m["treatment"].isin(["T1", "T2"]) & (m["gains_from_trade"] > 0)
    )
    full   = m[keep].copy()
    trades = full[full["agreement_dummy"] == 1].copy()
    trades["sec_per_offer"] = (
        trades["bargaining_time_full_sec"] / trades["total_offers"].replace(0, np.nan)
    )
    return full, trades


def _cost_effect(outcome, df, groups="participant_code"):
    """OLS outcome ~ has_cost + two_sided + has_cost:two_sided (pooled).
    Returns (no_cost_mean, cost_mean, coef, se, p).
    """
    d = df.dropna(subset=[outcome]).copy()
    mod = smf.ols(
        f"{outcome} ~ has_cost + two_sided + has_cost:two_sided", data=d
    ).fit(cov_type="cluster", cov_kwds={"groups": d[groups]})
    coef = mod.params["has_cost"]
    se   = mod.bse["has_cost"]
    p    = mod.pvalues["has_cost"]
    no_cost_mean = d[d["has_cost"] == 0][outcome].mean()
    cost_mean    = d[d["has_cost"] == 1][outcome].mean()
    return no_cost_mean, cost_mean, coef, se, p


def table_transaction_costs_summary(df):
    """
    Produce a LaTeX threeparttable body summarising the effect of transaction
    costs. Punchline: costs speed up the process without changing outcomes.

    Two panels:
      A – Process measures  (prediction: significant reduction)
      B – Outcome measures  (prediction: no significant change)
    """
    full, trades = _prepare(df)

    def _stars(p):
        if p < 0.01: return r"$^{***}$"
        if p < 0.05: return r"$^{**}$"
        if p < 0.10: return r"$^{*}$"
        return ""

    # ── collect rows ──────────────────────────────────────────────────────
    panels = {
        "A: Process Variables": [
            ("Negotiation duration (sec)",    "bargaining_time_full_sec", trades, "participant_code"),
            ("Number of offers",              "total_offers",             trades, "participant_code"),
            ("Time per offer (sec)",          "sec_per_offer",            trades, "participant_code"),
            ("Time to first offer (sec)",     "time_to_first_offer",      full,   "participant_code"),
        ],
        "B: Outcome Variables": [
            ("Trade rate",                    "agreement_dummy",          full,   "participant_code"),
            ("Buyer surplus share",           "buyer_share_realised",     trades, "participant_code"),
            ("Deal price",                    "deal_price",               trades, "participant_code"),
        ],
    }

    lines = []

    # header
    lines.append(
        r"\textbf{Outcome} & \textbf{Mean (no cost)} & \textbf{Mean (cost)} "
        r"& \textbf{Cost effect $\hat\beta$} & \textbf{$p$-value} \\"
    )
    lines.append(r"\hline")

    for panel_title, rows in panels.items():
        lines.append(
            rf"\multicolumn{{5}}{{l}}{{\textit{{{panel_title}}}}} \\"
        )
        for label, outcome, data, groups in rows:
            nc_m, c_m, coef, se, p = _cost_effect(outcome, data, groups)
            stars = _stars(p)

            lines.append(
                rf"{label} & {nc_m:.3f} & {c_m:.3f} & "
                rf"{coef:+.3f} ({se:.3f}){stars} & {p:.3f} \\"
            )
        lines.append(r"\hline")

    return "\n".join(lines)


if __name__ == "__main__":
    df  = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")
    tex = table_transaction_costs_summary(df=df)
    out = OVERLEAF_TABLES / "transaction_costs_summary.tex"
    out.write_text(tex)
    print(f"Saved to {out}")