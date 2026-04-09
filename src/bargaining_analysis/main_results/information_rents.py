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


def _reg_2x2(df, outcome, groups_col):
    """
    OLS outcome ~ two_sided + has_cost + two_sided:has_cost
    with clustered SEs. Returns the fitted model.
    """
    mod = smf.ols(
        f"{outcome} ~ two_sided + has_cost + two_sided:has_cost",
        data=df
    ).fit(cov_type="cluster", cov_kwds={"groups": df[groups_col]})
    print(f"\n  Regression: {outcome} ~ two_sided + has_cost + two_sided×has_cost")
    print(f"  (two_sided=1 for T1/T2; has_cost=1 for T2/T4; base cell = T3)")
    print(f"\n  {'Parameter':<30}  {'Coef':>8}  {'SE':>8}  {'p':>8}")
    print(f"  {'-'*58}")
    for name, coef, se, p in zip(
        mod.params.index, mod.params, mod.bse, mod.pvalues
    ):
        display = {
            "Intercept":             "Intercept (T3)",
            "two_sided":             "Two-sided (T1/T2 vs T3/T4)",
            "has_cost":              "Has cost (T2/T4 vs T1/T3)",
            "two_sided:has_cost":    "Two-sided × Has cost",
        }.get(name, name)
        print(f"  {display:<30}  {coef:>8.4f}  {se:>8.4f}  {p:>8.4f} {_stars(p)}")
    return mod


# ─── Prepare data ─────────────────────────────────────────────────────────────

def _prepare(df):
    buyers  = df[df["participant_role"] == "Buyer"].copy()
    sellers = df[df["participant_role"] == "Seller"][
        ["negotiation_id", "payoff", "first_offer", "number_of_offers",
         "offer_1", "own_offer_accepted", "participant_code"]
    ].rename(columns={
        "payoff":            "seller_payoff",
        "first_offer":       "first_offer_seller",
        "number_of_offers":  "number_of_offers_seller",
        "offer_1":           "offer_1_seller",
        "own_offer_accepted":"seller_offer_accepted",
        "participant_code":  "seller_code",
    })
    m = buyers.merge(sellers, on="negotiation_id", how="left")
    m.rename(columns={
        "first_offer":       "first_offer_buyer",
        "number_of_offers":  "number_of_offers_buyer",
        "offer_1":           "offer_1_buyer",
    }, inplace=True)
    m["total_offers"] = m["number_of_offers_buyer"] + m["number_of_offers_seller"]
    m["two_sided"]    = (m["information_asymmetry"] == "two-sided").astype(int)
    m["has_cost"]     = (m["TA_costs"] == 0.05).astype(int)

    # Restrict two-sided to positive gains_from_trade
    keep = (m["treatment"].isin(["T3", "T4"])) | (
        m["treatment"].isin(["T1", "T2"]) & (m["gains_from_trade"] > 0)
    )
    full = m[keep].copy()

    # Equal-split benchmark
    # One-sided  (T3/T4): seller_val ≈ 0 → equal split price = GFT / 2
    # Two-sided  (T1/T2): equal split price = seller_val + GFT / 2
    full["equal_split_price"] = np.where(
        full["two_sided"] == 0,
        full["gains_from_trade"] / 2,
        full["seller_valuation"] + full["gains_from_trade"] / 2,
    )
    full["price_deviation"] = full["deal_price"] - full["equal_split_price"]

    # Buyer share of realised surplus
    full["total_realised"] = full["payoff"] + full["seller_payoff"]
    full["buyer_share_realised"] = np.where(
        full["total_realised"] > 0,
        full["payoff"] / full["total_realised"],
        np.nan,
    )
    # Buyer share of available surplus
    full["buyer_share_available"] = np.where(
        full["gains_from_trade"] > 0,
        full["payoff"] / full["gains_from_trade"],
        np.nan,
    )

    # Per-cell subsets (trades only)
    trades = full[full["agreement_dummy"] == 1].copy()
    return full, trades


def _two_sample_ttest(a, b):
    """Welch t-test comparing means of two series. Returns (diff, p)."""
    a = a.dropna()
    b = b.dropna()
    diff = a.mean() - b.mean()
    _, p = stats.ttest_ind(a, b, equal_var=False)
    return diff, p


def _pval_label(p):
    if p < 0.01:
        return f"$p < 0.01$"
    elif p < 0.05:
        return f"$p = {p:.2f}$"
    else:
        return f"$p = {p:.2f}$"


def plot_buyer_surplus_share(df, figsize=(10, 5)):
    """
    Bar chart of average buyer surplus share across treatments (trades only).
    Left panel: T4 (BuyerCost) vs T2 (SymCost), right panel: T3 (BuyerNoCost) vs T1 (SymNoCost).
    Difference and p-value annotated between each pair.
    """
    set_plot_theme()

    _, trades = _prepare(df)

    TREATMENTS = {
        "T4": "BuyerCost",
        "T2": "SymCost",
        "T3": "BuyerNoCost",
        "T1": "SymNoCost",
    }

    series = {t: trades.loc[trades["treatment"] == t, "buyer_share_realised"].dropna()
              for t in TREATMENTS}

    def _ci95(s):
        n = len(s)
        se = s.std(ddof=1) / np.sqrt(n)
        return stats.t.ppf(0.975, df=n - 1) * se

    means  = {t: series[t].mean()  for t in TREATMENTS}
    ci95   = {t: _ci95(series[t])  for t in TREATMENTS}

    colors = COLOR_SCHEME

    # x positions: T4=0, T2=1, gap, T3=3, T1=4
    xs = {"T4": 0, "T2": 1, "T3": 3, "T1": 4}

    fig, ax = plt.subplots(figsize=figsize)

    for t, x in xs.items():
        color = colors[0] 
        ax.bar(x, means[t], width=0.6, color=color, label=TREATMENTS[t])
        ax.errorbar(x, means[t], yerr=ci95[t],
                    fmt="none", color="black", capsize=5, linewidth=1.5)

    # ── Annotation helper ──────────────────────────────────────────────────
    def _annotate_pair(left_t, right_t, x_left, x_right):
        diff, p = _two_sample_ttest(series[left_t], series[right_t])
        y_top = max(means[left_t] + ci95[left_t], means[right_t] + ci95[right_t])
        bracket_y = y_top + 0.02
        tip_y     = bracket_y - 0.01
        x_mid     = (x_left + x_right) / 2

        # bracket
        ax.plot([x_left, x_left, x_right, x_right],
                [tip_y, bracket_y, bracket_y, tip_y],
                color="black", lw=1.2)

        sign = "+" if diff >= 0 else ""
        ax.text(x_mid, bracket_y + 0.005,
                f"$\\Delta = {sign}{diff:.3f}$\n{_pval_label(p)}",
                ha="center", va="bottom", fontsize=10)

    _annotate_pair("T4", "T2", 0, 1)
    _annotate_pair("T3", "T1", 3, 4)

    # ── Axes formatting ────────────────────────────────────────────────────
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels([TREATMENTS[t] for t in xs], fontsize=12)
    ax.set_ylabel(r"Avg.\ buyer surplus share (trades only)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Equal split (0.5)")

    # Group labels
    ax.text(0.5, -0.14, "With Transaction Costs",
            ha="center", transform=ax.get_xaxis_transform(), fontsize=12, style="italic")
    ax.text(3.5, -0.14, "Without Transaction Costs",
            ha="center", transform=ax.get_xaxis_transform(), fontsize=12, style="italic")

    # Vertical separator
    ax.axvline(2, color="lightgrey", linestyle="-", linewidth=1)

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    finalize_plot(ax=ax)
    plt.close()
    return fig


def inject_values_information_rent_analysis(df):
    """
    Returns a dict of values to inject into main.tex.

    T1/T2 restricted to positive gains-from-trade (handled by _prepare).
    One-sided = T3/T4 (buyer knows seller's value).
    Two-sided = T1/T2 (neither knows the other's value).

    Buyer accepts seller's offer: buyer row with agreement_dummy==1 & own_offer_accepted==0.

    Seller concession rate (sk_test_5 definition):
        (offer_1_seller - deal_price) / offer_1_seller
    computed on seller-first trades where offer_1_seller > 0, clipped to [-1, 1].
    """
    full, trades = _prepare(df)

    # ── Buyer-accepts-seller probability ──────────────────────────────────
    one_sided = trades[trades["treatment"].isin(["T3", "T4"])]
    two_sided = trades[trades["treatment"].isin(["T1", "T2"])]

    def _prob_buyer_accepts_seller(group):
        buyer_accepts = (group["agreement_dummy"] == 1) & (group["own_offer_accepted"] == 0)
        return buyer_accepts.sum() / len(group)

    prob_one_sided = _prob_buyer_accepts_seller(one_sided)
    prob_two_sided = _prob_buyer_accepts_seller(two_sided)

    # ── Seller concession rate (sk_test_5) ────────────────────────────────
    seller_first = trades[trades["first_offer_seller"] == 1].dropna(
        subset=["offer_1_seller", "deal_price"]
    ).copy()
    seller_first = seller_first[seller_first["offer_1_seller"] > 0]

    seller_first["seller_conc_rate"] = (
        (seller_first["offer_1_seller"] - seller_first["deal_price"]) /
         seller_first["offer_1_seller"]
    ).clip(-1, 1)

    os_ = seller_first[seller_first["two_sided"] == 0]
    ts_ = seller_first[seller_first["two_sided"] == 1]

    conc_one_sided = os_["seller_conc_rate"].mean()
    conc_two_sided = ts_["seller_conc_rate"].mean()

    # Regression-based p-value for the difference (clustered on seller)
    os_2 = os_.copy(); os_2["_ts"] = 0
    ts_2 = ts_.copy(); ts_2["_ts"] = 1
    pool = pd.concat([os_2, ts_2])
    mod = smf.ols("seller_conc_rate ~ _ts", data=pool).fit(
        cov_type="cluster", cov_kwds={"groups": pool["seller_code"]}
    )
    p_conc_diff = mod.pvalues["_ts"]

    # ── Fraction of trades where buyer concedes more than seller ──────────
    # Restricted to multi-offer trades (both sides made at least one offer)
    multi = trades[trades["total_offers"] >= 2].dropna(
        subset=["offer_1_seller", "offer_1_buyer", "deal_price"]
    ).copy()
    multi["seller_concession"] = (multi["offer_1_seller"] - multi["deal_price"]).abs()
    multi["buyer_concession"]  = (multi["offer_1_buyer"]  - multi["deal_price"]).abs()
    multi["buyer_concedes_more"] = (multi["buyer_concession"] > multi["seller_concession"]).astype(float)

    frac_buyer_more_one_sided = multi[multi["two_sided"] == 0]["buyer_concedes_more"].mean()
    frac_buyer_more_two_sided = multi[multi["two_sided"] == 1]["buyer_concedes_more"].mean()

    return {
        "prob_buyer_accepts_seller_one_sided":  f"{prob_one_sided:.1%}",
        "prob_buyer_accepts_seller_two_sided":  f"{prob_two_sided:.1%}",
        "seller_concession_rate_one_sided":     f"{conc_one_sided:.2f}",
        "seller_concession_rate_two_sided":     f"{conc_two_sided:.2f}",
        "p_seller_concession_rate_diff":        f"{p_conc_diff:.3f}",
        "frac_buyer_concedes_more_one_sided":   f"{frac_buyer_more_one_sided * 100:.1f}",
        "frac_buyer_concedes_more_two_sided":   f"{frac_buyer_more_two_sided * 100:.1f}",
    }

