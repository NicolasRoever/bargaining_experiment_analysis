"""
surplus_share_analysis.py
─────────────────────────
Thorough analysis of who captures the larger share of the surplus
(buyer vs. seller) depending on the information structure
(one-sided vs. two-sided uncertainty).

Analyses
--------
1. Summary statistics with statistical tests (MWU + clustered OLS)
2. Bar chart: avg buyer surplus share across 4 treatments
3. Violin plot: distribution of buyer surplus share by info structure
4. Scatter + binned means: buyer surplus share vs. size of gains from trade
5. Regression table (LaTeX): buyer_share ~ info structure × transaction costs

Run with:
    conda run -n bargaining_analysis python -m src.bargaining_analysis.main_results.surplus_share_analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from pystout import pystout

from src.bargaining_analysis.config import (
    BLD, OVERLEAF_FIGURES, OVERLEAF_TABLES, OVERLEAF_ROOT, COLOR_SCHEME, VARLABELS_REGRESSION
)
from src.bargaining_analysis.helper import (
    set_plot_theme, finalize_plot, fix_pandas_append_error, inject_values
)


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def _prepare(df):
    """
    Merge buyer and seller rows per negotiation, compute surplus-share measures.

    Buyer share measures
    --------------------
    buyer_share_realised  : buyer payoff / (buyer + seller payoff)  [trades only]
    buyer_share_available : buyer payoff / gains_from_trade          [all rounds]
    split_gains_from_trade: already in data (buyer's fraction of GFT, trades)

    Sample restrictions
    -------------------
    Two-sided (T1/T2): keep only gains_from_trade > 0 (split is undefined otherwise).
    One-sided (T3/T4): no restriction (seller valuation ≈ 0 by design).
    """
    buyers = df[df["participant_role"] == "Buyer"].copy()
    sellers = df[df["participant_role"] == "Seller"][[
        "negotiation_id", "payoff", "offer_1", "participant_code",
        "number_of_offers", "first_offer",
    ]].rename(columns={
        "payoff":           "seller_payoff",
        "offer_1":          "offer_1_seller",
        "participant_code": "seller_code",
        "number_of_offers": "number_of_offers_seller",
        "first_offer":      "first_offer_seller",
    })

    m = buyers.merge(sellers, on="negotiation_id", how="left")
    m.rename(columns={
        "offer_1":          "offer_1_buyer",
        "number_of_offers": "number_of_offers_buyer",
        "first_offer":      "first_offer_buyer",
    }, inplace=True)

    m["two_sided"] = (m["information_asymmetry"] == "two-sided").astype(int)
    m["has_cost"]  = (m["TA_costs"] == 0.05).astype(int)

    # Sample restriction
    keep = (m["treatment"].isin(["T3", "T4"])) | (
        m["treatment"].isin(["T1", "T2"]) & (m["gains_from_trade"] > 0)
    )
    full = m[keep].copy()

    # Realised surplus shares
    full["total_realised"]       = full["payoff"] + full["seller_payoff"]
    full["buyer_share_realised"] = np.where(
        full["total_realised"] > 0,
        full["payoff"] / full["total_realised"],
        np.nan,
    )
    full["buyer_share_available"] = np.where(
        full["gains_from_trade"] > 0,
        full["payoff"] / full["gains_from_trade"],
        np.nan,
    )

    trades = full[full["agreement_dummy"] == 1].copy()
    return full, trades


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _mwu(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p


def _ols_diff(outcome, df, group_col="participant_code"):
    """
    OLS: outcome ~ two_sided, SE clustered by group_col.
    Returns (mean_one, mean_two, coef, se, p).
    """
    d = df.dropna(subset=[outcome]).copy()
    d["two_sided"] = (d["information_asymmetry"] == "two-sided").astype(int)
    mod = smf.ols(f"{outcome} ~ two_sided", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d[group_col]}
    )
    mean_one = d[d["two_sided"] == 0][outcome].mean()
    mean_two = d[d["two_sided"] == 1][outcome].mean()
    return mean_one, mean_two, mod.params["two_sided"], mod.bse["two_sided"], mod.pvalues["two_sided"]


def _ttest_vs_half(series):
    """One-sample t-test H0: mean == 0.5. Returns (mean, p)."""
    s = series.dropna()
    if len(s) < 2:
        return s.mean(), np.nan
    _, p = stats.ttest_1samp(s, 0.5)
    return s.mean(), p


def _ci95(s):
    s = s.dropna()
    se = s.std(ddof=1) / np.sqrt(len(s))
    return stats.t.ppf(0.975, df=len(s) - 1) * se


# ─────────────────────────────────────────────────────────────────────────────
# 1. Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_surplus_share_summary(df):
    """
    Print a comprehensive summary of buyer vs. seller surplus shares
    by information structure.
    """
    full, trades = _prepare(df)

    print()
    print("=" * 72)
    print("  SURPLUS SHARE ANALYSIS: BUYER vs. SELLER by INFORMATION STRUCTURE")
    print("=" * 72)

    # ── A. Buyer share of REALISED surplus (trades only) ──────────────────
    print()
    print("A. Buyer share of REALISED surplus (payoff_buyer / total_payoff) — trades only")
    print("-" * 72)
    print(f"  {'Measure':<38} {'One-sided':>10} {'Two-sided':>10} "
          f"{'Δ (2s−1s)':>10} {'p_MWU':>7} {'p_OLS':>7}")
    print(f"  {'-'*70}")

    m1, m2, coef, se, p_ols = _ols_diff("buyer_share_realised", trades)
    p_mwu = _mwu(
        trades[trades["information_asymmetry"] == "one-sided"]["buyer_share_realised"],
        trades[trades["information_asymmetry"] == "two-sided"]["buyer_share_realised"],
    )
    print(f"  {'Buyer share (realised)':<38} {m1:>10.3f} {m2:>10.3f} "
          f"{coef:>+10.3f} {p_mwu:>7.3f} {p_ols:>7.3f} {_stars(p_ols)}")

    # Seller share (complement)
    print(f"  {'Seller share (realised)':<38} {1-m1:>10.3f} {1-m2:>10.3f} "
          f"{-coef:>+10.3f}")

    # ── B. Split of gains from trade (split_gains_from_trade) ─────────────
    print()
    print("B. Buyer share of GAINS FROM TRADE (split_gains_from_trade) — trades only")
    print("-" * 72)

    gft_trades = trades[trades["gains_from_trade"] > 0].copy()
    m1g, m2g, coef_g, se_g, p_ols_g = _ols_diff("split_gains_from_trade", gft_trades)
    p_mwu_g = _mwu(
        gft_trades[gft_trades["information_asymmetry"] == "one-sided"]["split_gains_from_trade"],
        gft_trades[gft_trades["information_asymmetry"] == "two-sided"]["split_gains_from_trade"],
    )
    print(f"  {'Buyer split (% of GFT)':<38} {m1g:>10.3f} {m2g:>10.3f} "
          f"{coef_g:>+10.3f} {p_mwu_g:>7.3f} {p_ols_g:>7.3f} {_stars(p_ols_g)}")
    print(f"  {'Seller split (% of GFT)':<38} {1-m1g:>10.3f} {1-m2g:>10.3f}")

    # ── C. t-test vs. equal split (0.5) ───────────────────────────────────
    print()
    print("C. Deviation from equal split (0.5) — one-sample t-test")
    print("-" * 72)
    print(f"  {'Condition':<20} {'Measure':<30} {'Mean':>7} {'p (vs 0.5)':>11} {'Stars':>6}")
    print(f"  {'-'*70}")
    for info in ["one-sided", "two-sided"]:
        for col, label in [
            ("buyer_share_realised", "realised surplus share"),
            ("split_gains_from_trade", "GFT share"),
        ]:
            sub = gft_trades if col == "split_gains_from_trade" else trades
            s = sub[sub["information_asymmetry"] == info][col]
            mean, p = _ttest_vs_half(s)
            print(f"  {info:<20} {label:<30} {mean:>7.3f} {p:>11.3f} {_stars(p):>6}")

    # ── D. By treatment cell ───────────────────────────────────────────────
    print()
    print("D. Buyer realised surplus share by treatment — trades only")
    print("-" * 72)
    print(f"  {'Treatment':<10} {'Label':<25} {'N':>6} {'Mean':>7} {'SD':>7} {'vs 0.5 p':>10}")
    print(f"  {'-'*62}")
    labels = {"T1": "Sym-NoCost", "T2": "Sym-Cost", "T3": "Asym-NoCost", "T4": "Asym-Cost"}
    for t in ["T1", "T2", "T3", "T4"]:
        s = trades[trades["treatment"] == t]["buyer_share_realised"].dropna()
        _, p = _ttest_vs_half(s)
        print(f"  {t:<10} {labels[t]:<25} {len(s):>6} {s.mean():>7.3f} "
              f"{s.std(ddof=1):>7.3f} {p:>10.3f} {_stars(p)}")

    # ── E. Buyer share of available GFT (all rounds incl. non-trades) ─────
    print()
    print("E. Buyer share of AVAILABLE surplus (payoff / GFT) — all rounds, GFT>0")
    print("-" * 72)
    m1a, m2a, coef_a, se_a, p_a = _ols_diff("buyer_share_available", full)
    p_mwu_a = _mwu(
        full[full["information_asymmetry"] == "one-sided"]["buyer_share_available"],
        full[full["information_asymmetry"] == "two-sided"]["buyer_share_available"],
    )
    print(f"  {'Buyer share (available, all rounds)':<38} {m1a:>10.3f} {m2a:>10.3f} "
          f"{coef_a:>+10.3f} {p_mwu_a:>7.3f} {p_a:>7.3f} {_stars(p_a)}")

    # ── F. By cost condition within each info structure ────────────────────
    print()
    print("F. Buyer realised surplus share: info structure × transaction costs — trades")
    print("-" * 72)
    print(f"  {'Cell':<18} {'N':>6} {'Mean':>7} {'95% CI':>14} {'vs 0.5 p':>10}")
    print(f"  {'-'*58}")
    cells = [
        ("one-sided", 0, "Asym, no cost (T3)"),
        ("one-sided", 1, "Asym, cost    (T4)"),
        ("two-sided", 0, "Sym, no cost  (T1)"),
        ("two-sided", 1, "Sym, cost     (T2)"),
    ]
    for info, cost, label in cells:
        s = trades[
            (trades["information_asymmetry"] == info) &
            (trades["has_cost"] == cost)
        ]["buyer_share_realised"].dropna()
        if len(s) < 2:
            continue
        ci = _ci95(s)
        _, p = _ttest_vs_half(s)
        print(f"  {label:<18} {len(s):>6} {s.mean():>7.3f} "
              f"[{s.mean()-ci:>5.3f}, {s.mean()+ci:>5.3f}] "
              f"{p:>10.3f} {_stars(p)}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bar chart: avg buyer surplus share by treatment
# ─────────────────────────────────────────────────────────────────────────────

def plot_buyer_share_by_treatment(df, figsize=(10, 5)):
    """
    Four-bar chart showing avg buyer share of realised surplus (trades only)
    for T1–T4, grouped by transaction-cost condition.
    95% CIs shown. Δ and p-value annotated between one-sided vs. two-sided
    within each cost group.
    """
    set_plot_theme()
    _, trades = _prepare(df)

    LABELS = {"T4": "Asym.\n(cost)", "T2": "Sym.\n(cost)",
               "T3": "Asym.\n(no cost)", "T1": "Sym.\n(no cost)"}
    XS = {"T4": 0, "T2": 1, "T3": 3, "T1": 4}

    series = {t: trades[trades["treatment"] == t]["buyer_share_realised"].dropna()
              for t in LABELS}
    means  = {t: s.mean()  for t, s in series.items()}
    cis    = {t: _ci95(s)  for t, s in series.items()}

    # Colour: asymmetric=blue, symmetric=red
    bar_colors = {
        "T4": COLOR_SCHEME[0], "T3": COLOR_SCHEME[0],
        "T2": COLOR_SCHEME[1], "T1": COLOR_SCHEME[1],
    }

    fig, ax = plt.subplots(figsize=figsize)

    for t, x in XS.items():
        ax.bar(x, means[t], width=0.6, color=bar_colors[t], alpha=0.85)
        ax.errorbar(x, means[t], yerr=cis[t],
                    fmt="none", color="black", capsize=5, linewidth=1.5)

    # Annotate difference between paired bars
    def _bracket(left_t, right_t, x_left, x_right):
        diff, p = _mwu(series[left_t], series[right_t]), None
        # Use OLS for the annotated p
        tmp = pd.concat([
            series[left_t].to_frame("y").assign(group=0),
            series[right_t].to_frame("y").assign(group=1),
        ])
        mod = smf.ols("y ~ group", data=tmp).fit()
        p = mod.pvalues["group"]
        diff = means[right_t] - means[left_t]

        y_top = max(means[left_t] + cis[left_t], means[right_t] + cis[right_t])
        bk_y  = y_top + 0.025
        tip_y = bk_y  - 0.010
        x_mid = (x_left + x_right) / 2

        ax.plot([x_left, x_left, x_right, x_right],
                [tip_y, bk_y, bk_y, tip_y], color="black", lw=1.2)
        sign = "+" if diff >= 0 else ""
        ax.text(x_mid, bk_y + 0.005,
                f"$\\Delta = {sign}{diff:.3f}$\n$p = {p:.3f}${_stars(p)}",
                ha="center", va="bottom", fontsize=10)

    _bracket("T4", "T2", 0, 1)
    _bracket("T3", "T1", 3, 4)

    ax.set_xticks(list(XS.values()))
    ax.set_xticklabels([LABELS[t] for t in XS], fontsize=11)
    ax.set_ylabel(r"Avg.\ buyer share of realised surplus (trades only)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.30)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Equal split (0.5)")

    # Group labels
    ax.text(0.5, -0.16, "With Transaction Costs",
            ha="center", transform=ax.get_xaxis_transform(), fontsize=11, style="italic")
    ax.text(3.5, -0.16, "Without Transaction Costs",
            ha="center", transform=ax.get_xaxis_transform(), fontsize=11, style="italic")
    ax.axvline(2, color="lightgrey", linestyle="-", linewidth=1)

    # Legend
    asym_patch = mpatches.Patch(color=COLOR_SCHEME[0], alpha=0.85, label="One-sided (Asymmetric)")
    sym_patch  = mpatches.Patch(color=COLOR_SCHEME[1], alpha=0.85, label="Two-sided (Symmetric)")
    ax.legend(handles=[asym_patch, sym_patch], fontsize=10, loc="upper right")

    finalize_plot(ax=ax)
    plt.close()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Violin plot: distribution by information structure
# ─────────────────────────────────────────────────────────────────────────────

def plot_surplus_share_distribution(df, figsize=(9, 5)):
    """
    Violin plot of buyer share of realised surplus by information structure
    (one-sided vs. two-sided), trades only.
    Individual points overlaid (jittered). Mean ± 95% CI annotated.
    """
    set_plot_theme()
    _, trades = _prepare(df)

    plot_df = trades[["information_asymmetry", "buyer_share_realised"]].dropna()
    plot_df = plot_df.rename(columns={
        "information_asymmetry": "Information Structure",
        "buyer_share_realised":  "Buyer Share of Surplus",
    })
    plot_df["Information Structure"] = plot_df["Information Structure"].map({
        "one-sided": "One-sided\n(Asymmetric)",
        "two-sided": "Two-sided\n(Symmetric)",
    })

    order = ["One-sided\n(Asymmetric)", "Two-sided\n(Symmetric)"]
    palette = {order[0]: COLOR_SCHEME[0], order[1]: COLOR_SCHEME[1]}

    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=plot_df,
        x="Information Structure",
        y="Buyer Share of Surplus",
        order=order,
        palette=palette,
        cut=0,
        inner=None,
        ax=ax,
        alpha=0.6,
    )
    sns.stripplot(
        data=plot_df,
        x="Information Structure",
        y="Buyer Share of Surplus",
        order=order,
        palette=palette,
        size=2.0,
        alpha=0.3,
        jitter=True,
        ax=ax,
    )

    # Annotate mean ± CI
    for i, grp in enumerate(order):
        s = plot_df[plot_df["Information Structure"] == grp]["Buyer Share of Surplus"]
        mean = s.mean()
        ci   = _ci95(s)
        ax.errorbar(i, mean, yerr=ci, fmt="D", color="black",
                    capsize=6, linewidth=2, markersize=6, zorder=5)
        ax.text(i + 0.25, mean, f"{mean:.3f}", va="center", fontsize=10)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Equal split (0.5)")
    ax.set_ylabel(r"Buyer share of realised surplus")
    ax.set_xlabel("")
    ax.legend(fontsize=10)

    finalize_plot(ax=ax)
    plt.close()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Buyer surplus share vs. GFT size (binned means)
# ─────────────────────────────────────────────────────────────────────────────

def plot_surplus_share_by_gft_tercile(df, figsize=(9, 5)):
    """
    Bin gains_from_trade into terciles within each information-structure group,
    and plot mean buyer surplus share per bin × information structure.
    Shows how buyer advantage varies with the size of the pie.
    """
    set_plot_theme()
    _, trades = _prepare(df)

    sub = trades[trades["gains_from_trade"] > 0].copy()

    # Terciles computed within the full sample
    sub["gft_tercile"] = pd.qcut(sub["gains_from_trade"], q=3,
                                  labels=["Low GFT\n(bottom 33%)",
                                          "Mid GFT\n(33–67%)",
                                          "High GFT\n(top 33%)"])

    grouped = (
        sub.groupby(["gft_tercile", "information_asymmetry"])["buyer_share_realised"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    grouped["se"]   = grouped["std"] / np.sqrt(grouped["count"])
    grouped["ci95"] = grouped["se"] * stats.t.ppf(0.975, df=grouped["count"] - 1)

    order = ["Low GFT\n(bottom 33%)", "Mid GFT\n(33–67%)", "High GFT\n(top 33%)"]
    x = np.arange(len(order))
    width = 0.35

    info_labels = {"one-sided": "One-sided (Asymmetric)", "two-sided": "Two-sided (Symmetric)"}
    colors      = {"one-sided": COLOR_SCHEME[0], "two-sided": COLOR_SCHEME[1]}

    fig, ax = plt.subplots(figsize=figsize)

    for i, info in enumerate(["one-sided", "two-sided"]):
        sub_g = grouped[grouped["information_asymmetry"] == info].set_index("gft_tercile")
        means = [sub_g.loc[t, "mean"] if t in sub_g.index else np.nan for t in order]
        cis   = [sub_g.loc[t, "ci95"] if t in sub_g.index else 0       for t in order]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width=width,
               color=colors[info], alpha=0.85, label=info_labels[info])
        ax.errorbar(x + offset, means, yerr=cis,
                    fmt="none", color="black", capsize=4, linewidth=1.5)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Equal split (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=11)
    ax.set_ylabel(r"Avg.\ buyer share of realised surplus")
    ax.legend(fontsize=10)

    finalize_plot(ax=ax)
    plt.close()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Regression table (LaTeX)
# ─────────────────────────────────────────────────────────────────────────────

def table_surplus_share_regression(df, varlabels, output_path):
    """
    OLS regressions of buyer surplus share on information structure and
    transaction costs with interaction. Two outcome variables:
      (1) buyer_share_realised  — share of payoffs in trades
      (2) split_gains_from_trade — buyer's share of GFT in trades

    SE clustered by participant_code.
    """
    full, trades = _prepare(df)

    gft_trades = trades[trades["gains_from_trade"] > 0].copy()
    gft_full   = full[full["gains_from_trade"] > 0].copy()

    def _fit(formula, data, groups):
        d = data.dropna(subset=[c for c in formula.split()
                                 if c in data.columns]).copy()
        return smf.ols(formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d[groups]}
        )

    # (1) Realised surplus share, trades only
    m1 = _fit(
        "buyer_share_realised ~ C(information_asymmetry) + has_cost + gains_from_trade",
        trades, "participant_code"
    )
    # (2) Realised surplus share, trades only, with interaction
    m2 = _fit(
        "buyer_share_realised ~ C(information_asymmetry) + has_cost + gains_from_trade "
        "+ C(information_asymmetry):has_cost",
        trades, "participant_code"
    )
    # (3) Split of GFT, trades with GFT > 0
    m3 = _fit(
        "split_gains_from_trade ~ C(information_asymmetry) + has_cost + gains_from_trade",
        gft_trades, "participant_code"
    )
    # (4) Split of GFT, trades with GFT > 0, with interaction
    m4 = _fit(
        "split_gains_from_trade ~ C(information_asymmetry) + has_cost + gains_from_trade "
        "+ C(information_asymmetry):has_cost",
        gft_trades, "participant_code"
    )

    extra_labels = {
        "C(information_asymmetry)[T.two-sided]":          r"Two-sided (Sym.\ Uncertainty)",
        "has_cost":                                        r"Transaction Costs",
        "C(information_asymmetry)[T.two-sided]:has_cost": r"Two-sided $\times$ Trans.\ Costs",
        "gains_from_trade":                                 r"Size of Gains from Trade",
        "Intercept":                                       "Constant",
    }
    combined_labels = {**varlabels, **extra_labels}

    fix_pandas_append_error()

    pystout(
        models=[m1, m2, m3, m4],
        file=str(output_path),
        endog_names=[
            r"\shortstack{Buyer Share\\(Realised)}",
            r"\shortstack{Buyer Share\\(Realised)}",
            r"\shortstack{Buyer Share\\(GFT)}",
            r"\shortstack{Buyer Share\\(GFT)}",
        ],
        digits=3,
        stars={0.1: "*", 0.05: "**", 0.01: "***"},
        varlabels=combined_labels,
        modstat={"nobs": "Obs", "rsquared_adj": r"Adj.\ $R^2$"},
        addrows={
            "Sample":              ["Trades", "Trades", "Trades (GFT$>$0)", "Trades (GFT$>$0)"],
            "Interaction included":["No",     "Yes",    "No",               "Yes"],
        },
        exogvars=[
            "C(information_asymmetry)[T.two-sided]",
            "has_cost",
            "C(information_asymmetry)[T.two-sided]:has_cost",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Mechanism analysis: WHY do buyers extract more under two-sided uncertainty?
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_mechanism(df):
    """
    Extend _prepare() with mechanism variables needed for the three channels:
      (a) first_offer_split by role — all rounds with GFT > 0
      (b) Who ends up accepting (seller accepts buyer's offer vs. vice versa)
      (c) Buyer share conditional on acceptance type
    """
    buyers = df[df["participant_role"] == "Buyer"].copy()
    sellers = df[df["participant_role"] == "Seller"][[
        "negotiation_id", "payoff", "number_of_offers", "last_offer",
        "first_offer_split",
    ]].rename(columns={
        "payoff":            "seller_payoff",
        "number_of_offers":  "number_of_offers_seller",
        "last_offer":        "last_offer_seller",
        "first_offer_split": "first_offer_split_seller",
    })

    m = buyers.merge(sellers, on="negotiation_id", how="left")
    m["two_sided"] = (m["information_asymmetry"] == "two-sided").astype(int)
    m["has_cost"]  = (m["TA_costs"] == 0.05).astype(int)
    m["total_realised"]       = m["payoff"] + m["seller_payoff"]
    m["buyer_share_realised"] = np.where(
        m["total_realised"] > 0, m["payoff"] / m["total_realised"], np.nan
    )

    # All rounds with positive GFT (includes non-trades)
    gft_all = m[m["gains_from_trade"] > 0].copy()
    # Trades only (for acceptance-type and surplus-share analyses)
    trades = m[(m["agreement_dummy"] == 1) & (m["gains_from_trade"] > 0)].copy()

    # (b) Acceptance type: 1 = seller accepted buyer's offer; 0 = buyer accepted seller's
    # own_offer_accepted is already in the data for the buyer row

    # (c) Deal position: (deal_price - seller_val) / GFT  [seller's GFT share at deal]
    trades["seller_deal_gft_share"] = np.where(
        trades["gains_from_trade"] > 0,
        (trades["deal_price"] - trades["seller_valuation"]) / trades["gains_from_trade"],
        np.nan,
    )
    trades["buyer_deal_gft_share"] = 1 - trades["seller_deal_gft_share"]

    return gft_all, trades


def print_mechanism_summary(df):
    """
    Console output decomposing the buyer surplus advantage into three channels:
      (a) first_offer_split by role — all rounds with GFT > 0
      (b) Acceptance type distribution  — who ends up accepting?
      (c) Buyer share by acceptance type — does buyer capture more in both channels?
    """
    gft_all, trades = _prepare_mechanism(df)

    print()
    print("=" * 72)
    print("  MECHANISM: WHY DO BUYERS EXTRACT MORE UNDER TWO-SIDED UNCERTAINTY?")
    print("=" * 72)

    # ── (a) first_offer_split by role — all GFT > 0 rounds ───────────────────
    print()
    print("(a) FIRST OFFER SPLIT — fraction of GFT each party tries to capture")
    print("    (buyer: (buyer_val − offer_1) / GFT; seller: (offer_1 − seller_val) / GFT)")
    print("    Sample: all rounds with GFT > 0 (includes non-trades)")
    print("-" * 72)
    print(f"  {'Info structure':<22} {'Buyer first_offer_split':>24} {'Seller first_offer_split':>25}")
    for info in ["one-sided", "two-sided"]:
        sub = gft_all[gft_all["information_asymmetry"] == info]
        mean_buyer  = sub["first_offer_split"].mean()
        mean_seller = sub["first_offer_split_seller"].mean()
        print(f"  {info:<22} {mean_buyer:>24.3f} {mean_seller:>25.3f}")

    _, p_buyer = stats.mannwhitneyu(
        gft_all[gft_all["information_asymmetry"] == "one-sided"]["first_offer_split"].dropna(),
        gft_all[gft_all["information_asymmetry"] == "two-sided"]["first_offer_split"].dropna(),
        alternative="two-sided",
    )
    _, p_seller = stats.mannwhitneyu(
        gft_all[gft_all["information_asymmetry"] == "one-sided"]["first_offer_split_seller"].dropna(),
        gft_all[gft_all["information_asymmetry"] == "two-sided"]["first_offer_split_seller"].dropna(),
        alternative="two-sided",
    )
    print(f"  MWU p (buyer split difference):  {p_buyer:.4f} {_stars(p_buyer)}")
    print(f"  MWU p (seller split difference): {p_seller:.4f} {_stars(p_seller)}")

    # ── (b) Acceptance type ───────────────────────────────────────────────────
    print()
    print("(b) ACCEPTANCE TYPE — who capitulates?")
    print("    own_offer_accepted = 1: seller accepts buyer's offer")
    print("    own_offer_accepted = 0: buyer accepts seller's offer")
    print("-" * 72)
    print(f"  {'Info structure':<22} {'Seller accepts (frac)':>22} {'Buyer accepts (frac)':>22} {'N':>6}")
    for info in ["one-sided", "two-sided"]:
        sub = trades[trades["information_asymmetry"] == info]
        frac_seller = sub["own_offer_accepted"].mean()
        print(f"  {info:<22} {frac_seller:>22.3f} {1-frac_seller:>22.3f} {len(sub):>6}")

    _, p_acc = stats.mannwhitneyu(
        trades[trades["information_asymmetry"] == "one-sided"]["own_offer_accepted"].dropna(),
        trades[trades["information_asymmetry"] == "two-sided"]["own_offer_accepted"].dropna(),
        alternative="two-sided",
    )
    print(f"  MWU p-value (acceptance-type difference): {p_acc:.4f} {_stars(p_acc)}")

    # ── (c) Buyer share by acceptance type ───────────────────────────────────
    print()
    print("(c) BUYER SURPLUS SHARE conditional on acceptance type")
    print("-" * 72)
    print(f"  {'Info structure':<22} {'Seller accepts → buyer share':>30} {'Buyer accepts → buyer share':>29}")
    for info in ["one-sided", "two-sided"]:
        sub = trades[trades["information_asymmetry"] == info]
        sh_seller = sub[sub["own_offer_accepted"] == 1]["buyer_share_realised"].mean()
        sh_buyer  = sub[sub["own_offer_accepted"] == 0]["buyer_share_realised"].mean()
        print(f"  {info:<22} {sh_seller:>30.3f} {sh_buyer:>29.3f}")


    print()
    print("  Interpretation: In two-sided uncertainty, buyers open more aggressively,")
    print("  sellers accept buyer terms more often, and buyers capture a larger share")
    print("  in BOTH acceptance-type categories.")
    print()


def plot_mechanism_first_offer(df, figsize=(10, 5)):
    """
    Side-by-side bar chart illustrating buyer first-offer aggressiveness:
    For each information structure (one-sided / two-sided), show the average
    positions of:
      • Seller valuation (lower bound of GFT)
      • Buyer's first offer price
      • Deal price
      • Buyer valuation (upper bound of GFT)
    All scaled to the same [0, 1] GFT space (position relative to GFT range).

    This makes the anchoring effect immediately visible: under two-sided
    uncertainty, buyers open far below the seller's own valuation.
    """
    set_plot_theme()
    gft, trades = _prepare_mechanism(df)

    # Positions within GFT range: 0 = seller's val, 1 = buyer's val
    # pos_first_offer = seller's GFT share from buyer's first offer = 1 - first_offer_split
    gft["pos_first_offer"] = 1 - gft["first_offer_split"]
    gft["pos_deal_price"]  = (gft["deal_price"] - gft["seller_valuation"]) / gft["gains_from_trade"]

    info_labels = {
        "one-sided": "One-sided\n(Asymmetric)",
        "two-sided": "Two-sided\n(Symmetric)",
    }
    infos   = ["one-sided", "two-sided"]
    x_left  = [0, 2]
    colors  = [COLOR_SCHEME[0], COLOR_SCHEME[1]]

    fig, ax = plt.subplots(figsize=figsize)

    for idx, info in enumerate(infos):
        sub = gft[gft["information_asymmetry"] == info]
        x0  = x_left[idx]
        col = colors[idx]

        mean_first = sub["pos_first_offer"].mean()
        mean_deal  = sub["pos_deal_price"].mean()
        ci_first   = _ci95(sub["pos_first_offer"])
        ci_deal    = _ci95(sub["pos_deal_price"])

        # GFT range bar: 0 (seller val) to 1 (buyer val)
        ax.barh(x0 + 0.3, 1.0, left=0.0, height=0.25,
                color="lightgrey", alpha=0.6, label="GFT range" if idx == 0 else "")

        # First offer dot + CI
        ax.errorbar(mean_first, x0 + 0.3, xerr=ci_first,
                    fmt="v", color=col, markersize=10, linewidth=2, capsize=5,
                    label="Buyer's first offer" if idx == 0 else "")

        # Deal price dot + CI
        ax.errorbar(mean_deal, x0 + 0.3, xerr=ci_deal,
                    fmt="D", color=col, markersize=10, linewidth=2, capsize=5,
                    alpha=0.5, label="Deal price" if idx == 0 else "")

        # Seller-val line (0) and buyer-val line (1)
        ax.axvline(0, color="grey", linewidth=0.8, linestyle=":")
        ax.axvline(1, color="grey", linewidth=0.8, linestyle=":")
        ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--")

        ax.text(mean_first - 0.03, x0 + 0.55,
                f"{mean_first:.2f}", ha="center", fontsize=9, color=col)
        ax.text(mean_deal + 0.03, x0 + 0.55,
                f"deal={mean_deal:.2f}", ha="center", fontsize=9, color=col, alpha=0.7)

    ax.set_yticks([x + 0.3 for x in x_left])
    ax.set_yticklabels([info_labels[i] for i in infos], fontsize=11)
    ax.set_xlabel("Position in GFT space  (0 = seller's valuation, 1 = buyer's valuation)")
    ax.set_xlim(-1.4, 1.2)

    ax.text(0,   -0.3, "Seller val.", ha="center", fontsize=8, color="grey")
    ax.text(0.5, -0.3, "Midpoint",   ha="center", fontsize=8, color="grey")
    ax.text(1,   -0.3, "Buyer val.", ha="center", fontsize=8, color="grey")

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=10, loc="lower right")

    ax.set_title("Buyer Anchoring: First Offer vs. Deal Price in GFT Space", fontsize=12)
    finalize_plot(ax=ax)
    plt.close()
    return fig


def plot_mechanism_acceptance_and_share(df, figsize=(10, 5)):
    """
    Two-panel figure:
      Left:  Stacked bar — fraction of trades where seller accepts vs. buyer accepts,
             by information structure.
      Right: Grouped bar — buyer's surplus share conditional on acceptance type,
             by information structure.

    Reveals that under two-sided uncertainty both channels favour the buyer:
    sellers capitulate more AND the buyer captures a larger share in each category.
    """
    set_plot_theme()
    _, trades = _prepare_mechanism(df)

    info_labels = {"one-sided": "One-sided\n(Asym.)", "two-sided": "Two-sided\n(Sym.)"}
    infos  = ["one-sided", "two-sided"]
    x      = np.arange(len(infos))
    col    = {i: c for i, c in zip(infos, COLOR_SCHEME[:2])}

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)

    # ── Left: stacked acceptance-type bar ────────────────────────────────────
    frac_seller = [trades[trades["information_asymmetry"] == i]["own_offer_accepted"].mean()
                   for i in infos]
    frac_buyer  = [1 - f for f in frac_seller]

    bars_s = ax_left.bar(x, frac_seller, width=0.5,
                         color=[col[i] for i in infos], alpha=0.85,
                         label="Seller accepts buyer's offer")
    ax_left.bar(x, frac_buyer, width=0.5, bottom=frac_seller,
                color=[col[i] for i in infos], alpha=0.35,
                label="Buyer accepts seller's offer")

    for xi, (fs, fb) in enumerate(zip(frac_seller, frac_buyer)):
        ax_left.text(xi, fs / 2,        f"{fs:.0%}", ha="center", va="center",
                     fontsize=10, fontweight="bold", color="white")
        ax_left.text(xi, fs + fb / 2,   f"{fb:.0%}", ha="center", va="center",
                     fontsize=10, color="dimgrey")

    ax_left.set_xticks(x)
    ax_left.set_xticklabels([info_labels[i] for i in infos], fontsize=11)
    ax_left.set_ylabel("Fraction of trades")
    ax_left.set_ylim(0, 1.1)
    ax_left.set_title("Who capitulates?", fontsize=11)
    ax_left.legend(fontsize=9, loc="upper right")
    finalize_plot(ax=ax_left)

    # ── Right: grouped buyer share by acceptance type ─────────────────────────
    width = 0.3
    for xi, info in enumerate(infos):
        sub = trades[trades["information_asymmetry"] == info]
        sh_seller = sub[sub["own_offer_accepted"] == 1]["buyer_share_realised"]
        sh_buyer  = sub[sub["own_offer_accepted"] == 0]["buyer_share_realised"]

        mean_s, ci_s = sh_seller.mean(), _ci95(sh_seller)
        mean_b, ci_b = sh_buyer.mean(),  _ci95(sh_buyer)

        ax_right.bar(xi - width / 2, mean_s, width=width,
                     color=col[info], alpha=0.85,
                     label="Seller accepts" if xi == 0 else "")
        ax_right.bar(xi + width / 2, mean_b, width=width,
                     color=col[info], alpha=0.40,
                     label="Buyer accepts" if xi == 0 else "")
        ax_right.errorbar(xi - width / 2, mean_s, yerr=ci_s,
                          fmt="none", color="black", capsize=4, linewidth=1.5)
        ax_right.errorbar(xi + width / 2, mean_b, yerr=ci_b,
                          fmt="none", color="black", capsize=4, linewidth=1.5)

        ax_right.text(xi - width / 2, mean_s + ci_s + 0.005, f"{mean_s:.2f}",
                      ha="center", fontsize=9)
        ax_right.text(xi + width / 2, mean_b + ci_b + 0.005, f"{mean_b:.2f}",
                      ha="center", fontsize=9)

    ax_right.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Equal split (0.5)")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([info_labels[i] for i in infos], fontsize=11)
    ax_right.set_ylabel("Buyer share of realised surplus")
    ax_right.set_ylim(0, ax_right.get_ylim()[1] * 1.2)
    ax_right.set_title("Buyer share by acceptance type", fontsize=11)
    ax_right.legend(fontsize=9, loc="upper left")
    finalize_plot(ax=ax_right)

    plt.tight_layout()
    plt.close()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Inject mechanism values into main.tex
# ─────────────────────────────────────────────────────────────────────────────

def calculate_mechanism_values_surplus_share(df):
    """
    Returns a dict of values for injection into main.tex:
      - mean_gft_fraction_buyer_first_two_sided: avg (first_offer - seller_val) / GFT
        for two-sided trades (negative = buyer opens below seller's valuation)
      - seller_accepts_frac_one_sided / two_sided: fraction of trades where seller
        accepts the buyer's offer
      - buyer_accepts_frac_one_sided / two_sided: complement
      - n_trades_one_sided / two_sided: number of completed trades per group
      - mean_buyer_first_offer_split_{info}: avg buyer first_offer_split (all GFT>0 rounds)
      - mean_seller_first_offer_split_{info}: avg seller first_offer_split (all GFT>0 rounds)
    """
    gft_all, trades = _prepare_mechanism(df)

    result = {}

    for info in ["one-sided", "two-sided"]:
        key = info.replace("-", "_")

        sub_gft   = gft_all[gft_all["information_asymmetry"] == info]
        sub_trade = trades[trades["information_asymmetry"] == info]

        frac_seller = sub_trade["own_offer_accepted"].mean()
        result[f"seller_accepts_frac_{key}"] = round(frac_seller, 3)
        result[f"buyer_accepts_frac_{key}"]  = round(1 - frac_seller, 3)
        result[f"n_trades_{key}"]            = len(sub_trade)

        result[f"mean_buyer_first_offer_split_{key}"]  = round(
            sub_gft["first_offer_split"].mean(), 3
        )
        result[f"mean_seller_first_offer_split_{key}"] = round(
            sub_gft["first_offer_split_seller"].mean(), 3
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")

    # 1. Console summary
    print_surplus_share_summary(df)

    # 2. Bar chart
    fig_bar = plot_buyer_share_by_treatment(df)
    fig_bar.savefig(OVERLEAF_FIGURES / "surplus_share_by_treatment.pdf", bbox_inches="tight")
    print("Saved: surplus_share_by_treatment.pdf")

    # 3. Violin distribution
    fig_violin = plot_surplus_share_distribution(df)
    fig_violin.savefig(OVERLEAF_FIGURES / "surplus_share_distribution.pdf", bbox_inches="tight")
    print("Saved: surplus_share_distribution.pdf")

    # 4. By GFT tercile
    fig_tercile = plot_surplus_share_by_gft_tercile(df)
    fig_tercile.savefig(OVERLEAF_FIGURES / "surplus_share_by_gft_tercile.pdf", bbox_inches="tight")
    print("Saved: surplus_share_by_gft_tercile.pdf")

    # 5. Regression table
    table_surplus_share_regression(
        df,
        varlabels=VARLABELS_REGRESSION,
        output_path=OVERLEAF_TABLES / "surplus_share_regression.tex",
    )
    print("Saved: surplus_share_regression.tex")

    # 6. Mechanism analyses
    print_mechanism_summary(df)

    fig_anchor = plot_mechanism_first_offer(df)
    fig_anchor.savefig(OVERLEAF_FIGURES / "mechanism_first_offer.pdf", bbox_inches="tight")
    print("Saved: mechanism_first_offer.pdf")

    fig_accept = plot_mechanism_acceptance_and_share(df)
    fig_accept.savefig(OVERLEAF_FIGURES / "mechanism_acceptance_share.pdf", bbox_inches="tight")
    print("Saved: mechanism_acceptance_share.pdf")

    # 7. Inject mechanism values into main.tex
    mech_values = calculate_mechanism_values_surplus_share(df)
    inject_values(OVERLEAF_ROOT / "main.tex", **mech_values)
    print("Injected mechanism values into main.tex:", list(mech_values.keys()))
