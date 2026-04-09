"""
Tests of equilibrium predictions in the low-valuation region (valuation <= 7.72)
for treatment T4.

The model predicts that when the buyer's valuation is at or below the lower
cutoff b† = 7.72, the buyer should immediately terminate without trade and both
players earn zero.

Run with:
    conda run -n bargaining_analysis python -m src.bargaining_analysis.main_results.test_low_valuation_region
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.formula.api as smf
from src.bargaining_analysis.config import BLD, OVERLEAF_FIGURES
from src.bargaining_analysis.helper import finalize_plot, set_plot_theme

CUTOFF = 7.72
BUYER_ID = 2    # Buyer is id_in_group = 2 in T4
SELLER_ID = 1


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _header(title):
    w = 72
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def _result(label, value, note=""):
    flag = f"  [{note}]" if note else ""
    print(f"  {label:<52} {value}{flag}")


def _consistent(pred, obs):
    return "CONSISTENT" if pred == obs else "INCONSISTENT"


def _clustered_mean_test(series, groups, h0_mean=0.0):
    """
    OLS of series ~ 1, clustered SEs at group level.
    Returns (mean, se, t, p).
    """
    df = pd.DataFrame({"y": series, "group": groups}).dropna()
    mod = smf.ols("y ~ 1", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["group"]}
    )
    coef = mod.params["Intercept"]
    se   = mod.bse["Intercept"]
    t    = (coef - h0_mean) / se
    p    = 2 * (1 - stats.t.cdf(abs(t), df=mod.df_resid))
    return coef, se, t, p


# ─── Prepare negotiation-level data ──────────────────────────────────────────

def _prepare(df):
    """
    Return buyer-level rows for T4 (one row per negotiation) plus a merged
    dataset that also carries the matched seller's payoff and offer count.
    """
    t4 = df[df["treatment"] == "T4"].copy()

    buyers  = t4[t4["participant_role"] == "Buyer"].copy()
    sellers = t4[t4["participant_role"] == "Seller"][
        ["negotiation_id", "payoff", "number_of_offers", "participant_code"]
    ].rename(columns={
        "payoff":           "seller_payoff",
        "number_of_offers": "seller_n_offers",
        "participant_code": "seller_code",
    })

    merged = buyers.merge(sellers, on="negotiation_id", how="left")
    merged["total_offers"] = merged["number_of_offers"] + merged["seller_n_offers"]
    merged["buyer_terminated"] = (
        (merged["bargaining_outcome"] == "Player") &
        (merged["terminated_by_id_in_group"] == BUYER_ID)
    ).astype(int)
    merged["any_termination"] = (
        merged["bargaining_outcome"].isin(["Player", "Random_Termination"])
    ).astype(int)
    merged["player_termination"] = (
        merged["bargaining_outcome"] == "Player"
    ).astype(int)

    low  = merged[merged["valuation"] <= CUTOFF].copy()
    high = merged[merged["valuation"] >  CUTOFF].copy()
    return merged, low, high


# ─── Test 1: Termination rates ────────────────────────────────────────────────

def test1_termination_rates(low, high):
    _header("TEST 1 | Termination Rates  (player-initiated; theory: high in low-val region)")

    # Proportions
    n_low  = len(low);  term_low  = low["player_termination"].sum()
    n_high = len(high); term_high = high["player_termination"].sum()
    rate_low  = term_low  / n_low
    rate_high = term_high / n_high

    print(f"\n  {'Region':<28} {'N':>5}  {'Terminated':>10}  {'Rate':>8}")
    print(f"  {'-'*55}")
    print(f"  {'val <= 7.72':<28} {n_low:>5}  {term_low:>10}  {rate_low:>8.3f}")
    print(f"  {'val > 7.72':<28} {n_high:>5}  {term_high:>10}  {rate_high:>8.3f}")

    # Two-proportion z-test
    z, p = proportions_ztest([term_low, term_high], [n_low, n_high])
    print(f"\n  Two-proportion z-test:  z = {z:+.3f}   p = {p:.4f}")

    # Logistic regression with clustered SEs at participant level
    all_data = pd.concat([low, high]).copy()
    all_data["low_val"] = (all_data["valuation"] <= CUTOFF).astype(int)
    logit = smf.logit("player_termination ~ low_val", data=all_data).fit(
        cov_type="cluster", cov_kwds={"groups": all_data["participant_code"]},
        disp=False
    )
    coef = logit.params["low_val"]
    se   = logit.bse["low_val"]
    p_l  = logit.pvalues["low_val"]
    print(f"  Logistic (clustered SE): coef = {coef:+.3f}  SE = {se:.3f}  p = {p_l:.4f}")
    print(f"\n  Theory predicts HIGH termination in low-val region.")
    verdict = "CONSISTENT" if rate_low > rate_high else "INCONSISTENT"
    print(f"  Verdict: {verdict}  (rate_low={rate_low:.3f} {'>' if rate_low>rate_high else '<='} rate_high={rate_high:.3f})")


# ─── Test 2: Who terminates? ──────────────────────────────────────────────────

def test2_who_terminates(low):
    _header("TEST 2 | Who Terminates? (theory: buyer-initiated in low-val region)")

    term = low[low["bargaining_outcome"] == "Player"].copy()
    n_total       = len(term)
    n_buyer_init  = (term["terminated_by_id_in_group"] == BUYER_ID).sum()
    n_seller_init = (term["terminated_by_id_in_group"] == SELLER_ID).sum()
    n_unknown     = n_total - n_buyer_init - n_seller_init

    print(f"\n  Player-terminated negotiations in low-val region:  n = {n_total}")
    print()
    print(f"  {'Initiator':<30} {'Count':>7}  {'Share':>8}")
    print(f"  {'-'*48}")
    print(f"  {'Buyer (id=2)':<30} {n_buyer_init:>7}  {n_buyer_init/n_total:>8.3f}")
    print(f"  {'Seller (id=1)':<30} {n_seller_init:>7}  {n_seller_init/n_total:>8.3f}")
    if n_unknown > 0:
        print(f"  {'Unknown':<30} {n_unknown:>7}  {n_unknown/n_total:>8.3f}")

    # Binomial test: H0 = 50% buyer-initiated
    binom = stats.binomtest(n_buyer_init, n_total, p=0.5, alternative="greater")
    print(f"\n  Binomial test (H0: buyer_share = 0.5, one-sided greater):")
    print(f"    p = {binom.pvalue:.4f}")

    verdict = "CONSISTENT" if n_buyer_init > n_seller_init else "INCONSISTENT"
    print(f"\n  Theory predicts buyer-initiated termination.")
    print(f"  Verdict: {verdict}  (buyer={n_buyer_init/n_total:.3f}, seller={n_seller_init/n_total:.3f})")


# ─── Test 3: Trade rate ───────────────────────────────────────────────────────

def test3_trade_rate(low):
    _header("TEST 3 | Trade Rate in Low-Val Region  (theory: 0%)")

    n_total  = len(low)
    n_trades = low["agreement_dummy"].sum()
    rate     = n_trades / n_total

    print(f"\n  Negotiations:     {n_total}")
    print(f"  Agreements:       {n_trades}  ({rate:.3f})")

    # One-proportion z-test vs H0: rate = 0
    # With exact binomial since rate should be 0
    binom = stats.binomtest(int(n_trades), n_total, p=0.0, alternative="greater")
    print(f"\n  Exact binomial test (H0: trade rate = 0):")
    print(f"    p = {binom.pvalue:.4f}  (p-value is 0 if any trades occur since H0 prob=0)")

    # More useful: test vs some small epsilon — just flag
    print(f"\n  Theory predicts zero trade in this region.")
    verdict = "CONSISTENT" if n_trades == 0 else "INCONSISTENT"
    print(f"  Verdict: {verdict}  ({n_trades} trades observed, rate = {rate:.3f})")


# ─── Test 4: Average payoffs ──────────────────────────────────────────────────

def test4_average_payoffs(low):
    _header("TEST 4 | Average Payoffs in Low-Val Region  (theory: buyer = 0, seller = 0)")

    for role, series, groups in [
        ("Buyer",  low["payoff"],        low["participant_code"]),
        ("Seller", low["seller_payoff"], low["seller_code"]),
    ]:
        mean, se, t, p = _clustered_mean_test(series, groups, h0_mean=0.0)
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"\n  {role} payoff:")
        print(f"    Mean = {mean:+.4f}   SE (clustered) = {se:.4f}")
        print(f"    t = {t:+.3f}   p = {p:.4f} {sig}")
        verdict = "CONSISTENT" if p >= 0.05 else "INCONSISTENT"
        print(f"    Theory predicts mean = 0.  Verdict: {verdict}")


# ─── Test 5: Negotiation duration ────────────────────────────────────────────

def test5_duration(low, high):
    _header("TEST 5 | Negotiation Duration  (theory: immediate termination in low-val)")

    all_data = pd.concat([low, high]).copy()
    all_data["low_val"] = (all_data["valuation"] <= CUTOFF).astype(int)

    print(f"\n  {'Region':<28} {'N':>5}  {'Mean sec':>10}  {'SD sec':>8}  {'Mean offers':>13}  {'SD offers':>10}")
    print(f"  {'-'*80}")
    for label, grp in [("val <= 7.72", low), ("val > 7.72", high)]:
        ms = grp["bargaining_time_full_sec"].mean()
        ss = grp["bargaining_time_full_sec"].std()
        mo = grp["total_offers"].mean()
        so = grp["total_offers"].std()
        print(f"  {label:<28} {len(grp):>5}  {ms:>10.2f}  {ss:>8.2f}  {mo:>13.2f}  {so:>10.2f}")

    # OLS regressions with clustered SEs
    for outcome, label in [
        ("bargaining_time_full_sec", "Duration (sec)"),
        ("total_offers",             "Total offers"),
    ]:
        mod = smf.ols(f"{outcome} ~ low_val", data=all_data).fit(
            cov_type="cluster", cov_kwds={"groups": all_data["participant_code"]}
        )
        coef = mod.params["low_val"]
        se   = mod.bse["low_val"]
        p    = mod.pvalues["low_val"]
        print(f"\n  OLS {label}: coef(low_val) = {coef:+.3f}  SE = {se:.3f}  p = {p:.4f}")

    print(f"\n  Theory predicts shorter duration in low-val region.")
    verdict = "CONSISTENT" if low["bargaining_time_full_sec"].mean() < high["bargaining_time_full_sec"].mean() else "INCONSISTENT"
    print(f"  Verdict: {verdict}")


# ─── Test 6: Timing of termination ───────────────────────────────────────────

def test6_termination_timing(low):
    _header("TEST 6 | Timing of Termination in Low-Val Region  (theory: pre-offer)")

    term = low[low["bargaining_outcome"] == "Player"].copy()
    n    = len(term)

    print(f"\n  Player-terminated negotiations in low-val region: n = {n}")
    print(f"\n  Distribution of total offers exchanged before termination:")
    print(f"\n  {'Total offers':>14}  {'Count':>7}  {'Share':>8}")
    print(f"  {'-'*34}")
    for k, cnt in term["total_offers"].value_counts().sort_index().items():
        print(f"  {k:>14.0f}  {cnt:>7}  {cnt/n:>8.3f}")

    pre_offer = (term["total_offers"] == 0).sum()
    print(f"\n  Terminations with zero offers (pre-offer): {pre_offer} / {n} = {pre_offer/n:.3f}")

    # Also: fraction where BUYER made 0 offers
    buyer_zero = (term["number_of_offers"] == 0).sum()
    print(f"  Terminations where buyer made 0 offers:    {buyer_zero} / {n} = {buyer_zero/n:.3f}")

    print(f"\n  Theory predicts all terminations occur before any offer.")
    verdict = "CONSISTENT" if pre_offer / n > 0.5 else "INCONSISTENT"
    print(f"  Verdict: {verdict}  ({pre_offer/n:.3f} pre-offer share)")


# ─── Test 7: Mistake trades ───────────────────────────────────────────────────

def test7_mistake_trades(low):
    _header("TEST 7 | 'Mistake' Trades in Low-Val Region  (theory: no trade)")

    trades = low[low["agreement_dummy"] == 1].copy()
    n = len(trades)

    if n == 0:
        print("\n  No trades in low-val region. Consistent with theory.")
        return

    print(f"\n  Number of trades: {n}")
    print(f"\n  {'Statistic':<32}  {'Mean':>8}  {'Median':>8}  {'SD':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*76}")
    for col, label in [
        ("deal_price",    "Deal price"),
        ("payoff",        "Buyer payoff"),
        ("seller_payoff", "Seller payoff"),
        ("valuation",     "Buyer valuation"),
    ]:
        s = trades[col].dropna()
        print(f"  {label:<32}  {s.mean():>8.3f}  {s.median():>8.3f}  {s.std():>8.3f}  {s.min():>8.3f}  {s.max():>8.3f}")

    # Test if buyer payoff in mistake trades is significantly different from 0
    bp_mean, bp_se, bp_t, bp_p = _clustered_mean_test(
        trades["payoff"], trades["participant_code"]
    )
    print(f"\n  Buyer payoff in mistake trades (H0 = 0):")
    print(f"    Mean = {bp_mean:+.3f}   SE = {bp_se:.3f}   t = {bp_t:+.3f}   p = {bp_p:.4f}")

    sp_mean, sp_se, sp_t, sp_p = _clustered_mean_test(
        trades["seller_payoff"], trades["seller_code"]
    )
    print(f"  Seller payoff in mistake trades (H0 = 0):")
    print(f"    Mean = {sp_mean:+.3f}   SE = {sp_se:.3f}   t = {sp_t:+.3f}   p = {sp_p:.4f}")

    # Check if trades are systematic by valuation bucket
    print(f"\n  Breakdown by valuation:")
    print(f"  {'Valuation':>12}  {'N':>5}  {'Mean buyer payoff':>18}  {'Mean deal price':>16}")
    print(f"  {'-'*55}")
    for val, grp in trades.groupby("valuation"):
        print(
            f"  {val:>12.2f}  {len(grp):>5}  "
            f"{grp['payoff'].mean():>18.3f}  "
            f"{grp['deal_price'].mean():>16.3f}"
        )

    print(f"\n  Theory predicts no trades in this region.")
    print(f"  Verdict: INCONSISTENT  ({n} trades observed)")
    if abs(bp_mean) < 0.5 and bp_p >= 0.1:
        print(f"  Buyer payoff near zero — may reflect near-zero-surplus mistakes.")
    elif bp_mean < 0:
        print(f"  Buyer payoff is negative — buyer is making losses on these trades.")


# ─── Plot: buyer valuation vs. number of buyer offers ────────────────────────

def plot_valuation_vs_buyer_offers(
    df,
    figsize=(9, 5),
    jitter_x=0.08,
    jitter_y=0.10,
    random_state=42,
):
    """Scatter plot of buyer valuation (x) vs. number of buyer offers (y) for T4.

    A vertical dashed line marks the low-valuation cutoff b† = 7.72.
    """

    df = df[(df["participant_role"] == "Buyer")
            & (df["treatment"] == "T3")].copy()
    set_plot_theme()

    
    fig, ax = plt.subplots(figsize=figsize)
    color = sns.color_palette()[0]

    rng = np.random.default_rng(random_state)
    x_jittered = df["valuation"] + rng.normal(0, jitter_x, size=len(df))
    y_jittered = df["number_of_offers"] + rng.normal(0, jitter_y, size=len(df))

    ax.scatter(
        x_jittered,
        y_jittered,
        color=color,
        alpha=0.4,
        s=20,
    )
    ax.set_xlim(0, 7.5)
    ax.set_ylim(-0.5, 20)

    ax.set_xlabel(r"Buyer Valuation")
    ax.set_ylabel(r"Number of Buyer Offers")

    finalize_plot(ax=ax)
    plt.close()
    return fig


# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary(low, high):
    _header("SUMMARY: Theoretical Predictions vs. Observations (val <= 7.72, T4)")

    n_low      = len(low)
    rate_term  = low["player_termination"].sum() / n_low
    rate_trade = low["agreement_dummy"].sum() / n_low
    term       = low[low["bargaining_outcome"] == "Player"]
    n_term     = len(term)
    buyer_init_share = (term["terminated_by_id_in_group"] == BUYER_ID).mean() if n_term > 0 else np.nan
    pre_offer_share  = (term["total_offers"] == 0).mean() if n_term > 0 else np.nan

    rows = [
        ("Termination rate",       "High (close to 1)",  f"{rate_term:.3f}",
         "CONSISTENT" if rate_term > 0.5 else "INCONSISTENT"),
        ("Buyer-initiated share",  "1.0 (buyer only)",   f"{buyer_init_share:.3f}" if not np.isnan(buyer_init_share) else "N/A",
         "CONSISTENT" if buyer_init_share > 0.5 else "INCONSISTENT"),
        ("Trade rate",             "0.0",                f"{rate_trade:.3f}",
         "CONSISTENT" if rate_trade == 0 else "INCONSISTENT"),
        ("Pre-offer termination",  "1.0",                f"{pre_offer_share:.3f}" if not np.isnan(pre_offer_share) else "N/A",
         "CONSISTENT" if (pre_offer_share is not np.nan and pre_offer_share > 0.5) else "INCONSISTENT"),
        ("Buyer mean payoff",      "0.0",                f"{low['payoff'].mean():.3f}",
         "CONSISTENT" if abs(low['payoff'].mean()) < 0.5 else "INCONSISTENT"),
        ("Seller mean payoff",     "0.0",                f"{low['seller_payoff'].mean():.3f}",
         "CONSISTENT" if abs(low['seller_payoff'].mean()) < 0.5 else "INCONSISTENT"),
    ]

    print()
    print(f"  {'Outcome':<30}  {'Prediction':>22}  {'Observed':>10}  {'Verdict'}")
    print(f"  {'-'*80}")
    for name, pred, obs, verdict in rows:
        print(f"  {name:<30}  {pred:>22}  {obs:>10}  {verdict}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_all_tests(df):
    merged, low, high = _prepare(df)

    print(f"\nSample (T4 negotiations):  total = {len(merged)}")
    print(f"  val <= 7.72:  n = {len(low)}  ({len(low)/len(merged):.2%})")
    print(f"  val >  7.72:  n = {len(high)}  ({len(high)/len(merged):.2%})")

    test1_termination_rates(low, high)
    test2_who_terminates(low)
    test3_trade_rate(low)
    test4_average_payoffs(low)
    test5_duration(low, high)
    test6_termination_timing(low)
    test7_mistake_trades(low)
    print_summary(low, high)

    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)

    return merged


if __name__ == "__main__":
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")
    merged = run_all_tests(df)
    fig = plot_valuation_vs_buyer_offers(df)
    fig.savefig(OVERLEAF_FIGURES / "low_val_valuation_vs_buyer_offers.pdf")
