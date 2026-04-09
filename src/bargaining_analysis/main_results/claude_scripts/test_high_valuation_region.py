"""
Tests of equilibrium predictions in the high-valuation region (valuation >= 21.40)
for treatment T4.

Theoretical predictions:
  - Seller makes screening offer at p = b*/2 = 10.70 immediately
  - Buyer accepts immediately (zero delay, no counteroffer)
  - Trade rate = 100%; price = 10.70, invariant to buyer valuation
  - Buyer payoff = valuation - 10.70  (slope=1, intercept=-10.70)

Run with:
    conda run -n bargaining_analysis python -m src.bargaining_analysis.main_results.test_high_valuation_region
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

CUTOFF      = 21.40
PRED_PRICE  = 10.70     # predicted deal price
BUYER_ID    = 2
SELLER_ID   = 1


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _header(title):
    w = 72
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def _clustered_mean_test(series, groups, h0_mean=0.0):
    """OLS series ~ 1, clustered SEs. Returns (mean, se, t, p)."""
    df = pd.DataFrame({"y": series, "group": groups}).dropna()
    mod = smf.ols("y ~ 1", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["group"]}
    )
    coef = mod.params["Intercept"]
    se   = mod.bse["Intercept"]
    t    = (coef - h0_mean) / se
    p    = 2 * (1 - stats.t.cdf(abs(t), df=mod.df_resid))
    return coef, se, t, p


def _stars(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.1:   return "*"
    return ""


def _verdict(condition):
    return "CONSISTENT" if condition else "INCONSISTENT"


# ─── Prepare negotiation-level data ──────────────────────────────────────────

def _prepare(df):
    """
    Return buyer-level rows for T4 merged with matched seller info.
    Each row = one negotiation, identified by buyer valuation.
    """
    t4      = df[df["treatment"] == "T4"].copy()
    buyers  = t4[t4["participant_role"] == "Buyer"].copy()
    sellers = t4[t4["participant_role"] == "Seller"][
        ["negotiation_id", "offer_1", "offer_time_1", "number_of_offers",
         "first_offer", "participant_code"]
    ].rename(columns={
        "offer_1":           "offer_1_seller",
        "offer_time_1":      "offer_time_1_seller",
        "number_of_offers":  "number_of_offers_seller",
        "first_offer":       "first_offer_seller",
        "participant_code":  "seller_code",
    })
    merged = buyers.merge(sellers, on="negotiation_id", how="left")
    merged["total_offers"] = (
        merged["number_of_offers"] + merged["number_of_offers_seller"]
    )
    # first_offer on the buyer row = 1 if buyer made the first offer in the negotiation
    merged.rename(columns={
        "first_offer":       "first_offer_buyer",
        "number_of_offers":  "number_of_offers_buyer",
        "offer_1":           "offer_1_buyer",
    }, inplace=True)

    high  = merged[merged["valuation"] >= CUTOFF].copy()
    other = merged[merged["valuation"] <  CUTOFF].copy()
    return merged, high, other


# ─── Test 1: Trade rate ───────────────────────────────────────────────────────

def test1_trade_rate(high):
    _header("TEST 1 | Trade Rate  (theory: 100%)")

    n          = len(high)
    n_trade    = high["agreement_dummy"].sum()
    n_player   = (high["bargaining_outcome"] == "Player").sum()
    n_random   = (high["bargaining_outcome"] == "Random_Termination").sum()
    rate_trade = n_trade / n

    print(f"\n  Total negotiations (val >= {CUTOFF}): {n}")
    print()
    print(f"  {'Outcome':<35} {'Count':>6}  {'Rate':>8}")
    print(f"  {'-'*52}")
    print(f"  {'Agreement (trade)':<35} {n_trade:>6}  {rate_trade:>8.3f}")
    print(f"  {'Player termination':<35} {n_player:>6}  {n_player/n:>8.3f}")
    print(f"  {'Computer termination':<35} {n_random:>6}  {n_random/n:>8.3f}")

    # Player termination breakdown
    buyer_term  = ((high["bargaining_outcome"] == "Player") &
                   (high["terminated_by_id_in_group"] == BUYER_ID)).sum()
    seller_term = ((high["bargaining_outcome"] == "Player") &
                   (high["terminated_by_id_in_group"] == SELLER_ID)).sum()
    print(f"\n  Of player terminations: buyer-initiated = {buyer_term}, "
          f"seller-initiated = {seller_term}")

    # One-proportion z-test: H0 trade rate = 1
    # Equivalent to testing non-trade rate = 0
    n_no_trade = n - n_trade
    z, p = proportions_ztest(n_no_trade, n, value=0, alternative="larger")
    print(f"\n  One-proportion z-test (H0: trade rate = 1.0):")
    print(f"    Non-trade count = {n_no_trade},  z = {z:+.3f},  p = {p:.4f}")

    print(f"\n  Theory predicts trade rate = 1.0.")
    print(f"  Verdict: {_verdict(rate_trade > 0.9)}  (observed = {rate_trade:.3f})")


# ─── Test 2: Price level and invariance ──────────────────────────────────────

def test2_price(high):
    _header(f"TEST 2 | Deal Price  (theory: mean = {PRED_PRICE}, slope on valuation = 0)")

    trades = high[high["agreement_dummy"] == 1].dropna(subset=["deal_price"])
    n      = len(trades)

    mean_p   = trades["deal_price"].mean()
    median_p = trades["deal_price"].median()
    std_p    = trades["deal_price"].std()

    print(f"\n  Trades: {n}")
    print(f"  {'Statistic':<20} {'Value':>10}")
    print(f"  {'-'*32}")
    print(f"  {'Mean price':<20} {mean_p:>10.3f}")
    print(f"  {'Median price':<20} {median_p:>10.3f}")
    print(f"  {'Std dev':<20} {std_p:>10.3f}")

    # t-test vs predicted price 10.70, clustered SEs
    m, se, t, p = _clustered_mean_test(
        trades["deal_price"], trades["participant_code"], h0_mean=PRED_PRICE
    )
    print(f"\n  H0: mean deal price = {PRED_PRICE}")
    print(f"    t = {t:+.3f}   SE (clustered) = {se:.3f}   p = {p:.4f} {_stars(p)}")
    print(f"  Verdict: {_verdict(p >= 0.05)}")

    # Regress deal price on valuation — theory: slope = 0
    mod = smf.ols("deal_price ~ valuation", data=trades).fit(
        cov_type="cluster", cov_kwds={"groups": trades["participant_code"]}
    )
    slope  = mod.params["valuation"]
    se_sl  = mod.bse["valuation"]
    p_sl   = mod.pvalues["valuation"]
    print(f"\n  OLS deal_price ~ valuation (H0: slope = 0):")
    print(f"    slope = {slope:+.4f}   SE = {se_sl:.4f}   p = {p_sl:.4f} {_stars(p_sl)}")

    # Mean deal price by valuation bin
    print(f"\n  Mean deal price by valuation bucket:")
    print(f"  {'Bucket':<16}  {'N':>5}  {'Mean price':>12}")
    print(f"  {'-'*36}")
    bins   = [21, 24, 27, 30.1]
    labels = ["[21, 24)", "[24, 27)", "[27, 30]"]
    trades["val_bin"] = pd.cut(trades["valuation"], bins=bins, labels=labels, right=False)
    for lbl, grp in trades.groupby("val_bin", observed=True):
        print(f"  {str(lbl):<16}  {len(grp):>5}  {grp['deal_price'].mean():>12.3f}")

    print(f"\n  Theory predicts price invariant to valuation (slope = 0).")
    print(f"  Verdict: {_verdict(p_sl >= 0.05)}")


# ─── Test 3: Buyer payoff ─────────────────────────────────────────────────────

def test3_buyer_payoff(high):
    _header("TEST 3 | Buyer Payoff  (theory: payoff = valuation - 10.70; slope=1, intercept=-10.70)")

    trades = high[high["agreement_dummy"] == 1].dropna(subset=["payoff", "valuation"])

    # OLS payoff ~ valuation, full region
    mod = smf.ols("payoff ~ valuation", data=trades).fit(
        cov_type="cluster", cov_kwds={"groups": trades["participant_code"]}
    )
    slope     = mod.params["valuation"]
    intercept = mod.params["Intercept"]
    se_slope  = mod.bse["valuation"]
    se_int    = mod.bse["Intercept"]
    p_slope   = mod.pvalues["valuation"]
    p_int     = mod.pvalues["Intercept"]

    print(f"\n  OLS payoff ~ valuation (trades only, n={len(trades)}):")
    print(f"  {'Parameter':<18}  {'Estimate':>10}  {'SE':>8}  {'p':>8}  {'Theory':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Intercept':<18}  {intercept:>10.4f}  {se_int:>8.4f}  {p_int:>8.4f}  {-PRED_PRICE:>10.2f}")
    print(f"  {'Slope (valuation)':<18}  {slope:>10.4f}  {se_slope:>8.4f}  {p_slope:>8.4f}  {'1.00':>10}")

    # Joint Wald test: intercept = -10.70 AND slope = 1
    wald = mod.f_test([f"Intercept = {-PRED_PRICE}", "valuation = 1"])
    print(f"\n  Joint Wald test (intercept={-PRED_PRICE}, slope=1):")
    print(f"    F = {wald.fvalue:.3f}   p = {wald.pvalue:.4f} {_stars(wald.pvalue)}")
    print(f"  Verdict: {_verdict(wald.pvalue >= 0.05)}")

    # Compare observed vs. predicted mean payoff
    from src.bargaining_analysis.main_results.main_results import equilibrium_payoff
    trades = trades.copy()
    trades["pred_payoff"] = trades["valuation"].apply(equilibrium_payoff)
    obs_mean  = trades["payoff"].mean()
    pred_mean = trades["pred_payoff"].mean()
    print(f"\n  Mean observed payoff : {obs_mean:.3f}")
    print(f"  Mean predicted payoff: {pred_mean:.3f}")
    print(f"  Difference           : {obs_mean - pred_mean:+.3f}")


# ─── Test 4: Duration and first-offer acceptance ─────────────────────────────

def test4_duration(high):
    _header("TEST 4 | Duration & Immediate Acceptance  (theory: zero delay, seller's first offer accepted)")

    trades = high[high["agreement_dummy"] == 1].copy()
    n_t    = len(trades)

    print(f"\n  Trades: {n_t}")
    print(f"\n  {'Statistic':<40}  {'Mean':>8}  {'Median':>8}  {'SD':>8}")
    print(f"  {'-'*68}")
    for col, lbl in [
        ("bargaining_time_full_sec", "Duration (sec)"),
        ("total_offers",             "Total offers exchanged"),
        ("number_of_offers_buyer",   "Buyer offers"),
        ("number_of_offers_seller",  "Seller offers"),
    ]:
        s = trades[col].dropna()
        print(f"  {lbl:<40}  {s.mean():>8.3f}  {s.median():>8.3f}  {s.std():>8.3f}")

    # Test if duration is significantly > 0
    m, se, t, p = _clustered_mean_test(
        trades["bargaining_time_full_sec"], trades["participant_code"], h0_mean=0.0
    )
    print(f"\n  H0: mean duration = 0  →  t = {t:+.3f}, p = {p:.4f} {_stars(p)}")
    print(f"  Verdict: {_verdict(False)}  (duration is significantly > 0 as expected to be flagged)")

    # Fraction settled on seller's first offer (seller moves first, buyer makes 0 offers)
    seller_first_immediate = (
        (trades["first_offer_seller"] == 1) & (trades["number_of_offers_buyer"] == 0)
    ).sum()
    print(f"\n  Distribution of total offers in trades:")
    print(f"  {'Offers':>8}  {'Count':>7}  {'Share':>8}")
    print(f"  {'-'*28}")
    for k, cnt in trades["total_offers"].value_counts().sort_index().items():
        marker = " <-- seller first offer accepted" if k == 1 else ""
        print(f"  {k:>8.0f}  {cnt:>7}  {cnt/n_t:>8.3f}{marker}")

    print(f"\n  Seller moves first AND buyer accepts immediately (0 buyer offers): "
          f"{seller_first_immediate} / {n_t} = {seller_first_immediate/n_t:.3f}")
    print(f"  Theory predicts this fraction = 1.0.")
    print(f"  Verdict: {_verdict(seller_first_immediate/n_t > 0.5)}")


# ─── Test 5: Who moves first? ─────────────────────────────────────────────────

def test5_first_mover(high):
    _header("TEST 5 | First Mover  (theory: seller always moves first)")

    all_n   = len(high)
    trades  = high[high["agreement_dummy"] == 1]
    n_t     = len(trades)

    print(f"\n  {'Group':<30}  {'N':>5}  {'Seller-first':>14}  {'Buyer-first':>12}")
    print(f"  {'-'*65}")
    for label, grp in [("All negotiations", high), ("Trades only", trades)]:
        sf = (grp["first_offer_seller"] == 1).sum()
        bf = (grp["first_offer_buyer"]  == 1).sum()
        na = len(grp) - sf - bf
        print(f"  {label:<30}  {len(grp):>5}  {sf:>7} ({sf/len(grp):.3f})  {bf:>6} ({bf/len(grp):.3f})")

    # One-proportion z-test: H0 seller-first share = 1
    sf_all = (high["first_offer_seller"] == 1).sum()
    z, p = proportions_ztest(sf_all, all_n, value=1.0, alternative="smaller")
    print(f"\n  One-proportion z-test (H0: seller-first rate = 1.0):")
    print(f"    z = {z:+.3f}   p = {p:.4f} {_stars(p)}")

    print(f"\n  Theory predicts seller always moves first.")
    print(f"  Verdict: {_verdict(sf_all/all_n > 0.9)}  (seller-first rate = {sf_all/all_n:.3f})")


# ─── Test 6: Seller's first offer level ──────────────────────────────────────

def test6_seller_first_offer(high):
    _header(f"TEST 6 | Seller's First Offer  (theory: offer = {PRED_PRICE})")

    trades = high[high["agreement_dummy"] == 1].copy()
    # Restrict to cases where seller moved first
    seller_first = trades[trades["first_offer_seller"] == 1].dropna(subset=["offer_1_seller"])
    n = len(seller_first)

    mean_fo   = seller_first["offer_1_seller"].mean()
    median_fo = seller_first["offer_1_seller"].median()
    std_fo    = seller_first["offer_1_seller"].std()

    print(f"\n  Seller-first trades: {n}")
    print(f"  {'Statistic':<25}  {'Value':>10}")
    print(f"  {'-'*38}")
    print(f"  {'Mean seller 1st offer':<25}  {mean_fo:>10.3f}")
    print(f"  {'Median seller 1st offer':<25}  {median_fo:>10.3f}")
    print(f"  {'Std dev':<25}  {std_fo:>10.3f}")

    # t-test vs predicted 10.70, clustered SEs
    m, se, t, p = _clustered_mean_test(
        seller_first["offer_1_seller"], seller_first["seller_code"], h0_mean=PRED_PRICE
    )
    print(f"\n  H0: mean seller first offer = {PRED_PRICE}")
    print(f"    t = {t:+.3f}   SE (clustered) = {se:.3f}   p = {p:.4f} {_stars(p)}")

    # Distribution by valuation bucket
    print(f"\n  Mean seller 1st offer by valuation bucket:")
    print(f"  {'Bucket':<16}  {'N':>5}  {'Mean offer':>12}  {'Median':>8}")
    print(f"  {'-'*45}")
    bins   = [21, 24, 27, 30.1]
    labels = ["[21, 24)", "[24, 27)", "[27, 30]"]
    seller_first = seller_first.copy()
    seller_first["val_bin"] = pd.cut(
        seller_first["valuation"], bins=bins, labels=labels, right=False
    )
    for lbl, grp in seller_first.groupby("val_bin", observed=True):
        print(f"  {str(lbl):<16}  {len(grp):>5}  {grp['offer_1_seller'].mean():>12.3f}  "
              f"{grp['offer_1_seller'].median():>8.3f}")

    # Regress seller 1st offer on valuation — theory: slope = 0
    mod = smf.ols("offer_1_seller ~ valuation", data=seller_first).fit(
        cov_type="cluster", cov_kwds={"groups": seller_first["seller_code"]}
    )
    sl   = mod.params["valuation"]
    se_s = mod.bse["valuation"]
    p_s  = mod.pvalues["valuation"]
    print(f"\n  OLS seller_first_offer ~ valuation (H0: slope = 0):")
    print(f"    slope = {sl:+.4f}   SE = {se_s:.4f}   p = {p_s:.4f} {_stars(p_s)}")

    print(f"\n  Theory predicts seller offers exactly {PRED_PRICE} (invariant to valuation).")
    print(f"  Verdict: {_verdict(p >= 0.05)}  (mean = {mean_fo:.3f}, p-val vs {PRED_PRICE} = {p:.4f})")


# ─── Test 7: Terminations that shouldn't happen ──────────────────────────────

def test7_unexpected_terminations(high):
    _header("TEST 7 | Unexpected Terminations in High-Val Region  (theory: zero)")

    term = high[high["bargaining_outcome"] != "acceptance"].copy()
    n    = len(term)

    if n == 0:
        print("\n  No terminations in high-val region. Consistent with theory.")
        return

    n_player = (term["bargaining_outcome"] == "Player").sum()
    n_random = (term["bargaining_outcome"] == "Random_Termination").sum()
    buyer_init  = ((term["bargaining_outcome"] == "Player") &
                   (term["terminated_by_id_in_group"] == BUYER_ID)).sum()
    seller_init = ((term["bargaining_outcome"] == "Player") &
                   (term["terminated_by_id_in_group"] == SELLER_ID)).sum()

    print(f"\n  Total terminations: {n}  (of {len(high)} negotiations = {n/len(high):.3f})")
    print()
    print(f"  {'Type':<35}  {'Count':>6}  {'Share':>8}")
    print(f"  {'-'*52}")
    print(f"  {'Player termination (any)':<35}  {n_player:>6}  {n_player/n:>8.3f}")
    print(f"    of which buyer-initiated:          {buyer_init:>6}  {buyer_init/n:>8.3f}")
    print(f"    of which seller-initiated:         {seller_init:>6}  {seller_init/n:>8.3f}")
    print(f"  {'Computer termination':<35}  {n_random:>6}  {n_random/n:>8.3f}")

    # Valuation distribution
    print(f"\n  Valuation of terminated negotiations:")
    print(f"    Mean   = {term['valuation'].mean():.2f}")
    print(f"    Median = {term['valuation'].median():.2f}")
    print(f"    Min    = {term['valuation'].min():.2f}   Max = {term['valuation'].max():.2f}")

    # Offers exchanged before termination
    term["total_offers_term"] = (
        term["number_of_offers_buyer"] + term["number_of_offers_seller"]
    )
    print(f"\n  Offers exchanged before termination:")
    print(f"    Mean   = {term['total_offers_term'].mean():.2f}")
    print(f"    Median = {term['total_offers_term'].median():.2f}")
    print()
    print(f"  {'Total offers':>14}  {'Count':>7}  {'Share':>8}")
    print(f"  {'-'*34}")
    for k, cnt in term["total_offers_term"].value_counts().sort_index().items():
        print(f"  {k:>14.0f}  {cnt:>7}  {cnt/n:>8.3f}")

    print(f"\n  Theory predicts zero terminations in this region.")
    print(f"  Verdict: INCONSISTENT  ({n} terminations, {n/len(high):.1%} of negotiations)")


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(high):
    _header("SUMMARY: Theoretical Predictions vs. Observations (val >= 21.40, T4)")

    trades      = high[high["agreement_dummy"] == 1]
    n           = len(high)
    trade_rate  = len(trades) / n
    mean_price  = trades["deal_price"].mean() if len(trades) > 0 else np.nan
    sf_rate     = (high["first_offer_seller"] == 1).mean()
    sfi         = trades[(trades["first_offer_seller"] == 1) & (trades["number_of_offers_buyer"] == 0)]
    imm_rate    = len(sfi) / len(trades) if len(trades) > 0 else np.nan
    seller_first_trades = trades[trades["first_offer_seller"] == 1].dropna(subset=["offer_1_seller"])
    mean_fo     = seller_first_trades["offer_1_seller"].mean()
    mean_dur    = trades["bargaining_time_full_sec"].mean()

    # Slope of payoff on valuation
    mod = smf.ols("payoff ~ valuation", data=trades).fit()
    slope_payoff = mod.params["valuation"]

    # Slope of price on valuation
    mod2 = smf.ols("deal_price ~ valuation", data=trades).fit()
    slope_price = mod2.params["valuation"]

    rows = [
        ("Trade rate",             "1.000",           f"{trade_rate:.3f}",
         _verdict(trade_rate > 0.9)),
        ("Mean deal price",        f"{PRED_PRICE:.2f}",      f"{mean_price:.3f}",
         _verdict(abs(mean_price - PRED_PRICE) < 2.0)),
        ("Price slope on valuation","0.000",           f"{slope_price:.3f}",
         _verdict(abs(slope_price) < 0.2)),
        ("Seller-first rate",      "1.000",           f"{sf_rate:.3f}",
         _verdict(sf_rate > 0.9)),
        ("Immediate acceptance rate","1.000",          f"{imm_rate:.3f}",
         _verdict(imm_rate > 0.5)),
        ("Mean seller 1st offer",  f"{PRED_PRICE:.2f}",      f"{mean_fo:.3f}",
         _verdict(abs(mean_fo - PRED_PRICE) < 2.0)),
        ("Mean duration (sec)",    "~0",              f"{mean_dur:.2f}",
         _verdict(False)),
        ("Payoff slope on valuation","1.000",          f"{slope_payoff:.3f}",
         _verdict(abs(slope_payoff - 1.0) < 0.2)),
    ]

    print()
    print(f"  {'Outcome':<30}  {'Prediction':>12}  {'Observed':>10}  {'Verdict'}")
    print(f"  {'-'*72}")
    for name, pred, obs, verdict in rows:
        print(f"  {name:<30}  {pred:>12}  {obs:>10}  {verdict}")


# ─── Plot: gap closing through bargaining ─────────────────────────────────────

def plot_offer_convergence(high, figsize=(9, 5)):
    """Box plot of seller opening offer, deal price, and buyer opening offer for
    successful trades in the high-valuation region.

    Shows that despite the initial spread between opening positions, the deal
    price converges near the equilibrium prediction of 10.70.
    """
    set_plot_theme()

    trades = high[high["agreement_dummy"] == 1].copy()

    long = pd.concat([
        trades[["offer_1_seller"]].rename(columns={"offer_1_seller": "price"}).assign(
            stage="Seller Opening Offer"
        ),
        trades[["deal_price"]].rename(columns={"deal_price": "price"}).assign(
            stage="Deal Price"
        ),
        trades[["offer_1_buyer"]].rename(columns={"offer_1_buyer": "price"}).assign(
            stage="Buyer Opening Offer"
        ),
    ], ignore_index=True)

    order = ["Seller Opening Offer", "Deal Price", "Buyer Opening Offer"]
    palette = {
        "Seller Opening Offer": sns.color_palette()[0],
        "Deal Price":           sns.color_palette()[2],
        "Buyer Opening Offer":  sns.color_palette()[1],
    }

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=long, x="stage", y="price", order=order, palette=palette,
        width=0.5, flierprops=dict(marker="o", markersize=3, alpha=0.4), ax=ax,
    )
    ax.axhline(
        PRED_PRICE, color="black", linestyle="--", linewidth=1.2,
        label=r"Equilibrium $p^* = 10.70$",
    )
    ax.set_xlabel("")
    ax.set_ylabel(r"Price")
    ax.legend()

    finalize_plot(ax=ax)
    plt.close()
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_all_tests(df):
    _, high, other = _prepare(df)

    print(f"\nSample (T4 negotiations):  total = {len(high) + len(other)}")
    print(f"  val >= {CUTOFF}:  n = {len(high)}  ({len(high)/(len(high)+len(other)):.2%})")
    print(f"  val <  {CUTOFF}:  n = {len(other)}  ({len(other)/(len(high)+len(other)):.2%})")

    test1_trade_rate(high)
    test2_price(high)
    test3_buyer_payoff(high)
    test4_duration(high)
    test5_first_mover(high)
    test6_seller_first_offer(high)
    test7_unexpected_terminations(high)
    print_summary(high)

    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)

    return high


if __name__ == "__main__":
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")
    high = run_all_tests(df)
    fig = plot_offer_convergence(high)
    fig.savefig(OVERLEAF_FIGURES / "high_val_offer_convergence.pdf")
