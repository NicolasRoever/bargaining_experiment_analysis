"""
Tests of equilibrium predictions for the intermediate valuation region
(7.72 < buyer valuation < 21.40) in treatment T4.

Theoretical predictions:
  - Trade rate = 100% (all types eventually trade)
  - Seller makes initial offer at t = 0
  - Buyer delays Δ(b|b*) before making a counteroffer at price b/2
  - Deal price = b/2  (slope = 0.5, intercept = 0)
  - Delay is strictly decreasing in valuation
  - Buyer payoff ≈ b/2 minus time costs

Run with:
    conda run -n bargaining_analysis python -m src.bargaining_analysis.main_results.test_intermediate_region
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.formula.api as smf
from src.bargaining_analysis.config import BLD
from src.bargaining_analysis.main_results.main_results import equilibrium_payoff

B_DAGGER = 7.72
B_STAR   = 21.40
BUYER_ID  = 2
SELLER_ID = 1

VAL_BINS   = [7.72, 11, 14, 17, 21.4]
VAL_LABELS = ["(7.72,11]", "(11,14]", "(14,17]", "(17,21.4)"]


# ─── Calibrated delay function ────────────────────────────────────────────────

def predicted_delay(b, b_star=21.40, s=0, c=0.05, r=0.01):
    return (1 / r) * np.log((r * (b_star - s) + 2 * c) / (r * (b - s) + 2 * c))


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _header(title):
    w = 72
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def _clustered_mean_test(series, groups, h0_mean=0.0):
    """OLS series ~ 1, clustered SEs. Returns (mean, se, t, p)."""
    d = pd.DataFrame({"y": series, "g": groups}).dropna()
    mod = smf.ols("y ~ 1", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["g"]}
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
    return "   "


def _verdict(condition):
    return "CONSISTENT" if condition else "INCONSISTENT"


# ─── Prepare data ─────────────────────────────────────────────────────────────

def _prepare(df):
    t4      = df[df["treatment"] == "T4"].copy()
    buyers  = t4[t4["participant_role"] == "Buyer"].copy()
    sellers = t4[t4["participant_role"] == "Seller"][
        ["negotiation_id", "offer_1", "offer_time_1", "number_of_offers",
         "first_offer", "participant_code", "payoff", "own_offer_accepted",
         "last_offer", "last_offer_time"]
    ].rename(columns={
        "offer_1":           "offer_1_seller",
        "offer_time_1":      "offer_time_1_seller",
        "number_of_offers":  "number_of_offers_seller",
        "first_offer":       "first_offer_seller",
        "participant_code":  "seller_code",
        "payoff":            "seller_payoff",
        "own_offer_accepted":"seller_offer_accepted",
        "last_offer":        "last_offer_seller",
        "last_offer_time":   "last_offer_time_seller",
    })
    merged = buyers.merge(sellers, on="negotiation_id", how="left")
    merged.rename(columns={
        "first_offer":       "first_offer_buyer",
        "number_of_offers":  "number_of_offers_buyer",
        "offer_1":           "offer_1_buyer",
        "last_offer":        "last_offer_buyer",
    }, inplace=True)
    merged["total_offers"] = (
        merged["number_of_offers_buyer"] + merged["number_of_offers_seller"]
    )
    merged["pred_delay"] = merged["valuation"].apply(
        lambda b: predicted_delay(b) if B_DAGGER < b < B_STAR else np.nan
    )
    merged["pred_payoff"] = merged["valuation"].apply(
        lambda b: equilibrium_payoff(b) if B_DAGGER < b < B_STAR else np.nan
    )
    merged["val_bin"] = pd.cut(
        merged["valuation"], bins=VAL_BINS, labels=VAL_LABELS, right=True
    )

    mid    = merged[(merged["valuation"] > B_DAGGER) & (merged["valuation"] < B_STAR)].copy()
    trades = mid[mid["agreement_dummy"] == 1].copy()
    return mid, trades


# ─── Test 1: Trade rate ───────────────────────────────────────────────────────

def test1_trade_rate(mid):
    _header("TEST 1 | Trade Rate  (theory: 100% everywhere)")

    n         = len(mid)
    n_trade   = mid["agreement_dummy"].sum()
    n_player  = (mid["bargaining_outcome"] == "Player").sum()
    n_random  = (mid["bargaining_outcome"] == "Random_Termination").sum()

    print(f"\n  Total negotiations (7.72 < val < 21.40): {n}")
    print()
    print(f"  {'Outcome':<35} {'Count':>6}  {'Rate':>8}")
    print(f"  {'-'*52}")
    print(f"  {'Agreement (trade)':<35} {n_trade:>6}  {n_trade/n:>8.3f}")
    print(f"  {'Player termination':<35} {n_player:>6}  {n_player/n:>8.3f}")

    buyer_term  = ((mid["bargaining_outcome"] == "Player") &
                   (mid["terminated_by_id_in_group"] == BUYER_ID)).sum()
    seller_term = ((mid["bargaining_outcome"] == "Player") &
                   (mid["terminated_by_id_in_group"] == SELLER_ID)).sum()
    print(f"    of which buyer-initiated:          {buyer_term:>6}  {buyer_term/n:>8.3f}")
    print(f"    of which seller-initiated:         {seller_term:>6}  {seller_term/n:>8.3f}")
    print(f"  {'Computer termination':<35} {n_random:>6}  {n_random/n:>8.3f}")

    # One-proportion z-test: H0 trade rate = 1
    z, p = proportions_ztest(n - n_trade, n, value=0, alternative="larger")
    print(f"\n  One-proportion z-test (H0: trade rate = 1.0):  z = {z:+.3f},  p = {p:.4f}")

    # Trade rate by valuation bin
    print(f"\n  Trade rate by valuation bin:")
    print(f"  {'Bin':<16}  {'N':>5}  {'Trades':>7}  {'Rate':>8}")
    print(f"  {'-'*40}")
    for lbl, grp in mid.groupby("val_bin", observed=True):
        nt = grp["agreement_dummy"].sum()
        print(f"  {str(lbl):<16}  {len(grp):>5}  {nt:>7}  {nt/len(grp):>8.3f}")

    print(f"\n  Theory predicts trade rate = 1.0.")
    print(f"  Verdict: {_verdict(n_trade/n > 0.9)}  (observed = {n_trade/n:.3f})")


# ─── Test 2: Price = b/2 ─────────────────────────────────────────────────────

def test2_price(trades):
    _header("TEST 2 | Deal Price = b/2  (theory: slope = 0.5, intercept = 0)")

    n        = len(trades)
    ratio    = (trades["deal_price"] / trades["valuation"]).replace([np.inf, -np.inf], np.nan).dropna()

    print(f"\n  Trades: {n}")
    print(f"\n  {'Statistic':<35}  {'Value':>10}  {'Predicted':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Mean deal price':<35}  {trades['deal_price'].mean():>10.3f}")
    print(f"  {'Median deal price':<35}  {trades['deal_price'].median():>10.3f}")
    print(f"  {'Mean price / valuation':<35}  {ratio.mean():>10.3f}  {'0.500':>10}")

    # t-test on ratio vs 0.5
    m, se, t, p = _clustered_mean_test(
        trades["deal_price"] / trades["valuation"], trades["participant_code"], h0_mean=0.5
    )
    print(f"\n  H0: mean(price/valuation) = 0.5")
    print(f"    t = {t:+.3f}   SE (clustered) = {se:.4f}   p = {p:.4f} {_stars(p)}")

    # OLS deal_price ~ valuation — joint test slope=0.5, intercept=0
    mod = smf.ols("deal_price ~ valuation", data=trades).fit(
        cov_type="cluster", cov_kwds={"groups": trades["participant_code"]}
    )
    sl  = mod.params["valuation"];  se_sl = mod.bse["valuation"];  p_sl = mod.pvalues["valuation"]
    ic  = mod.params["Intercept"];  se_ic = mod.bse["Intercept"];  p_ic = mod.pvalues["Intercept"]
    print(f"\n  OLS deal_price ~ valuation:")
    print(f"  {'Parameter':<18}  {'Estimate':>10}  {'SE':>8}  {'p':>8}  {'Theory':>8}")
    print(f"  {'-'*56}")
    print(f"  {'Intercept':<18}  {ic:>10.4f}  {se_ic:>8.4f}  {p_ic:>8.4f}  {'0.00':>8}")
    print(f"  {'Slope':<18}  {sl:>10.4f}  {se_sl:>8.4f}  {p_sl:>8.4f}  {'0.50':>8}")

    wald = mod.f_test(["Intercept = 0", "valuation = 0.5"])
    print(f"\n  Joint Wald test (intercept=0, slope=0.5):")
    print(f"    F = {wald.fvalue:.3f}   p = {wald.pvalue:.4f} {_stars(wald.pvalue)}")

    # Binned scatter
    print(f"\n  Mean deal price by valuation bin:")
    print(f"  {'Bin':<16}  {'N':>5}  {'Mean price':>12}  {'Pred (b/2)':>12}")
    print(f"  {'-'*50}")
    for lbl, grp in trades.groupby("val_bin", observed=True):
        mid_b = grp["valuation"].mean()
        print(f"  {str(lbl):<16}  {len(grp):>5}  {grp['deal_price'].mean():>12.3f}  {mid_b/2:>12.3f}")

    print(f"\n  Theory predicts slope = 0.5, intercept = 0.")
    print(f"  Verdict: {_verdict(wald.pvalue >= 0.05)}")


# ─── Test 3: Delay decreasing in valuation ───────────────────────────────────

def test3_delay_slope(trades):
    _header("TEST 3 | Delay Decreasing in Valuation for buyers (theory: negative slope)")
    
    data = trades[trades["participant_role"] == "Buyer"].dropna(subset=["offer_time_1", "valuation"]).copy()

    mod = smf.ols("offer_time_1 ~ valuation + participant_code", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["participant_code"]}
    )
    sl  = mod.params["valuation"]
    se  = mod.bse["valuation"]
    p   = mod.pvalues["valuation"]

    rho, p_rho = stats.spearmanr(
        data["valuation"].dropna(),
        data["offer_time_1"].reindex(data["valuation"].dropna().index)
    )

    print(f"\n  OLS offer_time_1 ~ valuation (n={len(data)}):")
    print(f"    slope = {sl:+.4f}   SE = {se:.4f}   p = {p:.4f} {_stars(p)}")
    print(f"\n  Spearman rank correlation (offer_time_1, valuation):")
    print(f"    rho = {rho:+.4f}   p = {p_rho:.4f} {_stars(p_rho)}")

    # Binned means
    print(f"\n  Mean offer_time_1 (sec) by valuation bin:")
    print(f"  {'Bin':<16}  {'N':>5}  {'Mean offer_time_1':>10}  {'Pred delay':>12}")
    print(f"  {'-'*48}")
    for lbl, grp in data.groupby("val_bin", observed=True):
        mid_b  = grp["valuation"].mean()
        pred_d = predicted_delay(mid_b)
        print(f"  {str(lbl):<16}  {len(grp):>5}  {grp['offer_time_1'].mean():>10.2f}  {pred_d:>12.2f}")

    print(f"\n  Theory predicts negative slope (higher val → shorter delay).")
    print(f"  Verdict: {_verdict(sl < 0 and p < 0.05)}")


# ─── Test 4: Observed vs. predicted delay ────────────────────────────────────

def test4_delay_calibration(trades):
    _header("TEST 4 | Observed vs. Predicted Delay  (theory: slope=1, intercept=0)")

    d = trades.dropna(subset=["bargaining_time_full_sec", "pred_delay"]).copy()
    n = len(d)

    obs_mean  = d["bargaining_time_full_sec"].mean()
    pred_mean = d["pred_delay"].mean()

    print(f"\n  n = {n}")
    print(f"  Mean observed delay  : {obs_mean:.2f} sec")
    print(f"  Mean predicted delay : {pred_mean:.2f} sec")
    print(f"  Difference (obs-pred): {obs_mean-pred_mean:+.2f} sec")

    mod = smf.ols("bargaining_time_full_sec ~ pred_delay", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["participant_code"]}
    )
    sl  = mod.params["pred_delay"];   se_sl = mod.bse["pred_delay"]
    ic  = mod.params["Intercept"];    se_ic = mod.bse["Intercept"]
    p_s = mod.pvalues["pred_delay"];  p_i   = mod.pvalues["Intercept"]

    print(f"\n  OLS observed_duration ~ pred_delay:")
    print(f"  {'Parameter':<18}  {'Estimate':>10}  {'SE':>8}  {'p':>8}  {'Theory':>8}")
    print(f"  {'-'*56}")
    print(f"  {'Intercept':<18}  {ic:>10.4f}  {se_ic:>8.4f}  {p_i:>8.4f}  {'0.00':>8}")
    print(f"  {'Slope (pred_delay)':<18}  {sl:>10.4f}  {se_sl:>8.4f}  {p_s:>8.4f}  {'1.00':>8}")
    print(f"  R² = {mod.rsquared:.4f}")

    wald = mod.f_test(["Intercept = 0", "pred_delay = 1"])
    print(f"\n  Joint Wald test (intercept=0, slope=1):")
    print(f"    F = {wald.fvalue:.3f}   p = {wald.pvalue:.4f} {_stars(wald.pvalue)}")

    # Residual diagnostics
    resid = d["bargaining_time_full_sec"] - d["pred_delay"]
    print(f"\n  Raw residual (obs - pred) diagnostics:")
    print(f"    Mean = {resid.mean():+.2f}   Std = {resid.std():.2f}   "
          f"Median = {resid.median():+.2f}")
    _, p_norm = stats.shapiro(resid.sample(min(len(resid), 200), random_state=0))
    print(f"    Shapiro-Wilk normality (subsample n≤200): p = {p_norm:.4f}")

    print(f"\n  Theory predicts slope=1, intercept=0.")
    print(f"  Verdict: {_verdict(wald.pvalue >= 0.05)}")


# ─── Test 5: Buyer payoff ─────────────────────────────────────────────────────

def test5_buyer_payoff(trades):
    _header("TEST 5 | Buyer Payoff  (theory: slope ≈ 0.5, intercept ≈ 0)")

    mod = smf.ols("payoff ~ valuation", data=trades).fit(
        cov_type="cluster", cov_kwds={"groups": trades["participant_code"]}
    )
    sl  = mod.params["valuation"];  se_sl = mod.bse["valuation"]
    ic  = mod.params["Intercept"];  se_ic = mod.bse["Intercept"]
    p_s = mod.pvalues["valuation"]; p_i   = mod.pvalues["Intercept"]

    print(f"\n  OLS buyer_payoff ~ valuation (n={len(trades)}):")
    print(f"  {'Parameter':<18}  {'Estimate':>10}  {'SE':>8}  {'p':>8}  {'Theory':>8}")
    print(f"  {'-'*56}")
    print(f"  {'Intercept':<18}  {ic:>10.4f}  {se_ic:>8.4f}  {p_i:>8.4f}  {'~0':>8}")
    print(f"  {'Slope':<18}  {sl:>10.4f}  {se_sl:>8.4f}  {p_s:>8.4f}  {'~0.5':>8}")

    wald = mod.f_test(["Intercept = 0", "valuation = 0.5"])
    print(f"\n  Joint Wald test (intercept=0, slope=0.5):")
    print(f"    F = {wald.fvalue:.3f}   p = {wald.pvalue:.4f} {_stars(wald.pvalue)}")

    # Observed vs. equilibrium payoff prediction
    obs_mean  = trades["payoff"].mean()
    pred_mean = trades["pred_payoff"].mean()
    print(f"\n  Mean observed buyer payoff : {obs_mean:.4f}")
    print(f"  Mean predicted payoff (eq) : {pred_mean:.4f}")
    print(f"  Difference (obs - pred)    : {obs_mean - pred_mean:+.4f}")

    # Binned comparison
    print(f"\n  Mean payoff vs. equilibrium prediction by valuation bin:")
    print(f"  {'Bin':<16}  {'N':>5}  {'Obs payoff':>12}  {'Pred payoff':>12}  {'Diff':>8}")
    print(f"  {'-'*58}")
    for lbl, grp in trades.groupby("val_bin", observed=True):
        obs  = grp["payoff"].mean()
        pred = grp["pred_payoff"].mean()
        print(f"  {str(lbl):<16}  {len(grp):>5}  {obs:>12.3f}  {pred:>12.3f}  {obs-pred:>+8.3f}")

    print(f"\n  Theory predicts slope ≈ 0.5, intercept ≈ 0.")
    print(f"  Verdict: {_verdict(wald.pvalue >= 0.05)}")


# ─── Test 6: Who makes the first offer? ──────────────────────────────────────

def test6_first_mover(mid, trades):
    _header("TEST 6 | First Mover  (theory: seller always makes first offer)")

    print(f"\n  {'Group':<30}  {'N':>5}  {'Seller-first':>14}  {'Buyer-first':>12}")
    print(f"  {'-'*65}")
    for label, grp in [("All negotiations", mid), ("Trades only", trades)]:
        sf = (grp["first_offer_seller"] == 1).sum()
        bf = (grp["first_offer_buyer"]  == 1).sum()
        print(f"  {label:<30}  {len(grp):>5}  "
              f"{sf:>7} ({sf/len(grp):.3f})  {bf:>6} ({bf/len(grp):.3f})")

    # Test H0: seller-first rate = 1
    n_all = len(mid)
    sf_all = (mid["first_offer_seller"] == 1).sum()
    z, p = proportions_ztest(sf_all, n_all, value=1.0, alternative="smaller")
    print(f"\n  One-proportion z-test (H0: seller-first = 1.0):")
    print(f"    z = {z:+.3f}   p = {p:.4f} {_stars(p)}")

    # Conditional on trade: compare price and duration by first mover
    print(f"\n  Conditional on trade — mean outcomes by first mover:")
    print(f"  {'First mover':<16}  {'N':>5}  {'Mean price':>12}  {'Mean dur':>10}  {'Mean payoff':>12}")
    print(f"  {'-'*60}")
    for label, mask in [
        ("Seller first", trades["first_offer_seller"] == 1),
        ("Buyer first",  trades["first_offer_buyer"]  == 1),
    ]:
        g = trades[mask]
        if len(g) == 0:
            continue
        print(f"  {label:<16}  {len(g):>5}  {g['deal_price'].mean():>12.3f}  "
              f"{g['bargaining_time_full_sec'].mean():>10.2f}  {g['payoff'].mean():>12.3f}")

    print(f"\n  Theory predicts seller always moves first.")
    print(f"  Verdict: {_verdict(sf_all/n_all > 0.9)}  (seller-first rate = {sf_all/n_all:.3f})")


# ─── Test 7: Who makes the concession leading to trade? ──────────────────────

def test7_final_offer(trades):
    _header("TEST 7 | Final Offer  (theory: buyer makes the counteroffer that closes the deal)")

    # own_offer_accepted==1 for buyer row means buyer's offer was accepted by seller
    # seller_offer_accepted==1 means seller's offer was accepted by buyer
    multi = trades[trades["total_offers"] >= 2].copy()
    n     = len(multi)

    buyer_final  = multi["own_offer_accepted"].sum()
    seller_final = multi["seller_offer_accepted"].sum()

    print(f"\n  Multi-offer trades (≥2 offers): {n}")
    print()
    print(f"  {'Final offer maker':<30}  {'Count':>7}  {'Share':>8}")
    print(f"  {'-'*48}")
    print(f"  {'Buyer offer accepted by seller':<30}  {buyer_final:>7}  {buyer_final/n:>8.3f}")
    print(f"  {'Seller offer accepted by buyer':<30}  {seller_final:>7}  {seller_final/n:>8.3f}")

    # Binomial test: H0 buyer makes final offer with p=0.5
    binom = stats.binomtest(int(buyer_final), n, p=0.5)
    print(f"\n  Binomial test (H0: buyer-final share = 0.5):")
    print(f"    p = {binom.pvalue:.4f}")

    # Compare deal price when buyer vs seller makes final offer
    print(f"\n  Mean deal price by final offer maker:")
    print(f"  {'Final offer maker':<30}  {'N':>5}  {'Mean price':>12}  {'Mean payoff':>12}")
    print(f"  {'-'*62}")
    for label, mask in [
        ("Buyer offer accepted", multi["own_offer_accepted"]    == 1),
        ("Seller offer accepted", multi["seller_offer_accepted"] == 1),
    ]:
        g = multi[mask]
        print(f"  {label:<30}  {len(g):>5}  {g['deal_price'].mean():>12.3f}  {g['payoff'].mean():>12.3f}")

    print(f"\n  Theory predicts buyer makes the counteroffer (buyer-final share ≈ 1).")
    print(f"  Verdict: {_verdict(buyer_final/n > 0.5)}")


# ─── Test 8: Offer convergence ────────────────────────────────────────────────

def test8_offer_convergence(trades, df_full):
    _header("TEST 8 | Offer Convergence Pattern")

    multi = trades[trades["total_offers"] >= 2].copy()
    n     = len(multi)

    # First offer gap = seller's first offer - buyer's first offer (when both made offers)
    both_made = multi[
        (multi["number_of_offers_buyer"]  > 0) &
        (multi["number_of_offers_seller"] > 0)
    ].copy()
    both_made["first_gap"] = both_made["offer_1_seller"] - both_made["offer_1_buyer"]
    both_made["last_gap"]  = both_made["last_offer_seller"] - both_made["last_offer_buyer"]

    print(f"\n  Multi-offer trades: {n}")
    print(f"  Both parties made offers: {len(both_made)}")
    print()
    print(f"  {'Statistic':<40}  {'Mean':>8}  {'Median':>8}  {'SD':>8}")
    print(f"  {'-'*68}")

    for col, label in [
        ("offer_1_seller",   "Seller's 1st offer"),
        ("offer_1_buyer",    "Buyer's 1st offer"),
        ("first_gap",        "Initial gap (seller - buyer 1st offer)"),
        ("last_offer_seller","Seller's final offer"),
        ("last_offer_buyer", "Buyer's final offer"),
        ("last_gap",         "Final gap (seller - buyer last offer)"),
    ]:
        s = both_made[col].dropna()
        print(f"  {label:<40}  {s.mean():>8.3f}  {s.median():>8.3f}  {s.std():>8.3f}")

    gap_reduction = both_made["first_gap"].mean() - both_made["last_gap"].mean()
    print(f"\n  Gap reduction (initial → final): {gap_reduction:+.3f}")

    # Mean gap by round of offers exchanged
    # Use the raw offer columns to track convergence over rounds
    t4  = df_full[df_full["treatment"] == "T4"]
    buy = t4[t4["participant_role"] == "Buyer"]
    sel = t4[t4["participant_role"] == "Seller"]
    neg = buy.merge(
        sel[["negotiation_id"] + [f"offer_{i}" for i in range(1, 16)]].rename(
            columns={f"offer_{i}": f"s_offer_{i}" for i in range(1, 16)}
        ),
        on="negotiation_id", how="inner"
    )
    neg = neg[
        (neg["valuation"] > B_DAGGER) & (neg["valuation"] < B_STAR) &
        (neg["agreement_dummy"] == 1)
    ]

    print(f"\n  Mean gap between seller and buyer offers, by exchange round:")
    print(f"  {'Round':>7}  {'N pairs':>9}  {'Mean seller offer':>18}  {'Mean buyer offer':>17}  {'Mean gap':>10}")
    print(f"  {'-'*66}")
    for r in range(1, 8):
        s_col = f"s_offer_{r}"
        b_col = f"offer_{r}"
        if s_col not in neg.columns or b_col not in neg.columns:
            break
        pairs = neg[[s_col, b_col]].dropna()
        if len(pairs) < 5:
            break
        gap = pairs[s_col] - pairs[b_col]
        print(f"  {r:>7}  {len(pairs):>9}  {pairs[s_col].mean():>18.3f}  "
              f"{pairs[b_col].mean():>17.3f}  {gap.mean():>10.3f}")

    print(f"\n  Theory predicts convergence from high initial gap to zero final gap.")
    print(f"  Verdict: {_verdict(gap_reduction > 0)}")


# ─── Test 9: Seller payoff ────────────────────────────────────────────────────

def test9_seller_payoff(trades):
    _header("TEST 9 | Seller Payoff  (theory: slope = 0.5, intercept = 0)")

    mod = smf.ols("seller_payoff ~ valuation", data=trades).fit(
        cov_type="cluster", cov_kwds={"groups": trades["seller_code"]}
    )
    sl  = mod.params["valuation"];  se_sl = mod.bse["valuation"]
    ic  = mod.params["Intercept"];  se_ic = mod.bse["Intercept"]
    p_s = mod.pvalues["valuation"]; p_i   = mod.pvalues["Intercept"]

    print(f"\n  OLS seller_payoff ~ valuation (n={len(trades)}):")
    print(f"  {'Parameter':<18}  {'Estimate':>10}  {'SE':>8}  {'p':>8}  {'Theory':>8}")
    print(f"  {'-'*56}")
    print(f"  {'Intercept':<18}  {ic:>10.4f}  {se_ic:>8.4f}  {p_i:>8.4f}  {'~0':>8}")
    print(f"  {'Slope':<18}  {sl:>10.4f}  {se_sl:>8.4f}  {p_s:>8.4f}  {'~0.5':>8}")

    wald = mod.f_test(["Intercept = 0", "valuation = 0.5"])
    print(f"\n  Joint Wald test (intercept=0, slope=0.5):")
    print(f"    F = {wald.fvalue:.3f}   p = {wald.pvalue:.4f} {_stars(wald.pvalue)}")

    print(f"\n  Buyer vs. seller payoff comparison (trades):")
    bm = trades["payoff"].mean()
    sm = trades["seller_payoff"].mean()
    print(f"    Mean buyer payoff  : {bm:.4f}")
    print(f"    Mean seller payoff : {sm:.4f}")
    print(f"    Surplus captured by buyer: {bm/(bm+sm)*100:.1f}%")

    print(f"\n  Theory predicts seller_payoff ≈ b/2 (slope=0.5, intercept=0).")
    print(f"  Verdict: {_verdict(wald.pvalue >= 0.05)}")


# ─── Test 10a: Surplus share ─────────────────────────────────────────────────

def test10a_surplus_share(trades):
    _header("TEST 10a | Surplus Share: Who Captures More?  (theory: equal split — 50/50)")

    d = trades.copy()
    d["total_surplus"] = d["payoff"] + d["seller_payoff"]
    # Drop cases where total surplus is zero or negative (division undefined / misleading)
    d = d[d["total_surplus"] > 0].copy()
    d["buyer_share"] = d["payoff"] / d["total_surplus"]
    n = len(d)

    print(f"\n  Trades with positive realised surplus: {n}")

    # Summary statistics
    bs = d["buyer_share"]
    print(f"\n  {'Statistic':<35}  {'Buyer share':>12}  {'Seller share':>13}")
    print(f"  {'-'*62}")
    print(f"  {'Mean':<35}  {bs.mean():>12.3f}  {1-bs.mean():>13.3f}")
    print(f"  {'Median':<35}  {bs.median():>12.3f}  {1-bs.median():>13.3f}")
    print(f"  {'Std dev':<35}  {bs.std():>12.3f}")

    # Mean payoffs
    print(f"\n  Mean buyer payoff  : {d['payoff'].mean():.3f}")
    print(f"  Mean seller payoff : {d['seller_payoff'].mean():.3f}")
    print(f"  Mean total surplus : {d['total_surplus'].mean():.3f}")
    print(f"  Mean valuation     : {d['valuation'].mean():.3f}  "
          f"(efficiency = {d['total_surplus'].mean()/d['valuation'].mean():.3f})")

    # t-test: buyer share = 0.5
    m, se, t, p = _clustered_mean_test(d["buyer_share"], d["participant_code"], h0_mean=0.5)
    print(f"\n  H0: mean buyer share = 0.5 (equal split)")
    print(f"    Mean = {m:.3f}   SE (clustered) = {se:.4f}   t = {t:+.3f}   p = {p:.4f} {_stars(p)}")
    if p < 0.05:
        winner = "BUYER" if m > 0.5 else "SELLER"
        print(f"  → {winner} captures significantly more than half the surplus.")
    else:
        print(f"  → Cannot reject equal split.")

    # By valuation bin
    print(f"\n  Buyer's share of surplus by valuation bin:")
    print(f"  {'Bin':<16}  {'N':>5}  {'Buyer share':>12}  {'Seller share':>13}  {'Total surplus':>14}")
    print(f"  {'-'*64}")
    for lbl, grp in d.groupby("val_bin", observed=True):
        bs_bin = grp["buyer_share"].mean()
        ts_bin = grp["total_surplus"].mean()
        print(f"  {str(lbl):<16}  {len(grp):>5}  {bs_bin:>12.3f}  {1-bs_bin:>13.3f}  {ts_bin:>14.3f}")

    # Paired t-test: are buyer and seller payoffs significantly different?
    diff = d["payoff"] - d["seller_payoff"]
    _, p_paired = stats.ttest_rel(d["payoff"], d["seller_payoff"])
    print(f"\n  Paired t-test (buyer payoff vs. seller payoff):")
    print(f"    Mean difference (buyer − seller) = {diff.mean():+.3f}   p = {p_paired:.4f} {_stars(p_paired)}")

    print(f"\n  Theory predicts equal split (buyer share = 0.5).")
    print(f"  Verdict: {_verdict(p >= 0.05)}")


# ─── Test 10b: Seller share and buyer concession vs. valuation ───────────────

def test10b_leverage_channel(trades):
    _header(
        "TEST 10b | Outside-Option Leverage: Does the Seller Capture More\n"
        "           at Higher Valuations?  (prediction: both slopes > 0)"
    )

    d = trades.copy()
    d["seller_share"] = d["deal_price"] / d["valuation"]   # price / val in (0,1)

    # ── Part A: seller share ~ valuation ──────────────────────────────────────
    print(f"\n  PART A  Seller's share of surplus (price / valuation) ~ valuation")
    print(f"  If outside-option asymmetry drives the result, higher-valuation buyers")
    print(f"  should concede a larger fraction of the surplus to the seller.\n")

    mod_a = smf.ols("seller_share ~ valuation", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["participant_code"]}
    )
    sl_a = mod_a.params["valuation"]
    se_a = mod_a.bse["valuation"]
    p_a  = mod_a.pvalues["valuation"]
    ic_a = mod_a.params["Intercept"]

    print(f"  OLS seller_share ~ valuation (n={len(d)}):")
    print(f"    Intercept = {ic_a:+.4f}")
    print(f"    Slope     = {sl_a:+.4f}   SE = {se_a:.4f}   p = {p_a:.4f} {_stars(p_a)}")
    print(f"    R²        = {mod_a.rsquared:.4f}")

    rho_a, p_rho_a = stats.spearmanr(d["valuation"], d["seller_share"])
    print(f"\n  Spearman rank correlation (valuation, seller_share):")
    print(f"    rho = {rho_a:+.4f}   p = {p_rho_a:.4f} {_stars(p_rho_a)}")

    print(f"\n  Mean seller share (price/valuation) by valuation bin:")
    print(f"  {'Bin':<16}  {'N':>5}  {'Mean seller share':>18}  {'Mean buyer share':>17}")
    print(f"  {'-'*60}")
    for lbl, grp in d.groupby("val_bin", observed=True):
        ss = (grp["deal_price"] / grp["valuation"]).mean()
        print(f"  {str(lbl):<16}  {len(grp):>5}  {ss:>18.3f}  {1-ss:>17.3f}")

    verdict_a = _verdict(sl_a > 0 and p_a < 0.05)
    print(f"\n  Prediction: positive slope.")
    print(f"  Verdict: {verdict_a}  (slope = {sl_a:+.4f}, p = {p_a:.4f})")

    # ── Part B: P(buyer makes final concession) ~ valuation ───────────────────
    print(f"\n  {'─'*68}")
    print(f"\n  PART B  P(buyer accepts seller's offer) ~ valuation")
    print(f"  A buyer with a higher valuation has more to lose by walking away,")
    print(f"  so they should be more likely to accept the seller's final offer.\n")

    # seller_offer_accepted == 1 means buyer accepted the seller's offer
    # Restrict to multi-offer trades where someone had to make a concession
    multi = d[d["total_offers"] >= 2].copy()
    n_m   = len(multi)

    # Linear probability model (OLS) for interpretability + clustered SEs
    mod_b = smf.ols("seller_offer_accepted ~ valuation", data=multi).fit(
        cov_type="cluster", cov_kwds={"groups": multi["participant_code"]}
    )
    sl_b = mod_b.params["valuation"]
    se_b = mod_b.bse["valuation"]
    p_b  = mod_b.pvalues["valuation"]
    ic_b = mod_b.params["Intercept"]

    print(f"  OLS P(buyer accepts seller offer) ~ valuation (n={n_m}, multi-offer trades):")
    print(f"    Intercept = {ic_b:+.4f}")
    print(f"    Slope     = {sl_b:+.4f}   SE = {se_b:.4f}   p = {p_b:.4f} {_stars(p_b)}")
    print(f"    R²        = {mod_b.rsquared:.4f}")

    # Logit as robustness check
    logit_b = smf.logit("seller_offer_accepted ~ valuation", data=multi).fit(
        cov_type="cluster", cov_kwds={"groups": multi["participant_code"]}, disp=False
    )
    sl_l = logit_b.params["valuation"]
    se_l = logit_b.bse["valuation"]
    p_l  = logit_b.pvalues["valuation"]
    print(f"\n  Logit (marginal effect at mean):")
    from scipy.stats import norm
    mfx = sl_l * norm.pdf(logit_b.predict(multi).mean())
    print(f"    Coef = {sl_l:+.4f}   SE = {se_l:.4f}   p = {p_l:.4f} {_stars(p_l)}")

    rho_b, p_rho_b = stats.spearmanr(multi["valuation"], multi["seller_offer_accepted"])
    print(f"\n  Spearman rank correlation (valuation, buyer_accepts_seller):")
    print(f"    rho = {rho_b:+.4f}   p = {p_rho_b:.4f} {_stars(p_rho_b)}")

    print(f"\n  P(buyer accepts seller offer) by valuation bin:")
    print(f"  {'Bin':<16}  {'N':>5}  {'Buyer accepts (%)':>19}  {'Buyer gets accepted (%)':>23}")
    print(f"  {'-'*68}")
    for lbl, grp in multi.groupby("val_bin", observed=True):
        p_acc  = grp["seller_offer_accepted"].mean()
        p_own  = grp["own_offer_accepted"].mean()
        print(f"  {str(lbl):<16}  {len(grp):>5}  {p_acc:>19.3f}  {p_own:>23.3f}")

    verdict_b = _verdict(sl_b > 0 and p_b < 0.05)
    print(f"\n  Prediction: positive slope.")
    print(f"  Verdict: {verdict_b}  (slope = {sl_b:+.4f}, p = {p_b:.4f})")

    # ── Summary of the leverage story ─────────────────────────────────────────
    print(f"\n  {'─'*68}")
    print(f"\n  LEVERAGE STORY SUMMARY:")
    print(f"    Seller share slope on valuation : {sl_a:+.4f}  (p={p_a:.4f}) {_stars(p_a)}")
    print(f"    P(buyer concedes) slope         : {sl_b:+.4f}  (p={p_b:.4f}) {_stars(p_b)}")
    both_consistent = (sl_a > 0 and p_a < 0.05) and (sl_b > 0 and p_b < 0.05)
    print(f"    Overall verdict: {_verdict(both_consistent)}")


# ─── Test 10c: Price-level invariance ────────────────────────────────────────

def test10c_price_level_invariance(trades):
    _header(
        "TEST 10c | Price-Level Invariance: Does the Seller Target a Fixed Price?\n"
        "           (prediction: seller's opening offer and final price both flat in valuation)"
    )

    print(
        f"\n  If the seller is targeting a price level rather than a share of surplus,\n"
        f"  both the opening ask and the final deal price should be approximately\n"
        f"  invariant to buyer valuation (slope ≈ 0, not 0.5).\n"
        f"  The slope of deal price on valuation is the key discriminating statistic:\n"
        f"    slope ≈ 0    → seller sets a price, buyer type determines whether trade happens\n"
        f"    slope ≈ 0.5  → equal-split as theory predicts\n"
    )

    seller_first = trades[trades["first_offer_seller"] == 1].dropna(subset=["offer_1_seller"])
    n_sf  = len(seller_first)
    n_all = len(trades)

    def _ols_row(label, y_col, data, groups_col, h0_slope):
        mod = smf.ols(f"{y_col} ~ valuation", data=data).fit(
            cov_type="cluster", cov_kwds={"groups": data[groups_col]}
        )
        sl = mod.params["valuation"]
        se = mod.bse["valuation"]
        ic = mod.params["Intercept"]
        p  = mod.pvalues["valuation"]
        # One-sided t-test: slope < h0_slope
        t_vs_h0 = (sl - h0_slope) / se
        p_vs_h0 = stats.t.cdf(t_vs_h0, df=mod.df_resid)   # p(slope < h0)
        print(f"  {label}")
        print(f"    Intercept = {ic:+.3f}")
        print(f"    Slope     = {sl:+.4f}   SE = {se:.4f}   p(≠0) = {p:.4f} {_stars(p)}")
        print(f"    H0: slope = {h0_slope}  →  t = {t_vs_h0:+.3f},  "
              f"p(slope < {h0_slope}) = {p_vs_h0:.4f} {_stars(p_vs_h0)}")
        print(f"    R² = {mod.rsquared:.4f}")
        return sl, se, p, p_vs_h0

    # ── Seller's opening offer ────────────────────────────────────────────────
    print(f"  ── Seller's opening offer (seller-first trades, n={n_sf}) ──")
    sl_open, se_open, p_open, p_open_h0 = _ols_row(
        "OLS seller_first_offer ~ valuation:",
        "offer_1_seller", seller_first, "seller_code", h0_slope=0.5
    )

    rho_open, p_rho_open = stats.spearmanr(
        seller_first["valuation"], seller_first["offer_1_seller"]
    )
    print(f"    Spearman rho = {rho_open:+.4f}   p = {p_rho_open:.4f} {_stars(p_rho_open)}")

    print(f"\n  Mean seller opening offer by valuation bin:")
    print(f"  {'Bin':<16}  {'N':>5}  {'Mean opening offer':>20}  {'Mean valuation':>16}")
    print(f"  {'-'*62}")
    for lbl, grp in seller_first.groupby("val_bin", observed=True):
        print(f"  {str(lbl):<16}  {len(grp):>5}  {grp['offer_1_seller'].mean():>20.3f}  "
              f"{grp['valuation'].mean():>16.3f}")

    # ── Final deal price ──────────────────────────────────────────────────────
    print(f"\n  ── Final deal price (all trades, n={n_all}) ──")
    sl_price, se_price, p_price, p_price_h0 = _ols_row(
        "OLS deal_price ~ valuation:",
        "deal_price", trades, "participant_code", h0_slope=0.5
    )

    rho_price, p_rho_price = stats.spearmanr(trades["valuation"], trades["deal_price"])
    print(f"    Spearman rho = {rho_price:+.4f}   p = {p_rho_price:.4f} {_stars(p_rho_price)}")

    print(f"\n  Mean deal price by valuation bin:")
    print(f"  {'Bin':<16}  {'N':>5}  {'Mean price':>12}  {'Mean val/2 (pred)':>19}  {'Diff':>8}")
    print(f"  {'-'*60}")
    for lbl, grp in trades.groupby("val_bin", observed=True):
        mp   = grp["deal_price"].mean()
        half = grp["valuation"].mean() / 2
        print(f"  {str(lbl):<16}  {len(grp):>5}  {mp:>12.3f}  {half:>19.3f}  {mp-half:>+8.3f}")

    # ── Discrimination summary ────────────────────────────────────────────────
    print(f"\n  {'─'*68}")
    print(f"\n  DISCRIMINATION SUMMARY:")
    print(f"  {'Outcome':<35}  {'Slope':>8}  {'SE':>6}  {'p(≠0)':>8}  {'p(slope<0.5)':>14}")
    print(f"  {'-'*75}")
    print(f"  {'Seller opening offer':<35}  {sl_open:>8.4f}  {se_open:>6.4f}  {p_open:>8.4f}  {p_open_h0:>14.4f}")
    print(f"  {'Final deal price':<35}  {sl_price:>8.4f}  {se_price:>6.4f}  {p_price:>8.4f}  {p_price_h0:>14.4f}")
    print(
        f"\n  If both slopes are small AND significantly below 0.5, the data support\n"
        f"  the price-level story over the equal-split story."
    )
    price_level_story = (p_price_h0 < 0.05) and (p_open_h0 < 0.05)
    print(f"\n  Verdict: {_verdict(price_level_story)}")


# ─── Test 10: Unexpected terminations ────────────────────────────────────────

def test10_terminations(mid):
    _header("TEST 10 | Unexpected Terminations  (theory: zero)")

    term = mid[mid["bargaining_outcome"] != "acceptance"].copy()
    n    = len(term)

    if n == 0:
        print("\n  No terminations. Consistent with theory.")
        return

    n_player = (term["bargaining_outcome"] == "Player").sum()
    n_random = (term["bargaining_outcome"] == "Random_Termination").sum()
    buyer_t  = ((term["bargaining_outcome"] == "Player") &
                (term["terminated_by_id_in_group"] == BUYER_ID)).sum()
    seller_t = ((term["bargaining_outcome"] == "Player") &
                (term["terminated_by_id_in_group"] == SELLER_ID)).sum()

    print(f"\n  Total terminations: {n}  ({n/len(mid):.1%} of mid-region negotiations)")
    print()
    print(f"  {'Type':<35}  {'Count':>6}  {'Share':>8}")
    print(f"  {'-'*52}")
    print(f"  {'Player termination (any)':<35}  {n_player:>6}  {n_player/n:>8.3f}")
    print(f"    buyer-initiated:               {buyer_t:>6}  {buyer_t/n:>8.3f}")
    print(f"    seller-initiated:              {seller_t:>6}  {seller_t/n:>8.3f}")
    print(f"  {'Computer termination':<35}  {n_random:>6}  {n_random/n:>8.3f}")

    print(f"\n  Valuation of terminated negotiations:")
    print(f"    Mean = {term['valuation'].mean():.2f}   "
          f"Median = {term['valuation'].median():.2f}   "
          f"Min = {term['valuation'].min():.2f}   Max = {term['valuation'].max():.2f}")

    term["total_offers_t"] = term["number_of_offers_buyer"] + term["number_of_offers_seller"]
    print(f"\n  Offers exchanged before termination:")
    print(f"    Mean = {term['total_offers_t'].mean():.2f}   Median = {term['total_offers_t'].median():.2f}")

    # Cluster at low end of region?
    low_third  = (term["valuation"] <= 11).sum()
    print(f"\n  Terminations by valuation bin:")
    print(f"  {'Bin':<16}  {'Term count':>12}  {'Total in bin':>14}  {'Term rate':>10}")
    print(f"  {'-'*55}")
    for lbl, grp_mid in mid.groupby("val_bin", observed=True):
        grp_t = term[term["val_bin"] == lbl]
        rate  = len(grp_t) / len(grp_mid) if len(grp_mid) > 0 else np.nan
        print(f"  {str(lbl):<16}  {len(grp_t):>12}  {len(grp_mid):>14}  {rate:>10.3f}")

    print(f"\n  Theory predicts zero terminations; expects more at low end if any.")
    verdict_dir = low_third / n > 0.3
    print(f"  Verdict: INCONSISTENT  ({n} terminations; "
          f"terminations cluster near b†: {_verdict(verdict_dir)})")


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(mid, trades):
    _header("SUMMARY: Theoretical Predictions vs. Observations (7.72 < val < 21.40, T4)")

    n         = len(mid)
    trade_r   = len(trades) / n
    mean_ratio = (trades["deal_price"] / trades["valuation"]).mean()

    mod_price = smf.ols("deal_price ~ valuation", data=trades).fit()
    sl_price  = mod_price.params["valuation"]

    mod_dur   = smf.ols("bargaining_time_full_sec ~ valuation", data=trades).fit()
    sl_dur    = mod_dur.params["valuation"]

    mod_pay   = smf.ols("payoff ~ valuation", data=trades).fit()
    sl_pay    = mod_pay.params["valuation"]

    mod_sel   = smf.ols("seller_payoff ~ valuation", data=trades).fit()
    sl_sel    = mod_sel.params["valuation"]

    sf_rate   = (mid["first_offer_seller"] == 1).mean()
    multi     = trades[trades["total_offers"] >= 2]
    bf_share  = multi["own_offer_accepted"].mean() if len(multi) > 0 else np.nan

    d = trades.dropna(subset=["pred_delay"])
    delay_corr = trades["bargaining_time_full_sec"].corr(trades["pred_delay"])

    rows = [
        ("Trade rate",                "1.000",   f"{trade_r:.3f}",
         _verdict(trade_r > 0.9)),
        ("Mean price/valuation ratio","0.500",   f"{mean_ratio:.3f}",
         _verdict(abs(mean_ratio - 0.5) < 0.1)),
        ("Price slope on valuation",  "0.500",   f"{sl_price:.3f}",
         _verdict(abs(sl_price - 0.5) < 0.1)),
        ("Seller-first rate",         "1.000",   f"{sf_rate:.3f}",
         _verdict(sf_rate > 0.9)),
        ("Duration slope on valuation","< 0",    f"{sl_dur:.3f}",
         _verdict(sl_dur < 0)),
        ("Corr(obs dur, pred delay)", "≈ 1.0",   f"{delay_corr:.3f}",
         _verdict(delay_corr > 0.4)),
        ("Buyer payoff slope",        "≈ 0.5",   f"{sl_pay:.3f}",
         _verdict(abs(sl_pay - 0.5) < 0.15)),
        ("Seller payoff slope",       "≈ 0.5",   f"{sl_sel:.3f}",
         _verdict(abs(sl_sel - 0.5) < 0.15)),
        ("Buyer makes final offer",   "≈ 1.0",   f"{bf_share:.3f}",
         _verdict(bf_share > 0.5)),
    ]

    print()
    print(f"  {'Outcome':<35}  {'Prediction':>10}  {'Observed':>10}  {'Verdict'}")
    print(f"  {'-'*75}")
    for name, pred, obs, verdict in rows:
        print(f"  {name:<35}  {pred:>10}  {obs:>10}  {verdict}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_all_tests(df):
    mid, trades = _prepare(df)

    print(f"\nSample (T4, intermediate region 7.72 < val < 21.40):")
    print(f"  Total negotiations : {len(mid)}")
    print(f"  Trades             : {len(trades)}  ({len(trades)/len(mid):.2%})")
    print(f"  Valuation range    : [{mid['valuation'].min():.0f}, {mid['valuation'].max():.0f}]")

    test1_trade_rate(mid)
    test2_price(trades)
    test3_delay_slope(trades)
    test4_delay_calibration(trades)
    test5_buyer_payoff(trades)
    test6_first_mover(mid, trades)
    test7_final_offer(trades)
    test8_offer_convergence(trades, df)
    test9_seller_payoff(trades)
    test10a_surplus_share(trades)
    test10b_leverage_channel(trades)
    test10c_price_level_invariance(trades)
    test10_terminations(mid)
    print_summary(mid, trades)

    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")
    run_all_tests(df)
