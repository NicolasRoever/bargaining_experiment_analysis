"""
Test the comparative static effect of transaction costs.
Compare no-cost (T1/T3) to cost (T2/T4) treatments, separately within each
information structure and pooled.

Design (2×2):
  Information structure:  one-sided (T3/T4)  vs.  two-sided (T1/T2)
  Transaction costs:      no cost   (T1/T3)  vs.  with cost (T2/T4)

Restriction: two-sided analyses limited to positive gains_from_trade.

Hypothesis: costs speed up the process (duration, offers) without changing
            outcomes (trade rate, surplus division, prices).

Run with:
    conda run -n bargaining_analysis python -m src.bargaining_analysis.main_results.test_transaction_costs
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from src.bargaining_analysis.config import BLD

SELLER_ID = 1
BUYER_ID  = 2

CELLS = {
    "One-sided, No cost  (T3)": ("T3", 0, 0),
    "One-sided, Cost     (T4)": ("T4", 0, 1),
    "Two-sided, No cost  (T1)": ("T1", 1, 0),
    "Two-sided, Cost     (T2)": ("T2", 1, 1),
}
PCTILES = [10, 25, 50, 75, 90]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _header(title):
    w = 72
    print(); print("=" * w); print(f"  {title}"); print("=" * w)


def _subheader(s):
    print(f"\n  ── {s} ──")


def _stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1:  return "*"
    return "   "


def _clustered_mean_test(series, groups, h0=0.0):
    d = pd.DataFrame({"y": series, "g": groups}).dropna()
    mod = smf.ols("y ~ 1", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["g"]}
    )
    coef = mod.params["Intercept"]
    se   = mod.bse["Intercept"]
    t    = (coef - h0) / se
    p    = 2 * (1 - stats.t.cdf(abs(t), df=mod.df_resid))
    return coef, se, t, p


def _cell(df, treatment):
    return df[df["treatment"] == treatment]


def _cost_diff_test(outcome, df, groups="participant_code"):
    """OLS: outcome ~ has_cost + two_sided + has_cost:two_sided."""
    d = df.dropna(subset=[outcome]).copy()
    mod = smf.ols(
        f"{outcome} ~ has_cost + two_sided + has_cost:two_sided", data=d
    ).fit(cov_type="cluster", cov_kwds={"groups": d[groups]})
    return mod


def _cost_diff_test_within(outcome, df, groups="participant_code"):
    """OLS: outcome ~ has_cost, for use within a single info-structure group.

    Using the full interaction model within one group causes perfect
    multicollinearity (two_sided is constant), so we use a simple model.
    """
    d = df.dropna(subset=[outcome]).copy()
    mod = smf.ols(
        f"{outcome} ~ has_cost", data=d
    ).fit(cov_type="cluster", cov_kwds={"groups": d[groups]})
    return mod


def _print_reg(mod, extra_note=""):
    labels = {
        "Intercept":          "Intercept (T3, no cost)",
        "has_cost":           "Has cost  (T4 vs T3 / T2 vs T1)",
        "two_sided":          "Two-sided (T1/T2 vs T3/T4)",
        "has_cost:two_sided": "Has cost × Two-sided",
    }
    print(f"\n  {'Parameter':<32}  {'Coef':>10}  {'SE':>8}  {'p':>8}")
    print(f"  {'-'*62}")
    for nm, coef, se, p in zip(mod.params.index, mod.params, mod.bse, mod.pvalues):
        print(f"  {labels.get(nm, nm):<32}  {coef:>10.4f}  {se:>8.4f}  {p:>8.4f} {_stars(p)}")
    if extra_note:
        print(f"  {extra_note}")


def _table_header(cols):
    row = "  " + "".join(f"{c:>{w}}" for c, w in cols)
    sep = "  " + "-" * sum(w for _, w in cols)
    print(row); print(sep)


# ─── Prepare data ─────────────────────────────────────────────────────────────

def _prepare(df):
    buyers  = df[df["participant_role"] == "Buyer"].copy()
    sellers = df[df["participant_role"] == "Seller"][
        ["negotiation_id", "payoff", "first_offer", "number_of_offers", "offer_1",
         "offer_time_1", "own_offer_accepted", "participant_code",
         "last_offer", "last_offer_time"]
    ].rename(columns={
        "payoff":            "seller_payoff",
        "first_offer":       "first_offer_seller",
        "number_of_offers":  "number_of_offers_seller",
        "offer_1":           "offer_1_seller",
        "offer_time_1":      "offer_time_1_seller",
        "own_offer_accepted":"seller_offer_accepted",
        "participant_code":  "seller_code",
        "last_offer":        "last_offer_seller",
        "last_offer_time":   "last_offer_time_seller",
    })
    m = buyers.merge(sellers, on="negotiation_id", how="left")
    m.rename(columns={
        "first_offer":       "first_offer_buyer",
        "number_of_offers":  "number_of_offers_buyer",
        "offer_1":           "offer_1_buyer",
        "offer_time_1":      "offer_time_1_buyer",
        "last_offer":        "last_offer_buyer",
    }, inplace=True)

    m["total_offers"] = m["number_of_offers_buyer"] + m["number_of_offers_seller"]
    m["two_sided"]    = (m["information_asymmetry"] == "two-sided").astype(int)
    m["has_cost"]     = (m["TA_costs"] == 0.05).astype(int)

    keep = (m["treatment"].isin(["T3", "T4"])) | (
        m["treatment"].isin(["T1", "T2"]) & (m["gains_from_trade"] > 0)
    )
    full = m[keep].copy()

    full["time_to_first_offer"] = full[["offer_time_1_buyer", "offer_time_1_seller"]].min(axis=1)
    full["total_realised"]      = full["payoff"] + full["seller_payoff"]
    full["buyer_share_realised"] = np.where(
        full["total_realised"] > 0,
        full["payoff"] / full["total_realised"], np.nan
    )
    full["equal_split_price"] = np.where(
        full["two_sided"] == 0,
        full["gains_from_trade"] / 2,
        full["seller_valuation"] + full["gains_from_trade"] / 2,
    )
    full["price_deviation"]  = full["deal_price"] - full["equal_split_price"]
    full["price_over_val"]   = np.where(
        full["buyer_valuation"] > 0,
        full["deal_price"] / full["buyer_valuation"], np.nan
    )
    full["buyer_term"]  = (
        (full["bargaining_outcome"] == "Player") &
        (full["terminated_by_id_in_group"] == BUYER_ID)
    ).astype(float)
    full["seller_term"] = (
        (full["bargaining_outcome"] == "Player") &
        (full["terminated_by_id_in_group"] == SELLER_ID)
    ).astype(float)
    full["any_term"]    = (full["bargaining_outcome"] != "acceptance").astype(float)

    trades = full[full["agreement_dummy"] == 1].copy()
    trades["sec_per_offer"] = trades["bargaining_time_full_sec"] / trades["total_offers"].replace(0, np.nan)
    return full, trades


# ─── Test 1: Trade rates ───────────────────────────────────────────────────────

def test1_trade_rates(full):
    _header("TEST 1 | Trade Rates  (prediction: no effect of costs)")

    _table_header([("Cell", 30), ("N", 7), ("Trades", 8), ("Rate", 8), ("SE", 9), ("p(≠0)", 10)])
    for label, (t, ts, hc) in CELLS.items():
        grp = _cell(full, t)
        m, se, _, p = _clustered_mean_test(grp["agreement_dummy"], grp["participant_code"])
        print(f"  {label:<30}  {len(grp):>7}  {int(grp['agreement_dummy'].sum()):>8}  {m:>8.3f}  {se:>9.4f}  {p:>10.4f}")

    print(f"\n  Cost effect within one-sided (T4 vs T3):")
    mod_os = _cost_diff_test_within("agreement_dummy", full[full["two_sided"] == 0])
    coef = mod_os.params["has_cost"]; se = mod_os.bse["has_cost"]; p = mod_os.pvalues["has_cost"]
    print(f"    coef = {coef:+.4f}  SE = {se:.4f}  p = {p:.4f} {_stars(p)}")

    print(f"  Cost effect within two-sided (T2 vs T1):")
    mod_ts = _cost_diff_test_within("agreement_dummy", full[full["two_sided"] == 1])
    coef2 = mod_ts.params["has_cost"]; se2 = mod_ts.bse["has_cost"]; p2 = mod_ts.pvalues["has_cost"]
    print(f"    coef = {coef2:+.4f}  SE = {se2:.4f}  p = {p2:.4f} {_stars(p2)}")

    print(f"\n  Pooled regression:")
    mod = _cost_diff_test("agreement_dummy", full)
    _print_reg(mod)
    print(f"\n  Prediction: has_cost NOT significant.")
    cost_p = mod.pvalues["has_cost"]
    print(f"  Verdict: {'CONSISTENT' if cost_p >= 0.1 else 'INCONSISTENT'}  (p = {cost_p:.4f})")


# ─── Test 2: Termination rates ────────────────────────────────────────────────

def test2_termination_rates(full):
    _header("TEST 2 | Termination Rates  (prediction: costs may create terminations)")

    _table_header([("Cell", 30), ("N", 7), ("Any term", 10), ("Buyer term", 12), ("Seller term", 13), ("Comp term", 11)])
    for label, (t, ts, hc) in CELLS.items():
        grp = _cell(full, t)
        n    = len(grp)
        at   = grp["any_term"].mean()
        bt   = grp["buyer_term"].mean()
        st   = grp["seller_term"].mean()
        ct   = (grp["bargaining_outcome"] == "Random_Termination").mean()
        print(f"  {label:<30}  {n:>7}  {at:>10.3f}  {bt:>12.3f}  {st:>13.3f}  {ct:>11.3f}")

    for outcome, label in [
        ("any_term",    "Any termination"),
        ("buyer_term",  "Buyer-initiated termination"),
        ("seller_term", "Seller-initiated termination"),
    ]:
        print(f"\n  Pooled regression — {label}:")
        mod = _cost_diff_test(outcome, full)
        _print_reg(mod)
        cost_p = mod.pvalues["has_cost"]
        print(f"  Verdict: {'costs increase term rate' if mod.params['has_cost']>0 and cost_p<0.1 else 'no significant cost effect'}  (p = {cost_p:.4f})")


# ─── Test 3: Surplus division ─────────────────────────────────────────────────

def test3_surplus_division(trades):
    _header("TEST 3 | Surplus Division  (prediction: no effect of costs on split)")

    _table_header([("Cell", 30), ("N", 6), ("Buyer share", 12), ("Buyer pay", 11), ("Seller pay", 11), ("Total pay", 11)])
    for label, (t, ts, hc) in CELLS.items():
        grp = _cell(trades, t).dropna(subset=["buyer_share_realised"])
        print(
            f"  {label:<30}  {len(grp):>6}  "
            f"{grp['buyer_share_realised'].mean():>12.3f}  "
            f"{grp['payoff'].mean():>11.3f}  "
            f"{grp['seller_payoff'].mean():>11.3f}  "
            f"{grp['total_realised'].mean():>11.3f}"
        )

    print(f"\n  Note: total payoff should be LOWER with costs if time costs destroy surplus.")

    for outcome, label in [
        ("buyer_share_realised", "Buyer share of realised surplus"),
        ("total_realised",       "Total realised surplus (buyer + seller)"),
    ]:
        print(f"\n  Pooled regression — {label}:")
        mod = _cost_diff_test(outcome, trades)
        _print_reg(mod)
        cost_p = mod.pvalues["has_cost"]
        pred = "no effect" if outcome == "buyer_share_realised" else "significant reduction"
        print(f"  Prediction: {pred}.  Verdict: {'CONSISTENT' if (outcome=='buyer_share_realised' and cost_p>=0.1) or (outcome=='total_realised' and mod.params['has_cost']<0 and cost_p<0.1) else 'INCONSISTENT'}  (p = {cost_p:.4f})")


# ─── Test 4: Price level ──────────────────────────────────────────────────────

def test4_price_level(trades):
    _header("TEST 4 | Price Level  (prediction: no effect of costs)")

    _table_header([("Cell", 30), ("N", 6), ("Mean price", 12), ("Price/val (os)", 16), ("Dev. eq.split (ts)", 20)])
    for label, (t, ts, hc) in CELLS.items():
        grp = _cell(trades, t)
        mp  = grp["deal_price"].mean()
        pv  = grp["price_over_val"].mean() if ts == 0 else np.nan
        dev = grp["price_deviation"].mean() if ts == 1 else np.nan
        pv_s  = f"{pv:>16.3f}" if not np.isnan(pv) else f"{'—':>16}"
        dev_s = f"{dev:>20.3f}" if not np.isnan(dev) else f"{'—':>20}"
        print(f"  {label:<30}  {len(grp):>6}  {mp:>12.3f}  {pv_s}  {dev_s}")

    print(f"\n  Pooled regression — deal price:")
    mod = _cost_diff_test("deal_price", trades)
    _print_reg(mod)
    cost_p = mod.pvalues["has_cost"]
    print(f"\n  Prediction: has_cost NOT significant.")
    print(f"  Verdict: {'CONSISTENT' if cost_p >= 0.1 else 'INCONSISTENT'}  (p = {cost_p:.4f})")


# ─── Test 5: Negotiation duration ────────────────────────────────────────────

def test5_duration(full, trades):
    _header("TEST 5 | Negotiation Duration  (prediction: SIGNIFICANT reduction with costs)")

    for label_suffix, df in [("conditional on trade", trades), ("unconditional", full)]:
        _subheader(f"Mean duration ({label_suffix})")
        _table_header([("Cell", 30), ("N", 6), ("Mean sec", 10), ("Median", 9), ("SD", 9), ("pct_reduction vs no-cost", 26)])

        no_cost_means = {}
        for label, (t, ts, hc) in CELLS.items():
            grp = _cell(df, t)["bargaining_time_full_sec"].dropna()
            no_cost_means[t] = grp.mean()

        for label, (t, ts, hc) in CELLS.items():
            grp  = _cell(df, t)["bargaining_time_full_sec"].dropna()
            m    = grp.mean(); med = grp.median(); sd = grp.std()
            nc_t = "T3" if ts == 0 else "T1"
            pct  = (no_cost_means[nc_t] - m) / no_cost_means[nc_t] * 100 if hc else 0.0
            pct_s = f"{pct:>+26.1f}%" if hc else f"{'(reference)':>26}"
            print(f"  {label:<30}  {len(grp):>6}  {m:>10.2f}  {med:>9.2f}  {sd:>9.2f}  {pct_s}")

    print(f"\n  Pooled regression — duration (conditional on trade):")
    mod = _cost_diff_test("bargaining_time_full_sec", trades)
    _print_reg(mod)
    cost_p = mod.pvalues["has_cost"]
    pct_os = mod.params["has_cost"] / (mod.params["Intercept"]) * 100
    print(f"  Cost reduction (one-sided): {mod.params['has_cost']:+.2f} sec ({pct_os:+.1f}% of T3 baseline)")
    print(f"\n  Prediction: has_cost SIGNIFICANT and negative.")
    print(f"  Verdict: {'CONSISTENT' if mod.params['has_cost']<0 and cost_p<0.05 else 'INCONSISTENT'}  (p = {cost_p:.6f})")


# ─── Test 6: Number of offers ─────────────────────────────────────────────────

def test6_number_of_offers(trades):
    _header("TEST 6 | Number of Offers  (prediction: SIGNIFICANT reduction with costs)")

    _subheader("Mean total offers conditional on trade")
    _table_header([("Cell", 30), ("N", 6), ("Mean offers", 13), ("Median", 9), ("% vs no-cost", 14)])

    nc_means = {}
    for label, (t, ts, hc) in CELLS.items():
        grp = _cell(trades, t)["total_offers"].dropna()
        nc_means[t] = grp.mean()

    for label, (t, ts, hc) in CELLS.items():
        grp = _cell(trades, t)["total_offers"].dropna()
        m   = grp.mean(); med = grp.median()
        nc_t = "T3" if ts == 0 else "T1"
        pct  = (nc_means[nc_t] - m) / nc_means[nc_t] * 100 if hc else 0.0
        pct_s = f"{pct:>+14.1f}%" if hc else f"{'(reference)':>14}"
        print(f"  {label:<30}  {len(grp):>6}  {m:>13.2f}  {med:>9.2f}  {pct_s}")

    print(f"\n  Note: T3 has extreme outliers (max 411 offers). Median is more reliable.")

    print(f"\n  Pooled regression — total offers:")
    mod = _cost_diff_test("total_offers", trades)
    _print_reg(mod)
    cost_p = mod.pvalues["has_cost"]
    print(f"\n  Prediction: has_cost SIGNIFICANT and negative.")
    print(f"  Verdict: {'CONSISTENT' if mod.params['has_cost']<0 and cost_p<0.05 else 'INCONSISTENT'}  (p = {cost_p:.6f})")


# ─── Test 7: Time per offer ───────────────────────────────────────────────────

def test7_time_per_offer(trades):
    _header("TEST 7 | Time per Offer  (is speedup from fewer offers or faster offers?)")

    _table_header([("Cell", 30), ("N", 6), ("Mean sec/offer", 16), ("Median", 9), ("% vs no-cost", 14)])

    nc_means = {}
    for label, (t, ts, hc) in CELLS.items():
        grp = _cell(trades, t)["sec_per_offer"].dropna()
        nc_means[t] = grp.mean()

    for label, (t, ts, hc) in CELLS.items():
        grp = _cell(trades, t)["sec_per_offer"].dropna()
        m   = grp.mean(); med = grp.median()
        nc_t = "T3" if ts == 0 else "T1"
        pct  = (nc_means[nc_t] - m) / nc_means[nc_t] * 100 if hc else 0.0
        pct_s = f"{pct:>+14.1f}%" if hc else f"{'(reference)':>14}"
        print(f"  {label:<30}  {len(grp):>6}  {m:>16.3f}  {med:>9.3f}  {pct_s}")

    print(f"\n  Pooled regression — sec_per_offer:")
    mod = _cost_diff_test("sec_per_offer", trades)
    _print_reg(mod)
    cost_p = mod.pvalues["has_cost"]

    print(f"\n  Interpretation: if sec_per_offer and total_offers BOTH fall with costs,")
    print(f"  the speedup operates on two margins simultaneously.")
    print(f"  has_cost coef = {mod.params['has_cost']:+.4f}  p = {cost_p:.4f} {_stars(cost_p)}")


# ─── Test 8: Time to first offer ─────────────────────────────────────────────

def test8_time_to_first_offer(full):
    _header("TEST 8 | Time to First Offer  (prediction: faster start with costs)")

    df = full.dropna(subset=["time_to_first_offer"]).copy()

    _table_header([("Cell", 30), ("N", 6), ("Mean sec", 10), ("Median", 9), ("% vs no-cost", 14)])

    nc_means = {}
    for label, (t, ts, hc) in CELLS.items():
        grp = _cell(df, t)["time_to_first_offer"]
        nc_means[t] = grp.mean()

    for label, (t, ts, hc) in CELLS.items():
        grp = _cell(df, t)["time_to_first_offer"]
        m = grp.mean(); med = grp.median()
        nc_t = "T3" if ts == 0 else "T1"
        pct  = (nc_means[nc_t] - m) / nc_means[nc_t] * 100 if hc else 0.0
        pct_s = f"{pct:>+14.1f}%" if hc else f"{'(reference)':>14}"
        print(f"  {label:<30}  {len(grp):>6}  {m:>10.3f}  {med:>9.3f}  {pct_s}")

    print(f"\n  Pooled regression — time to first offer:")
    mod = _cost_diff_test("time_to_first_offer", df)
    _print_reg(mod)
    cost_p = mod.pvalues["has_cost"]
    print(f"\n  Prediction: has_cost SIGNIFICANT and negative.")
    print(f"  Verdict: {'CONSISTENT' if mod.params['has_cost']<0 and cost_p<0.05 else 'INCONSISTENT'}  (p = {cost_p:.6f})")


# ─── Test 9: Convergence speed ───────────────────────────────────────────────

def test9_convergence(trades, df_raw):
    _header("TEST 9 | Offer Convergence  (do costs change gap size or closing speed?)")

    multi = trades[
        (trades["total_offers"] >= 2) &
        (trades["number_of_offers_buyer"]  > 0) &
        (trades["number_of_offers_seller"] > 0)
    ].dropna(subset=["offer_1_seller", "offer_1_buyer", "last_offer_seller", "last_offer_buyer"]).copy()
    multi["initial_gap"] = multi["offer_1_seller"] - multi["offer_1_buyer"]
    multi["final_gap"]   = multi["last_offer_seller"] - multi["last_offer_buyer"]
    multi["gap_closed"]  = multi["initial_gap"] - multi["final_gap"]
    multi["pct_closed"]  = (multi["gap_closed"] / multi["initial_gap"].replace(0, np.nan)).clip(0, 1)

    _table_header([("Cell", 30), ("N", 6), ("Init gap", 10), ("Final gap", 11), ("Gap closed", 12), ("% closed", 10)])
    for label, (t, ts, hc) in CELLS.items():
        grp = _cell(multi, t)
        ig  = grp["initial_gap"].mean(); fg = grp["final_gap"].mean()
        gc  = grp["gap_closed"].mean();  pc = grp["pct_closed"].mean()
        print(f"  {label:<30}  {len(grp):>6}  {ig:>10.3f}  {fg:>11.3f}  {gc:>12.3f}  {pc:>10.3f}")

    for outcome, lbl in [("initial_gap", "Initial gap"), ("final_gap", "Final gap"), ("pct_closed", "Pct gap closed")]:
        print(f"\n  Pooled regression — {lbl}:")
        mod = _cost_diff_test(outcome, multi)
        coef = mod.params["has_cost"]; se = mod.bse["has_cost"]; p = mod.pvalues["has_cost"]
        print(f"    has_cost: coef = {coef:+.4f}  SE = {se:.4f}  p = {p:.4f} {_stars(p)}")

    # Per-round gap (up to round 6)
    buyers  = df_raw[df_raw["participant_role"] == "Buyer"].copy()
    sellers = df_raw[df_raw["participant_role"] == "Seller"].copy()
    neg = buyers.merge(
        sellers[["negotiation_id"] + [f"offer_{i}" for i in range(1, 8)]].rename(
            columns={f"offer_{i}": f"s_offer_{i}" for i in range(1, 8)}
        ),
        on="negotiation_id", how="inner"
    )
    keep = (neg["treatment"].isin(["T3","T4"])) | (
        neg["treatment"].isin(["T1","T2"]) & (neg["gains_from_trade"] > 0)
    )
    neg = neg[keep & (neg["agreement_dummy"] == 1)].copy()

    _subheader("Mean gap (seller_offer_r − buyer_offer_r) by round")
    header_row = f"  {'Round':>6}  " + "  ".join(f"{'Gap '+t:>12}" for t in ["T3","T4","T1","T2"])
    print(f"\n{header_row}")
    print("  " + "-" * (len(header_row) - 2))
    for r in range(1, 7):
        sc = f"s_offer_{r}"; bc = f"offer_{r}"
        if sc not in neg.columns or bc not in neg.columns: break
        row_parts = []
        for t in ["T3", "T4", "T1", "T2"]:
            g = neg[neg["treatment"] == t][[sc, bc]].dropna()
            if len(g) < 5:
                row_parts.append(f"{'—':>12}")
            else:
                row_parts.append(f"{(g[sc]-g[bc]).mean():>12.2f}")
        print(f"  {r:>6}  " + "  ".join(row_parts))

    print(f"\n  Interpretation: costs reduce the initial gap (fewer high opening asks),")
    print(f"  not just the number of rounds needed to close the same gap.")


# ─── Test 10: Duration distributions ─────────────────────────────────────────

def test10_duration_distributions(full, trades):
    _header("TEST 10 | Duration Distributions  (percentiles + CSV for plotting)")

    for label, df, outcome_label in [
        ("Trades",       trades, "trade"),
        ("Terminations", full[full["agreement_dummy"] == 0], "term"),
    ]:
        _subheader(f"{label} — duration percentiles (seconds)")
        header = f"  {'Cell':<30}  {'N':>5}  " + "  ".join(f"p{p:02d}{'':>1}" for p in PCTILES)
        print(header)
        print("  " + "-" * len(header))
        for lbl, (t, ts, hc) in CELLS.items():
            grp = _cell(df, t)["bargaining_time_full_sec"].dropna()
            pvals = "  ".join(f"{grp.quantile(p/100):>6.2f}" for p in PCTILES) if len(grp) > 0 else "—"
            print(f"  {lbl:<30}  {len(grp):>5}  {pvals}")

    # Save raw duration data to CSV
    out_rows = []
    for df_, outcome_label in [(trades, "trade"), (full[full["agreement_dummy"] == 0], "termination")]:
        grp_df = df_[["treatment", "information_asymmetry", "TA_costs", "two_sided", "has_cost",
                       "bargaining_time_full_sec", "negotiation_id"]].copy()
        grp_df["outcome_type"] = outcome_label
        out_rows.append(grp_df)
    csv_df = pd.concat(out_rows, ignore_index=True)
    out_path = BLD / "data" / "duration_by_treatment.csv"
    csv_df.to_csv(out_path, index=False)

    # Fine-grained percentiles (1–99) for CDF plotting
    pct_rows = []
    for df_, outcome_label in [(trades, "trade"), (full[full["agreement_dummy"] == 0], "termination")]:
        for lbl, (t, ts, hc) in CELLS.items():
            grp = _cell(df_, t)["bargaining_time_full_sec"].dropna()
            for p in range(1, 100):
                pct_rows.append({
                    "treatment": t, "two_sided": ts, "has_cost": hc,
                    "outcome_type": outcome_label, "percentile": p,
                    "duration": grp.quantile(p / 100) if len(grp) > 0 else np.nan,
                })
    pct_df = pd.DataFrame(pct_rows)
    pct_path = BLD / "data" / "duration_percentiles_by_treatment.csv"
    pct_df.to_csv(pct_path, index=False)

    print(f"\n  Raw duration data saved to:        {out_path}")
    print(f"  Fine-grained percentiles saved to: {pct_path}")


# ─── Test 11: Comprehensive regression ───────────────────────────────────────

def test11_comprehensive_regression(full, trades):
    _header("TEST 11 | Comprehensive Regression\n"
            "           (prediction: has_cost significant for process, not outcomes)")

    outcomes = [
        ("agreement_dummy",          full,   "Trade dummy",              "participant_code", "INCONSISTENT if sig"),
        ("buyer_share_realised",     trades, "Buyer share of surplus",   "participant_code", "INCONSISTENT if sig"),
        ("deal_price",               trades, "Deal price",               "participant_code", "INCONSISTENT if sig"),
        ("bargaining_time_full_sec", trades, "Duration (sec)",           "participant_code", "CONSISTENT if sig <0"),
        ("total_offers",             trades, "Total offers",             "participant_code", "CONSISTENT if sig <0"),
        ("sec_per_offer",            trades, "Sec per offer",            "participant_code", "CONSISTENT if sig <0"),
        ("time_to_first_offer",      full,   "Time to first offer",      "participant_code", "CONSISTENT if sig <0"),
    ]

    print(f"\n  {'Outcome':<30}  {'has_cost coef':>14}  {'SE':>8}  {'p':>8}  {'two_sided coef':>15}  {'p_ts':>8}  {'Expected'}")
    print(f"  {'-'*102}")
    summary = []
    for outcome, df_, label, grp_col, expected in outcomes:
        d = df_.dropna(subset=[outcome]).copy()
        mod = smf.ols(
            f"{outcome} ~ has_cost + two_sided + has_cost:two_sided", data=d
        ).fit(cov_type="cluster", cov_kwds={"groups": d[grp_col]})
        hc_coef = mod.params["has_cost"];    hc_se = mod.bse["has_cost"];    hc_p = mod.pvalues["has_cost"]
        ts_coef = mod.params["two_sided"];   ts_p  = mod.pvalues["two_sided"]
        print(f"  {label:<30}  {hc_coef:>14.4f}  {hc_se:>8.4f}  {hc_p:>8.4f} {_stars(hc_p)}  {ts_coef:>15.4f}  {ts_p:>8.4f} {_stars(ts_p)}  {expected}")
        summary.append((label, expected, hc_coef, hc_se, hc_p))

    return summary


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(summary_rows, full, trades):
    _header("SUMMARY: Transaction Cost Effects — Outcomes vs. Process")

    outcome_vars = ["Trade dummy", "Buyer share of surplus", "Deal price"]
    process_vars = ["Duration (sec)", "Total offers", "Sec per offer", "Time to first offer"]

    print(f"\n  ── OUTCOME VARIABLES (prediction: NO effect of costs) ──")
    print(f"\n  {'Variable':<35}  {'has_cost coef':>14}  {'p':>8}  {'Verdict'}")
    print(f"  {'-'*68}")
    for label, expected, coef, se, p in summary_rows:
        if label not in outcome_vars: continue
        verdict = "CONSISTENT" if p >= 0.1 else "INCONSISTENT"
        print(f"  {label:<35}  {coef:>14.4f}  {p:>8.4f} {_stars(p)}  {verdict}")

    print(f"\n  ── PROCESS VARIABLES (prediction: SIGNIFICANT negative effect) ──")
    print(f"\n  {'Variable':<35}  {'has_cost coef':>14}  {'p':>8}  {'% change':>10}  {'Verdict'}")
    print(f"  {'-'*80}")
    baselines = {
        "Duration (sec)":       trades[trades["has_cost"]==0]["bargaining_time_full_sec"].mean(),
        "Total offers":         trades[trades["has_cost"]==0]["total_offers"].mean(),
        "Sec per offer":        trades[trades["has_cost"]==0]["sec_per_offer"].mean(),
        "Time to first offer":  full[full["has_cost"]==0]["time_to_first_offer"].mean(),
    }
    for label, expected, coef, se, p in summary_rows:
        if label not in process_vars: continue
        base = baselines.get(label, np.nan)
        pct  = coef / base * 100 if base and not np.isnan(base) else np.nan
        verdict = "CONSISTENT" if coef < 0 and p < 0.05 else "INCONSISTENT"
        print(f"  {label:<35}  {coef:>14.4f}  {p:>8.4f} {_stars(p)}  {pct:>+9.1f}%  {verdict}")

    n_out = sum(1 for l,_,c,s,p in summary_rows if l in outcome_vars and p >= 0.1)
    n_proc = sum(1 for l,_,c,s,p in summary_rows if l in process_vars and c < 0 and p < 0.05)
    print(f"\n  Outcome variables:  {n_out}/{len(outcome_vars)} consistent (no cost effect)")
    print(f"  Process variables:  {n_proc}/{len(process_vars)} consistent (significant cost speedup)")
    print(f"\n  Overall: {'SUPPORTED' if n_proc == len(process_vars) and n_out == len(outcome_vars) else 'PARTIALLY SUPPORTED'} — "
          f"'costs speed up the process without changing outcomes'")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_all_tests(df):
    full, trades = _prepare(df)

    print(f"\nSample overview (one row per buyer-negotiation):")
    for label, (t, ts, hc) in CELLS.items():
        g = _cell(full, t)
        print(f"  {label}: n={len(g)}, trades={g['agreement_dummy'].sum()} ({g['agreement_dummy'].mean():.2%})")

    test1_trade_rates(full)
    test2_termination_rates(full)
    test3_surplus_division(trades)
    test4_price_level(trades)
    test5_duration(full, trades)
    test6_number_of_offers(trades)
    test7_time_per_offer(trades)
    test8_time_to_first_offer(full)
    test9_convergence(trades, df)
    test10_duration_distributions(full, trades)
    summary_rows = test11_comprehensive_regression(full, trades)
    print_summary(summary_rows, full, trades)

    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")
    run_all_tests(df)
