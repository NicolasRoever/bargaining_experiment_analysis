"""
Test whether buyers benefit from exclusive informational advantage.

Design (2×2):
  Information structure:  one-sided (T3/T4, buyer only informed)
                      vs. two-sided (T1/T2, both parties informed)
  Transaction costs:      no cost (T1/T3)
                      vs. with cost (T2/T4, c = 0.05/sec)

Restriction: two-sided analyses are limited to negotiations with positive
             gains_from_trade (as instructed).

Run with:
    conda run -n bargaining_analysis python -m src.bargaining_analysis.main_results.test_information_rents
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from src.bargaining_analysis.config import BLD

SELLER_ID = 1
BUYER_ID  = 2

CELLS = {
    "One-sided, No cost  (T3)": "T3",
    "One-sided, Cost     (T4)": "T4",
    "Two-sided, No cost  (T1)": "T1",
    "Two-sided, Cost     (T2)": "T2",
}
CELL_ORDER = list(CELLS.keys())


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _header(title):
    w = 72
    print(); print("=" * w); print(f"  {title}"); print("=" * w)


def _subheader(title):
    print(f"\n  ── {title} ──")


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


def _2x2_header():
    print(f"\n  {'Cell':<30}  {'N':>5}  {'Mean':>8}  {'SE':>8}  {'p(≠0)':>8}")
    print(f"  {'-'*64}")


def _2x2_row(label, vals, groups, h0=0.0):
    if len(vals) < 2:
        print(f"  {label:<30}  {'—':>5}")
        return np.nan
    m, se, t, p = _clustered_mean_test(vals, groups, h0=h0)
    print(f"  {label:<30}  {len(vals):>5}  {m:>8.3f}  {se:>8.4f}  {p:>8.4f} {_stars(p)}")
    return m


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


def _cell(df, treatment):
    return df[df["treatment"] == treatment]


# ─── Test 1: Buyer's share of realised surplus ────────────────────────────────

def test1_buyer_share_realised(full, trades):
    _header("TEST 1 | Buyer's Share of Realised Surplus  (theory: higher under one-sided)")

    _subheader("2×2 table — mean buyer_payoff / (buyer_payoff + seller_payoff)")
    _2x2_header()
    means = {}
    for label, t in CELLS.items():
        grp = _cell(trades, t).dropna(subset=["buyer_share_realised"])
        means[t] = _2x2_row(label, grp["buyer_share_realised"], grp["participant_code"])

    print(f"\n  One-sided advantage (T3−T1, no cost):  {means['T3']-means['T1']:+.3f}")
    print(f"  One-sided advantage (T4−T2, cost):     {means['T4']-means['T2']:+.3f}")

    reg_df = trades.dropna(subset=["buyer_share_realised"])
    _reg_2x2(reg_df, "buyer_share_realised", "participant_code")

    print(f"\n  Theory: one-sided buyers should capture a larger share (two_sided coef < 0).")


# ─── Test 2: Buyer's share of available surplus ───────────────────────────────

def test2_buyer_share_available(full, trades):
    _header("TEST 2 | Buyer's Share of Available Surplus  (payoff / gains_from_trade)")

    _subheader("2×2 table — mean buyer_payoff / gains_from_trade")
    _2x2_header()
    means = {}
    for label, t in CELLS.items():
        grp = _cell(trades, t).dropna(subset=["buyer_share_available"])
        means[t] = _2x2_row(label, grp["buyer_share_available"], grp["participant_code"])

    print(f"\n  One-sided advantage (T3−T1, no cost):  {means['T3']-means['T1']:+.3f}")
    print(f"  One-sided advantage (T4−T2, cost):     {means['T4']-means['T2']:+.3f}")

    reg_df = trades.dropna(subset=["buyer_share_available"])
    _reg_2x2(reg_df, "buyer_share_available", "participant_code")

    print(f"\n  Theory: one-sided buyers should capture a larger share (two_sided coef < 0).")


# ─── Test 3: Price deviation from equal split ─────────────────────────────────

def test3_price_deviation(full, trades):
    _header("TEST 3 | Price Deviation from Equal Split  (positive = seller captures more)")

    _subheader("2×2 table — mean (deal_price − equal_split_price)")
    _2x2_header()
    means = {}
    for label, t in CELLS.items():
        grp = _cell(trades, t).dropna(subset=["price_deviation"])
        means[t] = _2x2_row(label, grp["price_deviation"], grp["participant_code"])

    print(f"\n  Seller premium shift (T1−T3, no cost):  {means['T1']-means['T3']:+.3f}")
    print(f"  Seller premium shift (T2−T4, cost):     {means['T2']-means['T4']:+.3f}")

    reg_df = trades.dropna(subset=["price_deviation"])
    _reg_2x2(reg_df, "price_deviation", "participant_code")

    print(f"\n  Theory: one-sided buyer advantage → lower deviation (two_sided coef > 0).")


# ─── Test 4: Who concedes more? ───────────────────────────────────────────────

def test4_concessions(full, trades):
    _header("TEST 4 | Bargaining Concessions  (theory: one-sided buyers concede less)")

    multi = trades[trades["total_offers"] >= 2].copy()
    multi = multi.dropna(subset=["offer_1_seller", "offer_1_buyer", "deal_price"])

    # Absolute concession from first offer to deal price
    multi["seller_concession"] = (multi["offer_1_seller"] - multi["deal_price"]).abs()
    multi["buyer_concession"]  = (multi["offer_1_buyer"]  - multi["deal_price"]).abs()
    multi["buyer_concedes_more"] = (
        multi["buyer_concession"] > multi["seller_concession"]
    ).astype(float)

    _subheader("Part A — Mean absolute concession (first offer → deal price)")
    print(f"\n  {'Cell':<30}  {'N':>5}  {'Seller concedes':>16}  {'Buyer concedes':>15}  {'Buyer > Seller':>15}")
    print(f"  {'-'*86}")
    for label, t in CELLS.items():
        grp = _cell(multi, t)
        if len(grp) < 2: continue
        sc = grp["seller_concession"].mean()
        bc = grp["buyer_concession"].mean()
        bm = grp["buyer_concedes_more"].mean()
        print(f"  {label:<30}  {len(grp):>5}  {sc:>16.3f}  {bc:>15.3f}  {bm:>15.3f}")

    _subheader("Part B — P(buyer accepts seller's offer) by cell")
    _2x2_header()
    means = {}
    for label, t in CELLS.items():
        grp = _cell(multi, t).dropna(subset=["seller_offer_accepted"])
        means[t] = _2x2_row(label, grp["seller_offer_accepted"], grp["participant_code"])

    print(f"\n  One-sided advantage (T3−T1, no cost):  {means['T3']-means['T1']:+.3f}")
    print(f"  One-sided advantage (T4−T2, cost):     {means['T4']-means['T2']:+.3f}")

    reg_df = multi.dropna(subset=["seller_offer_accepted"])
    _reg_2x2(reg_df, "seller_offer_accepted", "participant_code")

    print(f"\n  Theory: one-sided buyers concede less → lower buyer-accepts-seller rate (two_sided coef > 0).")


# ─── Test 5: Trade rates ──────────────────────────────────────────────────────

def test5_trade_rates(full):
    _header("TEST 5 | Trade Rates by Cell  (theory: higher under one-sided — rents create gains from trade)")

    _subheader("2×2 table — fraction of negotiations ending in agreement")
    _2x2_header()
    means = {}
    for label, t in CELLS.items():
        grp = _cell(full, t)
        means[t] = _2x2_row(label, grp["agreement_dummy"], grp["participant_code"])

    print(f"\n  One-sided advantage (T3−T1, no cost):  {means['T3']-means['T1']:+.3f}")
    print(f"  One-sided advantage (T4−T2, cost):     {means['T4']-means['T2']:+.3f}")

    _reg_2x2(full, "agreement_dummy", "participant_code")

    print(f"\n  Theory: one-sided → higher trade rate (two_sided coef < 0).")


# ─── Test 6: Surplus-level heterogeneity ─────────────────────────────────────

def test6_surplus_heterogeneity(full, trades):
    _header("TEST 6 | Does the Information Advantage Matter More at High Surplus?")

    # Within each cell, split on median gains_from_trade
    print(f"\n  Buyer's share of realised surplus by cell × surplus bin:")
    print(f"  (Cutoff = within-cell median gains_from_trade)")
    print()
    print(f"  {'Cell':<30}  {'N low':>7}  {'Share low':>10}  {'N high':>8}  {'Share high':>11}  {'Diff':>8}")
    print(f"  {'-'*80}")

    reg_rows = []
    for label, t in CELLS.items():
        grp = _cell(trades, t).dropna(subset=["buyer_share_realised", "gains_from_trade"])
        if len(grp) < 4: continue
        median_gft = grp["gains_from_trade"].median()
        low  = grp[grp["gains_from_trade"] <= median_gft]
        high = grp[grp["gains_from_trade"] >  median_gft]
        sl   = low["buyer_share_realised"].mean()  if len(low)  > 0 else np.nan
        sh   = high["buyer_share_realised"].mean() if len(high) > 0 else np.nan
        diff = sh - sl if not (np.isnan(sl) or np.isnan(sh)) else np.nan
        print(f"  {label:<30}  {len(low):>7}  {sl:>10.3f}  {len(high):>8}  {sh:>11.3f}  {diff:>+8.3f}")
        for _, row in grp.iterrows():
            reg_rows.append({
                "buyer_share_realised": row["buyer_share_realised"],
                "two_sided":           row["two_sided"],
                "has_cost":            row["has_cost"],
                "high_surplus":        int(row["gains_from_trade"] > median_gft),
                "participant_code":    row["participant_code"],
            })

    reg_df = pd.DataFrame(reg_rows)
    mod = smf.ols(
        "buyer_share_realised ~ two_sided + has_cost + high_surplus + two_sided:high_surplus",
        data=reg_df
    ).fit(cov_type="cluster", cov_kwds={"groups": reg_df["participant_code"]})

    print(f"\n  Regression: buyer_share ~ two_sided + has_cost + high_surplus + two_sided×high_surplus")
    print(f"\n  {'Parameter':<35}  {'Coef':>8}  {'SE':>8}  {'p':>8}")
    print(f"  {'-'*62}")
    labels = {
        "Intercept":                 "Intercept",
        "two_sided":                 "Two-sided",
        "has_cost":                  "Has cost",
        "high_surplus":              "High surplus",
        "two_sided:high_surplus":    "Two-sided × High surplus",
    }
    for name, coef, se, p in zip(mod.params.index, mod.params, mod.bse, mod.pvalues):
        print(f"  {labels.get(name, name):<35}  {coef:>8.4f}  {se:>8.4f}  {p:>8.4f} {_stars(p)}")

    print(f"\n  Theory: two-sided penalty should be larger at high surplus (two_sided×high_surplus < 0).")


# ─── Test 7: Seller behaviour across information structures ───────────────────

def test7_seller_behaviour(full, trades):
    _header("TEST 7 | Seller Behaviour Across Information Structures")

    _subheader("Part A — Mean seller first offer (seller-first negotiations only)")
    seller_first = trades[trades["first_offer_seller"] == 1].dropna(subset=["offer_1_seller"])
    _2x2_header()
    for label, t in CELLS.items():
        grp = _cell(seller_first, t)
        _2x2_row(label, grp["offer_1_seller"], grp["seller_code"])

    _subheader("Part B — Seller concession rate (seller's offer accepted by buyer)")
    multi = trades[trades["total_offers"] >= 2].copy()
    _2x2_header()
    means_conc = {}
    for label, t in CELLS.items():
        grp = _cell(multi, t).dropna(subset=["seller_offer_accepted"])
        means_conc[t] = _2x2_row(label, grp["seller_offer_accepted"], grp["seller_code"])

    print(f"\n  [seller_offer_accepted = buyer accepted seller's offer = seller did NOT need to concede further]")
    print(f"  Higher rate = seller's offers are accepted more = seller less likely to need to make concessions.")

    _subheader("Part C — Seller-initiated termination rate (all negotiations)")
    _2x2_header()
    for label, t in CELLS.items():
        grp = _cell(full, t)
        seller_term = (
            (grp["bargaining_outcome"] == "Player") &
            (grp["terminated_by_id_in_group"] == SELLER_ID)
        ).astype(float)
        _2x2_row(label, seller_term, grp["seller_code"])

    _subheader("Regression: seller_offer_accepted ~ two_sided + has_cost + interaction")
    reg_df = multi.dropna(subset=["seller_offer_accepted"])
    mod = _reg_2x2(reg_df, "seller_offer_accepted", "seller_code")

    print(f"\n  Theory: informed sellers (T1/T2) bargain harder → lower seller_offer_accepted rate.")
    print(f"  (two_sided coef < 0 would mean sellers with private info extract more — buyer must accept)")


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary(full, trades):
    _header("SUMMARY: Does the Buyer Benefit from Exclusive Information?")

    multi = trades[trades["total_offers"] >= 2].dropna(
        subset=["seller_offer_accepted", "buyer_share_realised"]
    )

    def _reg_coef(outcome, df, groups):
        df = df.dropna(subset=[outcome])
        mod = smf.ols(
            f"{outcome} ~ two_sided + has_cost + two_sided:has_cost", data=df
        ).fit(cov_type="cluster", cov_kwds={"groups": df[groups]})
        coef = mod.params.get("two_sided", np.nan)
        se   = mod.bse.get("two_sided", np.nan)
        p    = mod.pvalues.get("two_sided", np.nan)
        return coef, se, p

    rows = [
        ("Buyer share of realised surplus", "two_sided < 0",
         *_reg_coef("buyer_share_realised",    trades, "participant_code")),
        ("Buyer share of available surplus","two_sided < 0",
         *_reg_coef("buyer_share_available",   trades, "participant_code")),
        ("Price deviation from eq. split",  "two_sided > 0",
         *_reg_coef("price_deviation",         trades, "participant_code")),
        ("Trade rate",                       "two_sided < 0",
         *_reg_coef("agreement_dummy",         full,   "participant_code")),
        ("P(buyer accepts seller offer)",    "two_sided > 0",
         *_reg_coef("seller_offer_accepted",   multi,  "participant_code")),
    ]

    print(f"\n  'two_sided' regression coefficient (two-sided vs. one-sided, controlling for cost):")
    print()
    print(f"  {'Outcome':<38}  {'Prediction':>16}  {'Coef':>8}  {'SE':>6}  {'p':>8}  {'Verdict'}")
    print(f"  {'-'*92}")
    for name, pred, coef, se, p in rows:
        if pred.endswith("< 0"):
            verdict = "CONSISTENT" if coef < 0 and p < 0.1 else "INCONSISTENT"
        else:
            verdict = "CONSISTENT" if coef > 0 and p < 0.1 else "INCONSISTENT"
        print(
            f"  {name:<38}  {pred:>16}  {coef:>8.4f}  {se:>6.4f}  {p:>8.4f}  {verdict}"
        )

    print()
    consistent = sum(
        1 for _, pred, coef, se, p in rows
        if (pred.endswith("< 0") and coef < 0 and p < 0.1)
        or (pred.endswith("> 0") and coef > 0 and p < 0.1)
    )
    print(f"  {consistent}/{len(rows)} margins consistent with buyer information rent hypothesis.")


# ═══════════════════════════════════════════════════════════════════════════════
# SURPLUS-KNOWLEDGE HYPOTHESIS
# Does knowing the seller's cost make buyers more accommodating?
# ═══════════════════════════════════════════════════════════════════════════════

def _ols_slope(df, y, x, groups):
    """OLS y ~ x with clustered SEs. Returns (intercept, slope, se_slope, p_slope)."""
    mod = smf.ols(f"{y} ~ {x}", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[groups]}
    )
    return (mod.params["Intercept"], mod.params[x],
            mod.bse[x], mod.pvalues[x], mod.rsquared)


def _compare_slopes(df_os, df_ts, y, x, groups, label_os="One-sided", label_ts="Two-sided"):
    """Run OLS y ~ x for each group and test slope equality via pooled interaction."""
    ic_os, sl_os, se_os, p_os, r2_os = _ols_slope(df_os.dropna(subset=[y,x]), y, x, groups)
    ic_ts, sl_ts, se_ts, p_ts, r2_ts = _ols_slope(df_ts.dropna(subset=[y,x]), y, x, groups)

    print(f"\n  {'Group':<18}  {'N':>5}  {'Intercept':>11}  {'Slope':>8}  {'SE':>8}  {'p':>8}  {'R²':>6}")
    print(f"  {'-'*72}")
    for lbl, df_, ic, sl, se, p, r2 in [
        (label_os, df_os, ic_os, sl_os, se_os, p_os, r2_os),
        (label_ts, df_ts, ic_ts, sl_ts, se_ts, p_ts, r2_ts),
    ]:
        n = df_.dropna(subset=[y,x]).shape[0]
        print(f"  {lbl:<18}  {n:>5}  {ic:>11.4f}  {sl:>8.4f}  {se:>8.4f}  {p:>8.4f} {_stars(p)}  {r2:>6.3f}")

    # Interaction test: pool both groups, add two_sided dummy and x:two_sided interaction
    df_os2 = df_os.dropna(subset=[y, x]).copy(); df_os2["_ts"] = 0
    df_ts2 = df_ts.dropna(subset=[y, x]).copy(); df_ts2["_ts"] = 1
    pool = pd.concat([df_os2, df_ts2])
    mod_pool = smf.ols(f"{y} ~ {x} + _ts + {x}:_ts", data=pool).fit(
        cov_type="cluster", cov_kwds={"groups": pool[groups]}
    )
    int_coef = mod_pool.params.get(f"{x}:_ts", np.nan)
    int_se   = mod_pool.bse.get(f"{x}:_ts", np.nan)
    int_p    = mod_pool.pvalues.get(f"{x}:_ts", np.nan)
    print(f"\n  Slope difference (two-sided − one-sided):  {sl_ts - sl_os:+.4f}")
    print(f"  Interaction test ({x}×two_sided):  coef = {int_coef:+.4f}  SE = {int_se:.4f}  p = {int_p:.4f} {_stars(int_p)}")
    return sl_os, sl_ts, int_p


# ─── SK Test 1: P(buyer accepts seller) ~ buyer valuation ────────────────────

def sk_test1_acceptance_vs_valuation(full, trades):
    _header("SK TEST 1 | P(Buyer Accepts Seller's Offer) ~ Buyer Valuation\n"
            "           (prediction: steeper slope under one-sided)")

    print(
        "\n  Under one-sided, valuation = surplus — higher val means more to lose, predicting\n"
        "  a positive slope. Under two-sided, buyer doesn't know surplus so the slope should\n"
        "  be weaker.\n"
    )
    multi = trades[trades["total_offers"] >= 2].dropna(
        subset=["seller_offer_accepted", "buyer_valuation"]
    )
    os_ = multi[multi["two_sided"] == 0]
    ts_ = multi[multi["two_sided"] == 1]

    sl_os, sl_ts, p_int = _compare_slopes(
        os_, ts_, "seller_offer_accepted", "buyer_valuation", "participant_code"
    )
    pred_consistent = (sl_os > sl_ts) and (p_int < 0.1)
    print(f"\n  Prediction: one-sided slope > two-sided slope.")
    print(f"  Verdict: {'CONSISTENT' if pred_consistent else 'INCONSISTENT'}  "
          f"(os={sl_os:+.4f}, ts={sl_ts:+.4f}, p_interaction={p_int:.4f})")


# ─── SK Test 2: Duration ~ buyer valuation ───────────────────────────────────

def sk_test2_duration_vs_valuation(full, trades):
    _header("SK TEST 2 | Negotiation Duration ~ Buyer Valuation\n"
            "           (prediction: steeper negative slope under one-sided)")

    print(
        "\n  If knowing the surplus makes high-valuation buyers eager to close, we should\n"
        "  see a steeper negative slope (longer delay at low val, shorter at high val)\n"
        "  under one-sided. Under two-sided the relationship should be flatter.\n"
    )
    d = trades.dropna(subset=["bargaining_time_full_sec", "buyer_valuation"])
    os_ = d[d["two_sided"] == 0]
    ts_ = d[d["two_sided"] == 1]

    sl_os, sl_ts, p_int = _compare_slopes(
        os_, ts_, "bargaining_time_full_sec", "buyer_valuation", "participant_code"
    )
    pred_consistent = (sl_os < sl_ts) and (p_int < 0.1)
    print(f"\n  Prediction: one-sided slope < two-sided slope (more negative).")
    print(f"  Verdict: {'CONSISTENT' if pred_consistent else 'INCONSISTENT'}  "
          f"(os={sl_os:+.4f}, ts={sl_ts:+.4f}, p_interaction={p_int:.4f})")


# ─── SK Test 3: Surplus knowledge vs. surplus level ──────────────────────────

def sk_test3_knowledge_vs_level(full, trades):
    _header("SK TEST 3 | Surplus Knowledge vs. Surplus Level\n"
            "           (at the same GFT, does one-sided still produce more accommodation?)")

    print(
        "\n  We compare P(buyer accepts) and buyer surplus share at the same actual GFT\n"
        "  level across one-sided and two-sided. Overlap region: GFT in [1, 30].\n"
        "  If it is knowledge (not surplus level) that matters, one-sided buyers should\n"
        "  be more accommodating even conditional on GFT.\n"
    )
    multi = trades[trades["total_offers"] >= 2].copy()
    overlap = multi[multi["gains_from_trade"].between(1, 30)].copy()
    overlap["gft_bin"] = pd.qcut(overlap["gains_from_trade"], q=4,
                                  labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])

    # Part A: P(buyer accepts) by GFT bin × info structure
    print(f"\n  Part A — P(buyer accepts seller offer) by GFT quartile:")
    print(f"  {'GFT bin':<14}  {'N (os)':>8}  {'Acc (os)':>10}  {'N (ts)':>8}  {'Acc (ts)':>10}  {'Diff':>8}")
    print(f"  {'-'*62}")
    for bin_, grp in overlap.groupby("gft_bin", observed=True):
        os_g = grp[grp["two_sided"] == 0]
        ts_g = grp[grp["two_sided"] == 1]
        acc_os = os_g["seller_offer_accepted"].mean() if len(os_g) > 0 else np.nan
        acc_ts = ts_g["seller_offer_accepted"].mean() if len(ts_g) > 0 else np.nan
        diff   = acc_os - acc_ts if not (np.isnan(acc_os) or np.isnan(acc_ts)) else np.nan
        print(f"  {str(bin_):<14}  {len(os_g):>8}  {acc_os:>10.3f}  {len(ts_g):>8}  {acc_ts:>10.3f}  {diff:>+8.3f}")

    # Part B: buyer surplus share by GFT bin × info structure
    trades_ov = trades[trades["gains_from_trade"].between(1, 30)].dropna(
        subset=["buyer_share_realised"]
    ).copy()
    trades_ov["gft_bin"] = pd.qcut(trades_ov["gains_from_trade"], q=4,
                                    labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])

    print(f"\n  Part B — Buyer's share of realised surplus by GFT quartile:")
    print(f"  {'GFT bin':<14}  {'N (os)':>8}  {'Share (os)':>12}  {'N (ts)':>8}  {'Share (ts)':>12}  {'Diff':>8}")
    print(f"  {'-'*68}")
    for bin_, grp in trades_ov.groupby("gft_bin", observed=True):
        os_g = grp[grp["two_sided"] == 0]
        ts_g = grp[grp["two_sided"] == 1]
        sh_os = os_g["buyer_share_realised"].mean() if len(os_g) > 0 else np.nan
        sh_ts = ts_g["buyer_share_realised"].mean() if len(ts_g) > 0 else np.nan
        diff  = sh_os - sh_ts if not (np.isnan(sh_os) or np.isnan(sh_ts)) else np.nan
        print(f"  {str(bin_):<14}  {len(os_g):>8}  {sh_os:>12.3f}  {len(ts_g):>8}  {sh_ts:>12.3f}  {diff:>+8.3f}")

    # Regression: seller_offer_accepted ~ two_sided + gft_bin + two_sided:gft_bin (continuous GFT)
    reg = overlap.dropna(subset=["seller_offer_accepted", "gains_from_trade"]).copy()
    mod = smf.ols(
        "seller_offer_accepted ~ two_sided + gains_from_trade + two_sided:gains_from_trade",
        data=reg
    ).fit(cov_type="cluster", cov_kwds={"groups": reg["participant_code"]})
    print(f"\n  Regression: P(accept) ~ two_sided + GFT + two_sided×GFT  (overlap region n={len(reg)}):")
    for nm, coef, se, p in zip(mod.params.index, mod.params, mod.bse, mod.pvalues):
        print(f"    {nm:<40}  coef={coef:+.4f}  SE={se:.4f}  p={p:.4f} {_stars(p)}")

    ts_main = mod.params.get("two_sided", np.nan)
    print(f"\n  Prediction: two_sided main effect < 0 (one-sided buyers more accommodating at same GFT).")
    print(f"  Verdict: {'CONSISTENT' if ts_main < 0 else 'INCONSISTENT'}  (two_sided coef = {ts_main:+.4f})")


# ─── SK Test 4: Buyer concession rate ────────────────────────────────────────

def sk_test4_buyer_concession_rate(full, trades):
    _header("SK TEST 4 | Buyer Concession Rate\n"
            "           (prediction: higher under one-sided — knowing s=0 makes buyers cave)")

    print(
        "\n  Concession rate = (deal_price − buyer_first_offer) / (buyer_valuation − buyer_first_offer)\n"
        "  Measures how far the buyer moved from their opening toward their maximum willingness to pay.\n"
        "  If knowing s=0 makes buyers accommodate, this rate should be higher under one-sided.\n"
    )
    multi = trades[trades["total_offers"] >= 2].copy()
    multi = multi[
        (multi["number_of_offers_buyer"] > 0) &
        (multi["buyer_valuation"] > 0)
    ].dropna(subset=["offer_1_buyer", "deal_price", "buyer_valuation"])
    multi = multi[multi["offer_1_buyer"] < multi["buyer_valuation"]].copy()
    multi["buyer_conc_rate"] = (
        (multi["deal_price"] - multi["offer_1_buyer"]) /
        (multi["buyer_valuation"]  - multi["offer_1_buyer"])
    ).clip(0, 1)

    os_ = multi[multi["two_sided"] == 0]
    ts_ = multi[multi["two_sided"] == 1]

    print(f"\n  {'Group':<18}  {'N':>5}  {'Mean':>8}  {'Median':>8}  {'SE (clust)':>12}")
    print(f"  {'-'*56}")
    for lbl, grp, groups_col in [("One-sided", os_, "participant_code"), ("Two-sided", ts_, "participant_code")]:
        m, se, t, p = _clustered_mean_test(grp["buyer_conc_rate"], grp[groups_col])
        print(f"  {lbl:<18}  {len(grp):>5}  {m:>8.3f}  {grp['buyer_conc_rate'].median():>8.3f}  {se:>12.4f}")

    sl_os, sl_ts, p_int = _compare_slopes(
        os_, ts_, "buyer_conc_rate", "buyer_valuation", "participant_code"
    )

    # Simple mean comparison with clustered test on the difference
    os_["_g"] = 0; ts_["_g"] = 1
    pool = pd.concat([os_, ts_])
    mod = smf.ols("buyer_conc_rate ~ _g", data=pool).fit(
        cov_type="cluster", cov_kwds={"groups": pool["participant_code"]}
    )
    diff_coef = mod.params["_g"]; diff_se = mod.bse["_g"]; diff_p = mod.pvalues["_g"]
    print(f"\n  Mean difference (two-sided − one-sided):  {diff_coef:+.4f}  SE={diff_se:.4f}  p={diff_p:.4f} {_stars(diff_p)}")

    pred_consistent = (diff_coef < 0) and (diff_p < 0.1)
    print(f"\n  Prediction: one-sided concession rate > two-sided (two_sided coef < 0).")
    print(f"  Verdict: {'CONSISTENT' if pred_consistent else 'INCONSISTENT'}")


# ─── SK Test 5: Seller behaviour as placebo ──────────────────────────────────

def sk_test5_seller_placebo(full, trades):
    _header("SK TEST 5 | Seller Behaviour as Placebo\n"
            "           (prediction: similar across info structures once normalized)")

    seller_first = trades[trades["first_offer_seller"] == 1].dropna(
        subset=["offer_1_seller", "deal_price"]
    ).copy()
    seller_first = seller_first[seller_first["offer_1_seller"] > 0]

    # Normalize seller's first offer by the design range (30 for os, 60 for ts)
    seller_first["seller_norm_offer"] = np.where(
        seller_first["two_sided"] == 0,
        seller_first["offer_1_seller"] / 30,
        seller_first["offer_1_seller"] / 60,
    )
    # Seller concession rate = (first_offer - deal_price) / first_offer
    seller_first["seller_conc_rate"] = (
        (seller_first["offer_1_seller"] - seller_first["deal_price"]) /
         seller_first["offer_1_seller"]
    ).clip(-1, 1)

    os_ = seller_first[seller_first["two_sided"] == 0]
    ts_ = seller_first[seller_first["two_sided"] == 1]

    print(f"\n  {'Metric':<42}  {'One-sided':>12}  {'Two-sided':>12}  {'Diff':>8}")
    print(f"  {'-'*78}")

    for col, lbl in [
        ("seller_norm_offer",  "Seller offer / max_val (30 or 60)"),
        ("seller_conc_rate",   "Seller concession rate (1 - price/ask)"),
    ]:
        mo = os_[col].mean(); mt = ts_[col].mean()
        print(f"  {lbl:<42}  {mo:>12.3f}  {mt:>12.3f}  {mo-mt:>+8.3f}")

        os_2 = os_.copy(); os_2["_ts"] = 0
        ts_2 = ts_.copy(); ts_2["_ts"] = 1
        pool = pd.concat([os_2, ts_2])
        mod = smf.ols(f"{col} ~ _ts", data=pool).fit(
            cov_type="cluster", cov_kwds={"groups": pool["seller_code"]}
        )
        coef = mod.params["_ts"]; se = mod.bse["_ts"]; p = mod.pvalues["_ts"]
        print(f"    Regression diff: coef={coef:+.4f}  SE={se:.4f}  p={p:.4f} {_stars(p)}")

    print(f"\n  Prediction: seller behaviour similar once normalized (differences not significant).")
    print(f"  If the norm offer and concession rate differ, the treatment effect has a seller-side component.")


# ─── SK Test 6: Buyer first-offer aggressiveness ─────────────────────────────

def sk_test6_buyer_opening_aggressiveness(full, trades):
    _header("SK TEST 6 | Buyer First-Offer Aggressiveness\n"
            "           (prediction: two-sided buyers open lower relative to valuation)")

    print(
        "\n  Buyer aggressiveness = buyer_first_offer / buyer_valuation.\n"
        "  Lower ratio = more aggressive (offering less relative to max willingness to pay).\n"
        "  If two-sided buyers are tougher, this ratio should be lower under two-sided.\n"
    )
    multi = trades[trades["total_offers"] >= 2].copy()
    multi = multi[
        (multi["number_of_offers_buyer"] > 0) &
        (multi["buyer_valuation"] > 0)
    ].dropna(subset=["offer_1_buyer", "buyer_valuation"])
    multi["buyer_rel_offer"] = (
        multi["offer_1_buyer"] / multi["buyer_valuation"]
    ).clip(0, 1)

    os_ = multi[multi["two_sided"] == 0]
    ts_ = multi[multi["two_sided"] == 1]

    print(f"\n  {'Group':<18}  {'N':>5}  {'Mean':>8}  {'Median':>8}  {'SE (clust)':>12}")
    print(f"  {'-'*56}")
    for lbl, grp in [("One-sided", os_), ("Two-sided", ts_)]:
        m, se, t, p = _clustered_mean_test(grp["buyer_rel_offer"], grp["participant_code"])
        print(f"  {lbl:<18}  {len(grp):>5}  {m:>8.3f}  {grp['buyer_rel_offer'].median():>8.3f}  {se:>12.4f}")

    # Mean comparison
    os_2 = os_.copy(); os_2["_ts"] = 0
    ts_2 = ts_.copy(); ts_2["_ts"] = 1
    pool = pd.concat([os_2, ts_2])
    mod = smf.ols("buyer_rel_offer ~ _ts", data=pool).fit(
        cov_type="cluster", cov_kwds={"groups": pool["participant_code"]}
    )
    coef = mod.params["_ts"]; se = mod.bse["_ts"]; p = mod.pvalues["_ts"]
    print(f"\n  Mean difference (two-sided − one-sided):  {coef:+.4f}  SE={se:.4f}  p={p:.4f} {_stars(p)}")

    # Also show slope of buyer_rel_offer on buyer_valuation for each group
    print(f"\n  OLS buyer_rel_offer ~ buyer_valuation (does aggressiveness vary with val?):")
    sl_os, sl_ts, p_int = _compare_slopes(
        os_, ts_, "buyer_rel_offer", "buyer_valuation", "participant_code"
    )

    pred_consistent = (coef > 0) and (p < 0.1)
    print(f"\n  Prediction: two-sided buyers open lower (two_sided coef < 0, i.e. coef > 0 means ts higher).")
    # Positive coef means ts is higher (less aggressive under two-sided)
    # Negative coef means ts is lower (more aggressive under two-sided) — that's what we predict
    print(f"  Verdict: {'CONSISTENT' if not pred_consistent else 'INCONSISTENT'}  "
          f"(two-sided buyers open {'higher' if coef>0 else 'lower'} by {abs(coef):.3f})")


# ─── SK Summary ───────────────────────────────────────────────────────────────

def sk_summary(full, trades):
    _header("SURPLUS-KNOWLEDGE SUMMARY\n"
            "  Does knowing the seller's cost make buyers more accommodating?")

    multi = trades[trades["total_offers"] >= 2].copy()
    multi_valid = multi[
        (multi["number_of_offers_buyer"] > 0) &
        (multi["buyer_valuation"] > 0)
    ].dropna(subset=["offer_1_buyer", "deal_price", "buyer_valuation"])
    multi_valid = multi_valid[
        multi_valid["offer_1_buyer"] < multi_valid["buyer_valuation"]
    ].copy()
    multi_valid["buyer_conc_rate"] = (
        (multi_valid["deal_price"] - multi_valid["offer_1_buyer"]) /
        (multi_valid["buyer_valuation"] - multi_valid["offer_1_buyer"])
    ).clip(0, 1)
    multi_valid["buyer_rel_offer"] = (
        multi_valid["offer_1_buyer"] / multi_valid["buyer_valuation"]
    ).clip(0, 1)

    def _diff_coef(outcome, df, groups="participant_code"):
        d = df.dropna(subset=[outcome]).copy()
        os_ = d[d["two_sided"] == 0].copy(); os_["_ts"] = 0
        ts_ = d[d["two_sided"] == 1].copy(); ts_["_ts"] = 1
        pool = pd.concat([os_, ts_])
        mod = smf.ols(f"{outcome} ~ _ts", data=pool).fit(
            cov_type="cluster", cov_kwds={"groups": pool[groups]}
        )
        return mod.params["_ts"], mod.bse["_ts"], mod.pvalues["_ts"]

    # Interaction slopes for tests 1 and 2
    def _interaction_p(df, y, x):
        os_ = df[df["two_sided"] == 0].dropna(subset=[y, x]).copy(); os_["_ts"] = 0
        ts_ = df[df["two_sided"] == 1].dropna(subset=[y, x]).copy(); ts_["_ts"] = 1
        pool = pd.concat([os_, ts_])
        mod = smf.ols(f"{y} ~ {x} + _ts + {x}:_ts", data=pool).fit(
            cov_type="cluster", cov_kwds={"groups": pool["participant_code"]}
        )
        sl_os = mod.params[x]
        sl_ts = sl_os + mod.params.get(f"{x}:_ts", 0)
        p = mod.pvalues.get(f"{x}:_ts", np.nan)
        return sl_os, sl_ts, p

    multi_acc = multi.dropna(subset=["seller_offer_accepted", "buyer_valuation"])
    sl_os_1, sl_ts_1, p_int_1 = _interaction_p(multi_acc, "seller_offer_accepted", "buyer_valuation")

    trades_dur = trades.dropna(subset=["bargaining_time_full_sec", "buyer_valuation"])
    sl_os_2, sl_ts_2, p_int_2 = _interaction_p(trades_dur, "bargaining_time_full_sec", "buyer_valuation")

    c_conc, se_conc, p_conc = _diff_coef("buyer_conc_rate", multi_valid)
    c_open, se_open, p_open = _diff_coef("buyer_rel_offer", multi_valid)

    seller_first = trades[trades["first_offer_seller"] == 1].dropna(
        subset=["offer_1_seller", "deal_price"]
    ).copy()
    seller_first = seller_first[seller_first["offer_1_seller"] > 0].copy()
    seller_first["seller_norm_offer"] = np.where(
        seller_first["two_sided"] == 0,
        seller_first["offer_1_seller"] / 30,
        seller_first["offer_1_seller"] / 60,
    )
    seller_first["seller_conc_rate"] = (
        (seller_first["offer_1_seller"] - seller_first["deal_price"]) /
         seller_first["offer_1_seller"]
    ).clip(-1, 1)
    c_snorm, se_snorm, p_snorm = _diff_coef("seller_norm_offer", seller_first, "seller_code")
    c_sconc, se_sconc, p_sconc = _diff_coef("seller_conc_rate", seller_first, "seller_code")

    rows = [
        ("P(accept) slope steeper under os",
         f"os={sl_os_1:+.4f} ts={sl_ts_1:+.4f}", f"p_int={p_int_1:.4f}",
         sl_os_1 > sl_ts_1 and p_int_1 < 0.1),
        ("Duration slope more negative under os",
         f"os={sl_os_2:+.4f} ts={sl_ts_2:+.4f}", f"p_int={p_int_2:.4f}",
         sl_os_2 < sl_ts_2 and p_int_2 < 0.1),
        ("Higher buyer concession rate under os",
         f"diff={c_conc:+.4f}", f"p={p_conc:.4f}",
         c_conc < 0 and p_conc < 0.1),
        ("Lower buyer opening offer under ts (tougher)",
         f"diff={c_open:+.4f}", f"p={p_open:.4f}",
         c_open < 0 and p_open < 0.1),
        ("Seller norm offer similar (placebo)",
         f"diff={c_snorm:+.4f}", f"p={p_snorm:.4f}",
         abs(c_snorm) < 0.05 or p_snorm >= 0.1),
        ("Seller conc rate similar (placebo)",
         f"diff={c_sconc:+.4f}", f"p={p_sconc:.4f}",
         abs(c_sconc) < 0.1 or p_sconc >= 0.1),
    ]

    print(f"\n  {'Test':<44}  {'Result':>28}  {'Stat':>18}  {'Verdict'}")
    print(f"  {'-'*102}")
    for name, result, stat, consistent in rows:
        print(f"  {name:<44}  {result:>28}  {stat:>18}  {'CONSISTENT' if consistent else 'INCONSISTENT'}")

    n_con = sum(r[3] for r in rows)
    print(f"\n  {n_con}/{len(rows)} tests consistent with 'surplus knowledge makes buyers accommodating'.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_all_tests(df):
    full, trades = _prepare(df)

    print(f"\nSample overview (negotiations, one row per buyer):")
    for t in ["T1", "T2", "T3", "T4"]:
        g = _cell(full, t)
        tr = g["agreement_dummy"].sum()
        print(f"  {t}: n={len(g)}, trades={tr} ({tr/len(g):.2%})")

    test1_buyer_share_realised(full, trades)
    test2_buyer_share_available(full, trades)
    test3_price_deviation(full, trades)
    test4_concessions(full, trades)
    test5_trade_rates(full)
    test6_surplus_heterogeneity(full, trades)
    test7_seller_behaviour(full, trades)
    print_summary(full, trades)

    sk_test1_acceptance_vs_valuation(full, trades)
    sk_test2_duration_vs_valuation(full, trades)
    sk_test3_knowledge_vs_level(full, trades)
    sk_test4_buyer_concession_rate(full, trades)
    sk_test5_seller_placebo(full, trades)
    sk_test6_buyer_opening_aggressiveness(full, trades)
    sk_summary(full, trades)

    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")
    run_all_tests(df)
