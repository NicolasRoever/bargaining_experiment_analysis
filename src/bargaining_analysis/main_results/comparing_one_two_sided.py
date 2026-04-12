"""
Comparing one-sided vs. two-sided uncertainty: key behavioral statistics.

Prints summary statistics and p-values for:
  1. Opening offers (sellers and buyers)
  2. Concession rates (sellers and buyers)
  3. Who accepts in case of trade
  4. Who terminates (buyer or seller)
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from src.bargaining_analysis.config import BLD, OVERLEAF_TABLES


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mwu_pvalue(a, b):
    """Two-sided Mann-Whitney U p-value, dropping NaN."""
    a = a.dropna()
    b = b.dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p


def _ols_pvalue(outcome, df, group_col="participant_code"):
    """
    OLS: outcome ~ two_sided, SE clustered by participant_code.
    Returns (mean_one, mean_two, coef, p).
    """
    d = df.dropna(subset=[outcome]).copy()
    d["two_sided"] = (d["information_asymmetry"] == "two-sided").astype(int)
    mod = smf.ols(f"{outcome} ~ two_sided", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d[group_col]}
    )
    mean_one = d[d["two_sided"] == 0][outcome].mean()
    mean_two = d[d["two_sided"] == 1][outcome].mean()
    coef = mod.params["two_sided"]
    p = mod.pvalues["two_sided"]
    return mean_one, mean_two, coef, p


def _header(title):
    print()
    print("=" * 65)
    print(title)
    print("=" * 65)


def _row(label, one_val, two_val, p_mwu, p_ols):
    print(f"  {label:<35} one={one_val:>7.3f}  two={two_val:>7.3f}  "
          f"p_MWU={p_mwu:>6.3f}  p_OLS={p_ols:>6.3f}")


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyse_opening_offers(df):
    """
    Opening offers of sellers and buyers in one- vs. two-sided uncertainty.
    Expressed as share of gains from trade (offer_1 / gains_from_trade).
    Only observations with gains_from_trade > 0 and a non-missing first offer.
    Also reports mean gains_from_trade to contextualise the EUR differences.
    """
    _header("1. OPENING OFFERS (one-sided vs. two-sided)")

    # Report mean GFT per group so reader can contextualise absolute offers
    for info in ["one-sided", "two-sided"]:
        gft = df[(df["participant_role"] == "Buyer") &
                 (df["information_asymmetry"] == info)]["gains_from_trade"].mean()
        print(f"  Mean gains_from_trade ({info}): {gft:.3f} EUR")

    print()
    for role in ["Buyer", "Seller"]:
        sub = df[
            (df["participant_role"] == role) &
            df["offer_1"].notna() &
            (df["gains_from_trade"] > 0)
        ].copy()
        sub["offer_1_pct_gft"] = sub["offer_1"] / sub["gains_from_trade"]

        one = sub[sub["information_asymmetry"] == "one-sided"]["offer_1_pct_gft"]
        two = sub[sub["information_asymmetry"] == "two-sided"]["offer_1_pct_gft"]
        p_mwu = _mwu_pvalue(one, two)
        m1, m2, _, p_ols = _ols_pvalue("offer_1_pct_gft", sub)
        _row(f"{role} opening offer (% of GFT)", m1, m2, p_mwu, p_ols)


def analyse_concession_rates(df):
    """
    Concession rates expressed as share of gains from trade.
    Buyer concession  = (last_offer - offer_1) / gains_from_trade  (positive → moved toward seller)
    Seller concession = (offer_1 - last_offer) / gains_from_trade  (positive → moved toward buyer)
    Only observations with >= 2 offers and gains_from_trade > 0.
    Also reports change in split_gains_from_trade from first to final offer as an
    alternative measure (requires trades to have deal_price for split calculation).
    """
    _header("2. CONCESSION RATES (% of gains from trade, one-sided vs. two-sided)")

    sub = df[
        (df["number_of_offers"] >= 2) &
        (df["gains_from_trade"] > 0)
    ].copy()

    for role, sign, label in [
        ("Buyer",  1,  "Buyer concession (% GFT, last-first)"),
        ("Seller", -1, "Seller concession (% GFT, first-last)"),
    ]:
        grp = sub[sub["participant_role"] == role].copy()
        grp["concession_pct"] = sign * (grp["last_offer"] - grp["offer_1"]) / grp["gains_from_trade"]
        one = grp[grp["information_asymmetry"] == "one-sided"]["concession_pct"]
        two = grp[grp["information_asymmetry"] == "two-sided"]["concession_pct"]
        p_mwu = _mwu_pvalue(one, two)
        m1, m2, _, p_ols = _ols_pvalue("concession_pct", grp)
        _row(label, m1, m2, p_mwu, p_ols)


def analyse_who_accepts(df):
    """
    Among rounds that ended in acceptance, who accepted (i.e. whose offer
    was NOT accepted — equivalently, who pressed Accept)?
    own_offer_accepted == 1  → the other player accepted this player's offer
                              → this player did NOT accept, the other did.
    own_offer_accepted == 0  → this player accepted the other's offer.
    We report the share of acceptances initiated by the Buyer.
    One row per negotiation_id in acceptance rounds.
    """
    _header("3. WHO ACCEPTS (share of acceptances by Buyer)")

    trades = df[(df["agreement_dummy"] == 1) & (df["participant_role"] == "Buyer")].copy()
    # Buyer accepted the seller's offer ↔ buyer's own offer was NOT accepted
    trades["buyer_accepted"] = (trades["own_offer_accepted"] == 0).astype(int)

    one = trades[trades["information_asymmetry"] == "one-sided"]["buyer_accepted"]
    two = trades[trades["information_asymmetry"] == "two-sided"]["buyer_accepted"]

    p_mwu = _mwu_pvalue(one, two)
    m1, m2, _, p_ols = _ols_pvalue("buyer_accepted", trades)
    _row("Share buyer accepted (vs seller)", m1, m2, p_mwu, p_ols)

    print(f"\n  (Complement = share seller accepted: "
          f"one={1-m1:.3f}, two={1-m2:.3f})")


def analyse_terminations(df):
    """
    Among rounds terminated by a player (bargaining_outcome == 'Player'),
    report the share terminated by the buyer.
    terminated_by_id_in_group == id_in_group → this player terminated.
    """
    _header("4. WHO TERMINATES (share of player-terminations by Buyer)")

    term = df[
        (df["bargaining_outcome"] == "Player") &
        (df["participant_role"] == "Buyer")
    ].copy()
    term["buyer_terminated"] = (
        term["terminated_by_id_in_group"] == term["id_in_group"]
    ).astype(int)

    one = term[term["information_asymmetry"] == "one-sided"]["buyer_terminated"]
    two = term[term["information_asymmetry"] == "two-sided"]["buyer_terminated"]
    p_mwu = _mwu_pvalue(one, two)
    m1, m2, _, p_ols = _ols_pvalue("buyer_terminated", term)
    _row("Share terminated by buyer", m1, m2, p_mwu, p_ols)

    print(f"\n  (Complement = share terminated by seller: "
          f"one={1-m1:.3f}, two={1-m2:.3f})")

    # Also: termination rate overall (player termination / all rounds)
    _header("4b. OVERALL TERMINATION RATE by player (all rounds)")
    all_rounds = df[df["participant_role"] == "Buyer"].copy()
    all_rounds["player_terminated"] = (all_rounds["bargaining_outcome"] == "Player").astype(int)

    one = all_rounds[all_rounds["information_asymmetry"] == "one-sided"]["player_terminated"]
    two = all_rounds[all_rounds["information_asymmetry"] == "two-sided"]["player_terminated"]
    p_mwu = _mwu_pvalue(one, two)
    m1, m2, _, p_ols = _ols_pvalue("player_terminated", all_rounds)
    _row("Player termination rate", m1, m2, p_mwu, p_ols)


def analyse_payoffs(df):
    """
    Mean payoff by role (Buyer / Seller) in one- vs. two-sided uncertainty,
    and p-value for the one- vs. two-sided difference within each role.
    All rounds included (payoff = 0 for non-trades).
    """
    _header("5. PAYOFFS BY ROLE (one-sided vs. two-sided)")

    df = df[df["gains_from_trade"] > 0].copy()  # Payoffs of 0 in no-gain rounds are not informative for payoff comparison

    for role in ["Buyer", "Seller"]:
        sub = df[df["participant_role"] == role].copy()
        one = sub[sub["information_asymmetry"] == "one-sided"]["payoff"]
        two = sub[sub["information_asymmetry"] == "two-sided"]["payoff"]
        p_mwu = _mwu_pvalue(one, two)
        m1, m2, _, p_ols = _ols_pvalue("payoff", sub)
        _row(f"{role} payoff (EUR)", m1, m2, p_mwu, p_ols)

    # Buyer-seller payoff gap within each information condition
    print()
    buyers = df[df["participant_role"] == "Buyer"].set_index("negotiation_id")["payoff"]
    sellers = df[df["participant_role"] == "Seller"].set_index("negotiation_id")["payoff"]
    gap = (buyers - sellers).rename("payoff_gap").reset_index()
    info = df[df["participant_role"] == "Buyer"][["negotiation_id", "information_asymmetry"]]
    gap = gap.merge(info, on="negotiation_id")

    one = gap[gap["information_asymmetry"] == "one-sided"]["payoff_gap"]
    two = gap[gap["information_asymmetry"] == "two-sided"]["payoff_gap"]
    p_mwu = _mwu_pvalue(one, two)
    m1, m2, _, p_ols = _ols_pvalue("payoff_gap", gap, group_col="negotiation_id")
    _row("Buyer - Seller payoff gap (EUR)", m1, m2, p_mwu, p_ols)


def analyse_surplus_shares(df):
    """
    Share of gains from trade going to each role (split_gains_from_trade),
    in one- vs. two-sided uncertainty.
    Restricted to trades with gains_from_trade > 0 (split is meaningful only then).
    split_gains_from_trade is the buyer's share; seller's share = 1 - split.
    """
    _header("6. SHARE OF GAINS FROM TRADE BY ROLE (one-sided vs. two-sided)")

    trades = df[
        (df["agreement_dummy"] == 1) &
        (df["gains_from_trade"] > 0) &
        (df["participant_role"] == "Buyer")
    ].copy()

    one = trades[trades["information_asymmetry"] == "one-sided"]["split_gains_from_trade"]
    two = trades[trades["information_asymmetry"] == "two-sided"]["split_gains_from_trade"]
    p_mwu = _mwu_pvalue(one, two)
    m1, m2, _, p_ols = _ols_pvalue("split_gains_from_trade", trades)
    _row("Buyer share of GFT", m1, m2, p_mwu, p_ols)

    print(f"\n  (Seller share of GFT: one={1-m1:.3f}, two={1-m2:.3f})")

    # Also report by role using own payoff / gains_from_trade
    print()
    for role in ["Buyer", "Seller"]:
        sub = df[
            (df["participant_role"] == role) &
            (df["agreement_dummy"] == 1) &
            (df["gains_from_trade"] > 0)
        ].copy()
        sub["own_share_gft"] = sub["payoff"] / sub["gains_from_trade"]
        one = sub[sub["information_asymmetry"] == "one-sided"]["own_share_gft"]
        two = sub[sub["information_asymmetry"] == "two-sided"]["own_share_gft"]
        p_mwu = _mwu_pvalue(one, two)
        m1, m2, _, p_ols = _ols_pvalue("own_share_gft", sub)
        _row(f"{role} payoff / GFT (trades only)", m1, m2, p_mwu, p_ols)


def _stars_vs_half(series):
    """Stars for one-sample t-test H0: mean == 0.5."""
    s = series.dropna()
    if len(s) < 2:
        return ""
    _, p = stats.ttest_1samp(s, 0.5)
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def table_buyer_seller_summary_compare_one_two_sided(df):
    """
    LaTeX summary table with three rows:
      1. Who accepts       — buyer/seller share among accepted rounds
      2. Who terminates    — buyer/seller share among player-terminated rounds
      3. Who first offers  — buyer/seller share of being the first mover

    Columns: one-sided (Buyer, Seller) | two-sided (Buyer, Seller)
    Stars indicate significant deviation from 0.5 (one-sample t-test).
    Returns a LaTeX tabular string ready for \\input{} in main.tex.
    """
    def _fmt(val, stars):
        return f"{val:.3f}{stars}"

    # ------------------------------------------------------------------
    # Row 1: Who accepts
    # ------------------------------------------------------------------
    trades = df[(df["agreement_dummy"] == 1) & (df["participant_role"] == "Buyer")].copy()
    trades["buyer_accepted"] = (trades["own_offer_accepted"] == 0).astype(float)

    for info in ["one-sided", "two-sided"]:
        s = trades[trades["information_asymmetry"] == info]["buyer_accepted"]
        if info == "one-sided":
            b_one_acc = (s.mean(), _stars_vs_half(s))
            s_one_acc = (1 - s.mean(), _stars_vs_half(1 - s))
        else:
            b_two_acc = (s.mean(), _stars_vs_half(s))
            s_two_acc = (1 - s.mean(), _stars_vs_half(1 - s))

    # ------------------------------------------------------------------
    # Row 2: Who terminates
    # ------------------------------------------------------------------
    term = df[
        (df["bargaining_outcome"] == "Player") &
        (df["participant_role"] == "Buyer")
    ].copy()
    term["buyer_terminated"] = (
        term["terminated_by_id_in_group"] == term["id_in_group"]
    ).astype(float)

    for info in ["one-sided", "two-sided"]:
        s = term[term["information_asymmetry"] == info]["buyer_terminated"]
        if info == "one-sided":
            b_one_ter = (s.mean(), _stars_vs_half(s))
            s_one_ter = (1 - s.mean(), _stars_vs_half(1 - s))
        else:
            b_two_ter = (s.mean(), _stars_vs_half(s))
            s_two_ter = (1 - s.mean(), _stars_vs_half(1 - s))

    # ------------------------------------------------------------------
    # Row 3: Who makes the first offer
    # ------------------------------------------------------------------
    buyers_t = df[df["participant_role"] == "Buyer"][
        ["negotiation_id", "information_asymmetry", "offer_time_1", "participant_code"]
    ].rename(columns={"offer_time_1": "buyer_offer_time_1"})
    sellers_t = df[df["participant_role"] == "Seller"][
        ["negotiation_id", "offer_time_1"]
    ].rename(columns={"offer_time_1": "seller_offer_time_1"})
    first = buyers_t.merge(sellers_t, on="negotiation_id")
    first = first.dropna(subset=["buyer_offer_time_1", "seller_offer_time_1"])
    first["buyer_first"] = (first["buyer_offer_time_1"] < first["seller_offer_time_1"]).astype(float)

    for info in ["one-sided", "two-sided"]:
        s = first[first["information_asymmetry"] == info]["buyer_first"]
        if info == "one-sided":
            b_one_fo = (s.mean(), _stars_vs_half(s))
            s_one_fo = (1 - s.mean(), _stars_vs_half(1 - s))
        else:
            b_two_fo = (s.mean(), _stars_vs_half(s))
            s_two_fo = (1 - s.mean(), _stars_vs_half(1 - s))

    # ------------------------------------------------------------------
    # Build LaTeX
    # ------------------------------------------------------------------
    rows = [
        ("Who accepts",          b_one_acc, s_one_acc, b_two_acc, s_two_acc),
        ("Who terminates",       b_one_ter, s_one_ter, b_two_ter, s_two_ter),
        ("Who makes first offer", b_one_fo,  s_one_fo,  b_two_fo,  s_two_fo),
    ]

    lines = []
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r" & \multicolumn{2}{c}{One-sided} & \multicolumn{2}{c}{Two-sided} \\"
    )
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    lines.append(r" & Buyer & Seller & Buyer & Seller \\")
    lines.append(r"\midrule")
    for label, b_one, s_one, b_two, s_two in rows:
        lines.append(
            f"{label} & {_fmt(*b_one)} & {_fmt(*s_one)} & {_fmt(*b_two)} & {_fmt(*s_two)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(
        r"\multicolumn{5}{l}{\footnotesize Stars: one-sample $t$-test against 0.5 "
        r"($^{*}p<.10$, $^{**}p<.05$, $^{***}p<.01$)} \\"
    )
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


def run_all(df):
    print("\n" + "*" * 65)
    print("  ONE-SIDED vs. TWO-SIDED UNCERTAINTY: BEHAVIOURAL COMPARISON")
    print("*" * 65)
    print("  p_MWU = Mann-Whitney U (non-parametric)")
    print("  p_OLS = OLS clustered by participant (parametric)")
    analyse_opening_offers(df)
    analyse_concession_rates(df)
    analyse_who_accepts(df)
    analyse_terminations(df)
    analyse_payoffs(df)
    analyse_surplus_shares(df)
    print()


if __name__ == "__main__":
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")
    run_all(df)
    latex = table_buyer_seller_summary_compare_one_two_sided(df)
    out_path = OVERLEAF_TABLES / "buyer_seller_summary.tex"
    out_path.write_text(latex)
