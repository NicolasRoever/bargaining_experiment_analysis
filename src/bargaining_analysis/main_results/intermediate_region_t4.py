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

B_DAGGER   = 7.72
B_STAR     = 21.40
VAL_BINS   = [7.72, 11, 14, 17, 21.4]
VAL_LABELS = ["(7.72,11]", "(11,14]", "(14,17]", "(17,21.4)"]

# ─── Prepare data ─────────────────────────────────────────────────────────────

def predicted_delay(b, b_star=21.40, s=0, c=0.05, r=0.01):
    return (1 / r) * np.log((r * (b_star - s) + 2 * c) / (r * (b - s) + 2 * c))

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


def inject_values_t4_middle_region(df): 
    mid, trades = _prepare(df)


    #First Offers
    sf = (mid["first_offer_seller"] == 1).sum()
    bf = (mid["first_offer_buyer"]  == 1).sum()
    n_total = len(mid)
    rate_sf = sf / n_total
    rate_bf = bf / n_total

    #Correlation Offer Time Valuation
    _corr_df = mid[["valuation", "offer_time_1"]].dropna()
    rho, p_rho = stats.spearmanr(_corr_df["valuation"], _corr_df["offer_time_1"])

    #Share of surplus
    d = trades.copy()
    d["total_surplus"] = d["gains_from_trade"]
    # Drop cases where total surplus is zero or negative (division undefined / misleading)
    d = d[d["total_surplus"] > 0].copy()
    d["buyer_share"] = d["split_gains_from_trade"] 
    n = len(d)
    average_buyer_share = d["buyer_share"].mean()
    average_seller_share = 1- average_buyer_share
    # t-test: buyer share = 0.5
    m, se, t, p_share_different_05_mid_t4 = _clustered_mean_test(d["buyer_share"], d["participant_code"], h0_mean=0.5)

    # First seller offer
    seller_first = trades[trades["first_offer_seller"] == 1]
    model = smf.ols(" offer_1_seller ~ valuation", data=seller_first).fit(cov_type="cluster", cov_kwds={"groups": seller_first["participant_code"]})
    slope_seller1_valuation_mid_t4 = model.params["valuation"]
    pval_valuation_seller = model.pvalues["valuation"]

    #Price Bargaining Mechanism
    sl_price, se_price, p_price, p_price_h0 = _ols_row(
        "OLS deal_price ~ valuation:",
        "deal_price", trades, "participant_code", h0_slope=0.5
    )



    return {
        "rate_seller_first_offer": f"{rate_sf*100:.0f}",
        "rate_buyer_first_offer":  f"{rate_bf*100:.0f}",
        "correlation_offer_time_valuation_mid_t4": f"{rho:.2f})", 
        "p_value_rho_mid_t4": f"{p_rho:.2f}", 
        "average_seller_share_mid_t4": f"{average_seller_share*100:.0f}",
        "p_value_mid_t4_share": f"{p_share_different_05_mid_t4:.2f}",
        "slope_price_valuation_mid_t4": f"{sl_price:.2f}",
        "slope_p_value_different_05": f"{p_price_h0:.2f}",
        "slope_seller1_valuation_mid_t4": f"{slope_seller1_valuation_mid_t4:.2f}",
        "p_value_valuation_seller_first_offer_mid_t4": f"{pval_valuation_seller:.2f}"


    }


def table_deal_price_by_val_bin_t4(df):
    """Return only the tabular body (rows) as a LaTeX fragment for threeparttable."""
    _, trades = _prepare(df)

    rows = []
    for lbl, grp in trades.groupby("val_bin", observed=True):
        n = len(grp)
        mean_price = grp["deal_price"].mean()
        mean_half_val = grp["valuation"].mean() / 2
        diff = mean_price - mean_half_val
        rows.append((str(lbl), n, mean_price, mean_half_val, diff))

    lines = []

    lines.append(
        r"\textbf{Buyer Valuation} & \textbf{N} & \textbf{Mean Deal Price} "
        r"& \textbf{Mean b/2} & \textbf{Diff} \\"
    )
    lines.append(r"\hline")

    # Data rows
    for lbl, n, mp, half, diff in rows:
        sign = "+" if diff >= 0 else ""
        lines.append(
            f"${lbl}$ & {n} & {mp:.2f} & {half:.2f} & {sign}{diff:.2f} \\\\"
        )
    lines.append(r"\hline")

    return "\n".join(lines)




