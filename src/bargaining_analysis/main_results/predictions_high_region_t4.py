import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
import statsmodels.formula.api as smf
import statsmodels.api as sm
from src.bargaining_analysis.config import BLD, OVERLEAF_FIGURES, COLOR_SCHEME

from src.bargaining_analysis.helper import set_plot_theme, finalize_plot



# ─── Prepare negotiation-level data ──────────────────────────────────────────

def _prepare(df, CUTOFF=21.40):
    """
    Return buyer-level rows for T4 merged with matched seller info.
    Each row = one negotiation, identified by buyer valuation.
    """
    t4      = df[df["treatment"] == "T4"].copy()
    buyers  = t4[t4["participant_role"] == "Buyer"].copy()
    sellers = t4[t4["participant_role"] == "Seller"][
        ["negotiation_id", "offer_1", "offer_time_1", "number_of_offers",
         "first_offer", "participant_code", "split_gains_from_trade", "payoff"]
    ].rename(columns={
        "offer_1":           "offer_1_seller",
        "offer_time_1":      "offer_time_1_seller",
        "number_of_offers":  "number_of_offers_seller",
        "first_offer":       "first_offer_seller",
        "participant_code":  "seller_code",
        "split_gains_from_trade": "seller_split_gains_from_trade",
        "payoff":            "seller_payoff"

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


def inject_values_t4_high_valuation_region(df, CUTOFF=21.40, PRED_PRICE=10.70):

    merged, high, other = _prepare(df, CUTOFF)

    # Trade Rate
    n          = len(high)
    n_trade    = high["agreement_dummy"].sum()
    rate_trade = n_trade / n

    # Trade Price
    trades = high[high["agreement_dummy"] == 1].dropna(subset=["deal_price"])
    n      = len(trades)
    mean_p   = trades["deal_price"].mean()
    sd_p     = trades["deal_price"].std()
    m, se, t, p = _clustered_mean_test(
        trades["deal_price"], trades["participant_code"], h0_mean=PRED_PRICE
    )

    #Mean Shares
    trades["buyer_share"]  = trades["split_gains_from_trade"]
    trades["seller_share"] = 1- trades["buyer_share"]

    mean_buyer_share  = trades["buyer_share"].mean()
    mean_seller_share = trades["seller_share"].mean()
    m, se, t, p_equality_share = _clustered_mean_test(
        trades["buyer_share"], trades["participant_code"], h0_mean=0.5
    )

    #Who moves first?
    all_n   = len(high)
    trades  = high[high["agreement_dummy"] == 1]
    n_t     = len(trades)
    sf = (high["first_offer_seller"] == 1).sum()
    bf = (high["first_offer_buyer"]  == 1).sum()
    share_sf = sf / all_n
    share_bf = bf / all_n

    #First Seller Offer Price
    mean_fo_seller   = high["offer_1_seller"].mean()
    sd_fo_seller     = high["offer_1_seller"].std()

    #Mean Number of offers
    mean_n_offers = high["total_offers"].mean()

    return dict(
        trade_rate_high_region_t4 = f"{rate_trade * 100:.0f}",
        trade_price_high_region_t4 = f"{mean_p:.0f}", 
        sd_trade_price_high_region_t4 = f"{sd_p:.0f}",
        p_value_clustered_t_test_trade_price_high_region_t4 = f"{p:.3f}",
        seller_first_offer_rate_high_region_t4 = f"{share_sf * 100:.0f}",
        buyer_first_offer_rate_high_region_t4 = f"{share_bf * 100:.0f}",
        first_seller_offer_price_high_region_t4 = f"{mean_fo_seller:.0f}",
        sd_first_seller_offer_price_high_region_t4 = f"{sd_fo_seller:.0f}",
        mean_number_of_offers_high_region_t4 = f"{mean_n_offers:.2f}",
        mean_buyer_share_high_region_t4 = f"{mean_buyer_share * 100:.0f}",
        mean_seller_share_high_region_t4 = f"{mean_seller_share * 100:.0f}",
        p_value_clustered_t_test_equality_shares_high_region_t4 = f"{p_equality_share:.3f}",
    )


def inject_values_surplus_split_high_t4(df):
    """
    Compute inject-ready statistics for the 'Sellers in High Region' subsection.

    Keys returned (all strings, ready for \\roever{}{}):
      average_buyer_share_high_t4               – mean buyer % of surplus
      p_value_buyer_share_equal_split_high_t4   – H0: buyer share = 0.5
      corr_seller_first_offer_val_high_t4       – Spearman ρ (seller offer, buyer val)
      p_corr_seller_first_offer_val_high_t4     – p-value for ρ
      slope_price_val_high_t4                   – OLS slope deal_price ~ valuation
      p_slope_price_val_high_t4                 – p-value for that slope
      mean_seller_payoff_high_t4                – mean seller payoff across trades
      slope_seller_payoff_val_high_t4           – OLS slope seller_payoff ~ valuation
      p_slope_seller_payoff_val_high_t4         – p-value for seller payoff slope
    """
    _, high, _ = _prepare(df)
    trades = high[high["agreement_dummy"] == 1].copy()
    trades["buyer_share"] = trades["split_gains_from_trade"]
    trades["seller_share"] = 1 - trades["buyer_share"]

    # Buyer surplus share and test vs equal split
    avg_buyer_share = trades["buyer_share"].mean()
    _, _, _, p_share = _clustered_mean_test(
        trades["buyer_share"], trades["participant_code"], h0_mean=0.5
    )

    # Spearman correlation and OLS slope: seller's opening offer vs buyer valuation
    # Use all high-region negotiations where seller made a first offer (not just trades)
    # to avoid conditioning on the outcome.
    with_offer = high.dropna(subset=["offer_1_seller"])
    rho, p_rho = stats.spearmanr(with_offer["valuation"], with_offer["offer_1_seller"])
    mod_fo = smf.ols("offer_1_seller ~ valuation", data=with_offer).fit(
        cov_type="cluster", cov_kwds={"groups": with_offer["seller_code"]}
    )
    slope_seller_fo   = mod_fo.params["valuation"]
    p_slope_seller_fo = mod_fo.pvalues["valuation"]

    # Slope of deal price on valuation (H0: slope = 0)
    mod_price = smf.ols("deal_price ~ valuation", data=trades).fit(
        cov_type="cluster", cov_kwds={"groups": trades["participant_code"]}
    )
    slope_price   = mod_price.params["valuation"]
    p_slope_price = mod_price.pvalues["valuation"]

    # Seller payoff: mean and slope on valuation
    mean_seller_payoff = trades["seller_payoff"].mean()
    mod_seller = smf.ols("seller_payoff ~ valuation", data=trades).fit(
        cov_type="cluster", cov_kwds={"groups": trades["participant_code"]}
    )
    slope_seller_payoff   = mod_seller.params["valuation"]
    p_slope_seller_payoff = mod_seller.pvalues["valuation"]
    t_slope_seller_payoff_lt_05 = (slope_seller_payoff - 0.5) / mod_seller.bse["valuation"]
    p_slope_seller_payoff_lt_05 = stats.t.cdf(t_slope_seller_payoff_lt_05, df=mod_seller.df_resid)

    return dict(
        average_buyer_share_high_t4=f"{avg_buyer_share * 100:.0f}",
        p_value_buyer_share_equal_split_high_t4=f"{p_share:.3f}",
        corr_seller_first_offer_val_high_t4=f"{rho:.2f}",
        p_corr_seller_first_offer_val_high_t4=f"{p_rho:.2f}",
        slope_seller_first_offer_val_high_t4=f"{slope_seller_fo:.3f}",
        p_slope_seller_first_offer_val_high_t4=f"{p_slope_seller_fo:.3f}",
        slope_price_val_high_t4=f"{slope_price:.2f}",
        p_slope_price_val_high_t4=f"{p_slope_price:.3f}",
        mean_seller_payoff_high_t4=f"{mean_seller_payoff:.2f}",
        slope_seller_payoff_val_high_t4=f"{slope_seller_payoff:.3f}",
        p_slope_seller_payoff_val_high_t4=f"{p_slope_seller_payoff:.3f}",
        p_value_slope_seller_payoff_lt_05=f"{p_slope_seller_payoff_lt_05:.3f}",
    )



def plot_offer_convergence_t4_high_region(df, figsize=(11, 5), PRED_PRICE=10.70, CUTOFF=21.40):
    """Box plot of seller opening offer, deal price, and buyer opening offer for
    successful trades in the high-valuation region.

    Shows that despite the initial spread between opening positions, the deal
    price converges near the equilibrium prediction of 10.70.
    """

    merged, high, other = _prepare(df, CUTOFF)
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