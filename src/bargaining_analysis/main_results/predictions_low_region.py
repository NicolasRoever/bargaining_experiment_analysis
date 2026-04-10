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

def _prepare(df, CUTOFF= 7.72, BUYER_ID=1, SELLER_ID=2):
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


def compare_termination_rates_low_rest(df):

    merged, low, high = _prepare(df)
 
    # Proportions
    n_low  = len(low);  term_low  = low["player_termination"].sum()
    n_high = len(high); term_high = high["player_termination"].sum()
    rate_low  = term_low  / n_low
    rate_high = term_high / n_high
    term = low[low["bargaining_outcome"] == "Player"].copy()
    n_total       = len(term)
    n_buyer_init  = (term["terminated_by_id_in_group"] == 2).sum()

    results = {
        "n_low_buyercost": n_low,
        "fraction_low_buyercost": f"{rate_low*100:.0f}",
        "n_high_buyercost": n_high,
        "fraction_high_buyercost": f"{rate_high*100:.0f}",
        "term_low_buyercost": term_low,
        "term_high_buyercost": term_high,
        "fraction_buyer_terminations_buyercost": f"{n_buyer_init / n_total * 100:.0f}"
    }

    return results

def payoff_for_trades_in_low_region(df):
    merged, high, low = _prepare(df)
    trades = low[low["agreement_dummy"] == 1].copy()

    mean_payoff = trades["payoff"].mean()

    return {"mean_payoff_trades_lowregion_t4": mean_payoff}
    

def timing_data_t4(df, CUTOFF = 7.72):
    merged, low, high = _prepare(df)

    return {
        "bargaining_time_low_t4": low["bargaining_time_full_sec"].mean(),
        "bargaining_time_high_t4": high["bargaining_time_full_sec"].mean(),
    }





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
            & (df["treatment"] == "T4")].copy()
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