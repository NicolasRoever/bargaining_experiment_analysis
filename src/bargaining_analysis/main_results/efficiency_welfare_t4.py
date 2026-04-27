"""
Efficiency and welfare decomposition for T4.

For each negotiation, computes:
  buyer_payoff    = realized buyer payoff (includes TA costs)
  seller_payoff   = realized seller payoff (includes TA costs)
  joint_surplus   = buyer_payoff + seller_payoff
  cost_of_delay   = cumulated_TA_costs_buyer + cumulated_TA_costs_seller
  foregone_trade  = max(gains_from_trade, 0) * (1 - agreement_dummy)
  available_surplus = gains_from_trade (= buyer valuation in T4, seller cost = 0)

Identity: buyer_payoff + seller_payoff + cost_of_delay + foregone_trade = available_surplus

Produces a waterfall-style stacked bar chart by buyer-valuation region:
  Low    : valuation < 7.72
  Middle : 7.72 <= valuation <= 21.40
  High   : valuation > 21.40

  conda run -n bargaining_analysis python -m src.bargaining_analysis.main_results.efficiency_welfare_t4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.bargaining_analysis.config import BLD, OVERLEAF_FIGURES, OVERLEAF_ROOT, COLOR_SCHEME
from src.bargaining_analysis.helper import set_plot_theme, finalize_plot, inject_values
from scipy import integrate

B_DAGGER = 7.72
B_STAR = 21.40


def _prepare_t4(df: pd.DataFrame) -> pd.DataFrame:
    """Merge T4 buyer and seller rows into one row per negotiation."""
    t4 = df[df["treatment"] == "T4"].copy()

    buyers = t4[t4["participant_role"] == "Buyer"][
        [
            "negotiation_id",
            "payoff",
            "cumulated_TA_costs",
            "valuation",
            "gains_from_trade",
            "agreement_dummy",
            "bargaining_time_full_sec",
            "participant_code",
        ]
    ].copy()

    sellers = t4[t4["participant_role"] == "Seller"][
        ["negotiation_id", "payoff", "cumulated_TA_costs"]
    ].rename(
        columns={
            "payoff": "seller_payoff",
            "cumulated_TA_costs": "seller_ta_costs",
        }
    )

    m = buyers.merge(sellers, on="negotiation_id", how="inner")
    m.rename(
        columns={
            "payoff": "buyer_payoff",
            "cumulated_TA_costs": "buyer_ta_costs",
        },
        inplace=True,
    )

    m["available_surplus"] = m["gains_from_trade"]
    m["cost_of_delay"] = m["buyer_ta_costs"] + m["seller_ta_costs"]
    m["foregone_trade"] = np.maximum(m["gains_from_trade"], 0) * (
        1 - m["agreement_dummy"]
    )

    # Assign valuation region
    m["region"] = pd.cut(
        m["valuation"],
        bins=[-np.inf, B_DAGGER, B_STAR, np.inf],
        labels=["Low\n($b < 7.72$)", "Middle\n($7.72 \\leq b \\leq 21.40$)", "High\n($b > 21.40$)"],
        right=False,
    )

    return m


def _compute_region_averages(m: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-negotiation metrics by region.

    Returns a DataFrame with columns:
      region, available_surplus, buyer_payoff, seller_payoff,
      cost_of_delay, foregone_trade, n_obs
    """
    agg = (
        m.groupby("region", observed=True)
        .agg(
            available_surplus=("available_surplus", "mean"),
            buyer_payoff=("buyer_payoff", "mean"),
            seller_payoff=("seller_payoff", "mean"),
            cost_of_delay=("cost_of_delay", "mean"),
            foregone_trade=("foregone_trade", "mean"),
            n_obs=("negotiation_id", "count"),
        )
        .reset_index()
    )
    return agg


def plot_efficiency_welfare_t4(
    df: pd.DataFrame,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    stacked bar chart decomposing average available surplus
    into buyer payoff, seller payoff, transaction costs, and foregone trade,
    separately for the three buyer-valuation regions in T4.

    The stacked segments satisfy:
      buyer_payoff + seller_payoff + cost_of_delay + foregone_trade
      = available_surplus   (by construction)

    Negative payoffs (e.g. TA costs with no agreement) extend below zero;
    the top of the stack always equals available_surplus.
    """
    m = _prepare_t4(df)
    agg = _compute_region_averages(m)

    set_plot_theme()

    colors = {
        "Buyer Payoff": COLOR_SCHEME[0],       # blue
        "Seller Payoff": COLOR_SCHEME[3],      # green
        "Transaction Costs": COLOR_SCHEME[2],  # teal
        "Foregone Trade": COLOR_SCHEME[1],     # red
    }

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(agg))
    width = 0.55

    regions = agg["region"].tolist()
    buyer_payoffs = agg["buyer_payoff"].values
    seller_payoffs = agg["seller_payoff"].values
    cost_of_delays = agg["cost_of_delay"].values
    foregone_trades = agg["foregone_trade"].values
    available_surpluses = agg["available_surplus"].values

    # Fix y-axis so negative territory is always clearly visible
    y_neg_pad = -2.0
    ax.set_ylim(bottom=y_neg_pad)

    # Shade the negative zone before drawing bars
    ax.axhspan(y_neg_pad, 0, color="#fee0d2", alpha=0.5, zorder=1, label="_nolegend_")
    ax.text(
        len(agg) - 0.5,
        y_neg_pad / 2,
        "Net loss zone",
        ha="right",
        va="center",
        fontsize=8,
        color="#d73027",
        style="italic",
    )

    # Stack segments bottom-up.
    # Segment 1: buyer_payoff (anchors the bottom; may be negative)
    bottoms_bp = np.zeros(len(agg))
    ax.bar(x, buyer_payoffs, width, bottom=bottoms_bp,
           color=colors["Buyer Payoff"], label="Buyer Payoff", zorder=3)

    # Segment 2: seller_payoff stacked on top of buyer_payoff
    bottoms_sp = buyer_payoffs
    ax.bar(x, seller_payoffs, width, bottom=bottoms_sp,
           color=colors["Seller Payoff"], label="Seller Payoff", zorder=3)

    # Segment 3: cost_of_delay stacked on joint payoff
    bottoms_cd = buyer_payoffs + seller_payoffs
    ax.bar(x, cost_of_delays, width, bottom=bottoms_cd,
           color=colors["Transaction Costs"], label="Transaction Costs", zorder=3)

    # Segment 4: foregone_trade stacked on top
    bottoms_ft = buyer_payoffs + seller_payoffs + cost_of_delays
    ax.bar(x, foregone_trades, width, bottom=bottoms_ft,
           color=colors["Foregone Trade"], label="Foregone Trade", zorder=3)

    # Annotate bars that dip below zero with their value
    for i, bp in enumerate(buyer_payoffs):
        if bp < 0:
            ax.text(
                x[i],
                bp - 0.1,
                f"{bp:.2f}",
                ha="center",
                va="top",
                fontsize=8,
                color=colors["Buyer Payoff"],
                fontweight="bold",
            )

    # Mark available surplus with a horizontal tick on each bar
    for i, avs in enumerate(available_surpluses):
        ax.hlines(
            avs,
            x[i] - width / 2,
            x[i] + width / 2,
            colors="black",
            linewidths=2,
            linestyles="--",
            zorder=5,
        )

    # Region sample sizes as annotation
    for i, row in agg.iterrows():
        ax.text(
            x[i],
            available_surpluses[i] + 1,
            f"$N={row['n_obs']}$",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Available surplus legend entry
    avs_handle = plt.Line2D(
        [], [],
        color="black",
        linewidth=2,
        linestyle="--",
        label="Available Surplus",
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + [avs_handle],
        labels + ["Available Surplus"],
        loc="upper left",
        framealpha=0.9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Average Payoff / Surplus (EUR)")

    # Prominent zero line
    ax.axhline(0, color="black", linewidth=1.8, zorder=4)

    # Extra top margin so N= annotations are not clipped
    y_top = max(available_surpluses) + 4
    ax.set_ylim(top=y_top)

    finalize_plot(ax=ax)
    plt.close()
    return fig


def delta(b, b_star=21.40, s=0.0, r=0.01, c=0.05):
    """Equilibrium delay for intermediate types predicted by the model."""
    return (1 / r) * np.log((r * (b_star - s) + 2 * c) / (r * (b - s) + 2 * c))


def joint_surplus(b, b_dagger=7.72, b_star=21.40, s=0.0, r=0.01, c=0.05):
    """Model-predicted joint surplus for buyer type b."""
    if b <= b_dagger:
        return 0.0
    elif b < b_star:
        d = delta(b, b_star=b_star, s=s, r=r, c=c)
        return np.exp(-r * d) * b - 2 * (c / r) * (1 - np.exp(-r * d))
    else:
        return b


def predicted_efficiency(b_lo, b_hi, b_dagger=7.72, b_star=21.40, s=0.0, r=0.01, c=0.05):
    """Model Predicted efficiency = E[joint surplus] / E[b] over b ~ U[b_lo, b_hi]."""
    mean_surplus = integrate.quad(
        joint_surplus, b_lo, b_hi, args=(b_dagger, b_star, s, r, c)
    )[0] / (b_hi - b_lo)
    mean_available = (b_lo + b_hi) / 2
    return mean_surplus / mean_available


def calculate_efficiency_delay_values(
    df: pd.DataFrame,
    b_dagger: float = B_DAGGER,
    b_star: float = B_STAR,
    b_max: float = 30.0,
) -> dict:
    """
    Returns a dictionary with predicted and empirical efficiency by valuation region,
    and predicted and actual average delay in the intermediate region.

    Empirical efficiency per region = sum(joint surplus) / sum(available surplus).
    Predicted efficiency uses the model's closed-form predictions.
    Delay in the intermediate region: empirical = mean bargaining_time_full_sec;
    predicted = mean of delta(b) evaluated at observed intermediate valuations.
    """
    m = _prepare_t4(df)
    m["joint_surplus_empirical"] = m["buyer_payoff"] + m["seller_payoff"]

    mask_low = m["valuation"] < b_dagger
    mask_mid = (m["valuation"] >= b_dagger) & (m["valuation"] < b_star)
    mask_high = m["valuation"] >= b_star

    def _emp_eff(mask):
        sub = m[mask]
        return sub["joint_surplus_empirical"].sum() / sub["available_surplus"].sum()

    emp_eff_low = _emp_eff(mask_low)
    emp_eff_mid = _emp_eff(mask_mid)
    emp_eff_high = _emp_eff(mask_high)
    emp_eff_overall = _emp_eff(pd.Series(True, index=m.index))

    pred_eff_low = predicted_efficiency(0, b_dagger)
    pred_eff_mid = predicted_efficiency(b_dagger, b_star)
    pred_eff_high = predicted_efficiency(b_star, b_max)
    pred_eff_overall = predicted_efficiency(0, b_max)

    mid = m[mask_mid].copy()
    emp_delay_mid = mid["bargaining_time_full_sec"].mean()
    pred_delay_mid = mid["valuation"].apply(lambda b: delta(b, b_star=b_star)).mean()

    return {
        "pred_eff_low_t4": f"{100 * pred_eff_low:.0f}",
        "pred_eff_mid_t4": f"{100 * pred_eff_mid:.0f}",
        "pred_eff_high_t4": f"{100 * pred_eff_high:.0f}",
        "pred_eff_overall_t4": f"{100 * pred_eff_overall:.0f}",
        "emp_eff_low_t4": f"{100 * emp_eff_low:.0f}",
        "emp_eff_mid_t4": f"{100 * emp_eff_mid:.0f}",
        "emp_eff_high_t4": f"{100 * emp_eff_high:.0f}",
        "emp_eff_overall_t4": f"{100 * emp_eff_overall:.0f}",
        "emp_delay_mid_t4": f"{emp_delay_mid:.2f}",
        "pred_delay_mid_t4": f"{pred_delay_mid:.2f}",
    }


if __name__ == "__main__":
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")
    fig = plot_efficiency_welfare_t4(df=df)
    fig.savefig(OVERLEAF_FIGURES / "efficiency_welfare_t4.pdf", bbox_inches="tight")

    values = calculate_efficiency_delay_values(df=df)
    for key, val in values.items():
        print(f"  {key}: {val}")
    inject_values(OVERLEAF_ROOT / "main.tex", **values)
