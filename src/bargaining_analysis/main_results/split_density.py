"""
Density plots of split_gains_from_trade by treatment cell.

Documents the strong bunching at the 50-50 equal-split norm across all four
treatment cells (T1–T4).  Only trade observations (bargaining_outcome ==
'acceptance') are used, since the split is only defined when trade occurs.

conda run -n bargaining_analysis python -m src.bargaining_analysis.main_results.split_density
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde
import numpy as np

from src.bargaining_analysis.config import BLD, OVERLEAF_FIGURES, COLOR_SCHEME
from src.bargaining_analysis.helper import set_plot_theme, finalize_plot


TREATMENT_LABELS = {
    "T1": "T1: Symmetric, no costs",
    "T2": "T2: Symmetric, costs",
    "T3": "T3: One-sided, no costs",
    "T4": "T4: One-sided, costs",
}


def plot_split_density(df, figsize=(10, 6)):
    """
    One figure with four KDE density curves — one per treatment cell — of the
    buyer's share of gains from trade, restricted to trade observations.

    A vertical dashed line at 0.5 marks the equal-split norm.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned dataset.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df_trade = df[df["bargaining_outcome"] == "acceptance"].dropna(
        subset=["split_gains_from_trade", "treatment"]
    ).copy()

    set_plot_theme()
    fig, ax = plt.subplots(figsize=figsize)

    x_grid = np.linspace(0, 1, 500)
    treatments = ["T1", "T2", "T3", "T4"]

    for i, treatment in enumerate(treatments):
        values = df_trade.loc[
            df_trade["treatment"] == treatment, "split_gains_from_trade"
        ].values

        if len(values) < 5:
            continue

        kde = gaussian_kde(values, bw_method="scott")
        ax.plot(
            x_grid,
            kde(x_grid),
            color=COLOR_SCHEME[i],
            linewidth=2.2,
            label=TREATMENT_LABELS[treatment],
        )

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.4, alpha=0.7,
               label="Equal split (50--50)")

    ax.set_xlabel("Buyer Share of Gains from Trade")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="black")

    finalize_plot(ax)
    return fig


def plot_split_histogram(df, bin_width=0.025, figsize=(11, 8)):
    """
    2x2 grid of histograms — one panel per treatment cell — showing the
    distribution of the buyer's share of gains from trade for trade
    observations.  Narrow bins (default 0.025) make the spike at the exact
    50-50 split clearly visible as a discrete mass point.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned dataset.
    bin_width : float
        Width of histogram bins (default 0.025 → 40 bins over [0, 1]).

    Returns
    -------
    matplotlib.figure.Figure
    """
    df_trade = df[df["bargaining_outcome"] == "acceptance"].dropna(
        subset=["split_gains_from_trade", "treatment"]
    ).copy()

    bins = np.arange(0, 1 + bin_width, bin_width)
    treatments = ["T1", "T2", "T3", "T4"]

    set_plot_theme()
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=False)
    axes_flat = axes.flatten()

    for i, treatment in enumerate(treatments):
        ax = axes_flat[i]
        values = df_trade.loc[
            df_trade["treatment"] == treatment, "split_gains_from_trade"
        ].values

        ax.hist(
            values,
            bins=bins,
            color=COLOR_SCHEME[i],
            edgecolor="white",
            linewidth=0.4,
            density=True,
        )
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1.4, alpha=0.8)

        n_half = np.sum(np.abs(values - 0.5) < bin_width / 2)
        share_half = n_half / len(values) * 100
        ax.set_title(
            TREATMENT_LABELS[treatment]
            + rf"  $-$  {share_half:.0f}\% at 50--50",
            fontsize=12,
        )
        ax.set_xlabel("Buyer Share of Gains from Trade")
        ax.set_ylabel("Density")

        finalize_plot(ax)

    fig.tight_layout()
    return fig


def plot_split_histogram_pooled(df, bin_width=0.025, figsize=(9, 5)):
    """
    Single histogram of split_gains_from_trade pooled across all treatment
    cells, restricted to trade observations.  Narrow bins make the spike at
    the exact 50-50 split clearly visible.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned dataset.
    bin_width : float
        Width of histogram bins (default 0.025).

    Returns
    -------
    matplotlib.figure.Figure
    """
    df_trade = df[df["bargaining_outcome"] == "acceptance"].dropna(
        subset=["split_gains_from_trade"]
    ).copy()

    values = df_trade["split_gains_from_trade"].values
    bins = np.arange(0, 1 + bin_width, bin_width)

    n_half = np.sum(np.abs(values - 0.5) < bin_width / 2)
    share_half = n_half / len(values) * 100

    set_plot_theme()
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        values,
        bins=bins,
        color=COLOR_SCHEME[0],
        edgecolor="white",
        linewidth=0.4,
        density=True,
    )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.4, alpha=0.8,
               label=rf"Equal split — {share_half:.0f}\% of trade obs.")

    ax.set_xlabel("Buyer Share of Gains from Trade")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="black")

    finalize_plot(ax)
    return fig


if __name__ == "__main__":
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")

    fig = plot_split_density(df=df)
    out = OVERLEAF_FIGURES / "split_density_by_treatment.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Figure saved to: {out}")

    fig2 = plot_split_histogram(df=df)
    out2 = OVERLEAF_FIGURES / "split_histogram_by_treatment.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    print(f"Figure saved to: {out2}")

    fig3 = plot_split_histogram_pooled(df=df)
    out3 = OVERLEAF_FIGURES / "split_histogram_pooled.pdf"
    fig3.savefig(out3, bbox_inches="tight")
    print(f"Figure saved to: {out3}")
