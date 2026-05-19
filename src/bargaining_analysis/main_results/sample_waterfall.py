"""Waterfall plot showing how the final analysis sample is constructed."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.bargaining_analysis.config import BLD, OVERLEAF_FIGURES
from src.bargaining_analysis.clean_data.functions_clean_data import filter_out_mistake_rows
from src.bargaining_analysis.helper import set_plot_theme, finalize_plot


_COLORS = {
    "start": "#3c5488",
    "loss": "#e64b35",
    "final": "#00a087",
}


def _count_negs(df: pd.DataFrame) -> int:
    return int(df["negotiation_id"].nunique())


def _compute_waterfall_steps(df: pd.DataFrame) -> list[tuple[str, int | None]]:
    """Derive waterfall deltas from merged_data_full.csv (before exclusions)."""
    n_total = _count_negs(df)

    # Pre-registered: filter mistakes then remove round <= 4
    df_no_mistakes = filter_out_mistake_rows(df)
    df_preregistered = df_no_mistakes[df_no_mistakes["round"] > 4]
    n_after_preregistered = _count_negs(df_preregistered)
    excl_preregistered = n_total - n_after_preregistered

    # Attrition: dE5arGFL (rounds 27-30) and lbnJrtKO (rounds 11-30)
    de5_mask = (
        (df_preregistered["session_id"] == "8jp2clvt")
        & (df_preregistered["participant_label"] == "dE5arGFL")
        & (df_preregistered["round"] > 29)
    )
    lbn_mask = (
        (df_preregistered["session_id"] == "8jp2clvt")
        & (df_preregistered["participant_label"] == "lbnJrtKO")
        & (df_preregistered["round"] > 13)
    )
    attrition_ids = set(df_preregistered.loc[de5_mask, "negotiation_id"].unique()) | set(
        df_preregistered.loc[lbn_mask, "negotiation_id"].unique()
    )
    df_no_attrition = df_preregistered[~df_preregistered["negotiation_id"].isin(attrition_ids)]
    n_after_attrition = _count_negs(df_no_attrition)
    excl_attrition = n_after_preregistered - n_after_attrition

    # Data errors: negotiations with both acceptance and termination recorded
    both_mask = (
        df_no_attrition["acceptance_time_raw"].notna()
        & df_no_attrition["termination_time_raw"].notna()
    )
    data_error_ids = df_no_attrition.loc[both_mask, "negotiation_id"].unique()
    df_final = df_no_attrition[~df_no_attrition["negotiation_id"].isin(data_error_ids)]
    n_final = _count_negs(df_final)
    excl_data_errors = n_after_attrition - n_final

    return [
        (f"All enrolled\n({n_total // 30} pairs × 30 rounds)", n_total),
        ("Pre-registered exclusions\n(mistakes + round 4 warm-up)", -excl_preregistered),
        ("Attrition\n(player dropouts)", -excl_attrition),
        ("Data errors\n(both acc. + termination recorded)", -excl_data_errors),
        ("Final\nanalysis sample", None),
    ]


def _build_waterfall_data(steps: list) -> list[dict]:
    """Return a list of dicts with bar drawing coordinates."""
    running = 0
    bars = []
    for label, delta in steps:
        if delta is None:
            bars.append(dict(label=label, bottom=0, height=running, kind="final"))
        elif delta > 0:
            bars.append(dict(label=label, bottom=0, height=delta, kind="start"))
            running += delta
        else:
            bars.append(dict(label=label, bottom=running + delta, height=-delta, kind="loss"))
            running += delta
    return bars


def plot_sample_waterfall(figsize=(10, 6)):
    df = pd.read_csv(BLD / "data" / "merged_data_full.csv")

    set_plot_theme()
    steps = _compute_waterfall_steps(df)
    bars = _build_waterfall_data(steps)
    x = np.arange(len(bars))
    labels = [b["label"] for b in bars]

    fig, ax = plt.subplots(figsize=figsize)

    for i, bar in enumerate(bars):
        color = _COLORS[bar["kind"]]
        ax.bar(
            x[i],
            bar["height"],
            bottom=bar["bottom"],
            color=color,
            width=0.55,
            edgecolor="white",
            linewidth=0.8,
        )
        top = bar["bottom"] + bar["height"]
        ax.text(x[i], top + 30, f"{int(top):,}", ha="center", va="bottom", fontsize=9)
        if bar["kind"] == "loss":
            mid = bar["bottom"] + bar["height"] / 2
            ax.text(
                x[i], mid, f"−{int(bar['height']):,}",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold",
            )

    running = 0
    for i, bar in enumerate(bars):
        if bar["kind"] == "start":
            running = bar["height"]
        elif bar["kind"] == "loss":
            prev_top = running
            ax.plot(
                [x[i - 1] + 0.28, x[i] - 0.28], [prev_top, prev_top],
                color="grey", linestyle="--", linewidth=0.8,
            )
            running -= bar["height"]
        elif bar["kind"] == "final":
            prev_top = running
            ax.plot(
                [x[i - 1] + 0.28, x[i] - 0.28], [prev_top, prev_top],
                color="grey", linestyle="--", linewidth=0.8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Number of negotiations")
    ax.set_ylim(0, 6400)

    legend_handles = [
        mpatches.Patch(color=_COLORS["start"], label="Total enrolled"),
        mpatches.Patch(color=_COLORS["loss"], label="Excluded observations"),
        mpatches.Patch(color=_COLORS["final"], label="Final sample"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    finalize_plot(ax=ax)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = plot_sample_waterfall()
    out_path = OVERLEAF_FIGURES / "sample_waterfall.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved to {out_path}")
