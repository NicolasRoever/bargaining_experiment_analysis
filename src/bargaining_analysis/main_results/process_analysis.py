"""
Run with: 
conda run -n bargaining_analysis python -m  src.bargaining_analysis.main_results.process_analysis

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from src.bargaining_analysis.config import BLD, OVERLEAF_FIGURES, OVERLEAF_ROOT, COLOR_SCHEME, B_DAGGER, B_STAR
from src.bargaining_analysis.helper import set_plot_theme, finalize_plot, inject_values

N_BINS = 15
MAX_OFFERS = 135

TREATMENT_LABELS = {"T1": "T1 (sym., no cost)", "T2": "T2 (sym., cost)",
                    "T3": "T3 (asym., no cost)", "T4": "T4 (asym., cost)"}


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def _melt_offers(df):
    """Reshape wide offer columns to one row per offer event."""
    id_cols = [
        "negotiation_id", "participant_code", "participant_role",
        "treatment", "information_asymmetry", "TA_costs", "number_of_offers",
    ]
    offer_cols = [f"offer_{i}" for i in range(1, MAX_OFFERS + 1) if f"offer_{i}" in df.columns]
    time_cols  = [f"offer_time_{i}" for i in range(1, MAX_OFFERS + 1) if f"offer_time_{i}" in df.columns]
    n = len(offer_cols)

    val_df  = df[id_cols + offer_cols].melt(
        id_vars=id_cols, value_vars=offer_cols,
        var_name="offer_col", value_name="offer_value",
    )
    val_df["offer_number"] = val_df["offer_col"].str.extract(r"(\d+)$").astype(int)

    time_df = df[["negotiation_id", "participant_code"] + time_cols].melt(
        id_vars=["negotiation_id", "participant_code"], value_vars=time_cols[:n],
        var_name="time_col", value_name="offer_time",
    )
    time_df["offer_number"] = time_df["time_col"].str.extract(r"(\d+)$").astype(int)

    long = val_df.merge(
        time_df[["negotiation_id", "participant_code", "offer_number", "offer_time"]],
        on=["negotiation_id", "participant_code", "offer_number"],
    )
    long = long.dropna(subset=["offer_value", "offer_time"]).drop(columns=["offer_col"])
    return long.reset_index(drop=True)


def clean_process_data(df):
    """
    Filter raw wide-format data and return offer-level long format.
    Applies time-analysis exclusion and keeps only multi-offer negotiations.
    """
    df = df[df["exclude_for_time_analysis"] == 0].copy()

    total_offers = (
        df.groupby("negotiation_id")["number_of_offers"].sum().rename("total_offers")
    )
    multi_offer_ids = total_offers[total_offers >= 2].index
    df = df[df["negotiation_id"].isin(multi_offer_ids)]

    return _melt_offers(df)


# ---------------------------------------------------------------------------
# Figure 1 – Convergence gap
# ---------------------------------------------------------------------------

def _normalize_and_bin(long, n_bins=N_BINS):
    """Add t_norm and bin columns, normalizing time per participant-round."""
    grp = long.groupby(["negotiation_id", "participant_code"])["offer_time"]
    long = long.copy()
    long["t_min"] = grp.transform("min")
    long["t_max"] = grp.transform("max")
    span = long["t_max"] - long["t_min"]
    long["t_norm"] = np.where(span > 0, (long["offer_time"] - long["t_min"]) / span, 0.0)
    long["bin"] = (long["t_norm"] * n_bins).astype(int).clip(upper=n_bins - 1)
    return long


def plot_convergence_gap(df, n_bins=N_BINS, figsize=(10, 6)):
    """
    Plot convergence gap (mean seller offer – mean buyer offer) across
    normalized-time bins, one line per treatment.
    """
    long = clean_process_data(df)
    long = _normalize_and_bin(long, n_bins)

    bin_means = (
        long.groupby(["treatment", "participant_role", "bin"])["offer_value"]
        .mean()
        .reset_index()
    )

    buyers  = bin_means[bin_means["participant_role"] == "Buyer"].rename(columns={"offer_value": "buyer_mean"})
    sellers = bin_means[bin_means["participant_role"] == "Seller"].rename(columns={"offer_value": "seller_mean"})
    merged  = buyers.merge(sellers, on=["treatment", "bin"])
    merged["gap"] = merged["seller_mean"] - merged["buyer_mean"]

    set_plot_theme()
    fig, ax = plt.subplots(figsize=figsize)

    treatments = sorted(merged["treatment"].unique())
    for i, t in enumerate(treatments):
        sub = merged[merged["treatment"] == t].sort_values("bin")
        ax.plot(sub["bin"], sub["gap"], color=COLOR_SCHEME[i % len(COLOR_SCHEME)],
                marker="o", label=TREATMENT_LABELS.get(t, t))

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Normalized offer time (bin, 0 = first offer, 14 = last)")
    ax.set_ylabel("Convergence gap (mean seller $-$ mean buyer offer)")
    ax.set_xticks(range(n_bins))
    ax.legend(title="Treatment")

    finalize_plot(ax=ax)
    plt.close()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 – Midpoint benchmark (BuyerCost treatments: T3, T4)
# ---------------------------------------------------------------------------

def _build_midpoint_data(long):
    """
    For each offer event in a negotiation, compute:
      - midpoint = (own last offer + opponent last offer) / 2
    Exclude events where the opponent has not yet submitted any offer.
    Normalize time at the match level (shared within-round clock).
    Returns a DataFrame with columns:
      negotiation_id, treatment, participant_role, bin, offer_value, midpoint
    """
    records = []

    for (neg_id, treatment), neg_df in long.groupby(["negotiation_id", "treatment"]):
        buyer_events  = neg_df[neg_df["participant_role"] == "Buyer"].sort_values("offer_time")
        seller_events = neg_df[neg_df["participant_role"] == "Seller"].sort_values("offer_time")

        # Build unified timeline sorted by time
        b = buyer_events[["offer_time", "offer_value"]].copy()
        b["role"] = "Buyer"
        s = seller_events[["offer_time", "offer_value"]].copy()
        s["role"] = "Seller"
        timeline = pd.concat([b, s]).sort_values("offer_time").reset_index(drop=True)

        if timeline.empty:
            continue

        t_min = timeline["offer_time"].min()
        t_max = timeline["offer_time"].max()
        span  = t_max - t_min

        last_buyer  = np.nan
        last_seller = np.nan

        for _, row in timeline.iterrows():
            role = row["role"]
            val  = row["offer_value"]
            t    = row["offer_time"]

            opponent_last = last_seller if role == "Buyer" else last_buyer
            own_last      = last_buyer  if role == "Buyer" else last_seller

            # Only record if opponent has already offered (midpoint defined)
            if not np.isnan(opponent_last):
                midpoint = (val + opponent_last) / 2.0
                t_norm   = (t - t_min) / span if span > 0 else 0.0
                bin_idx  = min(int(t_norm * N_BINS), N_BINS - 1)
                records.append({
                    "negotiation_id":   neg_id,
                    "treatment":        treatment,
                    "participant_role": role,
                    "bin":              bin_idx,
                    "offer_value":      val,
                    "midpoint":         midpoint,
                })

            # Update running last offer after processing
            if role == "Buyer":
                last_buyer = val
            else:
                last_seller = val

    return pd.DataFrame(records)


def plot_midpoint_benchmark(df, n_bins=N_BINS, figsize=(13, 5)):
    """
    For BuyerCost treatments (T3, T4): plot mean current offer and mean midpoint
    benchmark across normalized bargaining-time bins, separately for Buyer and Seller.
    """
    long = clean_process_data(df)
    long = long[long["treatment"] == "T4"].copy()

    midpoint_df = _build_midpoint_data(long)

    bin_stats = (
        midpoint_df.groupby(["participant_role", "bin"])[["offer_value", "midpoint"]]
        .mean()
        .reset_index()
    )

    set_plot_theme()
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    linestyles = {"offer_value": "-", "midpoint": "--"}
    series_labels = {"offer_value": "Mean offer", "midpoint": "Midpoint benchmark"}

    for ax_idx, role in enumerate(["Buyer", "Seller"]):
        ax  = axes[ax_idx]
        sub = bin_stats[bin_stats["participant_role"] == role].sort_values("bin")

        for s_idx, (series, ls) in enumerate(linestyles.items()):
            ax.plot(sub["bin"], sub[series],
                    color=COLOR_SCHEME[s_idx % len(COLOR_SCHEME)], linestyle=ls, marker="o",
                    label=series_labels[series])

        ax.set_title(role)
        ax.set_xlabel("Normalized bargaining time (bin)")
        ax.set_ylabel("Mean offer value (EUR)")
        ax.set_xticks(range(n_bins))
        ax.legend(fontsize=9)
        finalize_plot(ax=ax)

    plt.close()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 – Offer distance to predicted deal price (T4, High vs. Intermediate)
# ---------------------------------------------------------------------------

def _build_standing_offer_data(long, valuation_lookup):
    """
    For each offer event in a negotiation (once both roles have offered),
    record the current standing buyer and seller offers alongside the
    model-predicted deal price for that negotiation.

    valuation_lookup: dict {negotiation_id -> buyer_valuation}
    """
    records = []

    for neg_id, neg_df in long.groupby("negotiation_id"):
        buyer_val = valuation_lookup.get(neg_id)
        if buyer_val is None:
            continue

        if buyer_val > B_STAR:
            region = "High"
        elif buyer_val > B_DAGGER:
            region = "Intermediate"
        else:
            continue  # low-value region excluded

        predicted_price = buyer_val / 2

        b = neg_df[neg_df["participant_role"] == "Buyer"][["offer_time", "offer_value"]].copy()
        b["role"] = "Buyer"
        s = neg_df[neg_df["participant_role"] == "Seller"][["offer_time", "offer_value"]].copy()
        s["role"] = "Seller"
        timeline = pd.concat([b, s]).sort_values("offer_time").reset_index(drop=True)

        if timeline.empty:
            continue

        t_min = timeline["offer_time"].min()
        t_max = timeline["offer_time"].max()
        span  = t_max - t_min

        last_buyer  = np.nan
        last_seller = np.nan

        for _, row in timeline.iterrows():
            role = row["role"]
            val  = row["offer_value"]
            t    = row["offer_time"]

            if role == "Buyer":
                last_buyer = val
            else:
                last_seller = val

            if not (np.isnan(last_buyer) or np.isnan(last_seller)):
                t_norm  = (t - t_min) / span if span > 0 else 0.0
                bin_idx = min(int(t_norm * N_BINS), N_BINS - 1)
                records.append({
                    "negotiation_id":       neg_id,
                    "region":               region,
                    "bin":                  bin_idx,
                    "standing_buyer_offer":  last_buyer,
                    "standing_seller_offer": last_seller,
                    "predicted_deal_price":  predicted_price,
                })

    return pd.DataFrame(records)


def plot_offer_distance_to_prediction(df, n_bins=N_BINS, figsize=(13, 5)):
    """
    Two panels (High / Intermediate buyer valuation), each showing mean distance
    of standing seller and buyer offers from the model-predicted deal price
    across normalised bargaining-time bins. T4 only.
    """
    long = clean_process_data(df)
    long = long[long["treatment"] == "T4"].copy()

    buyers_t4 = df[
        (df["treatment"] == "T4") &
        (df["exclude_for_time_analysis"] == 0) &
        (df["participant_role"] == "Buyer")
    ]
    valuation_lookup = dict(zip(buyers_t4["negotiation_id"], buyers_t4["buyer_valuation"]))

    standing = _build_standing_offer_data(long, valuation_lookup)
    standing["seller_dist"] = standing["standing_seller_offer"] - standing["predicted_deal_price"]
    standing["buyer_dist"]  = standing["standing_buyer_offer"]  - standing["predicted_deal_price"]

    bin_stats = (
        standing.groupby(["region", "bin"])[["seller_dist", "buyer_dist"]]
        .mean()
        .reset_index()
    )

    set_plot_theme()
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    panels = [
        ("High",         r"High valuation ($b > 21.40$)"),
        ("Intermediate", r"Intermediate valuation ($7.72 < b \leq 21.40$)"),
    ]

    for ax, (region, title) in zip(axes, panels):
        sub = bin_stats[bin_stats["region"] == region].sort_values("bin")
        ax.plot(sub["bin"], sub["seller_dist"],
                color=COLOR_SCHEME[0], marker="o", label="Seller offer $-$ equal split")
        ax.plot(sub["bin"], sub["buyer_dist"],
                color=COLOR_SCHEME[1], marker="o", label="Buyer offer $-$ equal split")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Normalised bargaining time (bin)")
        ax.set_xticks(range(n_bins))
        ax.legend(fontsize=9)

        # Annotate first (bin 0) and last (bin 14) values for each series
        for series_col, color in [("seller_dist", COLOR_SCHEME[0]), ("buyer_dist", COLOR_SCHEME[1])]:
            for boundary_bin, ha in [(0, "right"), (n_bins - 1, "left")]:
                row = sub[sub["bin"] == boundary_bin]
                if row.empty:
                    continue
                y_val = row[series_col].iloc[0]
                ax.annotate(
                    f"{y_val:.2f}",
                    xy=(boundary_bin, y_val),
                    xytext=(4 if ha == "left" else -4, 4),
                    textcoords="offset points",
                    ha=ha,
                    va="bottom",
                    fontsize=7,
                    color=color,
                )

        finalize_plot(ax=ax)

    axes[0].set_ylabel("Distance to equal surplus split (EUR)")

    plt.close()
    return fig


def calculate_offer_distance_boundary_values(df):
    """
    Return a dict of start/end mean distances for seller and buyer offers
    from the predicted deal price, by region (High / Intermediate), for T4.
    Keys follow the pattern: offer_dist_{role}_{region}_{boundary}
      role    : seller | buyer
      region  : high | intermediate
      boundary: start (bin 0) | end (bin 14)
    """
    long = clean_process_data(df)
    long = long[long["treatment"] == "T4"].copy()

    buyers_t4 = df[
        (df["treatment"] == "T4") &
        (df["exclude_for_time_analysis"] == 0) &
        (df["participant_role"] == "Buyer")
    ]
    valuation_lookup = dict(zip(buyers_t4["negotiation_id"], buyers_t4["buyer_valuation"]))

    standing = _build_standing_offer_data(long, valuation_lookup)
    standing["seller_dist"] = standing["standing_seller_offer"] - standing["predicted_deal_price"]
    standing["buyer_dist"]  = standing["standing_buyer_offer"]  - standing["predicted_deal_price"]

    bin_stats = (
        standing.groupby(["region", "bin"])[["seller_dist", "buyer_dist"]]
        .mean()
        .reset_index()
    )

    results = {}
    region_keys = {"High": "high", "Intermediate": "intermediate"}
    boundary_bins = {"start": 0, "end": N_BINS - 1}

    for region, region_key in region_keys.items():
        sub = bin_stats[bin_stats["region"] == region]
        for boundary, bin_idx in boundary_bins.items():
            row = sub[sub["bin"] == bin_idx]
            if row.empty:
                continue
            results[f"offer_dist_seller_{region_key}_{boundary}"] = round(row["seller_dist"].iloc[0], 2)
            results[f"offer_dist_buyer_{region_key}_{boundary}"]  = round(row["buyer_dist"].iloc[0], 2)

    return results


# ---------------------------------------------------------------------------
# Inline statistics – first and last standing offers in T4
# ---------------------------------------------------------------------------

def calculate_first_and_last_offers_t4(df):
    """
    Calculate mean first and last standing offers for buyers and sellers in T4,
    separately for High (b > B_STAR) and Intermediate (B_DAGGER < b <= B_STAR)
    buyer valuation regions.

    Sample: BuyerCost negotiations (T4) with at least two offers submitted,
    excluding observations flagged for time-analysis inconsistencies.

    Returns a dict with keys:
      t4_{region}_{role}_{boundary}_offer
      region   : high | intermediate
      role     : buyer | seller
      boundary : first | last
    """
    long = clean_process_data(df)
    long = long[long["treatment"] == "T4"].copy()

    # Attach buyer valuation and region to every offer row
    buyers_t4 = df[
        (df["treatment"] == "T4") &
        (df["exclude_for_time_analysis"] == 0) &
        (df["participant_role"] == "Buyer")
    ][["negotiation_id", "buyer_valuation"]]
    long = long.merge(buyers_t4, on="negotiation_id", how="left")

    long["region"] = None
    long.loc[long["buyer_valuation"] > B_STAR,   "region"] = "high"
    long.loc[
        (long["buyer_valuation"] > B_DAGGER) & (long["buyer_valuation"] <= B_STAR),
        "region"
    ] = "intermediate"
    long = long[long["region"].notna()]

    # First offer per participant-round
    first = long[long["offer_number"] == 1]

    # Last standing offer per (negotiation_id, participant_code)
    last_idx = long.groupby(["negotiation_id", "participant_code"])["offer_number"].idxmax()
    last = long.loc[last_idx]

    results = {}
    for region in ("high", "intermediate"):
        for role, role_key in (("Buyer", "buyer"), ("Seller", "seller")):
            first_val = first[
                (first["region"] == region) & (first["participant_role"] == role)
            ]["offer_value"].mean()
            last_val = last[
                (last["region"] == region) & (last["participant_role"] == role)
            ]["offer_value"].mean()

            results[f"t4_{region}_{role_key}_first_offer"] = round(first_val, 2)
            results[f"t4_{region}_{role_key}_last_offer"]  = round(last_val,  2)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv(BLD / "data" / "merged_data_full_excluded.csv")

    dist_fig = plot_offer_distance_to_prediction(df)
    dist_fig.savefig(OVERLEAF_FIGURES / "offer_distance_to_prediction.pdf")
    print("Saved offer_distance_to_prediction.pdf")

    boundary_values = calculate_offer_distance_boundary_values(df)
    inject_values(OVERLEAF_ROOT / "main.tex", **boundary_values)
    print("Injected offer distance boundary values into main.tex")

    first_last_values = calculate_first_and_last_offers_t4(df)
    inject_values(OVERLEAF_ROOT / "main.tex", **first_last_values)
    print("Injected T4 first/last offer values into main.tex")
