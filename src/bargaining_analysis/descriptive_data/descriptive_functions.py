from src.bargaining_analysis.helper import finalize_plot
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
from src.bargaining_analysis.helper import finalize_plot, set_plot_theme

def plot_time_preference_switching_points(df, round_number=33):
    """
    Bar‐plots the count of switching points (1–7) in `time_preference_switching_points`
    for a given round.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain columns 'round' and 'time_preference_switching_points'.
    round_number : int, default 33
        Which round to filter on.

    Returns:
    --------
    matplotlib.axes.Axes
    """
    # filter
    df_r = df[df['round'] == round_number]

    # count 1–7, filling any missing with 0
    counts = (
        df_r['time_preference_switching_points']
        .value_counts()
        .reindex(range(1, 8), fill_value=0)
        .sort_index()
    )

    # human‐readable labels
    labels = [r'$\le5\%$', r'$10\%$', r'$15\%$', r'$20\%$', r'$25\%$', r'$30\%$', r'$>30\%$']

    # build figure
    fig, ax = plt.subplots()
    ax.bar(range(len(counts)), counts.values)

    # align ticks & labels
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # axis titles
    ax.set_xlabel('Switch to receiving money later at interest rate')
    ax.set_ylabel('Count')

    # prevent label cutoff
    fig.tight_layout()
    finalize_plot(ax=ax)

    return fig


def plot_gains_from_trade_histogram_two_sided(df):

    set_plot_theme()

    df_two_sided = df[(df["treatment"] == "T3") | (df["treatment"] == "T4")]
    df_two_sided = df_two_sided.drop_duplicates(subset='negotiation_id', keep='first')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_two_sided["gains_from_trade"], bins=30, kde=False)
    plt.xlabel('Gains from Trade in Negotiation')
    plt.ylabel('Count')
    finalize_plot(ax=ax)
    return fig



def plot_ultimatum_offer_histogram(df, round_number=33, bins=10, hist_color=None):
    """
    Plots a histogram of ultimatum offers for a given round on a fresh figure/axes.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain columns 'round' and 'ultimatum_offer'.
    round_number : int, default 33
        Which round to filter.
    bins : int or sequence, default 10
        Number of bins or bin edges for the histogram.
    hist_color : color spec, optional
        Color for the bars. Falls back to Matplotlib default if None.

    Returns:
    --------
    matplotlib.axes.Axes
        The axes containing the histogram.
    """
    # filter to the chosen round
    df_r = df[df['round'] == round_number]

    # create a fresh figure and axis
    fig, ax = plt.subplots()

    # plot the histogram
    ax.hist(df_r['ultimatum_offer'], bins=bins, color=hist_color)

    # label axes
    ax.set_xlabel('Money to Keep in Euros')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Ultimatum Offers (Round {round_number})')

    # apply any final styling (e.g. grid, spines) if your finalize_plot does that
    finalize_plot(ax=ax)

    # tighten layout so titles/labels don't overlap
    fig.tight_layout()

    return fig


def plot_risk_elicitation_choices(df):

    df_round33 = df[df["round"] == 33]

    risk_labels = [
        r"80\% chance of winning €2",
        r"70\% chance of winning €3",
        r"60\% chance of winning €4",
        r"50\% chance of winning €5",
        r"40\% chance of winning €6",
        r"30\% chance of winning €7"
    ]

    
    risk_counts = (
        df_round33["risk_elicitation_choice"]
        .value_counts()
        .reindex(range(-1, -7, -1), fill_value=0)
    )

    # Create figure & axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw horizontal bars on that Axes
    ax.barh(risk_labels, risk_counts.values)

    # Now set labels and title on the Axes
    ax.set_xlabel("Count")
    ax.set_ylabel("Risk Elicitation Choice")
    ax.set_title("Counts of Risk Elicitation Choices")

    # Tidy up
    fig.tight_layout()

    # If you have a finalize_plot() utility, pass the figure to it
    finalize_plot(ax=ax)

    return fig


def calculate_descriptive_table_values(df: pd.DataFrame) -> dict:
    """
    For round 33, compute by‐treatment:
      - mean_session_duration (in minutes)
      - mean_age
      - share_females
      - n_participants
      - n_groups (n_participants / 8)
    
    Returns a dict mapping:
      'mean_session_duration_T1' → '12.34'
      'mean_age_T2'              → '25.67'
      ... etc.
    """
    # 1) filter to round 33
    df33 = df[df['round'] == 33].copy()
    
    # 2) compute female indicator
    df33['share_females'] = (df33['gender'] == 2).astype(int)
    
    # 3) treatments to loop over
    treatments = ['T1','T2','T3','T4']
    out = {}
    
    for t in treatments:
        grp = df33[df33['treatment'] == t]
        
        # if no observations, leave blank
        if grp.empty:
            out[f'mean_session_duration_{t}'] = ''
            out[f'mean_age_{t}']              = ''
            out[f'share_females_{t}']         = ''
            out[f'n_participants_{t}']        = ''
            out[f'n_groups_{t}']              = ''
            continue

        
        # mean session duration in minutes
        # (assumes df['experiment_duration'] in seconds)
        mean_dur = grp['experiment_duration'].mean() / 60  
        out[f'mean_session_duration_{t}'] = f"{mean_dur:.2f}"
        
        # mean age
        mean_age = grp['age'].mean()
        out[f'mean_age_{t}'] = f"{mean_age:.2f}"
        
        # share females
        share = grp['share_females'].mean()
        out[f'share_females_{t}'] = f"{share:.2f}"
        
        # number participants
        n = len(grp)
        rounded_n = (n // 32) * 32
        out[f'n_participants_{t}'] = str(rounded_n)
        
        # number of groups = participants // 8
        n_groups = n // 8
        out[f'n_groups_{t}']      = str(n_groups)
    
    return out


def plot_bargaining_rounds_by_id(df_clean, negotiation_ids, max_offers=10):
    """
    Plot specified negotiations by their negotiation_id, with micro-step arrows
    and a dotted line marking the end of bargaining.

    Parameters:
    -----------
    df_clean : pandas.DataFrame
        The cleaned bargaining data. Must include columns:
        'negotiation_id', 'participant_role', 'offer_<i>', 'offer_time_<i>',
        'termination_time_sec', 'valuation', 'bargaining_outcome'
    negotiation_ids : list
        List of negotiation_id values to plot (e.g. length 4).
    max_offers : int
        Maximum number of offers recorded (default=10).
    """
    # apply your custom theme (which sets the color cycle)
    set_plot_theme()

    # grab first two theme colors for Buyer/Seller
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    buyer_color, seller_color = colors[0], colors[1]

    # Filter to only the specified negotiation IDs
    df_sel = df_clean[df_clean['negotiation_id'].isin(negotiation_ids)]

    # Layout: square-ish grid
    n = len(negotiation_ids)
    rows = cols = int(math.ceil(math.sqrt(n)))
    fig, axes = plt.subplots(rows, cols,
                              figsize=(5 * cols, 4 * rows),
                              squeeze=False)

    # Precompute global max for consistent y-axis
    all_off_vals = []
    for nid in negotiation_ids:
        part = df_sel[df_sel['negotiation_id'] == nid]
        for role in ('Buyer', 'Seller'):
            row = part[part['participant_role'] == role]
            if row.empty:
                continue
            for i in range(1, max_offers + 1):
                val = row.get(f'offer_{i}', pd.Series([np.nan])).iloc[0]
                if pd.notna(val):
                    all_off_vals.append(val)
    y_max = max(all_off_vals) * 1.1 if all_off_vals else 1

    def get_series(rec):
        """Extract lists of (time, offer) for a single participant."""
        times, offers = [], []
        for i in range(1, max_offers + 1):
            t = rec.get(f'offer_time_{i}', np.nan)
            o = rec.get(f'offer_{i}',        np.nan)
            if pd.notna(t) and pd.notna(o):
                times.append(t)
                offers.append(o)
        return times, offers

    # Plot each negotiation by ID
    for idx, nid in enumerate(negotiation_ids):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        part  = df_sel[df_sel['negotiation_id'] == nid]
        buyer = part[part['participant_role'] == 'Buyer'].iloc[0]
        seller= part[part['participant_role'] == 'Seller'].iloc[0]

        bt, bo = get_series(buyer)
        st, so = get_series(seller)

        # scatter once per role, using theme colors
        if bt:
            ax.scatter(bt, bo,
                       color=buyer_color,
                       alpha=0.7,
                       label='Buyer')
            ax.plot(bt, bo,
                    '--',
                    alpha=0.3,
                    color=buyer_color)
        if st:
            ax.scatter(st, so,
                       color=seller_color,
                       alpha=0.7,
                       label='Seller')
            ax.plot(st, so,
                    '--',
                    alpha=0.3,
                    color=seller_color)

        # end-line
        term_times = [t for t in (
            buyer.get('termination_time_sec'),
            seller.get('termination_time_sec')
        ) if pd.notna(t)]
        t_end = max(term_times) if term_times else max(bt + st, default=0)
        ax.axvline(t_end,
                   linestyle=':',
                   linewidth=1.5,
                   color='black')
        ax.text(
            t_end,
            y_max * 0.95,
            'End',
            rotation=90,
            va='top',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6)
        )

        # outcome & valuations
        outcome_map = {
            'acceptance': 'Acceptance',
            'player': 'Player Termination',
            'Random_Termination': 'Computer Termination'
        }
        mapped = outcome_map.get(buyer["bargaining_outcome"],
                                 buyer["bargaining_outcome"])
        ax.text(
            0.02,
            0.98,
            (f'Buyer Valuation: {buyer["valuation"]}, '
             f'Seller Valuation: {seller["valuation"]}\n'
             f'Outcome: {mapped}'),
            transform=ax.transAxes,
            va='top',
            fontsize=8,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8)
        )

        ax.set_title(f'Negotiation {nid}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Offer')
        ax.set_ylim(0, y_max)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=7)
        finalize_plot(ax=ax)

    # Hide any unused subplots
    for idx in range(len(negotiation_ids), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


