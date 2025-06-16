from src.bargaining_analysis.helper import finalize_plot
import pandas as pd
import matplotlib.pyplot as plt


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
        .reindex(range(1, 7), fill_value=0)
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



