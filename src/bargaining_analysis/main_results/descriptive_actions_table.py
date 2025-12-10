import pandas as pd
from pydantic import validate_call

def calculate_bargaining_actions_values(df: pd.DataFrame) -> dict:
    """
    For round 33, compute by‐treatment means and standard deviations for:
      - bargaining_time_full_sec
      - number_of_offers
      - payoff
      - avg_random_termination   (share where termination_mode=="Random_Termination")
      - avg_player_termination   (share where termination_mode=="Player")
      - avg_deals                (share where termination_mode=="")
    
    Returns a dict mapping, e.g.
      'bargaining_time_full_sec_T1'    → '123.45'
      'bargaining_time_full_sec_sd_T1' → ' 67.89'
      'avg_deals_T4'                   → '0.67'
      'avg_deals_sd_T4'                → '0.15'
    ready to pass into inject_values.
    """
    
    treatments = ['T1','T2','T3','T4']
    out = {}
    
    for t in treatments:
        grp = df[df['treatment'] == t]
        grp_time = df[(df['treatment'] == t) & (df["time_inconsistency_dummy"] == 0)]
        
        # compute continuous statistics
        for col in ['number_of_offers', 'payoff']:
            mean_val = grp[col].mean()
            sd_val   = grp[col].std(ddof=1)
            out[f'{col}_{t}']     = f"{mean_val:.2f}"
            out[f'{col}_sd_{t}']  = f"{sd_val:.2f}"

        for col in ['bargaining_time_full_sec']:
            mean_val = grp_time[col].mean()
            sd_val   = grp_time[col].std(ddof=1)
            out[f'{col}_{t}']     = f"{mean_val:.2f}"
            out[f'{col}_sd_{t}']  = f"{sd_val:.2f}"
        
        # compute termination shares & their SDs
        rand_ind   = (grp['termination_mode'] == "Random_Termination").astype(int)
        player_ind = (grp['termination_mode'] == "Player").astype(int)
        deal_ind   = grp['termination_mode'].isna().astype(int)
        
        for key, series in [
            ('avg_random_termination', rand_ind),
            ('avg_player_termination', player_ind),
            ('avg_deals', deal_ind)
        ]:
            mean_val = series.mean()
            sd_val   = series.std(ddof=1)
            out[f'{key}_{t}']    = f"{mean_val:.2f}"
            out[f'{key}_sd_{t}'] = f"{sd_val:.2f}"

        # compute WAR and WAR+
        war_val = calculate_war(df, t)
        war_plus_val = calculate_war_plus(df, t)
        out[f'WAR_{t}'] = f"{war_val:.2f}"
        out[f'WAR_plus_{t}'] = f"{war_plus_val:.2f}"
    
    return out


def calculate_war(df: pd.DataFrame, treatment: str) -> float:
    """
    WAR = sum(realized surplus) / sum(gains_from_trade)
    Realized surplus = gains_from_trade if bargaining_outcome == 'acceptance', else 0.
    Drops duplicate negotiation_id rows (keeps the first).
    """
    df = df.drop_duplicates(subset=['negotiation_id'], keep='first')
    df = df[df['treatment'] == treatment]

    df['realized_surplus'] = df.apply(
        lambda row: row['gains_from_trade'] if row['bargaining_outcome'] == 'acceptance' else 0,
        axis=1
    )

    numerator = df['realized_surplus'].sum()
    denominator = df[df["gains_from_trade"] > 0]["gains_from_trade"].sum()

    return numerator / denominator if denominator != 0 else float('nan')

def calculate_war_plus(df: pd.DataFrame, treatment: str) -> float:
    """
    WAR+ = (realized surplus + positive unrealized) / sum(gains_from_trade)
    Positive unrealized = gains_from_trade if gains_from_trade > 0 and not accepted, else 0.
    Drops duplicate negotiation_id rows (keeps the first).
    """
    df = df.drop_duplicates(subset=['negotiation_id'], keep='first')
    df = df[df['treatment'] == treatment]

    df['realized_surplus'] = df.apply(
        lambda row: row['gains_from_trade'] if row['bargaining_outcome'] == 'acceptance' else 0,
        axis=1
    )

    df['positive_realized'] = df.apply(
        lambda row: row['gains_from_trade'] if (row['bargaining_outcome'] == 'acceptance' and row['gains_from_trade'] > 0) else 0,
        axis=1
    )

    numerator = df['positive_realized'].sum()
    denominator = df[df["gains_from_trade"] > 0]["gains_from_trade"].sum()

    return numerator / denominator if denominator != 0 else float('nan')