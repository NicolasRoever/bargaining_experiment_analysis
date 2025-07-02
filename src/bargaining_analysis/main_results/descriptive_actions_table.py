import pandas as pd

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
        for col in ['number_of_offers', 'payoff', 'efficiency']:
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
    
    return out
