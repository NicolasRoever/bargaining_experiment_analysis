import itertools
from scipy.stats import ttest_ind
import pandas as pd

def calculate_balance_table_values(df):
    """
    Calculate balance‐test differences for round 33 across all 6 pairwise
    comparisons of treatments 1–4, for the five variables:
      - age
      - share_female  (computed from gender==2)
      - risk_elicitation_choice
      - ultimatum_offer
      - time_preference_switching_points
    
    Returns a dict mapping names like 'age_diff_t1_t2' → '0.12**'
    """
    # 1) subset to round 33
    df33 = df[df['round'] == 33].copy()
    
    # 2) compute share_female indicator
    df33['share_female'] = (df33['gender'] == 2).astype(int)
    
    # 3) map our variable keys to actual columns
    variables = {
        'age': 'age',
        'share_female': 'share_female',
        'risk_elicitation_choice': 'risk_elicitation_choice',
        'ultimatum_offer': 'ultimatum_offer',
        'time_preference_switching_points': 'time_preference_switching_points'
    }
    
    # 4) all treatment pairs (1–4)
    treatments = ["T1", "T2", "T3", "T4"]
    pairs = list(itertools.combinations(treatments, 2))
    
    results = {}
    
    # helper to assign stars
    def stars(p):
        if p < 0.01:
            return '***'
        elif p < 0.05:
            return '**'
        elif p < 0.10:
            return '*'
        else:
            return ''
    
    # 5) loop over each pair and variable
    for (i, j) in pairs:
        grp_i = df33[df33['treatment'] == i]
        grp_j = df33[df33['treatment'] == j]
        
        for var_key, col in variables.items():
            x = grp_i[col]
            y = grp_j[col]
            
            if len(x) == 0 or len(y) == 0:
                diff_str = ''
            else:
                diff = x.mean() - y.mean()
                tstat, pval = ttest_ind(x, y, equal_var=False)
                diff_str = f"{diff:.2f}{stars(pval)}"
            
            name = f"{var_key}_diff_{i}_{j}"
            results[name] = diff_str
    
    return results