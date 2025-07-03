import pandas as pd
from pydantic import validate_call, ConfigDict


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def compute_acceptance_rates(df: pd.DataFrame) -> dict:
    """
    Filter the DataFrame for gains_from_trade > 0, then compute the acceptance rate
    for each treatment T1–T4.
    """
    # 1) Filter for positive gains
    df_filtered = df[df['gains_from_trade'] > 0]

    # 2) Prepare output dict
    rates = {}

    # 3) Loop through treatments
    for t in ['T1', 'T2', 'T3', 'T4']:
        sub = df_filtered[df_filtered['treatment'] == t]
        rate = (sub['bargaining_outcome'] == 'acceptance').mean()
        rates[f'acceptance_{t}'] =  int(round(rate * 100))

    return rates