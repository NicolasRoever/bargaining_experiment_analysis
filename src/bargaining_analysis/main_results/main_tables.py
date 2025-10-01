import pandas as pd
from pydantic import validate_call, ConfigDict
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf



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

def compute_metrics_buyer_only_table(df: pd.DataFrame) -> dict:
    """
    Compute experimental results for three buyer valuation regions:
      1) < 7.72: fraction terminated (with 95% CI, t-approximation)
      2) [7.72, 21.40): OLS slope of trade price on valuation (with 95% CI)
      3) >= 21.40: mean trade price (with 95% CI)

    Returns:
      dict with keys matching LaTeX variable names, values rounded to 2 decimals.
    """

    plot_df = df[
    (df['participant_role'] == 'Buyer') & 
    (df["treatment"] == "T4")
    ].copy()

    # --- Region splits
    region_1 = plot_df[plot_df["valuation"] < 7.72]
    region_2 = plot_df[(plot_df["valuation"] >= 7.72) & (plot_df["valuation"] < 21.40)]
    region_3 = plot_df[plot_df["valuation"] >= 21.40]

    # --- 1) Fraction terminated (region 1)
    n1 = len(region_1)
    term_count = region_1["deal_price"].isna().sum()
    p = term_count / n1
    frac_term = round(p * 100, 2)
    se = np.sqrt(p * (1 - p) / n1)
    ci_low, ci_high = stats.t.interval(0.95, df=n1 - 1, loc=p, scale=se)
    frac_term_lb, frac_term_ub = round(ci_low * 100, 2), round(ci_high * 100, 2)

    # --- 2) Regression slope (region 2)
    model = smf.ols("deal_price ~ valuation", data=region_2).fit()
    slope_trade_price = round(model.params["valuation"], 2)
    ci_low, ci_high = model.conf_int().loc["valuation"]
    slope_trade_price_lb, slope_trade_price_ub = round(ci_low, 2), round(ci_high, 2)

    # --- 3) Mean trade price (region 3)
    trades_r3 = region_3["deal_price"]
    n3 = len(trades_r3)
    mean_trade_price = round(trades_r3.mean(), 2)
    std_r3 = trades_r3.std(ddof=1)
    ci_low, ci_high = stats.t.interval(
        0.95, df=n3 - 1, loc=trades_r3.mean(), scale=std_r3 / np.sqrt(n3)
    )
    mean_trade_price_lb, mean_trade_price_ub = round(ci_low, 2), round(ci_high, 2)

    return {
        "frac_term": frac_term,
        "frac_term_lb": frac_term_lb,
        "frac_term_ub": frac_term_ub,
        "slope_trade_price": slope_trade_price,
        "slope_trade_price_lb": slope_trade_price_lb,
        "slope_trade_price_ub": slope_trade_price_ub,
        "mean_trade_price": mean_trade_price,
        "mean_trade_price_lb": mean_trade_price_lb,
        "mean_trade_price_ub": mean_trade_price_ub,
    }