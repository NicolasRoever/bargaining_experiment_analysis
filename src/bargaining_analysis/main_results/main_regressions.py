import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from pystout import pystout
import matplotlib.lines as mlines
from pydantic import BaseModel
from src.bargaining_analysis.config import VARLABELS_REGRESSION
from src.bargaining_analysis.helper import fix_pandas_append_error, set_plot_theme, finalize_plot


def model_gains_from_trade_number_offers(df: pd.DataFrame, varlabels_regression,  output_path: str):
    """
    Run a regression model for gains from trade and number of offers.
    """


    data_t12 = df[df['treatment'].isin(['T1', 'T2']) & pd.notna(df['number_of_offers'])]
    model_t12_offers = smf.ols(formula='number_of_offers ~ C(participant_role):valuation + C(participant_code)', 
                      data=data_t12).fit(cov_type='cluster', cov_kwds={'groups': data_t12['participant_code']})
    
    model_t12_time = smf.ols(formula='bargaining_time_full_sec ~ C(participant_role):valuation + C(participant_code)', 
                      data=data_t12).fit(cov_type='cluster', cov_kwds={'groups': data_t12['participant_code']})
    
    data_t34 = df[df['treatment'].isin(['T3', 'T4']) & pd.notna(df['number_of_offers'])]
    model_t34_offers = smf.ols(formula='number_of_offers ~ C(participant_role):valuation + C(participant_code)', 
                      data=data_t34).fit(cov_type='cluster', cov_kwds={'groups': data_t34['participant_code']})
    
    model_t34_time = smf.ols(formula='bargaining_time_full_sec ~ C(participant_role):valuation + C(participant_code)',
                      data=data_t34).fit(cov_type='cluster', cov_kwds={'groups': data_t34['participant_code']})
    
    fix_pandas_append_error()
    
    pystout(models = [model_t12_offers, model_t34_offers, model_t12_time, model_t34_time],
    endog_names=[r" \shortstack{ Number Offers \\ (T1 and T2)}", r" \shortstack{ Number Offers \\ (T3 and T4)}",  r" \shortstack{ Bargaining Time (sec) \\ (T1 and T2)}", r" \shortstack{ Bargaining Time (sec) \\ (T3 and T4)}"],
    file=output_path,
    addrows={"Individual Fixed Effects":["\\checkmark","\\checkmark", "\\checkmark","\\checkmark"]},
    digits=2,
    stars={.1:'*',.05:'**',.01:'***'},
    varlabels=varlabels_regression,
    exogvars = ['C(participant_role)[Buyer]:valuation','C(participant_role)[Seller]:valuation'],
    modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
    )
    

