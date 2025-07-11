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
    

def run_signaling_regressions(df: pd.DataFrame, output_path: str):
    """
    Run signaling regressions for T3 and T4 treatments.
    """
    df_signal = df[(~np.isclose(df["id_in_group"], df['accepted_by_id_in_group'])) & 
                   (df['participant_role'] == 'Buyer') &
                   (df["bargaining_outcome"] == "acceptance") &
                   (df['treatment'].isin(['T3', 'T4']))]

    model = smf.ols('last_offer_time ~ valuation', data=df_signal).fit(
        cov_type='cluster', 
        cov_kwds={'groups': df_signal['participant_code']}
    )
    
    fix_pandas_append_error()
    
    pystout(models=[model],
            endog_names=[r" \shortstack{ Last Offer Time (sec) }"],
            file=output_path,
            digits=2,
            stars={.1:'*',.05:'**',.01:'***'},
            varlabels=VARLABELS_REGRESSION,
            exogvars=['valuation'],
            modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
           )
    

def run_information_efficiency_regression(df, varlabels_regression, output_path: str):
    """
    Run regression for information efficiency.
    """

    df_gft30 = df[(df["gains_from_trade"] >= 0) & (df["gains_from_trade"] <= 30) ].copy()
    negotiation_data = df_gft30.drop_duplicates(subset='negotiation_id', keep='first')

    model = smf.ols(formula='agreement_dummy ~ C(information_asymmetry)', data=negotiation_data).fit(cov_type='cluster',
    cov_kwds={
        'groups': negotiation_data['negotiation_id']
    })
    
    fix_pandas_append_error()
    
    pystout(models=[model],
            endog_names=[r" \shortstack{ Agreement Dummy }"],
            file=output_path,
            digits=2,
            stars={.1:'*',.05:'**',.01:'***'},
            varlabels=varlabels_regression,
            exogvars=['C(information_asymmetry)[T.two-sided]'],
            modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
        
)

