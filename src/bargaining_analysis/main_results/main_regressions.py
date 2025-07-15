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

    # Information Effect
    df_gft30 = df[(df["gains_from_trade"] >= 0) & (df["gains_from_trade"] <= 30) ].copy()
    negotiation_data = df_gft30.drop_duplicates(subset='negotiation_id', keep='first')

    model = smf.ols(formula='agreement_dummy ~ C(information_asymmetry)', data=negotiation_data).fit(cov_type='cluster',
    cov_kwds={
        'groups': negotiation_data['negotiation_id']
    })

    #Fairness Effect
    fairness_df = df[['player_termination_dummy', 'ultimatum_indicator', 'participant_label']].dropna()
    fairness = smf.ols(formula='player_termination_dummy ~ ultimatum_indicator', data=fairness_df).fit(
    cov_type='cluster',
    cov_kwds={'groups': fairness_df['participant_label']},
    )

    #Risk Effect
    df_with_TA = df[df["treatment"].isin(["T3", "T4"])].copy()
    termination_risk = smf.ols(formula='player_termination_dummy ~ risk_elicitation_choice', 
    data=df_with_TA).fit(
        cov_type='cluster',
        cov_kwds={'groups': df_with_TA['participant_label']},
    )


    fix_pandas_append_error()
    
    pystout(models=[model, fairness, termination_risk],
            endog_names=[r" \shortstack{ Agreement Dummy } ", r" \shortstack{ Player Termination \\ Dummy }", r" \shortstack{ Player Termination \\ Dummy }"],
            file=output_path,
            digits=2,
            stars={.1:'*',.05:'**',.01:'***'},
            varlabels=varlabels_regression,
            exogvars=['C(information_asymmetry)[T.two-sided]', 'ultimatum_indicator', 'risk_elicitation_choice'],
            modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
        
    )


def run_signaling_regression(df: pd.DataFrame, output_path: str):
   

   #On T3/T4
    df_t34_buyer = df[(df['treatment'].isin(["T3", "T4"])) & (df['participant_role'] == 'Buyer') & (df['time_inconsistency_dummy'] == 0)]
    df_t34_buyer = df_t34_buyer.dropna(subset=['valuation', 'last_offer_time'])

    model_t34_buyer_simple = smf.ols(formula='last_offer_time ~ valuation + C(participant_label)', 
    data=df_t34_buyer).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_t34_buyer['participant_label']},
    )

    model_t34_buyer_adv = smf.ols(formula='last_offer_time ~ valuation + I(valuation**2) + C(participant_label)', 
    data=df_t34_buyer).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_t34_buyer['participant_label']},
    )

    #On T1/T2
    df_t12 = df[(df['treatment'].isin(["T1", "T2"])) & (df['time_inconsistency_dummy'] == 0)]
    df_t12 = df_t12.dropna(subset=['valuation', 'last_offer_time'])
    model_t12_simple =  smf.ols(formula='last_offer_time ~ C(participant_role):valuation + gains_from_trade + C(participant_label)', 
    data=df_t12).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_t12['participant_label']},
    )

    model_t12_adv = smf.ols(formula='last_offer_time ~ C(participant_role):valuation + I(valuation**2) + gains_from_trade + C(participant_label)',
    data=df_t12).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_t12['participant_label']},
    )

    #Export results
    fix_pandas_append_error()
    
    pystout(models=[model_t34_buyer_simple, model_t34_buyer_adv, model_t12_simple, model_t12_adv],
            endog_names=[r" \shortstack{ Accepted Offer Time (sec) \\ (T3 and T4)}",],
            file=output_path,
            digits=2,
            stars={.1:'*',.05:'**',.01:'***'},
            varlabels=VARLABELS_REGRESSION,
            exogvars=['C(participant_role)[Buyer]', 'C(treatment)[T.T2]', 'C(treatment)[T.T4]'],
            modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
           )
    

def bargaining_power_regressions(df:pd.DataFrame, varlabels_regression, output_path: str):
    """
    Run regressions for bargaining power.
    """
    df01 = df[(df['split_gains_from_trade'] >= 0) & (df['split_gains_from_trade'] <= 1)]
    df01_time = df01[np.isclose(df01['time_inconsistency_dummy'], 0)]


    #Model One: First Offer
    data_time_model = df01_time.dropna(subset=['first_offer', 'information_asymmetry'])
    model_t12 = smf.ols(formula='split_gains_from_trade ~ first_offer:C(information_asymmetry)', 
                        data=data_time_model).fit(
        cov_type='cluster',
        cov_kwds={'groups': data_time_model['participant_code']})


    #Model 2: Individual Fixed Effects

    data_model_ind_fe = df01.dropna(subset=['split_gains_from_trade', 'participant_label'])
    model_ind_fe = smf.ols(formula='split_gains_from_trade ~ C(participant_label)', 
                        data=data_model_ind_fe).fit(
        cov_type='cluster',
        cov_kwds={'groups': data_model_ind_fe['participant_label']}
                        )
    

    #Model 3: Behavioral Factors
    data_model_behavioral = df01.dropna(subset=['split_gains_from_trade', 'ultimatum_indicator', 'risk_elicitation_choice', 'time_preference_switching_points'])
    model_behavioral = smf.ols(formula='split_gains_from_trade ~ ultimatum_indicator + risk_elicitation_choice + time_preference_switching_points', 
                        data=data_model_behavioral).fit(
        cov_type='cluster',
        cov_kwds={'groups': data_model_behavioral['participant_label']}
                        )

    fix_pandas_append_error()

    #Make Table
    
    pystout(models=[model_t12, model_ind_fe, model_behavioral],
            endog_names=[r" \shortstack{ Split \\ Gains from Trade}", r" \shortstack{ Split \\ Gains from Trade}", r" \shortstack{ Split \\ Gains from Trade}"],
            file=output_path,
            digits=2,
            stars={.1:'*',.05:'**',.01:'***'},
            varlabels=varlabels_regression,
            addrows={
                "Individual Fixed Effects": ["x", "\\checkmark", "x"],
            },
            exogvars=['first_offer:C(information_asymmetry)[one-sided]', 'first_offer:C(information_asymmetry)[two-sided]', 'ultimatum_indicator', 'risk_elicitation_choice', 'time_preference_switching_points', 'Intercept'],
            modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
           )
