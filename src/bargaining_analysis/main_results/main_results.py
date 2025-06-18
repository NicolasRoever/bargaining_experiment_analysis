import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from pystout import pystout
from src.bargaining_analysis.config import VARLABELS_REGRESSION
from src.bargaining_analysis.helper import fix_pandas_append_error

# Define the piecewise function
def equilibrium_payoff(x, c=0.05, r=0.01):
    if x <= 7.72:
        return 0
    elif x >= 21.40:
        return x/2
    else:
        b = x
        return ((r*b + 2*c)/(r*21.40 + 2*c)) * (b/2 + c/r) - c/r
    

def plot_T4_buyer_payoff_vs_valuation(
        df: pd.DataFrame,
        color_scheme: list[str]):
    
   
    df_buyers = df[(df["participant_role"] == "Buyer")]

    # Generate x values for the theoretical line
    x_values = np.linspace(0, 30, 1000)
    y_values = [equilibrium_payoff(x) for x in x_values]



    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    sns.set_style("white")


    # Define a mapping from outcome labels to colors
    outcome_color_map = {
        'Player': color_scheme[0],               # Player Termination
        'Random_Termination': color_scheme[1],   # Computer Termination
        'acceptance': color_scheme[2],           # Highlight Acceptance
    }

    # Assign color values based on the outcome
    colors = df_buyers['bargaining_outcome'].map(outcome_color_map)

    plt.figure(figsize=(12, 8))

    # Plot the theoretical line
    plt.plot(x_values, y_values, color=color_scheme[3], linewidth=2, label='Equilibrium Payoff Model')

    # Add the scatter plot with custom colors
    scatter = plt.scatter(
        df_buyers['valuation'],
        df_buyers['payoff'],
        c=colors,
        alpha=0.8
    )

    plt.xlabel('Buyer Valuation (Equals Surplus) ')
    plt.ylabel('Buyer Payoff')

    # Create custom legend
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([], [], color=color_scheme[0], marker='o', linestyle='None', label='Player Termination'),
        mlines.Line2D([], [], color=color_scheme[1], marker='o', linestyle='None', label='Computer Termination'),
        mlines.Line2D([], [], color=color_scheme[2], marker='o', linestyle='None', label='Acceptance'),
        mlines.Line2D([], [], color=color_scheme[3], linewidth=2, label='Equilibrium Payoff Model')
    ]

    plt.legend(handles=legend_elements, title="Bargaining Outcome")

    sns.despine()
    plt.tight_layout()

    return plt

def regression_table_all_treatments(data, path, varlables_regression):

    negotiation_data = data.drop_duplicates(subset='negotiation_id', keep='first')

    efficiency = smf.ols(formula='efficiency ~ C(TA_costs) + C(information_asymmetry) + positive_gains_symmetric_treatment + C(information_asymmetry):small_gains_from_trade_indicator', data=negotiation_data).fit()
    behavioral_efficiency = smf.ols(formula='efficiency ~ C(TA_costs) + C(information_asymmetry) + positive_gains_symmetric_treatment + C(information_asymmetry):small_gains_from_trade_indicator + ultimatum_indicator  + risk_elicitation_choice + time_preference_switching_points', data=data).fit()


    fix_pandas_append_error()

    pystout(
    models=[efficiency, behavioral_efficiency],
    file=path,
    digits=2,
    endog_names=['Efficiency', "Efficiency"],
    varlabels=varlables_regression,
    stars={.1:'*',.05:'**',.01:'***'},
    addrows={"Level of Observation":["Negotiation","Individual"]},
    modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
    )


def regression_table_asymmetric_treatment(data, path):

    #Filter Data
    df_one_sided = data[(data["treatment"] == "T3") | (data["treatment"] == "T4")]
    df_one_negotiations = df_one_sided.drop_duplicates(subset='negotiation_id', keep='first')

    # Calculate means of dependent variables
    mean_first_offer = df_one_sided['first_offer'].mean()
    mean_split_gains = df_one_sided['split_gains_from_trade'].mean()
    mean_efficiency = df_one_negotiations['efficiency'].mean()
    mean_agreement = df_one_negotiations['agreement_dummy'].mean()

    # Run Regressions
    first_offer = smf.ols(formula='first_offer ~ C(participant_role)', data=df_one_sided).fit()
    split_gains = smf.ols(formula='split_gains_from_trade ~ C(participant_role) + C(TA_costs)', data=df_one_sided).fit()
    efficiency = smf.ols(formula='efficiency ~ gains_from_trade + C(TA_costs)', data=df_one_negotiations).fit()
    agreement = smf.ols(formula='agreement_dummy ~ C(TA_costs) + gains_from_trade', data=df_one_negotiations).fit()


    fix_pandas_append_error()

    #Create Table
    pystout(
    models=[first_offer, split_gains, efficiency, agreement],
    file=path,
    digits=2,
    endog_names=['Dummy First Offer', "Split Gains from Trade", "Efficiency", "Dummy Agreement"],
    varlabels={'const':'Constant','first_offer':'First Offer', 'split_gains_from_trade':'Split Gains from Trade', 'efficiency':'Efficiency', 'agreement_dummy':'Dummy Agreement', 'C(participant_role)[T.Seller]':'Seller', 'C(TA_costs)[T.0.05]':'TA Costs $= 0.05$', 'gains_from_trade':'Gains from Trade'},
    addrows={"Level of Observation":["Player","Player","Negotiation","Negotiation"], 
             "Mean Dep. Variable":["{:.2f}".format(mean_first_offer), "{:.2f}".format(mean_split_gains), "{:.2f}".format(mean_efficiency), "{:.2f}".format(mean_agreement)]}, 
    stars={.1:'*',.05:'**',.01:'***'},
    modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
    )



def regression_table_symmetric_treatment(data, path, varlabels_regression):

    
    #Filter Data
    df_symmetric = data[(data["treatment"] == "T1") | (data["treatment"] == "T2")]
    df_symmetric_negotiations = df_symmetric.drop_duplicates(subset='negotiation_id', keep='first')

    # Calculate means of dependent variables
    mean_split_gains = df_symmetric['split_gains_from_trade'].mean()
    mean_offer_time = df_symmetric['offer_time_1'].mean()
    mean_efficiency = df_symmetric_negotiations['efficiency'].mean()
    mean_agreement = df_symmetric_negotiations['agreement_dummy'].mean()

    #Run Regressions
    first_mover = smf.ols(formula='split_gains_from_trade ~ first_offer', data=df_symmetric).fit()
    silent_signaling = smf.ols(formula='offer_time_1 ~ C(participant_role):valuation', data=df_symmetric).fit()
    breakpoint()
    efficiency = smf.ols(formula='efficiency ~ C(TA_costs)', data=df_symmetric_negotiations).fit()
    agreement = smf.ols(formula='agreement_dummy ~ C(TA_costs)', data=df_symmetric_negotiations).fit()


    fix_pandas_append_error()
    #Create Table
    pystout(models=[first_mover, silent_signaling, efficiency, agreement],
            file=path,
            digits=2,
            endog_names=['Split Gains from Trade', "First Offer (sec)", "Efficiency", " Agreement"],
            varlabels=varlabels_regression,
            stars= {.1:'*',.05:'**',.01:'***'},
            modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'},
            addrows={"Level of Observation":["Player","Player","Negotiation","Negotiation"], 
             "Mean Dep. Variable":["{:.2f}".format(mean_split_gains), "{:.2f}".format(mean_offer_time), "{:.2f}".format(mean_efficiency), "{:.2f}".format(mean_agreement)]}, 
            )


def regression_table_one_sided_treatment(data, path):
    
    first_offer = smf.ols(formula='first_offer ~ C(participant_role)', data=data).fit()

    efficiency = smf.ols(formula='efficiency ~ gains_from_trade + C(TA_costs) + time_preference_switching_points + risk_elicitation_choice + ultimatum_indicator + C(information_asymmetry)', data=data).fit()


    pystout(models=[first_offer, efficiency],
            file=path,
            addnotes=['First Offer is indicator equal to 1 if the player has made the first offer, efficiency is 1 if an agreement was reached when gains from trade are at least 0','And another one'],
            digits=2,
            endog_names=['First Offer', "Efficiency"],
            varlabels={'const':'Constant','first_offer':'First Offer', 'efficiency':'Efficiency'},
            modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
            )


def regression_table_master_negotiators(data, path, varlabels_regression):
    
    df_master = data.dropna(subset=['first_offer_split', 'large_gains_from_trade_indicator'])


    masters = smf.ols(formula='large_gains_from_trade_indicator ~ C(information_asymmetry):C(TA_costs) + first_offer + ultimatum_indicator + risk_elicitation_choice + time_preference_switching_points + bargaining_time_full_sec', data=df_master).fit()

    masters_with_offer_size = smf.ols(formula='large_gains_from_trade_indicator ~ C(information_asymmetry):C(TA_costs) + first_offer + first_offer_split + ultimatum_indicator + risk_elicitation_choice + time_preference_switching_points + bargaining_time_full_sec', data=df_master).fit()

    masters_with_fe = smf.ols(formula='large_gains_from_trade_indicator ~ C(information_asymmetry):C(TA_costs) + first_offer + first_offer_split + ultimatum_indicator + risk_elicitation_choice + time_preference_switching_points + bargaining_time_full_sec + C(participant_id)', data=df_master).fit()


    fix_pandas_append_error()

    pystout(models=[masters, masters_with_offer_size, masters_with_fe],
            file=path,
            exogvars=["C(TA_costs)[T.0.05]", "", "first_offer", "first_offer_split", "C(information_asymmetry)[T.two-sided]:C(TA_costs)[0.0]", "C(information_asymmetry)[T.two-sided]:C(TA_costs)[0.05]", "ultimatum_indicator", "risk_elicitation_choice", "time_preference_switching_points", "bargaining_time_full_sec"],
            varlabels=varlabels_regression,
            addrows={"Individual Fixed Effects":["\\texttimes","\\texttimes","\\checkmark"]},
            modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
            )

    

    

