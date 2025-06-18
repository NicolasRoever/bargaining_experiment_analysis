import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from pystout import pystout

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


def regression_table_asymmetric_treatment(data, path):

    #Filter Data
    df_one_sided = data[(data["treatment"] == "T3") | (data["treatment"] == "T4")]

    #Run Regressions

    first_offer = smf.ols(formula='first_offer ~ C(participant_role)', data=df_one_sided).fit()
    split_gains = smf.ols(formula='split_gains_from_trade ~ C(participant_role) + C(TA_costs)', data=df_one_sided).fit()
    efficiency = smf.ols(formula='efficiency ~ gains_from_trade + C(TA_costs)', data=df_one_sided).fit()
    agreement = smf.ols(formula='agreement_dummy ~ C(TA_costs) + gains_from_trade', data=df_one_sided).fit()


    #Create Table
    pystout(models=[first_offer, split_gains, efficiency, agreement],
    file=path,
    addnotes=['\\textit{Notes:} These regressions are run on the data from the treatments with asymmetric uncertainty. Dummy First Offer is indicator equal to 1 if the player has made the first offer, split gains from trade is percentage of the surplus the player extracts from a successfull trade (we exclude transaction costs for this measure), efficiency is indicator equal to 1 if an agreement was reached and 0 otherwise (surplus is always at leas zero in the considered treatmends) and Dummy Agreement is an indicator equal to 1 if an agreement was reached.'],
    digits=2,
    endog_names=['Dummy First Offer', "Split Gains from Trade", "Efficiency", "Dummy Agreement"],
    varlabels={'const':'Constant','first_offer':'First Offer', 'split_gains_from_trade':'Split Gains from Trade', 'efficiency':'Efficiency', 'agreement_dummy':'Dummy Agreement', 'C(participant_role)[T.Seller]':'Seller', 'C(TA_costs)[T.0.05]':'TA Costs $= 0.05$', 'gains_from_trade':'Gains from Trade'},
    modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
    )



def regression_table_symmetric_treatment(data, path):

    
    first_mover_advantage = smf.ols(formula='split_gains_from_trade ~ first_offer', data=data).fit()

    slient_signaling = smf.ols(formula='valuation ~ offer_time_1', data=data).fit()

    pystout(models=[first_mover_advantage, slient_signaling],
        file=path,
        addnotes=['Here is a little note','And another one'],
        digits=2,
        endog_names=['Split Gains from Trade', "Valuation"],
        varlabels={'const':'Constant','first_offer':'First Offer', 'offer_time_1':'First Offer Time'},
        modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
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



    

    
