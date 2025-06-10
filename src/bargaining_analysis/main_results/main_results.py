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


def regress_first_mover(data, path):

    # Model 1

    df_1 = data.dropna(subset=["payoff", "first_offer"])

    df_1["payoff"] = pd.to_numeric(df_1["payoff"], errors='coerce') 

    model_1 = smf.ols(formula='payoff ~ first_offer', data=df_1).fit()

    # Model 2

    df_2 = data[data["split_gains_from_trade"] > 0]


    df_2["payoff"] = pd.to_numeric(df_2["payoff"], errors='coerce') 

    model_2 = smf.ols(formula='payoff ~ first_offer', data=df_2).fit()

    #Model 3
    model_3 = smf.ols(formula='bargaining_time_full_sec ~ valuation', data=data).fit()

    pystout(models=[model_1, model_2, model_3],
        file=path,
        addnotes=['Here is a little note','And another one'],
        digits=2,
        endog_names=['Payoff','Split Gains from Trade', 'Bargaining Time'],
        varlabels={'const':'Constant','first_offer':'First Offer'},
        modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
        )


    

    
