import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from pystout import pystout
import matplotlib.lines as mlines
from src.bargaining_analysis.config import VARLABELS_REGRESSION
from src.bargaining_analysis.helper import fix_pandas_append_error, set_plot_theme, finalize_plot
from pydantic import BaseModel

# Define the piecewise function
def equilibrium_payoff(x, c=0.05, r=0.01):
    if x <= 7.72:
        return 0
    elif x >= 21.40:
        return x - 10.70
    else:
        b = x
        return ((r*b + 2*c)/(r*21.40 + 2*c)) * (b/2 + c/r) - c/r


def compute_sample_statistics(df: pd.DataFrame) -> dict:
    """
    Compute descriptive statistics for the analysis sample.
    Returns a dictionary with total number of negotiations and participants.
    """
    total_negotiations = df["negotiation_id"].nunique()
    total_participants = df["participant_code"].nunique()
    
    return {
        "total_negotiations": int(total_negotiations),
        "total_participants": int(total_participants),
    }
    

def plot_buyer_payoff_vs_valuation(
        df: pd.DataFrame,
        color_scheme: list[str],
        treatment_t4 = "no"):

    
   
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

    fig, ax = plt.subplots(figsize=(12, 8))
     
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
        mlines.Line2D([], [], color=color_scheme[2], marker='o', linestyle='None', label='Acceptance')
    ]

    if treatment_t4 == "yes":
        plt.plot(x_values, y_values, color=color_scheme[3], linewidth=2, label='Equilibrium Payoff Model')
        legend_elements.append(mlines.Line2D([], [], color=color_scheme[3], linewidth=2, label='Equilibrium Payoff Model'))
    else:
        plt.plot(x_values, 0.5*x_values, color=color_scheme[3], linestyle='--', linewidth=2, label='Equal Split Line')
        legend_elements.append(mlines.Line2D([], [], color=color_scheme[3], linestyle='--', linewidth=2, label='Equal Split Line'))
     

    plt.legend(handles=legend_elements, title="Bargaining Outcome")

    finalize_plot(ax)

    return fig




def regression_table_all_treatments(data, path, varlables_regression):

    negotiation_data = data.drop_duplicates(subset='negotiation_id', keep='first')


    efficiency_df = negotiation_data.dropna(subset=['efficiency', 'TA_costs', 'information_asymmetry', 'positive_gains_symmetric_treatment', 'small_gains_from_trade_indicator'])
    efficiency = smf.ols(formula='efficiency ~ C(TA_costs) + C(information_asymmetry) + positive_gains_symmetric_treatment + C(information_asymmetry):small_gains_from_trade_indicator', data=efficiency_df).fit(cov_type='cluster',
    cov_kwds={
        'groups': efficiency_df['negotiation_id']
    })

    behavioral_efficiency_df = data.dropna(subset=['efficiency', 'TA_costs', 'information_asymmetry', 'positive_gains_symmetric_treatment', 'small_gains_from_trade_indicator', 'ultimatum_indicator', 'risk_elicitation_choice', 'time_preference_switching_points'])
    behavioral_efficiency = smf.ols(formula='efficiency ~ C(TA_costs) + C(information_asymmetry) + positive_gains_symmetric_treatment + C(information_asymmetry):small_gains_from_trade_indicator + ultimatum_indicator  + risk_elicitation_choice + time_preference_switching_points', data=behavioral_efficiency_df).fit(cov_type='cluster',
    cov_kwds={
        'groups': behavioral_efficiency_df['participant_code']
    })



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


def regression_table_asymmetric_treatment(data, path, varlabels_regression):

    #Filter Data
    df_one_sided = data[(data["treatment"] == "T3") | (data["treatment"] == "T4")]
    df_one_sided_time = df_one_sided[df_one_sided["time_inconsistency_dummy"] == 0]
    df_one_negotiations = df_one_sided.drop_duplicates(subset='negotiation_id', keep='first')

    # Run Regressions
    df_first_offer = df_one_sided_time.dropna(subset=['first_offer'])
    first_offer = smf.ols(formula='first_offer ~ C(participant_role)', 
                        data=df_first_offer).fit(cov_type='cluster',
        cov_kwds={
            'groups': df_first_offer['participant_code']
        })
    
    bargaining_time = smf.ols(formula='bargaining_time_full_sec ~ C(TA_costs)', 
                        data=df_one_sided_time).fit(cov_type='cluster',
        cov_kwds={
            'groups': df_one_sided_time['participant_code']
        })
    
    number_of_offers = smf.ols(formula='number_of_offers ~ C(TA_costs)', 
                      data=df_one_sided).fit(cov_type='cluster',
    cov_kwds={
        'groups': df_one_sided['participant_code']
    })

    efficiency = smf.ols(formula='efficiency ~ gains_from_trade + C(TA_costs)', data=df_one_negotiations).fit(cov_type='cluster',
    cov_kwds={
        'groups': df_one_negotiations['participant_code']
    })
    agreement = smf.ols(formula='agreement_dummy ~ C(TA_costs)', data=df_one_negotiations).fit(cov_type='cluster',
    cov_kwds={
        'groups': df_one_negotiations['participant_code']
    })
    termination = smf.ols(formula='player_terminated ~ C(participant_role) + C(TA_costs)', data=df_one_negotiations).fit(cov_type='cluster',
    cov_kwds={
        'groups': df_one_negotiations['participant_code']
    })


    fix_pandas_append_error()

    #Create Table
    pystout(
    models=[first_offer, bargaining_time, number_of_offers,  termination, agreement, efficiency],
    file=path,
    digits=2,
    endog_names=[r" \shortstack{ Dummy \\ First  Offer}", r"Bargaining Time", r"Number of Offers", r"\shortstack{ Dummy \\ Player  Termination}", r"\shortstack{ Dummy \\ Agreement}", r"Efficiency"], 
    varlabels=varlabels_regression,
    addrows={"Level of Observation":["Player","Player", "Player", "Negotiation","Negotiation","Negotiation"]}, 
    stars={.1:'*',.05:'**',.01:'***'},
    modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
    )



def regression_table_symmetric_treatment(data, path, varlabels_regression):

    
    #Filter Data
    df_symmetric = data[(data["treatment"] == "T1") | (data["treatment"] == "T2")]
    df_symmetric_time = df_symmetric[df_symmetric["time_inconsistency_dummy"] == 0]
    df_symmetric_negotiations = df_symmetric.drop_duplicates(subset='negotiation_id', keep='first')


    #Run Regressions
    split_gains_df = df_symmetric_time.dropna(subset=['split_gains_from_trade'])
    split_gains = smf.ols(formula='split_gains_from_trade ~ C(participant_role)', data=split_gains_df).fit(cov_type='cluster',
    cov_kwds={
        'groups': split_gains_df['participant_code']
    })

    bargaining_time_df = df_symmetric_time.dropna(subset=['bargaining_time_full_sec'])
    bargaining_time = smf.ols(formula='bargaining_time_full_sec ~ C(TA_costs)', data=bargaining_time_df).fit(cov_type='cluster',
    cov_kwds={
        'groups': bargaining_time_df['participant_code']
    })

    first_move_df = df_symmetric_time.dropna(subset=['first_offer'])
    first_move = smf.ols(formula='first_offer ~ C(participant_role)', data=first_move_df).fit(cov_type='cluster',
    cov_kwds={
        'groups': first_move_df['participant_code']
    })

    first_offer_df = df_symmetric_time.dropna(subset=['first_offer','split_gains_from_trade'])
    first_offer = smf.ols(formula='split_gains_from_trade ~ first_offer', data=first_offer_df).fit(cov_type='cluster',
    cov_kwds={
        'groups': first_offer_df['participant_code']
    })

    agreement_df = df_symmetric_negotiations.dropna(subset=['agreement_dummy'])
    agreement = smf.ols(formula='agreement_dummy ~ C(TA_costs)', data=agreement_df).fit(cov_type='cluster',
    cov_kwds={
        'groups': agreement_df['participant_code']
    })

    termination_df = df_symmetric_negotiations.dropna(subset=['player_terminated'])
    termination = smf.ols(formula='player_terminated ~ C(participant_role)', data=termination_df).fit(cov_type='cluster',
    cov_kwds={
        'groups': termination_df['participant_code']
    })

    efficiency_df = df_symmetric_negotiations.dropna(subset=['efficiency'])
    efficiency_1 = smf.ols(formula='efficiency ~ C(TA_costs)', data=efficiency_df).fit(cov_type='cluster',
    cov_kwds={
        'groups': efficiency_df['participant_code']
    })

    efficiency_2 = smf.ols(formula='efficiency ~ C(TA_costs) + small_gains_from_trade_indicator', data=df_symmetric_negotiations).fit(cov_type='cluster',
    cov_kwds={
        'groups': df_symmetric_negotiations['participant_code']
    })


    fix_pandas_append_error()
    #Create Table
    pystout(models=[first_move, split_gains, first_offer, bargaining_time, agreement, termination, efficiency_1, efficiency_2],
            file=path,
            digits=2,
            endog_names=[r'\shortstack{Dummy \\ First Offer}', r'\shortstack{Split \\ Gains from Trade}', r'\shortstack{Split \\ Gains from Trade }', r"\shortstack{ Bargaining \\ Time}", r"\shortstack{ Dummy \\ Agreement}", r"\shortstack{ Dummy \\ Player  Termination}", r"Efficiency", r"Efficiency"],
            varlabels=varlabels_regression,
            stars= {.1:'*',.05:'**',.01:'***'},
            modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'},
            addrows={"Level of Observation":["Player","Player","Player","Player","Negotiation","Negotiation","Negotiation","Negotiation"]}
            )


def regression_table_master_negotiators(data, path, varlabels_regression):
    
    df_master = data.dropna(subset=['first_offer_split', 'large_gains_from_trade_indicator'])
    df_master_time = df_master[df_master["time_inconsistency_dummy"] == 0]


    masters = smf.ols(formula='large_gains_from_trade_indicator ~ C(information_asymmetry):C(TA_costs) + first_offer + ultimatum_indicator + risk_elicitation_choice + time_preference_switching_points + bargaining_time_full_sec', data=df_master_time).fit(cov_type='cluster',
    cov_kwds={
        'groups': df_master_time['participant_code']
    })

    masters_with_offer_size = smf.ols(formula='large_gains_from_trade_indicator ~ C(information_asymmetry):C(TA_costs) + first_offer + first_offer_split + ultimatum_indicator + risk_elicitation_choice + time_preference_switching_points + bargaining_time_full_sec', data=df_master_time).fit(cov_type='cluster',
    cov_kwds={
        'groups': df_master_time['participant_code']
    })

    masters_with_fe = smf.ols(formula='large_gains_from_trade_indicator ~ C(information_asymmetry):C(TA_costs) + first_offer + first_offer_split + ultimatum_indicator + risk_elicitation_choice + time_preference_switching_points + bargaining_time_full_sec + C(participant_code)', data=df_master_time).fit(cov_type='cluster',
    cov_kwds={
        'groups': df_master_time['participant_code']
    })


    fix_pandas_append_error()

    pystout(models=[masters, masters_with_offer_size, masters_with_fe],
            file=path,
            exogvars=["C(TA_costs)[T.0.05]",  "first_offer", "first_offer_split", "C(information_asymmetry)[T.two-sided]:C(TA_costs)[0.0]", "C(information_asymmetry)[T.two-sided]:C(TA_costs)[0.05]", "ultimatum_indicator", "risk_elicitation_choice", "time_preference_switching_points", "bargaining_time_full_sec"],
            varlabels=varlabels_regression,
            addrows={"Individual Fixed Effects":["\\texttimes","\\texttimes","\\checkmark"]},
            modstat={'nobs':'Obs','rsquared_adj':'Adj. R\sym{2}'}
            )

    

    
def plot_buyer_first_offer_vs_valuation(df):


    df_t34 = df[(df["treatment"] == "T3") | (df["treatment"] == "T4")]
    df_t34_buyer_time = df_t34[(df_t34["participant_role"] == "Buyer") & (df_t34["time_inconsistency_dummy"] == 0)].dropna(subset=["offer_time_1"])
    
    

    # 1) Define your mapping
    label_map = {
        'T4': 'Costly, Buyer Only Uncertainty',
        'T3': 'No Cost, Buyer Only Uncertainty'
    }

    # 2) Create a new label column
    df_t34_buyer_time = df_t34_buyer_time.copy()
    df_t34_buyer_time['treatment_label'] = df_t34_buyer_time['treatment'].map(label_map)

    # Winsorize at 98%
    # Demean the 'offer_time_1' variable by individual level mean
    df_t34_buyer_time['demeaned_offer_time_1'] = df_t34_buyer_time.groupby('participant_code')['offer_time_1'].transform(lambda x: x - x.mean())
    
    # Winsorize the demeaned variable at 98%
    upper_bound = df_t34_buyer_time['demeaned_offer_time_1'].quantile(0.98)
    df_t34_buyer_time['demeaned_offer_time_1'] = df_t34_buyer_time['demeaned_offer_time_1'].clip( upper=upper_bound)

    set_plot_theme()

    # 3) Create figure + axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # 4) Scatter on the axes
    sns.scatterplot(
        x='valuation',
        y='demeaned_offer_time_1',
        data=df_t34_buyer_time,
        hue='treatment_label',
        alpha=0.7,
        ax=ax
    )

    # 5) Add 3rd‐order polynomial fits per label on the same axes
    for label in df_t34_buyer_time['treatment_label'].unique():
        sns.regplot(
            x='valuation',
            y='demeaned_offer_time_1',
            data=df_t34_buyer_time[df_t34_buyer_time['treatment_label'] == label],
            scatter=False,
            order=3,
            label=f'Fit for {label}',
            line_kws={"linewidth": 2},
            ax=ax
        )

    # 6) Tidy up
    ax.set_xlabel('Valuation')
    ax.set_ylabel('Demeaned First Offer Time (in Seconds)')
    ax.legend(title='', facecolor='white')

    finalize_plot(ax)

    return fig


def plot_two_sided_signaling(df):

    df_twosided = df[(df["treatment"] == "T2") | (df["treatment"] == "T1")]
    df_twosided_time = df_twosided[df_twosided["time_inconsistency_dummy"] == 0]

    # Demean
    df_twosided_time['demeaned_offer_time_1'] = df_twosided_time.groupby('participant_code')['offer_time_1'].transform(lambda x: x - x.mean())
    
    # Winsorize at 98%
    lower_bound = df_twosided_time['demeaned_offer_time_1'].quantile(0.01)
    upper_bound = df_twosided_time['demeaned_offer_time_1'].quantile(0.99)
    df_twosided_time['demeaned_offer_time_1'] = df_twosided_time['demeaned_offer_time_1'].clip(lower=lower_bound, upper=upper_bound)

    set_plot_theme()
    # Create the figure + axes
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='valuation',
        y='demeaned_offer_time_1',
        hue='participant_role',
        data=df_twosided_time,
        alpha=0.7,
        ax=ax
    )

    # Add 1st order polynomial fits for each participant role
    for role in df_twosided_time['participant_role'].unique():
        sns.regplot(
            x='valuation',
            y='demeaned_offer_time_1',
            data=df_twosided_time[df_twosided_time['participant_role'] == role],
            scatter=False,
            order=1,
            label=f'Fit for {role}',
            line_kws={"linewidth": 2},
            ax=ax
        )

    # Tidy up the axis labels and legend
    ax.set_xlabel('Valuation')
    ax.set_ylabel('Demeaned First Offer Time (in Seconds)')
    ax.legend(title='', facecolor='white')
    finalize_plot(ax)

    return fig


def plot_split_gains_by_treatment_role_grouped_t34(df, color_scheme):
    """
    Create a single plot with grouped boxplots for all treatment-role combinations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    color_scheme : list
        List of colors to use for the plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Filter for one-sided treatments (T3 and T4)
    df_one_sided = df[(df["treatment"] == "T3") | (df["treatment"] == "T4")]
    
    # Create a combined label for treatment-role
    df_one_sided = df_one_sided.copy()
    df_one_sided['treatment_role'] = df_one_sided['treatment'] + ' - ' + df_one_sided['participant_role']
    
    # Drop rows with missing split_gains_from_trade
    df_plot = df_one_sided.dropna(subset=['split_gains_from_trade'])
    
    # Winsorize the split_gains_from_trade at 98%
    from scipy.stats.mstats import winsorize
    df_plot['split_gains_from_trade'] = winsorize(df_plot['split_gains_from_trade'], limits=[0.01, 0.01])
    
    set_plot_theme()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create boxplot
    bp = ax.boxplot([df_plot[df_plot['treatment_role'] == label]['split_gains_from_trade'] 
                     for label in sorted(df_plot['treatment_role'].unique())],
                    labels=sorted(df_plot['treatment_role'].unique()),
                    patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], color_scheme[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize the plot
    ax.set_title('Split Gains from Trade by Treatment and Role')
    ax.set_ylabel('Split Gains from Trade')
    ax.set_xlabel('Treatment - Role')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45)
    
    # Adjust layout
    finalize_plot()

    
    return fig

def plot_buyer_payoff_vs_gains_from_trade(
        df: pd.DataFrame,
        color_scheme: list[str], 
        legend_elements,
        treatment_pattern: str = "T1|T2", 
      ):
    
    df_buyers = df[df["participant_role"].str.contains("Buyer") & df["treatment"].str.contains(treatment_pattern)]

    # Generate x values for the theoretical line
    x_values = np.linspace(0, 60, 1000)

    # Define a mapping from outcome labels to colors
    outcome_color_map = {
        'Player': color_scheme[0],               # Player Termination
        'Random_Termination': color_scheme[1],   # Computer Termination
        'acceptance': color_scheme[2],           # Highlight Acceptance
    }

    # Assign color values based on the outcome
    colors = df_buyers['bargaining_outcome'].map(outcome_color_map)

    set_plot_theme()

    fig, ax = plt.subplots(figsize=(12, 10))
     
    # Add the scatter plot with custom colors
    scatter = plt.scatter(
        df_buyers['valuation'],
        df_buyers['payoff'],
        c=colors,
        alpha=0.8
    )

 

    plt.plot(x_values, 0.5*x_values, color=color_scheme[3], linestyle='--', linewidth=2, label='Equal Split Line')
    legend_elements.append(mlines.Line2D([], [], color=color_scheme[3], linestyle='--', linewidth=2, label='Equal Split Line'))
     

    plt.xlabel('Gains from Trade')
    plt.ylabel('Buyer Payoff')

    plt.legend(handles=legend_elements, title="Bargaining Outcome")

    finalize_plot(ax)

    return fig

def plot_buyer_payoff_vs_gains_from_trade_t12(
        df: pd.DataFrame,
        color_scheme: list[str], 
        legend_elements: list[mlines.Line2D],
        treatment_pattern: str = "T1|T2"):
    
    df_buyers = df[df["participant_role"].str.contains("Buyer") & df["treatment"].str.contains(treatment_pattern)]

    # Generate x values for the theoretical line
    x_values = np.linspace(0, 60, 1000)

    # Define a mapping from outcome labels to colors
    outcome_color_map = {
        'Player': color_scheme[0],               # Player Termination
        'Random_Termination': color_scheme[1],   # Computer Termination
        'acceptance': color_scheme[2],           # Highlight Acceptance
    }

    # Assign color values based on the outcome
    colors = df_buyers['bargaining_outcome'].map(outcome_color_map)

    set_plot_theme()

    fig, ax = plt.subplots(figsize=(12,8))
     
    # Add the scatter plot with custom colors
    scatter = plt.scatter(
        df_buyers['gains_from_trade'],
        df_buyers['payoff'],
        c=colors,
        alpha=0.8
    )

    plt.plot(x_values, 0.5*x_values, color=color_scheme[3], linestyle='--', linewidth=2, label='Equal Split Line')
    legend_elements.append(mlines.Line2D([], [], color=color_scheme[3], linestyle='--', linewidth=2, label='Equal Split Line'))
     
    plt.xlim(-60, 60)  # Set x-axis limits

    plt.xlabel('Gains from Trade')
    plt.ylabel('Buyer Payoff')

    plt.legend(handles=legend_elements, title="Bargaining Outcome")

    finalize_plot(ax)

    return fig


def plot_offer_time_vs_valuation_demeaned_t34_buyer(
        df: pd.DataFrame,
        color_scheme: list[str],
        treatment_pattern: str = "T3|T4"):
    """
    Plot demeaned offer time vs valuation with individual fixed effects removed.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    color_scheme : list[str]
        List of colors to use for different treatments
    treatment_pattern : str
        Regex pattern to filter treatments (default: all treatments)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Filter data for buyers and specified treatments
    df_buyer = df[
        (df["participant_role"] == "Buyer") & 
        (df["treatment"].str.contains(treatment_pattern))
    ].copy()
    
    # Calculate the mean offer_time_1 for each participant
    participant_means = df_buyer.groupby('participant_label')['offer_time_1'].mean()
    
    # Subtract the participant mean from offer_time_1 to get the fixed effect corrected offer time
    df_buyer['offer_time_1_demeaned'] = df_buyer.apply(
        lambda row: row['offer_time_1'] - participant_means[row['participant_label']], axis=1
    )
    
    # Winsorize the offer_time_1_corrected at 98%
    upper_bound = df_buyer['offer_time_1_demeaned'].quantile(0.99)
    df_buyer['offer_time_1_corrected_wins'] = df_buyer['offer_time_1_demeaned'].clip(upper=upper_bound)
    
    set_plot_theme()
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create scatter plot with treatment colors
    sns.scatterplot(
        x='valuation', 
        y='offer_time_1_corrected_wins', 
        data=df_buyer, 
        hue='treatment', 
        alpha=0.7,
        ax=ax
    )
    
    # Add 3rd order polynomial regression line for each treatment
    for i, treatment in enumerate(df_buyer['treatment'].unique()):
        treatment_data = df_buyer[df_buyer['treatment'] == treatment]
        if len(treatment_data) > 0:
            sns.regplot(
                x='valuation', 
                y='offer_time_1_corrected_wins', 
                data=treatment_data, 
                scatter=False, 
                order=3, 
                label=f'Fit for {treatment}',
                line_kws={"linewidth": 2},
                ci=None,  # Disable standard errors
                ax=ax
            )
    
    # Add a dotted line at y = 0
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Valuation')
    ax.set_ylabel('Time of First Offer (demeaned by individual fixed effect)')
    ax.legend(
    title='Treatment',
    frameon=True,        # ensure the frame is drawn
    facecolor='white',   # background color
    edgecolor='black',   # border color
    framealpha=1         # full opacity
    )
    finalize_plot(ax)
    
    return fig


def plot_offer_time_vs_valuation_demeaned_t34(
        df: pd.DataFrame,
        color_scheme: list[str],
        treatment_pattern: str = "T1|T2"):
    """
    Plot demeaned offer time vs valuation with individual fixed effects removed.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    color_scheme : list[str]
        List of colors to use for different treatments
    treatment_pattern : str
        Regex pattern to filter treatments (default: all treatments)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Filter data for buyers and specified treatments
    df_buyer = df[
        (df["treatment"].str.contains(treatment_pattern))
    ].copy()
    
    # Calculate the mean offer_time_1 for each participant
    participant_means = df_buyer.groupby('participant_label')['offer_time_1'].mean()
    
    # Subtract the participant mean from offer_time_1 to get the fixed effect corrected offer time
    df_buyer['offer_time_1_demeaned'] = df_buyer.apply(
        lambda row: row['offer_time_1'] - participant_means[row['participant_label']], axis=1
    )
    
    # Winsorize the offer_time_1_corrected at 98%
    upper_bound = df_buyer['offer_time_1_demeaned'].quantile(0.99)
    df_buyer['offer_time_1_corrected_wins'] = df_buyer['offer_time_1_demeaned'].clip(upper=upper_bound)
    
    set_plot_theme()
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create scatter plot with treatment colors
    sns.scatterplot(
        x='valuation', 
        y='offer_time_1_corrected_wins', 
        data=df_buyer, 
        hue='participant_role', 
        alpha=0.4,
        ax=ax
    )
    
    # Add 3rd order polynomial regression line for each treatment
    for i, participant_role in enumerate(df_buyer['participant_role'].unique()):
        treatment_data = df_buyer[df_buyer['participant_role'] == participant_role]
        if len(treatment_data) > 0:
            sns.regplot(
                x='valuation', 
                y='offer_time_1_corrected_wins', 
                data=treatment_data, 
                scatter=False, 
                order=3, 
                label=f'Fit for {participant_role}',
                line_kws={"linewidth": 2},
                ci=None,  # Disable standard errors
                ax=ax
            )
    
    # Add a dotted line at y = 0
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Valuation')
    ax.set_ylabel('Time of First Offer (demeaned by individual fixed effect)')
    ax.legend(
    title='Role',
    frameon=True,        # ensure the frame is drawn
    facecolor='white',   # background color
    edgecolor='black',   # border color
    framealpha=1         # full opacity
    )
    finalize_plot(ax)
    
    return fig

def plot_offer_time_vs_valuation_demeaned_t12(
        df: pd.DataFrame,
        color_scheme: list[str],
        treatment_pattern: str = "T1|T2"):
    """
    Plot demeaned offer time vs valuation with individual fixed effects removed.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    color_scheme : list[str]
        List of colors to use for different treatments
    treatment_pattern : str
        Regex pattern to filter treatments (default: all treatments)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Filter data for buyers and specified treatments
    df_buyer = df[
        (df["treatment"].str.contains(treatment_pattern))
    ].copy()
    
    # Calculate the mean offer_time_1 for each participant
    participant_means = df_buyer.groupby('participant_label')['offer_time_1'].mean()
    
    # Subtract the participant mean from offer_time_1 to get the fixed effect corrected offer time
    df_buyer['offer_time_1_demeaned'] = df_buyer.apply(
        lambda row: row['offer_time_1'] - participant_means[row['participant_label']], axis=1
    )
    
    # Winsorize the offer_time_1_corrected at 98%
    upper_bound = df_buyer['offer_time_1_demeaned'].quantile(0.99)
    df_buyer['offer_time_1_corrected_wins'] = df_buyer['offer_time_1_demeaned'].clip(upper=upper_bound)
    
    set_plot_theme()
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create scatter plot with treatment colors
    sns.scatterplot(
        x='valuation', 
        y='offer_time_1_corrected_wins', 
        data=df_buyer, 
        hue='participant_role', 
        alpha=0.4,
        ax=ax
    )
    
    # Add 3rd order polynomial regression line for each treatment
    for i, participant_role in enumerate(df_buyer['participant_role'].unique()):
        treatment_data = df_buyer[df_buyer['participant_role'] == participant_role]
        if len(treatment_data) > 0:
            sns.regplot(
                x='valuation', 
                y='offer_time_1_corrected_wins', 
                data=treatment_data, 
                scatter=False, 
                order=3, 
                label=f'Fit for {participant_role}',
                line_kws={"linewidth": 2},
                ci=None,  # Disable standard errors
                ax=ax
            )
    
    # Add a dotted line at y = 0
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Valuation')
    ax.set_ylabel('Time of First Offer (demeaned by individual fixed effect)')
    ax.legend(
    title='Role',
    frameon=True,        # ensure the frame is drawn
    facecolor='white',   # background color
    edgecolor='black',   # border color
    framealpha=1         # full opacity
    )
    finalize_plot(ax)
    
    return fig
