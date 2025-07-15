from src.bargaining_analysis.helper import set_plot_theme, finalize_plot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from pydantic import validate_call
import numpy as np
from itertools import product
import statsmodels.formula.api as smf

def plot_boxplots_buyer_split_gains_from_trade(df: pd.DataFrame):

    set_plot_theme()

    # 1. Filter buyers in T1–T4
    df_plot = df[
        df['participant_role'].eq('Buyer') &
        df['treatment'].isin(['T1','T2','T3','T4'])
    ].copy()

    # 2. Make T1+T2 vs T3+T4 groups
    df_plot['group'] = df_plot['treatment'].map({
        'T1':'T1 and T2', 'T2':'T1 and T2',
        'T3':'T3 and T4', 'T4':'T3 and T4'
    })

    # 3. Shift your round numbers so the min becomes 1
    min_round = df_plot['round'].min()          # e.g. 4
    df_plot['round_num'] = df_plot['round'] - min_round + 1
    n_rounds = df_plot['round_num'].max()       # should be 30 if rounds ran 4–33

    # 4. Draw the side-by-side boxplots
    fig, ax = plt.subplots(figsize=(14,7))
    ax = sns.boxplot(
        x='round_num',
        y='split_gains_from_trade',
        hue='group',
        data=df_plot,
        order=range(1, n_rounds+1),
        palette={'T1 and T2':'C0','T3 and T4':'C1'},
        dodge=True
    )
    ax.axhline(0.5, linestyle='--', color='gray', linewidth=1)

    # 5. Force the x-ticks to 1…n_rounds
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Split Gains From Trade', fontsize=12)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks(range(1, n_rounds+1))
    ax.set_xticklabels(range(1, n_rounds+1), rotation=45)
    ax.legend(title='Treatment Group')
    
    finalize_plot(ax)
    
    return fig

def plot_boxplots_buyer_number_of_offers(df_plot: pd.DataFrame):

    set_plot_theme()


    # 2. Make T1+T2 vs T3+T4 groups
    df_plot['group'] = df_plot['treatment'].map({
        'T1':'T1 and T2', 'T2':'T1 and T2',
        'T3':'T3 and T4', 'T4':'T3 and T4'
    })

    # 3. Shift your round numbers so the min becomes 1
    min_round = df_plot['round'].min()          # e.g. 4
    df_plot['round_num'] = df_plot['round'] - min_round + 1
    n_rounds = df_plot['round_num'].max()       # should be 30 if rounds ran 4–33

    # 4. Draw the side-by-side boxplots
    fig, ax = plt.subplots(figsize=(14,7))
    ax = sns.boxplot(
        x='round_num',
        y='number_of_offers',
        hue='group',
        data=df_plot,
        order=range(1, n_rounds+1),
        palette={'T1 and T2':'C0','T3 and T4':'C1'},
        dodge=True
    )

    # 5. Force the x-ticks to 1…n_rounds
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Split Gains From Trade', fontsize=12)
    ax.set_ylim(0, 10)
    ax.set_xticks(range(1, n_rounds+1))
    ax.set_xticklabels(range(1, n_rounds+1), rotation=45)
    leg = ax.legend(
        title='Treatment Group',
        frameon=True,     # turn on the frame
        fancybox=False    # straight corners
    )
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1)

    finalize_plot(ax)

    return fig

def compare_seller_split_gains_from_trade_by_treatment(df: pd.DataFrame):

    set_plot_theme()

    # 1. Filter for Sellers
    df_seller = df[df['participant_role'] == 'Seller'].copy()

    # 2. Create pooled treatment groups
    df_seller['pooled_treatment'] = df_seller['treatment'].map({
        'T1': 'T1 and T2', 'T2': 'T1 and T2',
        'T3': 'T3 and T4', 'T4': 'T3 and T4'
    })

    # 3. Draw 2 boxplots (one per pooled treatment group)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x='pooled_treatment',
        y='split_gains_from_trade',
        data=df_seller,
        order=['T1 and T2', 'T3 and T4'],
        palette='Set2'
    )
    plt.xlabel('Pooled Treatment', fontsize=12)
    plt.ylabel('Split Gains From Trade', fontsize=12)
    plt.title('Seller: Split Gains From Trade by Pooled Treatment', fontsize=14)
    plt.ylim(-0.5, 1.5)

    # 4. t-test between pooled treatment groups
    g1 = df_seller.loc[df_seller['pooled_treatment'] == 'T1 and T2', 'split_gains_from_trade']
    g2 = df_seller.loc[df_seller['pooled_treatment'] == 'T3 and T4', 'split_gains_from_trade']
    tstat, pval = ttest_ind(g1, g2, equal_var=False, nan_policy='omit')

    # Add p-value to the plot
    plt.text(0.5, 1.4, f'p-value of mean difference: {pval:.3f}', ha='center', fontsize=12, color='black')

    finalize_plot(ax)

    return fig

def plot_gains_from_trade_number_offers(df, treatments):
    """
    Scatterplot of Gains from Trade vs. Number of Offers colored by treatment,
    with a single pooled regression line across all selected treatments.
    """
    # Filter for specified treatments
    df_plot = df[df['treatment'].isin(treatments)]

    set_plot_theme()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot, colored by treatment
    sns.scatterplot(
        x='gains_from_trade',
        y='number_of_offers',
        data=df_plot,
        alpha=0.7,
        ax=ax
    )

    # Pooled regression line across all treatments
    sns.regplot(
        x='gains_from_trade',
        y='number_of_offers',
        data=df_plot,
        scatter=False,
        ax=ax,
        line_kws={
            'label': 'Pooled fit'
        }
    )

    # Axis labels and legend
    ax.set_xlabel('Gains from Trade (in Euros)')
    ax.set_ylabel('Number of Offers')
    plt.ylim(0, 15)
    plt.tight_layout()

    finalize_plot(ax)

    return fig

def plot_acceptance_rates(df: pd.DataFrame):
    """
    Plot acceptance rates for each treatment condition, showing the fraction of
    acceptance where gains from trade are non-negative.
    """

 # --- 1. Filter --------------------------------------------------------------
    df_filtered = df[df['gains_from_trade'] >= 0].copy()

    # --- 2. Human-readable treatment labels -------------------------------------
    labels_map = {
        'T1': 'No Cost,\nSymmetric Uncertainty',
        'T2': 'Costly,\nSymmetric Uncertainty',
        'T3': 'No Cost,\nBuyer-only Uncertainty',
        'T4': 'Costly,\nBuyer-only Uncertainty'
    }
    df_filtered['treatment_label'] = df_filtered['treatment'].map(labels_map)

    # --- 3. Acceptance rates and 95 % CIs ---------------------------------------
    grouped = df_filtered.groupby('treatment_label')['bargaining_outcome']
    p   = grouped.apply(lambda x: (x == 'acceptance').mean())   # proportion
    n   = grouped.count()
    se  = np.sqrt(p * (1 - p) / n)      # standard error
    ci95 = 1.96 * se                    # 95 % half-width (Wald)

    # --- 4. Plot ----------------------------------------------------------------
    set_plot_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(p.index, p, yerr=ci95, capsize=5)

    ax.set_ylabel('Fraction of Acceptance where  \n Gains from Trade $\geq$ 0')
    ax.set_xticklabels(p.index, rotation=45, ha='right')

    # --- Write the percentage inside each bar -----------------------------------
    for rect, frac in zip(bars, p):
        bar_mid_y = rect.get_height() / 2
        ax.text(
            rect.get_x() + rect.get_width() / 2,  # bar centre (x)
            bar_mid_y,                            # halfway up the bar (y)
            f'{frac*100:.1f} %',                  # e.g. “74.3 %”
            ha='center', va='center',
            color='white', fontweight='bold', fontsize=9
        )

    finalize_plot(ax)
    return fig


def plot_last_offer_time_vs_valuation_t34(df: pd.DataFrame):

    df_signal = df[(~np.isclose(df["id_in_group"], df['accepted_by_id_in_group'])) & 
    (df['participant_role'] == 'Buyer') &
    (df["bargaining_outcome"]== "acceptance") &
    (df['treatment'].isin(['T3', 'T4'])) & 
    (np.isclose(df['time_inconsistency_dummy'], 0))]


    set_plot_theme()
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1) scatter
    sns.scatterplot(
        x='valuation',
        y='last_offer_time',
        data=df_signal,
        hue='treatment',
        alpha=0.7
    )

    # 2) linear regression line (through all points)
    sns.regplot(
        x='valuation',
        y='last_offer_time',
        data=df_signal,
        scatter=False,
        label='Linear fit'
    )

    plt.xlabel('Valuation (Equals Gains from Trade)')
    plt.ylabel('Time of Accepted Offer (Seconds)')
    plt.legend(title='Treatment')
    finalize_plot(ax)
    
    return fig

def plot_last_offer_time_vs_valuation_t12(df: pd.DataFrame):

    df_signal = df[(~np.isclose(df["id_in_group"], df['accepted_by_id_in_group'])) & 
    (df['treatment'].isin(['T1', 'T2'])) & np.isclose(df['time_inconsistency_dummy'], 0)]

    set_plot_theme()

    fig, ax = plt.subplots(figsize=(8, 6))

    # scatter by role
    sns.scatterplot(
        x='valuation',
        y='last_offer_time',
        data=df_signal,
        hue='participant_role',
        alpha=0.4
    )

    # linear fit for Buyers
    sns.regplot(
        x='valuation',
        y='last_offer_time',
        data=df_signal[df_signal['participant_role'] == 'Buyer'],
        scatter=False,
        label='Buyer fit',
        order=2
    )

    # linear fit for Sellers
    sns.regplot(
        x='valuation',
        y='last_offer_time',
        data=df_signal[df_signal['participant_role'] == 'Seller'],
        scatter=False,
        label='Seller fit',
        order=2
    )

    plt.xlabel('Valuation')
    plt.ylabel('Time of Accepted Offer (Seconds)')
    plt.legend(title='Participant Role')
    finalize_plot(ax)
    
    return fig


def plot_split_gains_from_trade_vs_valuation_t34(df: pd.DataFrame):

    set_plot_theme()

    buyers_t3_t4 = df[
    (df['participant_role'] == 'Buyer') &
    (df['treatment'].isin(['T3', 'T4']))
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    # 1) scatter
    sns.scatterplot(
        x='valuation',
        y='split_gains_from_trade',
        data=buyers_t3_t4,
        hue='treatment',
        alpha=0.7
    )

    # 2) linear regression line (through all points)
    sns.regplot(
        x='valuation',
        y='split_gains_from_trade',
        data=buyers_t3_t4,
        scatter=False,
        label='Linear fit'
    )

    plt.xlabel('Buyer Valuation (Equals Gains from Trade)')
    plt.ylabel('Split of Gains from Trade')
    plt.legend(title='Treatment')
    plt.ylim(-.5, 1.5)
    finalize_plot(ax)
    
    return fig

def plot_split_gains_from_trade_vs_valuation_t12(df: pd.DataFrame):

    set_plot_theme()

    all_t1_t2 = df[
    (df['treatment'].isin(['T1', 'T2']))
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1) scatter
    sns.scatterplot(
        x='valuation',
        y='split_gains_from_trade',
        data=all_t1_t2,
        hue='participant_role',
        alpha=0.7
    )

    # linear fit for Buyers
    sns.regplot(
        x='valuation',
        y='split_gains_from_trade',
        data=all_t1_t2[all_t1_t2['participant_role'] == 'Buyer'],
        scatter=False,
        label='Buyer fit',
    )

    # linear fit for Sellers
    sns.regplot(
        x='valuation',
        y='split_gains_from_trade',
        data=all_t1_t2[all_t1_t2['participant_role'] == 'Seller'],
        scatter=False,
        label='Seller fit',
    )

    plt.xlabel('Buyer Valuation')
    plt.ylabel('Split of Gains from Trade')
    plt.legend(title='Treatment')
    plt.ylim(-.5, 1.5)
    finalize_plot(ax)
    
    return fig


def plot_mean_payoff_t3t4(df: pd.DataFrame):

    df_t34 = df[(df['treatment'].isin(['T3', 'T4'])) & (df['bargaining_outcome'] == 'acceptance')].copy()
    df_model_t34 = df_t34.dropna(subset=['payoff'])
    # 1. Build grid of all (treatment × role) combos
    roles = df_model_t34['participant_role'].unique()
    treatments = ['T3', 'T4']
    grid = pd.DataFrame(list(product(treatments, roles)),
                        columns=['treatment', 'participant_role'])

    # 2. Get model’s predicted means & CIs
    model_t34 = smf.ols(formula='payoff ~ C(treatment):C(participant_role)', 
                      data=df_model_t34).fit(
    cov_type='cluster',
     cov_kwds={'groups': df_model_t34['participant_code']}
                      )
    pred = model_t34.get_prediction(grid)
    pred_df = pred.summary_frame(alpha=0.05)

    grid = grid.assign(
        mean  = pred_df['mean'],
        lower = pred_df['mean_ci_lower'],
        upper = pred_df['mean_ci_upper']
    )

    # 3. Pivot so treatments are the index, roles the columns
    plot_df = grid.pivot(index='treatment',
                        columns='participant_role',
                        values=['mean', 'lower', 'upper'])

    # 4. Extract arrays
    means = plot_df['mean']      # DataFrame: index=treatments, cols=roles
    lower = plot_df['lower']
    upper = plot_df['upper']

    n_treat = len(treatments)
    n_roles = len(roles)
    x = np.arange(n_treat)
    width = 0.8 / n_roles  # total bar‐group width of 0.8

    set_plot_theme()
    # 5. Plot
    fig, ax = plt.subplots()

    for i, role in enumerate(roles):
        y = means[role].values
        err_low  = y - lower[role].values
        err_high = upper[role].values - y
        yerr = np.vstack([err_low, err_high])
        
        # center the group around each x[i]
        offset = (i - (n_roles-1)/2) * width
        ax.bar(x + offset, y, width,
            yerr=yerr, capsize=5,
            label=role)

    # 6. Tweak axes/legend
    ax.set_xticks(x)
    ax.set_xticklabels(['T3: No Transaction Costs', 'T4: Transaction Costs'])
    ax.set_ylabel('Average Payoff')
    ax.legend(title='Role')
    finalize_plot(ax)
    
    return fig


def plot_boxplots_seller_gains_from_trade(df: pd.DataFrame):
    """
    Boxplots of Seller Gains from Trade by Treatment.
    """
    # 1. Subset seller data
    seller = df[df['participant_role'] == 'Seller']

    # 2. Pull out the two groups
    g1 = seller[seller['treatment'].isin(['T1','T2'])]['split_gains_from_trade'].dropna()
    g2 = seller[seller['treatment'].isin(['T3','T4'])]['split_gains_from_trade'].dropna()

    # 3. Compute 1% and 99% bounds for each
    low1, high1 = g1.quantile([0.05, 0.95])
    low2, high2 = g2.quantile([0.05, 0.95])

    # 4. Winsorize (clip) each series
    g1_w = g1.clip(lower=low1, upper=high1)
    g2_w = g2.clip(lower=low2, upper=high2)

    set_plot_theme()
    # 5. Plot
    fig, ax = plt.subplots()
    ax.boxplot([g1_w, g2_w], labels=['T1 and T2', 'T3 and T4'], 
            medianprops={'color': sns.color_palette()[0] },)
    ax.set_ylabel('Split Gains from Trade (winsorized)')
    finalize_plot(ax)
    

    return fig

