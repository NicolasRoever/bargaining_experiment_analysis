from src.bargaining_analysis.config import SRC, BLD, COLOR_SCHEME
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pytask



COLOR_SCHEME=["#3c5488", "#e64b35", "#4dbbd5", "#00a087", "#f39b7f"]
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
sns.set_style("white")



def task_plot_comp_term_times(
    depends_on=SRC / "data" / "environment_data" / "termination_times_low_prob.pkl",
    color_scheme = COLOR_SCHEME,
    produces= BLD / "figures" / "comp_term_times.pdf"):

    term_times = pd.read_pickle(depends_on)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(term_times)), term_times, marker='o', color=color_scheme[0])
    plt.xlabel('Round')
    plt.ylabel('Computer Termination Time in Seconds')
    sns.despine()
    
    plt.savefig(produces)


def task_plot_buyer_vals_onesided(
        depends_on = SRC / "data" / "environment_data" / "participant_data_1_groups_one-sided.pkl",
        produces = BLD / "figures" / "buyer_vals_onesided.pdf",
        color_scheme = COLOR_SCHEME):

    one_sided = pd.read_pickle(depends_on)
    all_buyer_vals = one_sided[one_sided['Role'] == 'Buyer']['Valuation'].explode().values

    plt.figure(figsize=(12, 8))

    plt.figure(figsize=(12, 8))
    plt.hist(all_buyer_vals, bins=50, edgecolor='none', color=color_scheme[0] )
    plt.xlabel('Valuation')
    plt.ylabel('Frequency')
    plt.grid(False)
    sns.despine()
    
    plt.savefig(produces)


def task_plot_buyer_vals_twosided(
        depends_on = SRC / "data" / "environment_data" / "participant_data_1_groups_two-sided.pkl",
        produces = BLD / "figures" / "buyer_vals_twosided.pdf",
        color_scheme = COLOR_SCHEME):

    two_sided = pd.read_pickle(depends_on)
    all_buyer_vals = two_sided[two_sided['Role'] == 'Buyer']['Valuation'].explode().values
    seller_vals = two_sided[two_sided['Role'] == 'Seller']['Valuation']
    all_seller_vals = seller_vals.explode().values

    plt.figure(figsize=(12, 8))

    # Plot histogram for buyer valuations
    plt.hist(all_buyer_vals, bins=60, label='Buyer Valuations', edgecolor='none', color=color_scheme[0])

    # Plot histogram for seller valuations
    plt.hist(all_seller_vals, bins=60, alpha=0.5,  label='Seller Valuations', edgecolor='none', color=color_scheme[2])

    plt.xlabel('Valuation')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(False)
    sns.despine()
    
    plt.savefig(produces)

        