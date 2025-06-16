from src.bargaining_analysis.config import SRC, BLD, COLOR_SCHEME, OVERLEAF_FIGURES, OVERLEAF_TABLES
from src.bargaining_analysis.descriptive_data.descriptive_functions import plot_time_preference_switching_points, plot_ultimatum_offer_histogram, plot_risk_elicitation_choices, calculate_descriptive_table_values
from src.bargaining_analysis.helper import inject_values
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import pytask



def task_plot_comp_term_times(
    depends_on=SRC / "data" / "environment_data" / "termination_times_low_prob.pkl",
    color_scheme = COLOR_SCHEME,
    produces= OVERLEAF_FIGURES / "comp_term_times.pdf"):

    term_times = pd.read_pickle(depends_on)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(term_times)), term_times, marker='o', color=color_scheme[0])
    plt.xlabel('Round')
    plt.ylabel('Computer Termination Time in Seconds')
    sns.despine()
    
    plt.savefig(produces)


def task_plot_buyer_vals_onesided(
        depends_on = SRC / "data" / "environment_data" / "participant_data_1_groups_one-sided.pkl",
        produces = OVERLEAF_FIGURES / "buyer_vals_onesided.pdf",
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
        produces = OVERLEAF_FIGURES / "buyer_vals_twosided.pdf",
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


def task_write_descriptive_table(
    depends_on = BLD / "data" / "merged_data_full.csv"
):
    print("?t")
    df = pd.read_csv(depends_on)

    descriptive_table_values = calculate_descriptive_table_values(df)

    inject_values(
        OVERLEAF_TABLES / "descriptives_table.tex",
        **descriptive_table_values
    )




def task_inject_values_for_mistakes(
    depends_on = BLD / "data" / "merged_data_full_excluded.csv",
):
    df = pd.read_csv(depends_on)

    total_number_mistakes = df["mistake"].sum()
    total_number_negotiations = (len(df)) / 2
    average_session_duration = round(df["experiment_duration"].mean() / 60, 2)

    inject_values(
        OVERLEAF_TABLES.parents[1] / "main.tex",
        total_number_mistakes = total_number_mistakes,
        total_number_negotiations = total_number_negotiations,
        average_session_duration = average_session_duration
    )



def task_plot_time_preference_switching_points(
    depends_on = BLD / "data" / "merged_data_full_excluded.csv",
    produces = OVERLEAF_FIGURES / "time_preference_switching_points.pdf"
):
    print("?t")
    one_sided = pd.read_csv(depends_on)
    plot_time_preference_switching_points(one_sided)
    plt.savefig(produces)


def task_plot_ultimatum_offer_histogram(
    depends_on = BLD / "data" / "merged_data_full_excluded.csv",
    produces = OVERLEAF_FIGURES / "ultimatum_offer_histogram.pdf"
):
    print("?t")
    one_sided = pd.read_csv(depends_on)
    plot_ultimatum_offer_histogram(one_sided)
    plt.savefig(produces)


def task_plot_risk_elicitation_choices(
    depends_on = BLD / "data" / "merged_data_full_excluded.csv",
    produces = OVERLEAF_FIGURES / "risk_elicitation_choices.pdf"
):
    print("?t")
    one_sided = pd.read_csv(depends_on)
    plot_risk_elicitation_choices(one_sided)
    plt.savefig(produces)


        