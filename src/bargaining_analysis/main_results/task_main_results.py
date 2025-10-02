from src.bargaining_analysis.main_results.main_results import plot_buyer_payoff_vs_valuation, regression_table_symmetric_treatment,regression_table_asymmetric_treatment, regression_table_symmetric_treatment, regression_table_all_treatments, regression_table_master_negotiators, plot_buyer_first_offer_vs_valuation, plot_two_sided_signaling, plot_split_gains_by_treatment_role_grouped_t34, plot_buyer_payoff_vs_gains_from_trade_t12, plot_offer_time_vs_valuation_demeaned_t34_buyer, plot_offer_time_vs_valuation_demeaned_t12

from src.bargaining_analysis.main_results.main_plots import plot_boxplots_buyer_split_gains_from_trade, plot_boxplots_buyer_number_of_offers, compare_seller_split_gains_from_trade_by_treatment, plot_gains_from_trade_number_offers, plot_acceptance_rates, plot_last_offer_time_vs_valuation_t34, plot_last_offer_time_vs_valuation_t12, plot_split_gains_from_trade_vs_valuation_t34, plot_split_gains_from_trade_vs_valuation_t12, plot_mean_payoff_t3t4, plot_boxplots_seller_gains_from_trade, plot_buyer_share_as_function_of_surplus, plot_agreement_prob_by_gft_ma3, plot_cox_by_tacosts, plot_deviation_from_equal_split_sym, plot_logit_fit_for_agreement_sym,  plot_first_offer_regression, densities_by_first_offer_symno

from src.bargaining_analysis.main_results.main_regressions import model_gains_from_trade_number_offers, run_signaling_regressions, run_information_efficiency_regression, bargaining_power_regressions


from src.bargaining_analysis.main_results.main_tables import compute_acceptance_rates, compute_metrics_buyer_only_table

from src.bargaining_analysis.config import BLD, OVERLEAF_FIGURES, COLOR_SCHEME, OVERLEAF_TABLES, VARLABELS_REGRESSION, LEGEND_ELEMENTS_OUTCOME
from src.bargaining_analysis.main_results.descriptive_actions_table import calculate_bargaining_actions_values
from src.bargaining_analysis.helper import inject_values
import pandas as pd
import matplotlib.pyplot as plt
from pytask import task


#--------------------------------------------------------------
# Tables
#--------------------------------------------------------------


def task_make_table_buyer_only(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        out_path = OVERLEAF_TABLES / "buyer_only_equ_predictions.tex"
):
    print("Hello")
    df = pd.read_csv(depends_on)
    metrics = compute_metrics_buyer_only_table(df)
    inject_values(out_path, **metrics)

def task_regression_gains_from_trade_number_offers(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        varlabels_regression = VARLABELS_REGRESSION,
        out_path = OVERLEAF_TABLES / "gains_from_trade_number_offers.tex"
):
    df = pd.read_csv(depends_on)
    model_gains_from_trade_number_offers(df, varlabels_regression, out_path)


def task_write_regression_table_all_treatments(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        varlabels_regression = VARLABELS_REGRESSION,
        out_path = OVERLEAF_TABLES / "regression_table_all_treatments.tex"
):
    df = pd.read_csv(depends_on)
    regression_table_all_treatments(df, out_path, varlabels_regression)

    

def task_make_plotgrid_buyer_payoff_vs_valuation(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = [OVERLEAF_FIGURES / "buyer_payoff_vs_valuation_grid_T12.pdf", OVERLEAF_FIGURES / "buyer_payoff_vs_valuation_grid_T3.pdf", OVERLEAF_FIGURES / "buyer_payoff_vs_valuation_grid_T4.pdf"]
):
    df = pd.read_csv(depends_on)



    plot_t12 = plot_buyer_payoff_vs_valuation(df[df["treatment"].isin(["T1", "T2"])], COLOR_SCHEME, treatment_t4="no")
    plot_t3 = plot_buyer_payoff_vs_valuation(df[df["treatment"] == "T3"], COLOR_SCHEME, treatment_t4="no")
    plot_t4 = plot_buyer_payoff_vs_valuation(df[df["treatment"] == "T4"], COLOR_SCHEME, treatment_t4="yes")

    plot_t12.savefig(produces[0])
    plot_t3.savefig(produces[1])
    plot_t4.savefig(produces[2])



def task_write_descriptive_actions_table(
    depends_on = BLD / "data" / "merged_data_full_excluded.csv",
    out_path = OVERLEAF_TABLES / "descriptive_actions_table.tex"
):

    df = pd.read_csv(depends_on)
  
    descriptive_actions_values = calculate_bargaining_actions_values(df)
    inject_values(out_path, **descriptive_actions_values)

def task_inject_values_acceptance_table(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        out_path = OVERLEAF_TABLES / "comparison_efficiency.tex"
    ):
    df = pd.read_csv(depends_on)
    acceptance_rates = compute_acceptance_rates(df)
    inject_values(out_path, **acceptance_rates)  


def task_write_regression_table_symmetric_uncertainty(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        varlabels_regression = VARLABELS_REGRESSION,
        out_path = OVERLEAF_TABLES / "regression_table_symmetric_uncertainty.tex"
):
    df = pd.read_csv(depends_on)
    regression_table_symmetric_treatment(df, out_path, varlabels_regression)  


def task_write_regression_table_one_sided_treatment(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        varlabels_regression = VARLABELS_REGRESSION,
        out_path = OVERLEAF_TABLES / "regression_table_asymmetric_treatment.tex"
):
    df = pd.read_csv(depends_on)
    regression_table_asymmetric_treatment(df, out_path, varlabels_regression)

def task_run_signaling_regressions(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        out_path = OVERLEAF_TABLES / "signaling_regressions_table.tex"
):
    df = pd.read_csv(depends_on)
    run_signaling_regressions(df, out_path)

#--------------------------------------------------------------
# Plots
#--------------------------------------------------------------


def task_densities_by_first_offer_symno(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "densities_by_first_offer_symno.pdf"
):
    df = pd.read_csv(depends_on)
    plot = densities_by_first_offer_symno(df)
    plot.savefig(produces)


def task_plot_first_offer_regression_symno(
    depends_on = BLD / "data" / "merged_data_full_excluded.csv",
    produces = OVERLEAF_FIGURES / "coefplot_firstoffer_symno.pdf"
):
    df = pd.read_csv(depends_on)
    df_plot = df[(df["treatment"] == "T1")]
    plot = plot_first_offer_regression(df)
    plot.savefig(produces)

def task_plot_first_offer_regression_symcost(
    depends_on = BLD / "data" / "merged_data_full_excluded.csv",
    produces = OVERLEAF_FIGURES / "coefplot_firstoffer_symcost.pdf"
):
    df = pd.read_csv(depends_on)
    df_plot = df[(df["treatment"] == "T2")]
    plot = plot_first_offer_regression(df)
    plot.savefig(produces)

def task_plot_logit_fit_for_agreement_symno(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "agreement_frontier_symno.pdf"
):
    df = pd.read_csv(depends_on)
    plot_df = df[(df["treatment"] == "T1")]
    plot = plot_logit_fit_for_agreement_sym(plot_df)
    plot.savefig(produces)

def task_plot_logit_fit_for_agreement_symcost(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "agreement_frontier_symcost.pdf"
):
    df = pd.read_csv(depends_on)
    plot_df = df[(df["treatment"] == "T2")]
    plot = plot_logit_fit_for_agreement_sym(plot_df)
    plot.savefig(produces)

def task_plot_deviation_from_equal_splitsymcost(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "dev_from_equal_split_vs_gft_symcost.pdf"
):
    df = pd.read_csv(depends_on)
    df_plot = df[(df["participant_role"] == "Buyer") &
                (df["treatment"] == "T2") & 
                (df["gains_from_trade"] > 0)
                ].copy()
    plot = plot_deviation_from_equal_split_sym(df_plot, binning='width', num_bins=8, time_inconsistency=False)
    plot.savefig(produces)

for configuration in [(8, False), (12, False), (6, False), (8, True)]:

    num_bins, time_inconsistency = configuration

    @task
    def task_plot_deviation_from_equal_splitsymnocost(
            depends_on = BLD / "data" / "merged_data_full_excluded.csv",
            num_bins = num_bins,
            time_inconsistency = time_inconsistency,
            produces = OVERLEAF_FIGURES / f"deviation_from_equal_split_main_{num_bins}_bins_timeinconsistency_{time_inconsistency}.pdf"
    ):
        df = pd.read_csv(depends_on)
        df_plot = df[(df["participant_role"] == "Buyer") &
                (df["treatment"] == "T1") & 
                (df["gains_from_trade"] > 0)
                ].copy()
        plot = plot_deviation_from_equal_split_sym(df_plot, binning='width', num_bins=num_bins, time_inconsistency=time_inconsistency)
        plot.savefig(produces)




def task_plot_cox_by_tacosts(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "cox_by_tacosts.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_cox_by_tacosts(df)
    plot.savefig(produces)


def task_plot_agreement_prob_by_gft_ma3(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "agreement_prob_by_gft_ma3.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_agreement_prob_by_gft_ma3(df, binning="nearest")
    plot.savefig(produces)


def task_plot_buyer_share_as_function_of_surplus(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "buyer_share_as_function_of_surplus.pdf"
):
    
    df = pd.read_csv(depends_on)
    plot = plot_buyer_share_as_function_of_surplus(df)
    plot.savefig(produces)


def task_plot_boxplots_seller_gains_from_trade(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "boxplots_seller_gains_from_trade.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_boxplots_seller_gains_from_trade(df)
    plot.savefig(produces)

def task_plot_buyer_first_offer_vs_valuation(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "buyer_first_offer_vs_valuation_asymmetric.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_buyer_first_offer_vs_valuation(df)
    plot.savefig(produces)


def task_plot_two_sided_signaling(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "two_sided_signaling.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_two_sided_signaling(df)
    plot.savefig(produces)

def task_plot_split_gains_by_treatment_role_grouped_t34(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "split_gains_by_treatment_role_grouped_t34.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_split_gains_by_treatment_role_grouped_t34(df, COLOR_SCHEME)
    plot.savefig(produces)

def task_plot_buyer_payoff_vs_gains_from_trade_t12(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "buyer_payoff_vs_gains_from_trade_t12.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_buyer_payoff_vs_gains_from_trade_t12(df, COLOR_SCHEME, LEGEND_ELEMENTS_OUTCOME)
    plot.savefig(produces)


def task_plot_offer_time_vs_valuation_demeaned_t34_buyer(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "offer_time_vs_valuation_demeaned_t34_buyer.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_offer_time_vs_valuation_demeaned_t34_buyer(df, COLOR_SCHEME)
    plot.savefig(produces)


def task_plot_offer_time_vs_valuation_demeaned_t12(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "offer_time_vs_valuation_demeaned_t12.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_offer_time_vs_valuation_demeaned_t12(df, COLOR_SCHEME)
    plot.savefig(produces)

def task_plot_boxplots_buyer_split_gains_from_trade(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "boxplots_buyer_split_gains_from_trade_rounds.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_boxplots_buyer_split_gains_from_trade(df)
    plot.savefig(produces)

def task_plot_boxplots_buyer_number_of_offers(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "boxplots_buyer_number_of_offers_rounds.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_boxplots_buyer_number_of_offers(df)
    plot.savefig(produces)

def task_plot_compare_seller_split_gains_from_trade_by_treatment(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "compare_seller_split_gains_from_trade_by_treatment.pdf"
):
    df = pd.read_csv(depends_on)
    plot = compare_seller_split_gains_from_trade_by_treatment(df)
    plot.savefig(produces)


for treatments in [
    ["T1", "T2"],
    ["T3", "T4"]
    ]:

    @task
    def task_plot_gains_from_trade_number_offers(
            depends_on = BLD / "data" / "merged_data_full_excluded.csv",
            treatments = treatments,
            produces = OVERLEAF_FIGURES / f"gains_from_trade_number_offers_{treatments}.pdf"
    ):
        df = pd.read_csv(depends_on)
        plot = plot_gains_from_trade_number_offers(df, treatments=treatments)
        plot.savefig(produces)

def task_plot_acceptance_rates(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "acceptance_rates_plot.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_acceptance_rates(df)
    plot.savefig(produces)

def task_plot_last_offer_time_vs_valuation_t34(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "last_offer_time_vs_valuation_t34.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_last_offer_time_vs_valuation_t34(df)
    plot.savefig(produces)

def task_plot_last_offer_time_vs_valuation_t12(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "last_offer_time_vs_valuation_t12.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_last_offer_time_vs_valuation_t12(df)
    plot.savefig(produces)

def task_plot_split_gains_from_trade_vs_valuation_t34(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "split_gains_from_trade_vs_valuation_t34.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_split_gains_from_trade_vs_valuation_t34(df)
    plot.savefig(produces)

def task_plot_split_gains_from_trade_vs_valuation_t12(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "split_gains_from_trade_vs_valuation_t12.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_split_gains_from_trade_vs_valuation_t12(df)
    plot.savefig(produces)

def task_plot_mean_payoff_t3t4(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "mean_payoff_t3t4.pdf"
):
    df = pd.read_csv(depends_on)
    plot = plot_mean_payoff_t3t4(df)
    plot.savefig(produces)



#--------------------------------------------------------------
# Regression Tables
#--------------------------------------------------------------

def task_run_information_efficiency_regression(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        varlabels_regression = VARLABELS_REGRESSION,
        out_path = OVERLEAF_TABLES / "information_efficiency_regression.tex"
):
    df = pd.read_csv(depends_on)
    run_information_efficiency_regression(df, varlabels_regression, out_path)

def task_bargaining_power_regressions(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        varlabels_regression = VARLABELS_REGRESSION,
        out_path = OVERLEAF_TABLES / "bargaining_power_regressions.tex"
):
    df = pd.read_csv(depends_on)
    bargaining_power_regressions(df, varlabels_regression, out_path)






# def task_generate_first_mover_effect_table(
#         depends_on = BLD / "data" / "two_sided_without_TA.pkl",
#         produces = OVERLEAF_TABLES / "first_mover_table.tex"
# ):
#     df = pd.read_pickle(depends_on)
#     regress_first_mover(df, produces)