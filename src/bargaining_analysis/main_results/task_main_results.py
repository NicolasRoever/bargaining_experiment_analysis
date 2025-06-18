from src.bargaining_analysis.main_results.main_results import plot_T4_buyer_payoff_vs_valuation, regression_table_symmetric_treatment, regression_table_one_sided_treatment, regression_table_asymmetric_treatment, regression_table_symmetric_treatment, regression_table_all_treatments, regression_table_master_negotiators
from src.bargaining_analysis.config import BLD, OVERLEAF_FIGURES, COLOR_SCHEME, OVERLEAF_TABLES, VARLABELS_REGRESSION
from src.bargaining_analysis.main_results.descriptive_actions_table import calculate_bargaining_actions_values
from src.bargaining_analysis.helper import inject_values
import pandas as pd




def task_write_regression_table_master_negotiators(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        varlabels_regression = VARLABELS_REGRESSION,
        out_path = OVERLEAF_TABLES / "regression_table_master_negotiators.tex"
):
    df = pd.read_csv(depends_on)
    regression_table_master_negotiators(df, out_path, varlabels_regression)

def task_write_regression_table_all_treatments(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        varlabels_regression = VARLABELS_REGRESSION,
        out_path = OVERLEAF_TABLES / "regression_table_all_treatments.tex"
):
    df = pd.read_csv(depends_on)
    regression_table_all_treatments(df, out_path, varlabels_regression)


def task_plot_T4_buyer_payoff_vs_valuation(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        produces = OVERLEAF_FIGURES / "T4_buyer_payoff_vs_valuation.pdf"
):
    df = pd.read_csv(depends_on)
    df_t4 = df[df["treatment"] == "T4"]
    plot = plot_T4_buyer_payoff_vs_valuation(df_t4, COLOR_SCHEME)
    plot.savefig(produces)
    


def task_write_descriptive_actions_table(
    depends_on = BLD / "data" / "merged_data_full_excluded.csv",
    out_path = OVERLEAF_TABLES / "descriptive_actions_table.tex"
):

    df = pd.read_csv(depends_on)
  
    descriptive_actions_values = calculate_bargaining_actions_values(df)
    inject_values(out_path, **descriptive_actions_values)


def task_write_regression_table_symmetric_uncertainty(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        varlabels_regression = VARLABELS_REGRESSION,
        out_path = OVERLEAF_TABLES / "regression_table_symmetric_uncertainty.tex"
):
    df = pd.read_csv(depends_on)
    regression_table_symmetric_treatment(df, out_path, varlabels_regression)  


def task_write_regression_table_one_sided_treatment(
        depends_on = BLD / "data" / "merged_data_full_excluded.csv",
        out_path = OVERLEAF_TABLES / "regression_table_one_sided_treatment.tex"
):
    df = pd.read_csv(depends_on)
    df_one_sided = df[(df["treatment"] == "T3") | (df["treatment"] == "T4")]
    regression_table_one_sided_treatment(df_one_sided, out_path)


# def task_generate_first_mover_effect_table(
#         depends_on = BLD / "data" / "two_sided_without_TA.pkl",
#         produces = OVERLEAF_TABLES / "first_mover_table.tex"
# ):
#     df = pd.read_pickle(depends_on)
#     regress_first_mover(df, produces)