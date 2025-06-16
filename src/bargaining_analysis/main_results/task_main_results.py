from src.bargaining_analysis.main_results.main_results import plot_T4_buyer_payoff_vs_valuation, regress_first_mover, regression_table_symmetric_treatment    
from src.bargaining_analysis.config import BLD, OVERLEAF_FIGURES, COLOR_SCHEME, OVERLEAF_TABLES
from src.bargaining_analysis.main_results.descriptive_actions_table import calculate_bargaining_actions_values
from src.bargaining_analysis.helper import inject_values
import pandas as pd


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
        out_path = OVERLEAF_TABLES / "regression_table_symmetric_uncertainty.tex"
):
    df = pd.read_csv(depends_on)
    df_symmetric = df[df["treatment"] == "T1" | df["treatment"] == "T2"]
    regression_table_symmetric_treatment(df_symmetric, out_path)


# def task_generate_first_mover_effect_table(
#         depends_on = BLD / "data" / "two_sided_without_TA.pkl",
#         produces = OVERLEAF_TABLES / "first_mover_table.tex"
# ):
#     df = pd.read_pickle(depends_on)
#     regress_first_mover(df, produces)