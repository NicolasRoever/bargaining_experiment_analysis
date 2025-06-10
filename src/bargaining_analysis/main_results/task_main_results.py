from src.bargaining_analysis.main_results.main_results import plot_T4_buyer_payoff_vs_valuation, regress_first_mover    
from src.bargaining_analysis.config import BLD, OVERLEAF_FIGURES, COLOR_SCHEME, OVERLEAF_TABLES

import pandas as pd


def task_plot_T4_buyer_payoff_vs_valuation(
        depends_on = BLD / "data" / "one_sided_with_TA.pkl",
        produces = OVERLEAF_FIGURES / "T4_buyer_payoff_vs_valuation.pdf"
):
    df = pd.read_pickle(depends_on)
    plot = plot_T4_buyer_payoff_vs_valuation(df, COLOR_SCHEME)
    plot.savefig(produces)


def task_generate_first_mover_effect_table(
        depends_on = BLD / "data" / "two_sided_without_TA.pkl",
        produces = OVERLEAF_TABLES / "first_mover_table.tex"
):
    df = pd.read_pickle(depends_on)
    regress_first_mover(df, produces)