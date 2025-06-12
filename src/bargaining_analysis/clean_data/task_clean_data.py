from src.bargaining_analysis.clean_data.functions_clean_data import clean_data_asymmetric_TA, clean_zero_TA_costs_two_sided_data
from src.bargaining_analysis.helper import set_plot_theme
from src.bargaining_analysis.config import SRC, BLD
import os
import glob

import pandas as pd


set_plot_theme()


def task_clean_data_asymmetric_TA(
        depends_on = SRC / "data" / "main" / "asymmetric_TA" / "asymmetric_TA_1.csv",
        produces = BLD / "data" / "one_sided_with_TA.pkl"
):
    df = pd.read_csv(depends_on)
    df = clean_data_asymmetric_TA(df)
    df.to_pickle(produces)



def task_create_merged_data(
        produces = BLD / "data" / "merged_data.csv"
):

    data_dir = produces.parent
    pkl_files = data_dir.glob("*.pkl")
   
    dfs = [pd.read_pickle(p) for p in pkl_files]
    
    if len(dfs) == 1:
        merged_df = dfs[0]
    else:
        merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(produces, index=False)



# def task_clean_two_sided_without_TA(
#         depends_on = SRC / "data" / "DataCollection130052025" / "all_apps_wide-2025-05-14.csv",
#         produces = BLD / "data" / "two_sided_without_TA.pkl"
# ):
#     df = pd.read_csv(depends_on)
#     df = clean_zero_TA_costs_two_sided_data(df)

#     df.to_pickle(produces)