from src.bargaining_analysis.clean_data.functions_clean_data import clean_data_asymmetric_TA, clean_symmetric_TA_data, clean_zero_TA_costs_two_sided_data, apply_exclusion_criteria, clean_data_asymmetric_no_TA
from src.bargaining_analysis.helper import set_plot_theme       
from src.bargaining_analysis.config import SRC, BLD
import os
import glob

import pandas as pd


set_plot_theme()


def task_clean_data_asymmetric_TA(
        depends_on = SRC / "data" / "main" / "asymmetric_TA" / "asymmetric_TA_1.csv",
        produces = BLD / "data" / "asymmetric_TA.pkl"
):
    df = pd.read_csv(depends_on)
    df = clean_data_asymmetric_TA(df)
    df.to_pickle(produces)


def task_clean_data_symmetric_TA(
        depends_on = SRC / "data" / "main" / "symmetric_TA" / "data_june_13.csv",
        produces = BLD / "data" / "symmetric_TA.pkl"
):
    df = pd.read_csv(depends_on)
    df = clean_symmetric_TA_data(df)
    df.to_pickle(produces)


def task_clean_data_asymmetric_no_TA(
        depends_on = SRC / "data" / "main" / "asymmetric_No_TA" / "june_16.csv",
        produces = BLD / "data" / "asymmetric_no_TA.pkl"
):
    df = pd.read_csv(depends_on)
    df = clean_data_asymmetric_no_TA(df)
    df.to_pickle(produces)


create_merged_data_dependencies = [
    BLD / "data" / "asymmetric_TA.pkl",
    BLD / "data" / "asymmetric_no_TA.pkl", 
    BLD / "data" / "symmetric_TA.pkl"
]


def task_create_merged_data(
        depends_on = create_merged_data_dependencies,
        produces = BLD / "data" / "merged_data_full.csv"
):

    dfs = [pd.read_pickle(p) for p in depends_on]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(produces, index=False)


def task_apply_exclusion_criteria(
        depends_on = BLD / "data" / "merged_data_full.csv",
        produces = BLD / "data" / "merged_data_full_excluded.csv"
):
    df = pd.read_csv(depends_on)
    df = apply_exclusion_criteria(df)
    df.to_csv(produces, index=False)






# def task_clean_two_sided_without_TA(
#         depends_on = SRC / "data" / "DataCollection130052025" / "all_apps_wide-2025-05-14.csv",
#         produces = BLD / "data" / "two_sided_without_TA.pkl"
# ):
#     df = pd.read_csv(depends_on)
#     df = clean_zero_TA_costs_two_sided_data(df)

#     df.to_pickle(produces)