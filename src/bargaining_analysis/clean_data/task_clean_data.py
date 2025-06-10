from src.bargaining_analysis.clean_data.functions_clean_data import clean_data, clean_zero_TA_costs_two_sided_data
from src.bargaining_analysis.config import SRC, BLD

import pandas as pd

def task_clean_one_sided_with_TA(
        depends_on = SRC / "data" / "Data_Collection_03_04_2025" / "raw_data.csv",
        produces = BLD / "data" / "one_sided_with_TA.pkl"
):
    df = pd.read_csv(depends_on)
    df = clean_data(df)
    df.to_pickle(produces)


def task_clean_two_sided_without_TA(
        depends_on = SRC / "data" / "DataCollection130052025" / "all_apps_wide-2025-05-14.csv",
        produces = BLD / "data" / "two_sided_without_TA.pkl"
):
    df = pd.read_csv(depends_on)
    df = clean_zero_TA_costs_two_sided_data(df)

    df.to_pickle(produces)