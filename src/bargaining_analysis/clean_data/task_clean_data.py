from src.bargaining_analysis.clean_data.functions_clean_data import clean_data_asymmetric_TA, clean_symmetric_TA_data, apply_exclusion_criteria, clean_data_asymmetric_no_TA, clean_data_symmetric_no_TA, create_time_inconsistency_dummy, check_time_data_consistency, add_group_id_in_session, print_time_inconsistency_summary
from src.bargaining_analysis.helper import set_plot_theme       
from src.bargaining_analysis.config import SRC, BLD
import os
import glob

import pandas as pd


set_plot_theme()

def task_clean_data_symmetric_no_TA(
        depends_on = SRC / "data" / "main" / "symmetric_No_TA" / "june_17_data.csv",
        produces = BLD / "data" / "symmetric_no_TA.pkl"
):
    df = pd.read_csv(depends_on)
    print("\n Data Quality Check: for symmetric_no_TA \n --------------------------------")
    df = clean_data_symmetric_no_TA(df)
    print_time_inconsistency_summary(df)
  
    df.to_pickle(produces)


task_clean_data_asymmetric_TA_dependencies = [
   SRC / "data" / "main" / "asymmetric_TA" / "asymmetric_TA_1.csv",
    SRC / "data" / "main" / "asymmetric_TA" / "july_8.csv"
    ]


def task_clean_data_asymmetric_TA(
        depends_on = task_clean_data_asymmetric_TA_dependencies,
        produces = BLD / "data" / "asymmetric_TA.pkl"
):
    
    output = pd.DataFrame()
    for depends_on in depends_on:
        df = pd.read_csv(depends_on)
        print(f"\n Data Quality Check: for {depends_on} \n --------------------------------")
        clean_df = clean_data_asymmetric_TA(df)
        print_time_inconsistency_summary(clean_df)
        output = pd.concat([output, clean_df], ignore_index=True)


    output.to_pickle(produces)



task_clean_data_asymmetric_TA_dependencies = [
    SRC / "data" / "main" / "symmetric_TA" / "data_june_13.csv", 
    SRC / "data" / "main" / "symmetric_TA" / "july_9.csv"]
def task_clean_data_symmetric_TA(
        depends_on = task_clean_data_asymmetric_TA_dependencies,
        produces = BLD / "data" / "symmetric_TA.pkl"
):
    
    output = pd.DataFrame()
    for depends_on in depends_on:
        df = pd.read_csv(depends_on)
        print(f"\n Data Quality Check: for {depends_on} \n --------------------------------")
        df = clean_symmetric_TA_data(df)
        print_time_inconsistency_summary(df)
        output = pd.concat([output, df], ignore_index=True)
        
    output.to_pickle(produces)


task_clean_data_asymmetric_no_TA_dependencies = [
    SRC / "data" / "main" / "asymmetric_No_TA" / "june_16.csv",
    SRC / "data" / "main" / "asymmetric_No_TA" / "july_10.csv"]

def task_clean_data_asymmetric_no_TA(
        depends_on = task_clean_data_asymmetric_no_TA_dependencies,
        produces = BLD / "data" / "asymmetric_no_TA.pkl"
):

    output = pd.DataFrame()
    for depends_on in depends_on:
        df = pd.read_csv(depends_on)
        print(f"\n Data Quality Check: for {depends_on} \n --------------------------------")
        df = clean_data_asymmetric_no_TA(df) 
        print_time_inconsistency_summary(df)
        output = pd.concat([output, df], ignore_index=True)
    
    output.to_pickle(produces)


create_merged_data_dependencies = [
    BLD / "data" / "asymmetric_TA.pkl",
    BLD / "data" / "asymmetric_no_TA.pkl", 
    BLD / "data" / "symmetric_TA.pkl", 
    BLD / "data" / "symmetric_no_TA.pkl", 
]


def task_create_merged_data(
        depends_on = create_merged_data_dependencies,
        groupings_df_path = SRC / "data" / "environment_data" / "participant_data_4_groups_one-sided.pkl",
        produces = BLD / "data" / "merged_data_full.csv"
):

    dfs = [pd.read_pickle(p) for p in depends_on]
    merged_df = pd.concat(dfs, ignore_index=True)

    merged_df['negotiation_id'] = merged_df.groupby(['session_id', 'round', 'group_id_in_round']).ngroup()


    #Error Handling for TIme Inconsistencies
    merged_df["time_inconsistency_dummy"] = create_time_inconsistency_dummy(merged_df)
    check_time_data_consistency(merged_df)

    #Add group_id_in_session
    groupings_df = pd.read_pickle(groupings_df_path)
    merged_df = add_group_id_in_session(merged_df, groupings_df)
    merged_df["group_id"] = merged_df.groupby(['session_id', 'group_id_in_session']).ngroup()

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