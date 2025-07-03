import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import pdb
import re
from pydantic import validate_call, ConfigDict


#------------------------------------------------------
# Exclusion Criteria Function
#------------------------------------------------------


def apply_exclusion_criteria(cleaned_data):
    """
    Apply the pre-registered exclusion criteria to the cleaned data.
    """

    #Remove Mistakes (as defined in the pre-registration)
    cleaned_data = filter_out_mistake_rows(cleaned_data)

    #Remove first 4 rounds (practice rounds and first real bargaining round)
    cleaned_data = cleaned_data[cleaned_data['round'] > 4] 

    return cleaned_data



#------------------------------------------------------
# High Level Data Cleaning Functions
# one for each treatment
#------------------------------------------------------

def clean_data_asymmetric_TA(raw_data):
    """
    Clean the raw data by reshaping it and adding the termination times.
    """
    df_clean = clean_data(raw_data)
    
    return df_clean


def clean_symmetric_TA_data(raw_data):
    """
    Clean the raw data by reshaping it and adding the termination times.
    """

    df_clean = clean_data(raw_data)

    return df_clean


def clean_data_asymmetric_no_TA(raw_data):
    """
    Clean the raw data by reshaping it and adding the termination times.
    """

    df_clean = clean_data(raw_data)

    return df_clean

def clean_data_symmetric_no_TA(raw_data):
    """
    Clean the raw data by reshaping it and adding the termination times.
    """

    df_clean = clean_data(raw_data)

    return df_clean


#------------------------------------------------------
# Large Main Data Cleaning Function used for all treatments
#------------------------------------------------------



def clean_data(raw_data):
    """
    Clean the raw data by reshaping it and adding all the columns needed for the analysis.
    """

    df_clean = pd.DataFrame()
    df_long_wide = reshape_raw_bargaining_data(raw_data)


    #Descriptive Variables

    df_clean['participant_id_in_session'] = df_long_wide['participant.id_in_session']
    df_clean['participant_label'] = df_long_wide['participant.label']
    df_clean["participant_code"] = df_long_wide["participant.code"]
    df_clean['participant_role'] = df_long_wide['participant.role_in_game']
    df_clean["group_id_in_round"] = df_long_wide["id_in_subsession"]
    df_clean["id_in_group"] = df_long_wide["id_in_group"]
    df_clean['round'] = df_long_wide['round']
    df_clean["session_id"] = df_long_wide["session.code"]
    df_clean["information_asymmetry"] = df_long_wide["session.config.information_asymmetry"]
    df_clean["TA_costs"] = df_long_wide["session.config.transaction_costs"]
    df_clean["treatment"] = determine_treatment_category(df_clean["information_asymmetry"], df_clean["TA_costs"])
    df_clean["subsession.is_practice_round"] = df_long_wide["subsession.is_practice_round"]
    df_clean["Role_Seller"] = np.where(df_clean["participant_role"] == "Seller", 1, 0)
    check_participant_code_uniqueness(df_clean)

    #Transaction Costs
    df_clean["cumulated_TA_costs"] = df_long_wide["cumulated_TA_costs"]


    #Time Variables
    df_clean["acceptance_time_raw"] = df_long_wide["acceptance_time"]
    df_clean["termination_time_raw"] = df_long_wide["termination_time"].astype('Float64')
    df_clean["bargain_start_time_unix"] = df_long_wide["bargain_start_time"].astype('Float64')
    df_clean = add_acceptance_time_sec(df_clean)
    df_clean = add_termination_time_sec(df_clean)
    df_clean["bargaining_time_full_sec"] = df_clean["acceptance_time_sec"].combine_first(df_clean["termination_time_sec"])

    #Bargaining Outcome
    df_clean["termination_mode"] = df_long_wide["termination_mode"]
    df_clean['bargaining_outcome'] = (
    df_clean['termination_mode']
    .fillna('acceptance')
    )


    #Individual Offers and Offer Times
    df_clean = add_offer_columns(
        raw_df=df_long_wide,
        df_clean=df_clean,
        list_col='amount_proposed_list',
        prefix='offer'
    )
    
    df_clean = add_offer_columns(
    raw_df=df_long_wide,
    df_clean=df_clean,
    list_col='offer_time_list',
    prefix='offer_time'
    )

    
    df_clean = adjust_offer_times(df_clean) #Fix for the first two sessions
    df_clean["offer_amount_list"] = df_long_wide["amount_proposed_list"]
    df_clean["offer_time_list"] = df_long_wide["offer_time_list"]
    df_clean["last_offer"] = obtain_last_offer(df_clean)
    df_clean["last_offer_time"] = last_offer_time(df_clean)
    offer_time_columns = [col for col in df_clean.columns if col.startswith("offer_time_")]
    df_clean["number_of_offers"] = df_clean[offer_time_columns].count(axis=1)


    #Acceptance Information
    df_clean["accepted_by_id_in_group"] = df_long_wide["accepted_by"].astype('Int64')
    df_clean["agreement_dummy"] = np.where(
        df_clean["bargaining_outcome"] == "acceptance",
        1,
        0
    )

    #Valuation and Payoff
    df_clean["deal_price"] = df_long_wide["deal_price"]
    #df_clean["deal_price"] = correct_deal_price(df_clean)
    df_clean["valuation"] = df_long_wide["valuation"].astype('Int64')
    df_clean['payoff'] = calculate_payoff(df_clean)
    df_clean["id_in_group"] = pd.to_numeric(df_long_wide["id_in_group"], errors='coerce')
    df_clean["group_id_in_round"] = df_long_wide["id_in_subsession"]
    df_clean["terminated_by_id_in_group"] = df_long_wide["terminated_by"].astype('Int64')

    df_clean["player_terminated"] =  np.where(
        df_clean['terminated_by_id_in_group'].isna(),
        0,
        1)
    
    df_clean['relative_valuation'] = calculate_relative_valuation(df_clean)

    # define your bins and labels
    bins = [-np.inf, 7.72, 21.40, np.inf]
    labels = ['Low', 'Medium', 'High']

    # create the new column
    df_clean['valuation_bucket'] = pd.cut(
        df_clean['valuation'],
        bins=bins,
        labels=labels,
        right=True,        # intervals are (...,] by default
        include_lowest=True
    )

    #Gains from Trade
    df_clean["gains_from_trade"] = calculate_gains_from_trade(df_clean)
    df_clean["split_gains_from_trade"] = calculate_split_gains_from_trade(df_clean)
    df_clean["gains_from_trade_dummy"] = np.where(
        df_clean["gains_from_trade"] >= 0,
        1,
        0
    )
    df_clean["large_gains_from_trade_indicator"] = np.where(
        df_clean["gains_from_trade"] >= 0.8,
        1,
        0
    )

    df_clean["majority_gains_from_trade_indicator"] = (
    df_clean["split_gains_from_trade"]
      .ge(0.5)           # yields True/False/NA
      .astype("Int64")   # maps True→1, False→0, NA→<NA>
    )

    df_clean['positive_gains_symmetric_treatment'] = (
    df_clean['gains_from_trade_dummy'] *
    (df_clean['information_asymmetry'] == 'two-sided').astype(int)
    )

    df_clean["small_gains_from_trade_indicator"] = np.where(
        (df_clean["gains_from_trade"] <= 10) & (df_clean["gains_from_trade"] >= 0),
        1,
        0
    )

    #First Offer Split
    df_clean["first_offer"] = determine_first_offer(df_clean)
    df_clean["first_offer_split"] = calculate_first_offer_split(df_clean)

    #Efficiency
    df_clean["efficiency"] = calculate_efficiency(df_clean)

    #Add Dummy for information asymmetry
    df_clean["seller_info_public"] = np.where(
        (df_clean["information_asymmetry"] == "one-sided") & (df_clean["participant_role"] == "Seller"),
        1,
        0
    )

   
    #Add Ultimatum Offer

    df_clean["ultimatum_offer"] = map_round_variable(df_long_wide, df_clean, "ultimatum_offer")
    df_clean["ultimatum_indicator"] = np.where(
        df_clean["ultimatum_offer"]<= 25,
        1,
        0
    )

    # Add Risk Choice
    df_clean["risk_elicitation_choice"] = map_round_variable(df_long_wide, df_clean, "risk_elicitation_choice")


    #Add time preferences
    df_clean = add_time_row_columns(df_long_wide, df_clean)
    check_monotonicity_for_time_preferences(df_clean)
    df_clean["time_preference_switching_points"] = find_time_preference_switching_points(df_clean)

    #Add Bargain Beginning and End
    df_clean["experiment_start_time"] = map_round_variable(
        df_long_wide, df_clean, "experiment_start_time", round_number=1)
    df_clean["experiment_end_time"] = map_round_variable(df_long_wide, df_clean, "experiment_end_time")
    df_clean["experiment_duration"] = df_clean["experiment_end_time"] - df_clean["experiment_start_time"]


    #Add Age
    df_clean["age"] = map_round_variable(df_long_wide, df_clean, "age")
    #Add Gender
    df_clean["gender"] = map_round_variable(df_long_wide, df_clean, "gender")

    #Add Strategy Question
    df_clean["strategy_answer"] = map_round_variable(df_long_wide, df_clean, "question_strategy", dtype="str", round_number=1)
    
    #Calculate Mistakes
    df_clean["mistake"] = np.where(
        (-df_clean["cumulated_TA_costs"])  > df_clean["payoff"],
        1,
        0
    )


    #Remove Practice Rounds
    df_clean = df_clean[np.isclose(df_clean["subsession.is_practice_round"], 0)]
    
    return df_clean


#------------------------------------------------------
# Lowest Level Data Cleaning Functions Generating Individual Columns
#------------------------------------------------------




def reshape_raw_bargaining_data(raw_data):
    """
    Reshape bargaining data from wide to long format.
    
    Parameters:
    -----------
    raw_data : pandas.DataFrame
        The raw bargaining data in wide format
        
    Returns:
    --------
    pandas.DataFrame
        Reshaped data in long format with one row per player per round
    """
    # 1. Define ID variables to keep
    id_vars = ['participant.id_in_session', 'participant.label', 'participant.role_in_game', 'session.code','participant.code', 'session.config.information_asymmetry', 'session.config.transaction_costs']

    # 2. Identify round-specific measures
    value_vars = [col for col in raw_data.columns if col.startswith('bargain_live')]
    
    # 3. Melt into long form
    df_long = raw_data.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='variable',
        value_name='value'
    )
    
    # 4. Extract round number and field from variable names
    regex = r'bargain_live\.(?P<round>\d+)\.(?P<field>.+)'
    extracted = df_long['variable'].str.extract(regex)
    df_long = df_long.join(extracted)
    
    # 5. Convert round to integer
    df_long['round'] = df_long['round'].astype(int)
    
    # 6. Create unique column names for each measure
    df_long['var_field'] = df_long['field']
    
    # 7. Pivot to create one column per measure
    df_long_wide = (
        df_long
        .pivot_table(
            index=id_vars + ['round'],
            columns='var_field',
            values='value',
            aggfunc='first'
        )
        .reset_index()
    )
    
    # 8. Sort by player and round
    df_long_wide = df_long_wide.sort_values(by=id_vars + ['round'])
    
    # 9. Clean up column names
    df_long_wide.columns = [col.replace('player.', '').replace('group.', '') 
                          for col in df_long_wide.columns]
    
    
    return df_long_wide


#------------------------------------------------------
# Lowest Level Data Cleaning Functions Generating Individual Columns
#------------------------------------------------------
    
    


def calculate_gains_from_trade(df: pd.DataFrame) -> pd.Series:
    # +1 for Buyer, -1 for Seller
    weight = df['participant_role'].map({'Buyer': 1, 'Seller': -1})
    # multiply valuations by those weights
    weighted_vals = df['valuation'] * weight
    # sum weighted_vals within each (round, group_id_in_round) and align back to rows
    gains = weighted_vals.groupby(
        [df['round'], df['group_id_in_round']]
    ).transform('sum')
    # cast to pandas nullable Int64 to match your test
    return gains.astype('Int64')



def fix_acceptance_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the acceptance time by adding the bargaining start time.
    """
    pass

def add_offer_columns(raw_df: pd.DataFrame,
                      df_clean: pd.DataFrame,
                      list_col: str = 'amount_proposed_list',
                      prefix: str = 'offer') -> pd.DataFrame:
    """
    Given your raw-format data (with stringified lists of offers) and
    your partially cleaned long-form df, parse out the lists and
    append offer_1, offer_2, … columns to df_clean.
    
    Parameters
    ----------
    raw_df : pd.DataFrame
        The original wide-format DataFrame containing `list_col` as strings
        like "[15.00000000001, 29.00000000001, …]".
    df_clean : pd.DataFrame
        Your long-form DataFrame (must be aligned/indexed the same as raw_df).
    list_col : str
        Column in raw_df holding the string-lists.
    prefix : str
        Base name for new columns; will produce prefix_1, prefix_2, …
    
    Returns
    -------
    pd.DataFrame
        A copy of df_clean with new offer_X columns filled with the parsed
        numbers (and NaN where there was no Xth offer).
    """
    # make a working copy
    out = df_clean.copy()
    
    # parse the string → real list of floats
    parsed = (
        raw_df[list_col]
        .fillna('[]')
        .apply(ast.literal_eval)
        .apply(lambda lst: [float(x) for x in lst])
    )
    
    # decide how many columns we need
    max_n = int(parsed.map(len).max())
    
    # for each possible slot, pull it out (or NaN)
    for i in range(max_n):
        col = f"{prefix}_{i+1}"
        out[col] = parsed.map(lambda lst: lst[i] if i < len(lst) else np.nan)
    
    return out

# Define the piecewise function
def equilibrium_payoff(x, c=0.05, r=0.01):
    if x <= 7.72:
        return 0
    elif x >= 21.40:
        return x/2
    else:
        b = x
        return ((r*b + 2*c)/(r*21.40 + 2*c)) * (b/2 + c/r) - c/r


def determine_first_offer(df: pd.DataFrame) -> pd.Series:
    """
    For each group (defined by round and group_id_in_round), return a Series indicating
    which player made the first offer based on offer_time_1:
      - 1 if this player made the first offer
      - 0 otherwise
      - <NA> if offer times are not comparable (both missing)

    Returns a Series aligned with df's index, in the same order as df.
    """
    def label_group(group: pd.DataFrame) -> pd.Series:
        t = pd.to_numeric(group['offer_time_1'], errors='coerce')

        # both missing → all <NA>
        if t.isna().all():
            return pd.Series([pd.NA] * len(t), index=group.index, dtype='Int64')
        # exactly one non‐missing → that one gets 1
        if t.isna().sum() == 1:
            return t.notna().astype('Int64')
        # two or more non‐missing → min gets 1, rest 0
        result = pd.Series(0, index=group.index, dtype='Int64')
        result[t.idxmin()] = 1
        return result

    # 1) group without sorting the keys, keep only the values from label_group
    labels = (
        df
        .groupby(['round', 'group_id_in_round'], sort=False, group_keys=False)
        .apply(label_group)
    )

    # 2) re‐order to match exactly df's index ordering
    return labels.loc[df.index]

def last_offer_time(df: pd.DataFrame) -> pd.Series:
    """
    For each row, find the last non-null value among columns
    offer_time_1, offer_time_2, ..., offer_time_n. Ignores any
    similarly-named columns that don't end in a number.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing your offer_time_1, offer_time_2, … columns.

    Returns
    -------
    pd.Series
        The last non-null offer time per row (or NaN if none exists).
    """
    # 1) Compile regex to match only suffixes that are digits
    pattern = re.compile(r'^offer_time_(\d+)$')

    # 2) Filter & sort by the numeric suffix
    offer_cols = sorted(
        (col for col in df.columns if pattern.match(col)),
        key=lambda col: int(pattern.match(col).group(1))
    )

    # 3) Forward-fill across each row, then take the rightmost column
    filled = df[offer_cols].ffill(axis=1)
    return filled.iloc[:, -1]



def obtain_last_offer(df: pd.DataFrame) -> pd.Series:
    # 1. Grab and sort all offer columns by their numeric suffix
    offer_cols = sorted(
        df.filter(regex=r"^offer_\d+$").columns,
        key=lambda c: int(c.split("_", 1)[1])
    )
    offers = df[offer_cols]
    # 2. Reverse the column order so the "last" becomes first,
    #    then forward-fill across columns and take the first column.
    return offers.iloc[:, ::-1] \
                 .bfill(axis=1) \
                 .iloc[:, 0]


def add_row_with_buyer_valuation(df: pd.DataFrame) -> pd.Series:
    """
    Add a row with the buyer valuation to the dataframe.
    """
    pass



def filter_out_mistake_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out all rows for any (round, group_id_in_round) where the payoff is larger than termination
    """
    # keep only those groups whose minimum gain is >= -15
    cleaned = (
        df
        .groupby(['round', 'group_id_in_round'])
        .filter(lambda g: g['payoff'].min() >= -15)
    )
    # reset_index so the result matches your expected
    return cleaned.reset_index(drop=True)


def add_time_row_columns(raw_df: pd.DataFrame,
                        df_clean: pd.DataFrame,
                        prefix: str = 'time_row') -> pd.DataFrame:
    """
    Add time row columns (1-6) from round 33 to all rounds for each participant.
    
    Parameters
    ----------
    raw_df : pd.DataFrame
        The original wide-format DataFrame containing time_row_1 through time_row_6 columns
    df_clean : pd.DataFrame
        Your long-form DataFrame (must be aligned/indexed the same as raw_df)
    prefix : str
        Base name for the columns (default: 'time_row')
    
    Returns
    -------
    pd.DataFrame
        A copy of df_clean with the time row columns added for all rounds
    """
    out = df_clean.copy()
    
    # Get time row values from round 33
    time_row_map = (
        raw_df
        .loc[raw_df["round"] == 33]
        .set_index("participant.id_in_session")
    )
    
    # Add each time row column by mapping from round 33 values
    for i in range(1, 7):
        col = f"{prefix}_{i}"
        out[col] = out["participant_id_in_session"].map(time_row_map[col]).astype('Float64')
    
    return out



def find_time_preference_switching_points(df: pd.DataFrame) -> pd.Series:
    """
    Scan each row's time_row_1 … time_row_6 columns in order,
    return the index (1–6) of the first value equal to 2.0.
    If no 2.0 appears in that row, return 7.0.
    """
    # grab and sort the six time_row columns
    cols = sorted(
        (c for c in df.columns if c.startswith("time_row_")),
        key=lambda c: int(c.rsplit("_", 1)[-1])
    )
    n = len(cols)
    # for each row, find first 2.0 or default to n+1
    def _first2(row):
        for i, col in enumerate(cols, start=1):
            if row[col] == 2.0:
                return int(i)
        return int(n + 1)
    return df.apply(_first2, axis=1)


def check_monotonicity_for_time_preferences(df: pd.DataFrame) -> None:
    """
    Ensures each row's time_row_1 … time_row_6 values switch from 1→2 at most once,
    and never switch back from 2→1. Raises ValueError if any row violates this.
    """
    # pick out & sort the six time_row columns
    cols = sorted(
        (c for c in df.columns if c.startswith("time_row_")),
        key=lambda c: int(c.rsplit("_", 1)[-1])
    )

    for idx, row in df[cols].iterrows():
        # drop any NAs and keep the sequence of actual 1s/2s
        vals = [v for v in row.tolist() if pd.notna(v)]
        transitions = 0
        for prev, curr in zip(vals, vals[1:]):
            if prev == 1 and curr == 2:
                transitions += 1
                if transitions > 1:
                    raise ValueError(
                        f"Row {idx!r} has more than one 1→2 transition: {vals}"
                    )
            elif prev == 2 and curr == 1:
                raise ValueError(
                    f"Row {idx!r} switches back from 2→1: {vals}"
                )


def map_round_variable(raw_df: pd.DataFrame, 
                        df_clean: pd.DataFrame, 
                        variable: str, 
                        dtype: str = 'Float64', 
                        round_number: int = 33) -> pd.Series:
    """
    Maps a variable from a single round to all rounds for each participant.
    
    Parameters
    ----------
    raw_df : pd.DataFrame
        The original wide-format DataFrame containing the variable
    df_clean : pd.DataFrame
        Your long-form DataFrame to add the mapped variable to
    variable : str
        Name of the variable to map from round 33
    dtype : str, optional
        Data type to convert the mapped values to (default: 'Float64')
    
    Returns
    -------
    pd.Series
        The mapped variable as a Series with the same index as df_clean
    """
    # Check if the variable exists in the DataFrame
    if variable not in raw_df.columns:
        return pd.Series(pd.NA, index=df_clean.index, dtype=dtype)
    
    # Create mapping from round 33
    var_map = (
        raw_df
        .loc[raw_df["round"] == round_number]
        .set_index("participant.id_in_session")[variable]
    )
    
    # Map to all rounds and convert to specified dtype
    return df_clean["participant_id_in_session"].map(var_map).astype(dtype)


def determine_treatment_category(information_asymmetry: pd.Series,
                                 TA_costs: pd.Series) -> pd.Series:
    # make an output Series with the same index
    result = pd.Series(index=information_asymmetry.index, dtype='object')
    
    # two-sided vs one-sided masks
    two_sided = information_asymmetry == "two-sided"
    one_sided = information_asymmetry == "one-sided"
    
    # fill with .loc on that index
    result.loc[two_sided & (TA_costs == 0)] = "T1"
    result.loc[two_sided & (TA_costs  > 0)] = "T2"
    result.loc[one_sided & (TA_costs == 0)] = "T3"
    result.loc[one_sided & (TA_costs  > 0)] = "T4"
    
    return result

def add_acceptance_time_sec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds df['acceptance_time_sec'].

    Note: We had an error in how we computed the acceptance time in the first two experiments.
    This is why we need this special calculation for these two sessions.

    We recover the acceptance time from the transaction costs which are 5 cents per second.
    """
    special_sessions_mask = df["session_id"].isin({"o1rrqa15", "1a8klj6g"})


    df["acceptance_time_sec"] = np.where(
        special_sessions_mask,
        df["cumulated_TA_costs"] / 0.05,
        df["acceptance_time_raw"]
    )

    return df


def add_termination_time_sec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds df['termination_time_sec'].

    Note: We had an error in how we computed the termination time in the first two experiments.
    This is why we need this special calculation for these two sessions.
    We recover the termination time from the transaction costs which are 5 cents per second.

    """
    
    mask = (
        df["session_id"].isin({"o1rrqa15", "1a8klj6g"})
    )

    df["termination_time_sec"] = np.where(
        mask,
        df["cumulated_TA_costs"] / 0.05,
        df["termination_time_raw"]
    )

    return df


def calculate_efficiency(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the efficiency of the bargaining process.
    
    Efficiency = 1 if:
      - gains_from_trade >= 0  AND  bargaining_outcome == 'acceptance', OR
      - gains_from_trade <= 0  AND  bargaining_outcome in ['Player', 'Random_Termination']
    Otherwise 0.
    """
    df = pd.DataFrame(df)
    
    # Case 1: non‐negative gains and acceptance
    cond1 = (
        (df['gains_from_trade'] >= 0) &
        (df['bargaining_outcome'] == 'acceptance')
    )
    
    # Case 2: non‐positive gains and a "no‐deal" termination
    cond2 = (
        (df['gains_from_trade'] <= 0) &
        (df['bargaining_outcome'].isin(['Player', 'Random_Termination']))
    )
    
    return (cond1 | cond2).astype(int)


def adjust_offer_times(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Note: We had an error in how we computed the offer times in the first two experiments.
    This is why we need this special calculation for these two sessions.

    For rows whose `session_id` is in SPECIAL_SESSIONS, add the offset
        (bargain_start_time_unix / 1000) - bargain_start_time_unix
    to every column that starts with `prefix + "_"` (e.g. offer_1, offer_2 …).

    All other rows are left unchanged.

    Returns a **new** dataframe; original df is not modified.
    """
    out = df.copy()

    # columns like 'offer_1', 'offer_2', ...
    offer_cols = [c for c in out.columns if c.startswith(f"offer_time_")]

    # mask of rows that need the correction
    mask = out["session_id"].isin({"o1rrqa15", "1a8klj6g"})

    if mask.any():
        # row-wise offset: a Series aligned to the rows that need fixing
        offset = (out.loc[mask, "bargain_start_time_unix"] / 1000.0) - out.loc[mask, "bargain_start_time_unix"]

        # broadcast the offset across every offer_* column in one go
        out.loc[mask, offer_cols] = out.loc[mask, offer_cols].add(offset, axis=0)

    return out

def calculate_split_gains_from_trade(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the split gains from trade for each participant.

    For sellers:
        (deal_price − valuation) / gains_from_trade
    For buyers:
        (valuation − deal_price) / gains_from_trade

    Parameters
    ----------
    df : pd.DataFrame or dict
        Must contain columns 'gains_from_trade', 'valuation',
        'participant_role', and 'deal_price'.

    Returns
    -------
    pd.Series
        Split of gains from trade, indexed same as input.
    """

    # 1. Numerator: seller vs. buyer
    num = (df['deal_price'] - df['valuation']).where(
        df['participant_role'] == 'Seller',
        df['valuation'] - df['deal_price']
    )

    # 2) Prepare output as a nullable Float64 Series full of NA
    split = pd.Series(pd.NA, index=df.index, dtype="Float64")

    # 3) Only divide where gains_from_trade ≠ 0
    good = df['gains_from_trade'] != 0
    split.loc[good] = num.loc[good] / df.loc[good, 'gains_from_trade']
    return split


def calculate_first_offer_split(df: pd.DataFrame) -> pd.Series:
    """
    For rows where first_offer == 1:
      - if participant_role == 'Seller', returns (offer_1 - valuation) / gains_from_trade
      - if participant_role == 'Buyer',  returns (valuation - offer_1) / gains_from_trade
    All other rows get pd.NA.
    """
    # create a nullable Float64 series filled with NA
    split = pd.Series(pd.NA, index=df.index, dtype="Float64")

    # masks
    first = df['first_offer'] == 1
    seller = first & (df['participant_role'] == 'Seller')
    buyer  = first & (df['participant_role'] == 'Buyer')

    # compute splits
    split.loc[seller] = (
        df.loc[seller, 'offer_1'] - df.loc[seller, 'valuation']
    ) / df.loc[seller, 'gains_from_trade']

    split.loc[buyer] = (
        df.loc[buyer, 'valuation'] - df.loc[buyer, 'offer_1']
    ) / df.loc[buyer, 'gains_from_trade']

    # set infinites to pd.NA
    split.replace([np.inf, -np.inf], pd.NA, inplace=True)

    return split

def check_time_data_consistency(df: pd.DataFrame, tol: float = 3.0) -> None:
    """
    Raise ValueError if any row violates the three timing rules.

    Hard-coded column names
    -----------------------
    - acceptance_time_sec
    - bargaining_time_full_sec
    - offer_time_1
    - termination_time_sec
    """

    df_filter = df[df['time_inconsistency_dummy'] == 0]
    # Rule masks
    r1 = df_filter["acceptance_time_sec"] > df_filter["bargaining_time_full_sec"] + tol
    r2 = df_filter["last_offer_time"] > df_filter["bargaining_time_full_sec"] + tol
    r3 = df_filter["offer_time_1"] < -3

    any_bad = r1 | r2 | r3
    if any_bad.any():
        bad_rows = df_filter.index[any_bad].tolist()
        msg = [
            "Consistency check failed on rows: " + ", ".join(map(str, bad_rows)),
            "  • Rule 1: acceptance_time_sec > bargaining_time_full_sec + 3",
        ]
        raise ValueError("\n".join(msg))
    


def create_time_inconsistency_dummy(df: pd.DataFrame) -> pd.Series:
    """
    Create a 0/1 dummy indicating, for each row, whether its negotiation_id
    is "time‐inconsistent." A negotiation is time‐inconsistent if any of its
    participants ever have last_offer_time > bargaining_time_full_sec + 3.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns:
          - 'participant_code'
          - 'negotiation_id'
          - 'last_offer_time'
          - 'bargaining_time_full_sec'

    Returns
    -------
    pd.Series
        A 0/1 Series (indexed the same as df), where 1 means the row's
        negotiation_id has at least one participant_code with
        last_offer_time > bargaining_time_full_sec + 3.
    """
    # 1) Find all participant_codes who ever exceed the time threshold
    mask = ((df['last_offer_time'] > df['bargaining_time_full_sec'] + 3) | (df["acceptance_time_sec"] > df["bargaining_time_full_sec"] + 3) | (df["offer_time_1"] < -3))
    offenders = df.loc[mask, 'participant_code'].unique()

    # 2) Find all negotiation_ids in which those participants appear
    bad_negs = df.loc[df['participant_code'].isin(offenders), 'negotiation_id'].unique()

    # 3) Mark any row whose negotiation_id is in that set
    time_inconsistency_dummy = df['negotiation_id'].isin(bad_negs).astype(int)

    # Print the number of 1's as a percentage of the whole dataset
    percent_ones = (time_inconsistency_dummy.sum() / len(df)) * 100
    print(f"Percentage of time-inconsistent negotiations: {percent_ones:.2f}%")

    print(f"Offernders codes: {offenders}")

    return time_inconsistency_dummy

def check_participant_code_uniqueness(df: pd.DataFrame,
                                      round_to_check: int = 23,
                                      participant_col: str = 'participant_code',
                                      round_col: str = 'round') -> None:
    """
    Verify that each participant_code appears at most once in the specified round.
    Raises a ValueError listing any duplicates if the check fails.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check. Must contain `participant_col` and `round_col`.
    round_to_check : int
        The round number to validate (default is 23).
    participant_col : str
        Name of the column holding participant codes.
    round_col : str
        Name of the column holding round identifiers.

    Raises
    ------
    ValueError
        If any participant_code appears more than once in the specified round.
    """
    # Subset to only the requested round
    df_sub = df[df[round_col] == round_to_check]
    
    # Find duplicates within that round
    dup_mask = df_sub.duplicated(subset=[participant_col], keep=False)
    if dup_mask.any():
        # Get the offending participant codes
        dup_codes = (
            df_sub.loc[dup_mask, participant_col]
                  .unique()
        )
        codes_str = ", ".join(map(str, dup_codes))
        raise ValueError(
            f"Round {round_to_check} has duplicate participant_code(s): {codes_str}"
        )
    


def add_group_id_in_session(
    df_part: pd.DataFrame,
    df_groups: pd.DataFrame,
    participant_col: str = "participant_id_in_session",
    lookup_id_col: str = "Participant_ID",
    group_id_col: str = "Group_ID",
    out_col: str = "group_id_in_session"
) -> pd.DataFrame:
    """
    Take df_part with participant_col and df_groups with lookup_id_col & group_id_col,
    and return df_part with an added column out_col containing the group ID (or NaN if no match).
    """
    # Merge on participant ID
    merged = df_part.merge(
        df_groups[[lookup_id_col, group_id_col]],
        how="left",
        left_on=participant_col,
        right_on=lookup_id_col
    )
    # Rename the group column and drop the extra lookup column
    merged = merged.rename(columns={group_id_col: out_col})
    merged = merged.drop(columns=[lookup_id_col])
    return merged


    

def calculate_payoff(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the payoff for each participant.
    If bargaining_outcome is acceptance, calculate based on deal price.
    If bargaining_outcome is not acceptance, payoff is just -cumulated_TA_costs.
    """
    p = df["participant_role"].eq("Seller")
    a = df["bargaining_outcome"].eq("acceptance")
    base = df["cumulated_TA_costs"]
    deal, val = df["deal_price"], df["valuation"]

    arr = np.where(
        a,
        np.where(p, deal - val - base, val - deal - base),
        -base
    )
    return pd.Series(arr, index=df.index, name="payoff")
    

def correct_deal_price(df: pd.DataFrame) -> pd.Series:
    """
    Correct the deal price. In the first 4 sessions, we used a regex to save the deal price in the database; 
    it incorrectly converted very long accepted offers (like 19.00000000001) into 1900. We fix this by filtering for these offers
    and dividing by 10. 
    """
    mask = df["deal_price"] > 60
    df.loc[mask, "deal_price"] = df.loc[mask, "deal_price"] / 10
    percent_replaced = (mask.sum() / len(df)) * 100
    print(f"Replaced {percent_replaced:.2f}% of deal prices.")
    return df["deal_price"]

@validate_call(
    config=ConfigDict(arbitrary_types_allowed=True),
)
def calculate_relative_valuation(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the relative valuation for each participant.
    In t3, t4, it is valuation / 30. 
    In t1, t2 it is (valuaion - 30) / 30 for buyers and (30 - valuation) / 30 for sellers.
    """

    val  = df['valuation']
    trt  = df['treatment']
    role = df['participant_role']

    result = np.select(
        [
            trt.isin(['T3', 'T4']),
            trt.isin(['T1', 'T2']) & (role == 'Buyer'),
            trt.isin(['T1', 'T2']) & (role == 'Seller'),
        ],
        [
            val / 30,
            (val - 30) / 30,
            (30 - val) / 30,
        ],
    )
    return pd.Series(result, index=df.index)
    