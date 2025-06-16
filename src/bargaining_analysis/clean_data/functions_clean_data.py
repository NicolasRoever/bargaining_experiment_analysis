import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import pdb


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


    df_clean["full_bargaining_duration_corr"] = df_clean["cumulated_TA_costs"]/0.05
    
    return df_clean


def clean_symmetric_TA_data(raw_data):
    """
    Clean the raw data by reshaping it and adding the termination times.
    """

    df_clean = clean_data(raw_data)

    df_clean["full_bargaining_duration_corr"] = df_clean["cumulated_TA_costs"]/0.05

    return df_clean


def clean_data_asymmetric_no_TA(raw_data):
    """
    Clean the raw data by reshaping it and adding the termination times.
    """

    df_clean = clean_data(raw_data)

    return df_clean


def clean_zero_TA_costs_two_sided_data(raw_data):
    """
    Clean the raw data by reshaping it and adding the termination times.
    """

    
    df_clean = clean_data(raw_data)

    # 1. turn both series into plain float64 NumPy arrays
    payoff = df_clean["payoff"].to_numpy(dtype="float64")
    gft    = df_clean["gains_from_trade"].to_numpy(dtype="float64")

    # 2. do the division; /0 → ±inf (no Python exception)
    tmp = pd.Series(payoff / gft)

    # 3. replace the infinities with pd.NA and give it back the nullable dtype
    df_clean["split_gains_from_trade"] = tmp.replace([np.inf, -np.inf], pd.NA).astype("Float64")


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

    df_clean['participant_id'] = df_long_wide['participant.id_in_session']
    df_clean['participant_role'] = df_long_wide['participant.role_in_game']
    df_clean["group_id_in_round"] = df_long_wide["id_in_subsession"]
    df_clean['round'] = df_long_wide['round']
    df_clean["session_id"] = df_long_wide["session.code"]
    df_clean["information_asymmetry"] = df_long_wide["session.config.information_asymmetry"]
    df_clean["TA_costs"] = df_long_wide["session.config.transaction_costs"]
    df_clean["treatment"] = determine_treatment_category(df_clean["information_asymmetry"], df_clean["TA_costs"])
    df_clean["subsession.is_practice_round"] = df_long_wide["subsession.is_practice_round"]
    df_clean["Role_Seller"] = np.where(df_clean["participant_role"] == "Seller", 1, 0)

    #Transaction Costs
    df_clean["cumulated_TA_costs"] = df_long_wide["cumulated_TA_costs"]

    #Time variables
    df_clean["bargaining_duration"] = df_long_wide["bargaining_duration"]
    df_clean['bargain_start_time_unix'] = df_long_wide['bargain_start_time'].astype('Float64')
    df_clean["acceptance_time_raw"] = df_long_wide["acceptance_time"].astype('Float64')
    df_clean["termination_time_raw"] = df_long_wide["termination_time"].astype('Float64')


    df_clean = add_acceptance_time_sec(df_clean)
    df_clean = add_termination_time_sec(df_clean)

    df_clean["bargaining_time_full_sec"] = df_clean["acceptance_time_sec"].combine_first(df_clean["termination_time_sec"])

    df_clean["total_TA_costs"] = df_long_wide["current_payoff_terminate"]
    

    

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
    df_clean["first_offer"] = determine_first_offer(df_clean)


    offer_time_columns = [col for col in df_clean.columns if col.startswith("offer_time_")]
    df_clean["number_of_offers"] = df_clean[offer_time_columns].count(axis=1)


    #Valuation and Payoff
    df_clean["valuation"] = df_long_wide["valuation"].astype('Int64')
    df_clean['payoff'] = df_long_wide['payoff']
    df_clean["id_in_group"] = pd.to_numeric(df_long_wide["id_in_group"], errors='coerce')
    df_clean["group_id_in_round"] = df_long_wide["id_in_subsession"]
    df_clean["terminated_by_id_in_group"] = df_long_wide["terminated_by"].astype('Int64')
    df_clean["termination_mode"] = df_long_wide["termination_mode"]
    df_clean["player_terminated"] =  np.where(
        df_clean['terminated_by_id_in_group'].isna(),
        pd.NA,
        1)

    df_clean['bargaining_outcome'] = (
    df_clean['termination_mode']
      .fillna('acceptance')
    )

    #Acceptance Information
    df_clean["accepted_by_id_in_group"] = df_long_wide["accepted_by"].astype('Int64')
    df_clean["deal_price"] = df_long_wide["deal_price"]
    df_clean["agreement_dummy"] = np.where(
        df_clean["bargaining_outcome"] == "acceptance",
        1,
        0
    )

    #Gains from Trade
    df_clean["gains_from_trade"] = calculate_gains_from_trade(df_clean)
    df_clean["split_gains_from_trade"] = compute_split_gains_from_trade(df_clean)
    df_clean["gains_from_trade_dummy"] = np.where(
        df_clean["gains_from_trade"] >= 0,
        1,
        0
    )

    #Efficiency
    df_clean["efficiency"] = calculate_efficiency(df_clean)

    #Add Dummy for information asymmetry
    df_clean["valuation_info_public"] = np.where(
        (df_clean["information_asymmetry"] == "one_sided") & (df_clean["participant_role"] == "Seller"),
        1,
        0
    )

    #Add Dummy for interaction seller and information asymmetry
    df_clean["interaction_seller_info_public"] = df_clean["Role_Seller"] * df_clean["valuation_info_public"]

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
        df_clean["total_TA_costs"]  > df_clean["payoff"],
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
    id_vars = ['participant.id_in_session', 'participant.label', 'participant.role_in_game', 'session.code', 'session.config.information_asymmetry', 'session.config.transaction_costs']

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





def compute_split_gains_from_trade(df: pd.DataFrame) -> pd.Series:
    """
    Compute the split of gains from trade for each row.

    - If gains_from_trade == 0: return <NA>
    - Else if participant_role == 'Seller': deal_price / gains_from_trade
    - Else: 1 - (deal_price / gains_from_trade)

    Returns
    -------
    pd.Series
        A nullable-Float Series indexed like `df`, with name 'split_gains_from_trade'.
    """
    # pull into numpy floats so division by zero gives inf, not exception
    deal = df["deal_price"].to_numpy(dtype=float)
    gain = df["gains_from_trade"].to_numpy(dtype=float)

    # elementwise division: inf where gain==0
    ratio = deal / gain
    # mask out those infinities (and any -inf) back to NaN
    ratio[np.isinf(ratio)] = np.nan

    # apply seller vs. buyer
    split = np.where(
        df["participant_role"] == "Seller",
        ratio,
        1 - ratio
    )

    # wrap as a pandas Series of nullable floats
    return pd.Series(
        split,
        dtype="Float64"
    )

    
    


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



def add_row_with_buyer_valuation(df: pd.DataFrame) -> pd.Series:
    """
    Add a row with the buyer valuation to the dataframe.
    """
    pass



def plot_bargaining_rounds(df_clean, round1, round2):
    """
    Plot bargaining processes for two specified rounds.
    
    Parameters:
    -----------
    df_clean : pandas.DataFrame
        The cleaned bargaining data
    round1 : int
        First round to visualize
    round2 : int
        Second round to visualize
    """
    # Filter data for the specified rounds
    df_rounds = df_clean[df_clean['round'].isin([round1, round2])]

    # Get unique pairs for each round
    round_pairs = {}
    for round_num in [round1, round2]:
        round_pairs[round_num] = df_rounds[df_rounds['round'] == round_num]['group_id_in_round'].unique()

    # Calculate number of pairs per round
    max_pairs = max(len(pairs) for pairs in round_pairs.values())

    # Find the maximum offer value across all pairs and rounds
    max_offer = 0
    for round_num in [round1, round2]:
        for group_id in round_pairs[round_num]:
            pair_data = df_rounds[(df_rounds['round'] == round_num) & 
                                (df_rounds['group_id_in_round'] == group_id)]
            buyer = pair_data[pair_data['participant_role'] == 'Buyer'].iloc[0]
            seller = pair_data[pair_data['participant_role'] == 'Seller'].iloc[0]
            
            # Check all offers for both buyer and seller
            for i in range(1, 11):
                if pd.notna(buyer[f'offer_{i}']):
                    max_offer = max(max_offer, buyer[f'offer_{i}'])
                if pd.notna(seller[f'offer_{i}']):
                    max_offer = max(max_offer, seller[f'offer_{i}'])

    # Add some padding to the maximum value
    max_offer = max_offer * 1.1

    # Create a figure with subplots for each pair in each round
    fig, axes = plt.subplots(2, max_pairs, figsize=(5*max_pairs, 10))
    fig.suptitle(f'Bargaining Process: Rounds {round1} and {round2}', fontsize=16)

    for round_idx, round_num in enumerate([round1, round2]):
        pairs = round_pairs[round_num]
        for pair_idx, group_id in enumerate(pairs):
            ax = axes[round_idx, pair_idx]
            
            # Get data for this pair
            pair_data = df_rounds[(df_rounds['round'] == round_num) & 
                                (df_rounds['group_id_in_round'] == group_id)]
            
            # Get buyer and seller data
            buyer = pair_data[pair_data['participant_role'] == 'Buyer'].iloc[0]
            seller = pair_data[pair_data['participant_role'] == 'Seller'].iloc[0]
            
            # Plot buyer's offers
            for i in range(1, 11):
                if pd.notna(buyer[f'offer_{i}']) and pd.notna(buyer[f'offer_time_{i}']):
                    ax.scatter(buyer[f'offer_time_{i}'], buyer[f'offer_{i}'], 
                             color='blue', alpha=0.7, label='Buyer Offer' if i == 1 else None)
            
            # Plot seller's offers
            for i in range(1, 11):
                if pd.notna(seller[f'offer_{i}']) and pd.notna(seller[f'offer_time_{i}']):
                    ax.scatter(seller[f'offer_time_{i}'], seller[f'offer_{i}'], 
                             color='red', alpha=0.7, label='Seller Offer' if i == 1 else None)
            
            # Add valuation information
            ax.text(0.02, 0.98, 
                    f'Buyer Val: {buyer["valuation"]}\nSeller Val: {seller["valuation"]}\nOutcome: {buyer["bargaining_outcome"]}',
                    transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add lines connecting offers for each player
            buyer_times = [buyer[f'offer_time_{i}'] for i in range(1, 11) if pd.notna(buyer[f'offer_{i}']) and pd.notna(buyer[f'offer_time_{i}'])]
            buyer_offers = [buyer[f'offer_{i}'] for i in range(1, 11) if pd.notna(buyer[f'offer_{i}']) and pd.notna(buyer[f'offer_time_{i}'])]
            seller_times = [seller[f'offer_time_{i}'] for i in range(1, 11) if pd.notna(seller[f'offer_{i}']) and pd.notna(seller[f'offer_time_{i}'])]
            seller_offers = [seller[f'offer_{i}'] for i in range(1, 11) if pd.notna(seller[f'offer_{i}']) and pd.notna(seller[f'offer_time_{i}'])]
            
            if buyer_times:
                ax.plot(buyer_times, buyer_offers, 'b--', alpha=0.3)
            if seller_times:
                ax.plot(seller_times, seller_offers, 'r--', alpha=0.3)
            
            # Add termination time if applicable
            if pd.notna(buyer['termination_time_sec']):
                ax.axvline(x=buyer['termination_time_sec'], color='black', linestyle='--', alpha=0.5)
                ax.text(buyer['termination_time_sec'], max_offer * 0.95, 'Termination', 
                       rotation=90, verticalalignment='top')
            
            # Set title and labels
            ax.set_title(f'Round {round_num}, Pair {group_id}')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Offer Amount')
            ax.set_ylim(0, max_offer)  # Set consistent y-axis range
            ax.grid(True, alpha=0.3)
            ax.legend()

    # Hide empty subplots
    for round_idx in range(2):
        for pair_idx in range(max_pairs):
            if pair_idx >= len(round_pairs[round_idx + round1]):
                axes[round_idx, pair_idx].set_visible(False)

    plt.tight_layout()
    plt.show()



def filter_out_mistake_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out all rows for any (round, group_id_in_round) where
    one of the trading partners has payoffs < -15. Note that this only approximates our pre-specified criterion, but this should be fine
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
        out[col] = out["participant_id"].map(time_row_map[col]).astype('Float64')
    
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
    return df_clean["participant_id"].map(var_map).astype(dtype)


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

    For session_id in SPECIAL_SESSIONS:
        acceptance_time_sec = acceptance_time_raw
                              + (bargain_start_time_unix / 1000)
                              -  bargain_start_time_unix
    Else:
        acceptance_time_sec = acceptance_time_raw

    Additionally, if acceptance_time_raw < 100, use:
        acceptance_time_sec = df["cumulated_TA_costs"] / 0.05
    """
    special_sessions_mask = df["session_id"].isin({"o1rrqa15", "1a8klj6g"})
    low_acceptance_time_mask = df["acceptance_time_raw"].notna() & (df["acceptance_time_raw"] < 100)

    df["acceptance_time_sec"] = np.where(
        special_sessions_mask,
        df["acceptance_time_raw"]
        + (df["bargain_start_time_unix"] / 1000.0)
        - df["bargain_start_time_unix"],
        df["acceptance_time_raw"]
    )

    df["acceptance_time_sec"] = np.where(
        low_acceptance_time_mask,
        df["cumulated_TA_costs"] / 0.05,
        df["acceptance_time_sec"]
    )

    return df




def add_termination_time_sec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds df['termination_time_sec'].

    Note: We had an error in how we computed the termination time in the first two experiments.
    This is why we need this special calculation for these two sessions.


    Rule:
      if  session_id ∈ SPECIAL_SESSIONS
          AND termination_time_raw >= 500
      then
          termination_time_sec = termination_time_raw
                                 + (bargain_start_time_unix / 1000)
                                 -  bargain_start_time_unix
      else
          termination_time_sec = termination_time_raw
    """
    mask = (
        df["session_id"].isin({"o1rrqa15", "1a8klj6g"})
        & df["termination_time_raw"].notna()
        & (df["termination_time_raw"] >= 500)
    )

    df["termination_time_sec"] = np.where(
        mask,
        df["termination_time_raw"]
        + (df["bargain_start_time_unix"] / 1000.0)
        -  df["bargain_start_time_unix"],
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
    
    # Case 2: non‐positive gains and a “no‐deal” termination
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