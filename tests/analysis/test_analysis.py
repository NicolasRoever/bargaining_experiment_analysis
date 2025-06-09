import pandas as pd
from src.bargaining_analysis.analysis_03042025.clean_data import determine_first_offer, add_row_with_buyer_valuation, calculate_gains_from_trade, filter_out_mistake_rows
from pandas.testing import assert_series_equal, assert_frame_equal
import pdb

def test_determine_first_offer():
    # Create test data
    test_df = pd.DataFrame({
        'round': [1, 1, 1, 1, 2, 2, 2, 2],
        'group_id_in_round': [1, 1, 2, 2, 1, 2, 1, 2] ,
        'offer_time_1': [10.0, 20.0, None, 15.0, 30.0, None, None, None]
    })

    # Expected results
    expected = pd.Series(
        [1, 0, 0, 1, 1, pd.NA, 0, pd.NA],
        dtype='Int64'
    )

    # Apply function
    actual = determine_first_offer(test_df)

    # Compare results
    assert_series_equal(actual, expected, check_index=False)



def test_add_buyer_valuation_column():
    test_df = pd.DataFrame({
        'round': [1, 1, 1, 1, 2, 2, 2, 2],
        'group_id_in_round': [1, 1, 2, 2, 1, 2, 1, 2] ,
        'participant_role': ['Buyer', 'Seller', 'Buyer', 'Seller', 'Buyer', 'Buyer', 'Seller', 'Seller'],
        'valuation': [10,0,20, 0, 15, 5, 0, 0]
    })

    expected = pd.Series(
        [10, 10, 20, 20, 15, 5, 15, 5],
        dtype='Int64'
    )

    actual = add_row_with_buyer_valuation(test_df)

    assert_series_equal(actual, expected, check_index=False)


def test_calculate_gains_from_trade():
    test_df = pd.DataFrame({
        'round': [1, 1, 1, 1, 2, 2, 2, 2],
        'group_id_in_round': [1, 1, 2, 2, 1, 2, 1, 2] ,
        'participant_role': ['Buyer', 'Seller', 'Buyer', 'Seller', 'Buyer', 'Buyer', 'Seller', 'Seller'],
        'valuation': [10,4,40, 20, 15, 5, 20, 30]
    })

    expected = pd.Series(
        [6, 6, 20, 20, -5, -25, -5, -25],
        dtype='Int64'
    )

    actual = calculate_gains_from_trade(test_df)

    assert_series_equal(actual, expected, check_index=False)


def test_filter_out_mistake_rows():
    test_df = pd.DataFrame({
        'round': [1, 1, 1, 1, 2, 2, 2, 2],
        'group_id_in_round': [1, 1, 2, 2, 1, 2, 1, 2],
        'participant_role': ['Buyer', 'Seller', 'Buyer', 'Seller', 'Buyer', 'Buyer', 'Seller', 'Seller'],
        'gains_from_trade': [10, 30, 0, -40, 15, 5, 20, 30]
    })

    expected = test_df.drop([2, 3]).reset_index(drop=True)



    actual = filter_out_mistake_rows(test_df)

    assert_frame_equal(actual, expected, check_index=False)


