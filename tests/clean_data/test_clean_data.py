import pytest

import pandas as pd
import numpy as np

from src.bargaining_analysis.clean_data.functions_clean_data import find_time_preference_switching_points, calculate_split_gains_from_trade, calculate_first_offer_split, create_time_inconsistency_dummy, obtain_last_offer, calculate_deal_price



def test_find_time_preference_switching_points():
    # sample DataFrame
    data = {
        'time_row_1': [2.0, 1.0, 1.0, 1.0, 1.0],
        'time_row_2': [2.0, 2.0, 1.0, 1.0, 1.0],
        'time_row_3': [2.0, 2.0, 2.0, 1.0, 1.0],
        'time_row_4': [2.0, 2.0, 2.0, 1.0, 2.0],
        'time_row_5': [2.0, 2.0, 2.0, 1.0, 2.0],
        'time_row_6': [2.0, 2.0, 2.0, 1.0, 2.0],
    }
    df = pd.DataFrame(data)

    # expected switch times:
   
    expected = pd.Series([1, 2, 3, 7, 4])

    result = find_time_preference_switching_points(df)
    pd.testing.assert_series_equal(result, expected)


def test_calculate_split_gains_from_trade():
    # sample DataFrame
    data = pd.DataFrame({
        'gains_from_trade': [10, 15, 10, 15],
        'valuation': [10, 10, 20, 25],
        'participant_role': ['Seller', 'Seller', 'Buyer', 'Buyer'],
        'group_id_in_round': [1, 2, 1, 2],
        'deal_price': [15, 20, 15, 20],
        'session_id': ['1', '1', '1', '1']

    })
    expected = pd.Series([0.5, 2/3, 0.5, 1/3], dtype="Float64")

    actual = calculate_split_gains_from_trade(data)
    pd.testing.assert_series_equal(actual, expected)


def test_calculate_split_first_offer():
    # sample DataFrame
    data = pd.DataFrame({
        'gains_from_trade': [10, 15, 10, 15],
        'valuation': [10, 10, 20, 25],
        'participant_role': ['Seller', 'Seller', 'Buyer', 'Buyer'],
        'group_id_in_round': [1, 2, 1, 2],
        'offer_1': [18, 10, 14, 20],
        'first_offer': [1, 0, 0, 1]

    })
    expected = pd.Series([0.8, pd.NA, pd.NA, 1/3])

    actual = calculate_first_offer_split(data)
    pd.testing.assert_series_equal(actual, expected, check_names=False, check_dtype=False)


def test_create_time_inconsistency_dummy():
    # sample DataFrame
    data = pd.DataFrame({
        'bargaining_time_full_sec': [10, 10, 10, 10, 10, 10],
        'group_id_in_round': [1, 2, 1, 2, 1, 2],
        'negotiation_id': [1, 1, 2, 2, 3,3],
        'last_offer_time': [30, 7, 30, 8, 5, 5,],
        'participant_code': ['1', '2', '1', '2', '3', '4'],
        'accepted_by_id_in_group': [1, 1, 2, 2, 3, 3],
        'terminated_by_id_in_group': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
        'id_in_group': [1, 2, 1, 2, 1, 2],

    })
    expected = pd.Series([1, 1, 1, 1, 0, 0])

    actual = create_time_inconsistency_dummy(data)

    pd.testing.assert_series_equal(actual, expected, check_names=False, check_dtype=False)



def test_obtain_last_offer():
    data = pd.DataFrame({
        'offer_1': [18, 10, 14, pd.NA],
        'offer_2': [17, 11, pd.NA, pd.NA],
        'offer_3': [16, 12, pd.NA, pd.NA],
    })
    expected = pd.Series([16, 12, 14, pd.NA])

    actual = obtain_last_offer(data)

    pd.testing.assert_series_equal(actual, expected, check_names=False, check_dtype=False)

def test_calculate_deal_price():
    data = pd.DataFrame({
        'round': [1, 1, 2, 2],
        'gains_from_trade': [10, 15, 10, 15],
        'valuation': [10, 10, 20, 25],
        'participant_role': ['Seller', 'Seller', 'Buyer', 'Buyer'],
        'group_id_in_round': [1, 1, 2, 2],
        'id_in_group': [1, 2, 1, 2],
        'accepted_by_id_in_group': [1, 1, 2, 2],
        'last_offer': [20, 15, 8, 110]})
    
    expected = pd.Series([15, 15, 8, 8])

    actual = calculate_deal_price(data)

    pd.testing.assert_series_equal(actual, expected, check_names=False, check_dtype=False)