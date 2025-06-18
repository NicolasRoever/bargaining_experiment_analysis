import pytest

import pandas as pd
import numpy as np

from src.bargaining_analysis.clean_data.functions_clean_data import find_time_preference_switching_points, calculate_split_gains_from_trade, calculate_first_offer_split



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
    expected = pd.Series([0.5, 2/3, 0.5, 1/3])

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