import pytest

import pandas as pd
import numpy as np

from src.bargaining_analysis.clean_data.functions_clean_data import find_time_preference_switching_points



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