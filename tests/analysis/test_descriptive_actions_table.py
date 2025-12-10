import pytest
import pandas as pd
import numpy as np

from src.bargaining_analysis.main_results.descriptive_actions_table import calculate_war, calculate_war_plus

@pytest.fixture
def negotiation_data():
    """
    Fixture returning a simplified dataframe with duplicate negotiation_ids and mixed outcomes.
    """
    return pd.DataFrame({
        'negotiation_id': [1, 1, 2, 3, 4],
        'treatment': ['T1', 'T1', 'T1', 'T1', 'T2'],
        'gains_from_trade': [10, 10, 5, -4, 8],
        'bargaining_outcome': [
            'acceptance',  # keep first for id=1
            'acceptance',  # drop duplicate
            'rejection',
            'acceptance',
            'acceptance'
        ],
    })


def test_calculate_war_t1(negotiation_data):
    """
    WAR = realized surplus / total possible gains_from_trade for treatment T1.
    Realized surplus = 10 (id1) + 0 (id2 rejection) + (-4) (id3 acceptance)
    Total possible gains = 10 + 5 = 15
    WAR = 6 / 15 = 0.4
    """
    expected_result = 6 / 15
    actual_result = calculate_war(negotiation_data, 'T1')
    assert np.isclose(actual_result, expected_result, atol=1e-12)


def test_calculate_war_plus_t1(negotiation_data):
    """
    WAR+ = (positive realized) / total possible gains_from_trade
    Positive unrealized: none (id2 had 0 realized, but 5 was unrealized)
    WAR+ = (10) / 15 = 0.73333
    """
    expected_result = 2/3
    actual_result = calculate_war_plus(negotiation_data, 'T1')
    assert np.isclose(actual_result, expected_result, atol=1e-12)


def test_calculate_war_t2(negotiation_data):
    """
    WAR for T2:
    realized surplus = 8 (acceptance)
    total possible = 8
    WAR = 1.0
    """
    expected_result = 1.0
    actual_result = calculate_war(negotiation_data, 'T2')
    assert np.isclose(actual_result, expected_result, atol=1e-12)


def test_calculate_war_plus_t2(negotiation_data):
    """
    WAR+ for T2:
    same as WAR = 1.0
    """
    expected_result = 1.0
    actual_result = calculate_war_plus(negotiation_data, 'T2')
    assert np.isclose(actual_result, expected_result, atol=1e-12)