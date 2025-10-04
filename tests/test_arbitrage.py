"""Unit tests for the arbitrage module."""

from unittest.mock import patch
import pandas as pd
import pytest
from src.arbitrage import calculate_correlation, run_arbitrage_strategy


@pytest.fixture(name="sample_data1")
def _sample_data1() -> pd.DataFrame:
    """Fixture for sample data with a positive trend."""
    return pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                    "2023-01-07",
                    "2023-01-08",
                    "2023-01-09",
                    "2023-01-10",
                ]
            ),
            "Close": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        }
    )


@pytest.fixture(name="sample_data2")
def _sample_data2() -> pd.DataFrame:
    """Fixture for sample data with a positive trend, strongly correlated to sample_data1."""
    return pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                    "2023-01-07",
                    "2023-01-08",
                    "2023-01-09",
                    "2023-01-10",
                ]
            ),
            "Close": [200, 220, 240, 260, 280, 300, 320, 340, 360, 380],
        }
    )


@pytest.fixture(name="sample_data3")
def _sample_data3() -> pd.DataFrame:
    """Fixture for sample data with a negative trend, constructed to have returns
    that are negatively correlated with sample_data1."""
    prices = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    inverted_prices = [20000 / p for p in prices]
    return pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                    "2023-01-07",
                    "2023-01-08",
                    "2023-01-09",
                    "2023-01-10",
                ]
            ),
            "Close": inverted_prices,
        }
    )


def test_calculate_correlation_positive(sample_data1, sample_data2):
    """Test for a strong positive correlation."""
    corr = calculate_correlation(sample_data1, sample_data2, "s1", "s2")
    assert corr > 0.9


def test_calculate_correlation_negative(sample_data1, sample_data3):
    """Test for a strong negative correlation."""
    corr = calculate_correlation(sample_data1, sample_data3, "s1", "s3")
    assert corr < -0.9


def test_calculate_correlation_no_correlation():
    """Test for no correlation."""
    data1 = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            ),
            "Close": [100, 200, 200, 0],
        }
    )
    data2 = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
            ),
            "Close": [100, 100, 200, 200],
        }
    )
    corr = calculate_correlation(data1, data2, "s1", "s2")
    assert abs(corr) < 0.5


def test_calculate_correlation_no_variance():
    """Test with data that has no variance."""
    data1 = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Close": [100, 100, 100],
        }
    )
    data2 = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Close": [200, 200, 200],
        }
    )
    corr = calculate_correlation(data1, data2, "s1", "s2")
    assert pd.isna(corr)


@patch("src.arbitrage.calculate_correlation")
@patch("src.arbitrage.fetch_market_data")
def test_run_arbitrage_strategy_calls_fetch_and_calculate(mock_fetch, mock_calc):
    """Test that run_arbitrage_strategy calls fetch_market_data and calculate_correlation."""
    run_arbitrage_strategy(["s1", "s2"], "1d", "1h")

    assert mock_fetch.call_count == 2
    mock_calc.assert_called_once()


@pytest.mark.parametrize(
    "symbols",
    [
        (["s1"]),  # Too few symbols
        (["s1", "s2", "s3"]),  # Too many symbols
    ],
)
def test_run_arbitrage_strategy_wrong_number_of_symbols(symbols):
    """Test that run_arbitrage_strategy raises ValueError for wrong number of symbols."""
    with pytest.raises(
        ValueError, match="Exactly two symbols are required for the arbitrage strategy."
    ):
        run_arbitrage_strategy(symbols, "1d", "1h")
