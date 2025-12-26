"""Unit tests for the arbitrage module."""

from unittest.mock import patch
import pandas as pd
import pytest
from src.arbitrage import (
    calculate_correlation,
    run_arbitrage_strategy,
    calculate_spread,
    generate_signals,
)


@pytest.fixture(name="sample_data1")
def _sample_data1() -> pd.DataFrame:
    """Fixture for sample data with a positive trend."""
    dates = pd.date_range(start="2023-01-01", periods=50)
    return pd.DataFrame(
        {
            "Datetime": dates,
            "Close": range(100, 150),
        }
    )


@pytest.fixture(name="sample_data2")
def _sample_data2() -> pd.DataFrame:
    """Fixture for sample data with a positive trend, strongly correlated to sample_data1."""
    dates = pd.date_range(start="2023-01-01", periods=50)
    return pd.DataFrame(
        {
            "Datetime": dates,
            "Close": [x * 2 for x in range(100, 150)],
        }
    )


@pytest.fixture(name="sample_data3")
def _sample_data3() -> pd.DataFrame:
    """Fixture for sample data with a negative trend, constructed to have returns
    that are negatively correlated with sample_data1."""
    dates = pd.date_range(start="2023-01-01", periods=50)
    prices = range(100, 150)
    inverted_prices = [20000 / p for p in prices]
    return pd.DataFrame(
        {
            "Datetime": dates,
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
def test_run_arbitrage_strategy_calls_fetch_and_calculate(
    mock_fetch, mock_calc, sample_data1
):
    """Test that run_arbitrage_strategy calls fetch_market_data and calculate_correlation."""
    # Configure the mock to return a valid DataFrame
    mock_fetch.return_value = sample_data1
    run_arbitrage_strategy(["s1", "s2"], "1d", "1h")

    assert mock_fetch.call_count == 2
    # Check that it was called once (or check args if needed)
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


@pytest.fixture(name="spread_data1")
def _spread_data1() -> pd.DataFrame:
    """Fixture for spread calculation test."""
    dates = pd.date_range(start="2023-01-01", periods=50)
    return pd.DataFrame(
        {
            "Datetime": dates,
            "Close": range(100, 150),
        }
    )


@pytest.fixture(name="spread_data2")
def _spread_data2() -> pd.DataFrame:
    """Fixture for spread calculation test."""
    dates = pd.date_range(start="2023-01-01", periods=50)
    return pd.DataFrame(
        {
            "Datetime": dates,
            "Close": [x * 0.5 for x in range(100, 150)],
        }
    )


def test_calculate_spread(spread_data1, spread_data2):
    """Test the calculate_spread function."""
    # Use a small window for testing
    window = 10
    spread_df = calculate_spread(spread_data1, spread_data2, "s1", "s2", window=window)

    assert "Spread" in spread_df.columns
    assert not spread_df["Spread"].isnull().all()

    # Check that we have fewer rows due to rolling window dropna
    assert len(spread_df) == len(spread_data1) - window + 1


def test_generate_signals():
    """Test the generate_signals function."""
    zscore_data = pd.DataFrame(
        {
            "Z_Score": [0, 1, 2.5, 1, 0, -1, -2.5, -1, 0],
        }
    )

    # Expected signals:
    # 0: 0 (Flat)
    # 1: 0 (Flat)
    # 2.5: -1 (Short Entry > 2)
    # 1: -1 (Hold Short)
    # 0: 0 (Exit Short <= 0)
    # -1: 0 (Flat)
    # -2.5: 1 (Long Entry < -2)
    # -1: 1 (Hold Long)
    # 0: 0 (Exit Long >= 0)

    signals_df = generate_signals(zscore_data, entry_threshold=2.0, exit_threshold=0.0)

    assert "Signal" in signals_df.columns
    expected_signals = [0, 0, -1, -1, 0, 0, 1, 1, 0]
    assert signals_df["Signal"].tolist() == expected_signals
