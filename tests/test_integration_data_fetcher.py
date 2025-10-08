"""Integration tests for the data_fetcher module."""

import pytest
import pandas as pd
from src.data_fetcher import fetch_market_data


@pytest.mark.integration
def test_fetch_market_data_integration():
    """Test fetching real market data from Yahoo Finance."""
    # Arrange
    symbol = "BTC-USD"
    period = "7d"
    interval = "1d"

    # Act
    data = fetch_market_data(symbol, period, interval)

    # Assert
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

    # Data Quality Checks
    expected_columns = ["Datetime", "Close", "High", "Low", "Open", "Volume"]
    assert all(col in data.columns for col in expected_columns)

    # Check for missing values
    assert data.isnull().sum().sum() == 0

    # Check for logical consistency
    assert (data["High"] >= data["Low"]).all()

    # Check for positive prices and volume
    assert (data[["Open", "High", "Low", "Close", "Volume"]] >= 0).all().all()
