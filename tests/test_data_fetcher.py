"""Unit tests for the data_fetcher module."""

import pandas as pd
import pytest
from src.data_fetcher import fetch_market_data


@pytest.fixture
def _mock_yfinance_download(mocker):
    """Fixture to mock the yf.download function."""
    return mocker.patch("yfinance.download")


def test_fetch_market_data_success(_mock_yfinance_download):
    """Test successful fetching of market data."""
    # Arrange
    mock_data = pd.DataFrame(
        {
            ("Open", ""): [100],
            ("High", ""): [105],
            ("Low", ""): [95],
            ("Close", ""): [102],
            ("Volume", ""): [1000],
        },
        index=pd.to_datetime(["2023-01-01"]),
    )
    mock_data.index.name = "Datetime"
    _mock_yfinance_download.return_value = mock_data

    # Act
    result = fetch_market_data("BTC-USD", "1d", "1h")

    # Assert
    assert not result.empty
    assert "Open" in result.columns
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2023-01-01"]),
                "Open": [100],
                "High": [105],
                "Low": [95],
                "Close": [102],
                "Volume": [1000],
            }
        ),
    )


def test_fetch_market_data_no_data_none(_mock_yfinance_download):
    """Test ValueError when yf.download returns None."""
    # Arrange
    _mock_yfinance_download.return_value = None

    # Act & Assert
    with pytest.raises(ValueError, match="No data fetched for the given symbol."):
        fetch_market_data("BTC-USD", "1d", "1h")


def test_fetch_market_data_no_data_empty(_mock_yfinance_download):
    """Test ValueError when yf.download returns an empty DataFrame."""
    # Arrange
    _mock_yfinance_download.return_value = pd.DataFrame()

    # Act & Assert
    with pytest.raises(ValueError, match="No data fetched for the given symbol."):
        fetch_market_data("BTC-USD", "1d", "1h")
