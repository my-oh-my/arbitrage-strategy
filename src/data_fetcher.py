"""Module for fetching market data."""

import pandas as pd
import yfinance as yf


def fetch_market_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetches historical market data from Yahoo Finance.

    Args:
        symbol: The market symbol to fetch data for (e.g., "BTC-USD").
        period: The time period to fetch data for (e.g., "1d", "1mo", "1y").
        interval: The data interval (e.g., "1m", "1h", "1d").

    Returns:
        A pandas DataFrame containing the historical market data, with columns
        for Open, High, Low, Close, Volume, and a Datetime index.

    Raises:
        ValueError: If no data is fetched for the given symbol.
    """
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    if data is None or data.empty:
        raise ValueError("No data fetched for the given symbol.")

    # Reset column names and index
    data.columns = data.columns.droplevel(1)
    data.reset_index(inplace=True)

    if "Date" in data.columns:
        data.rename(columns={"Date": "Datetime"}, inplace=True)

    return data
