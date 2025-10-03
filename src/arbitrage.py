"""Module for running arbitrage strategy."""

import pandas as pd
from src.data_fetcher import fetch_market_data


def calculate_correlation(
    data1: pd.DataFrame, data2: pd.DataFrame, symbol1: str, symbol2: str
) -> float:
    """Calculates and assesses the correlation between two symbols based on their rate of return.

    Args:
        data1: DataFrame for the first symbol.
        data2: DataFrame for the second symbol.
        symbol1: The first symbol.
        symbol2: The second symbol.

    Returns:
        The correlation value between the rate of return of the 'Close' prices of the two symbols.
    """
    # Calculate the rate of return for each symbol
    data1["Return"] = data1["Close"].pct_change()
    data2["Return"] = data2["Close"].pct_change()

    # Merge the two dataframes on the datetime column
    merged_data = pd.merge(
        data1[["Datetime", "Return"]],
        data2[["Datetime", "Return"]],
        on="Datetime",
        suffixes=(f"_{symbol1}", f"_{symbol2}"),
    )

    # Calculate the correlation on the returns
    correlation = merged_data[[f"Return_{symbol1}", f"Return_{symbol2}"]].corr()

    print("\nCorrelation Matrix (based on returns):")
    print(correlation)

    # Assess the correlation
    correlation_matrix = correlation.to_numpy()
    corr_value = correlation_matrix[0, 1]

    return float(corr_value)


def run_arbitrage_strategy(symbols: list[str], period: str, interval: str):
    """Fetches data for two symbols, calculates their correlation, and prints the results.

    Args:
        symbols: A list containing two stock symbols.
        period: The time period to fetch data for (e.g., '1d', '1mo', '1y').
        interval: The data interval (e.g., '1m', '1h', '1d').
    """
    try:
        data1 = fetch_market_data(symbols[0], period, interval)
        data2 = fetch_market_data(symbols[1], period, interval)

        correlation_value = calculate_correlation(data1, data2, symbols[0], symbols[1])
        print(f"\nCorrelation Value: {correlation_value}")

    except ValueError as e:
        print(f"Error fetching data: {e}")
