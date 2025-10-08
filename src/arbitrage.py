"""Module for running arbitrage strategy."""

import pandas as pd
from statsmodels.tsa.stattools import coint
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


def test_cointegration(
    data1: pd.DataFrame, data2: pd.DataFrame, symbol1: str, symbol2: str
) -> float:
    """Performs a cointegration test and returns the p-value."""
    merged_data = pd.merge(
        data1[["Datetime", "Close"]],
        data2[["Datetime", "Close"]],
        on="Datetime",
        suffixes=(f"_{symbol1}", f"_{symbol2}"),
    )
    coint_result = coint(
        merged_data[f"Close_{symbol1}"], merged_data[f"Close_{symbol2}"]
    )
    return float(coint_result[1])


def calculate_spread(
    data1: pd.DataFrame, data2: pd.DataFrame, symbol1: str, symbol2: str
) -> pd.DataFrame:
    """Calculates the spread between two symbols."""
    merged_data = pd.merge(
        data1[["Datetime", "Close"]],
        data2[["Datetime", "Close"]],
        on="Datetime",
        suffixes=(f"_{symbol1}", f"_{symbol2}"),
    )
    merged_data["Ratio"] = (
        merged_data[f"Close_{symbol1}"] / merged_data[f"Close_{symbol2}"]
    )
    merged_data["Spread"] = (
        merged_data[f"Close_{symbol1}"]
        - merged_data["Ratio"] * merged_data[f"Close_{symbol2}"]
    )
    return merged_data


def calculate_zscore(spread_data: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Z-score of the spread."""
    spread_data["Z_Score"] = (
        spread_data["Spread"] - spread_data["Spread"].mean()
    ) / spread_data["Spread"].std()
    return spread_data


def run_arbitrage_strategy(symbols: list[str], period: str, interval: str):
    """Fetches data for two symbols, calculates their correlation, and prints the results.

    Args:
        symbols: A list containing two stock symbols.
        period: The time period to fetch data for (e.g., '1d', '1mo', '1y').
        interval: The data interval (e.g., '1m', '1h', '1d').
    """
    if len(symbols) != 2:
        raise ValueError("Exactly two symbols are required for the arbitrage strategy.")

    try:
        data1 = fetch_market_data(symbols[0], period, interval)
        data2 = fetch_market_data(symbols[1], period, interval)

        correlation_value = calculate_correlation(data1, data2, symbols[0], symbols[1])
        print(f"\nCorrelation Value: {correlation_value}")

        p_value = test_cointegration(data1, data2, symbols[0], symbols[1])
        print(f"\nCointegration Test P-value: {p_value}")

        if p_value < 0.05:
            print("The two symbols are likely cointegrated.")
            spread_data = calculate_spread(data1, data2, symbols[0], symbols[1])
            zscore_data = calculate_zscore(spread_data)
            print("\nSpread and Z-Score:")
            print(zscore_data[["Datetime", "Spread", "Z_Score"]].tail())
        else:
            print("The two symbols are not cointegrated.")

    except ValueError as e:
        print(f"Error fetching data: {e}")
