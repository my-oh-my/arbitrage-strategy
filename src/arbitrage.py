"""Module for running arbitrage strategy."""

from dataclasses import dataclass
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.regression.rolling import RollingOLS
from src.data_fetcher import fetch_market_data


@dataclass
class StrategyConfig:
    """Configuration key parameters for the arbitrage strategy.

    Attributes:
        spread_window: Rolling window size for the spread calculation (OLS).
        zscore_window: Rolling window size for the Z-score calculation.
        entry_threshold: Z-score value to trigger a position entry.
        exit_threshold: Z-score value to trigger a position exit.
    """

    spread_window: int = 30
    zscore_window: int = 20
    entry_threshold: float = 2.0
    exit_threshold: float = 0.0


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
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    symbol1: str,
    symbol2: str,
    window: int = 30,
) -> pd.DataFrame:
    """Calculates the spread between two symbols using rolling OLS regression."""
    merged_data = pd.merge(
        data1[["Datetime", "Close"]],
        data2[["Datetime", "Close"]],
        on="Datetime",
        suffixes=(f"_{symbol1}", f"_{symbol2}"),
    )

    # Use RollingOLS to find the time-varying hedge ratio
    y = merged_data[f"Close_{symbol1}"]
    x = sm.add_constant(merged_data[f"Close_{symbol2}"])

    model = RollingOLS(y, x, window=window)
    rolling_res = model.fit()

    # Get the hedge ratio (coefficient for the second symbol)
    hedge_ratio = rolling_res.params[f"Close_{symbol2}"]

    # Calculate the spread
    merged_data["Spread"] = (
        merged_data[f"Close_{symbol1}"] - hedge_ratio * merged_data[f"Close_{symbol2}"]
    )

    # Store the hedge ratio for plotting
    merged_data["HedgeRatio"] = hedge_ratio

    # Drop NaN values resulting from the rolling window
    merged_data.dropna(inplace=True)

    return merged_data


def calculate_zscore(spread_data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates the rolling Z-score of the spread."""
    spread_mean = spread_data["Spread"].rolling(window=window).mean()
    spread_std = spread_data["Spread"].rolling(window=window).std()

    spread_data["Z_Score"] = (spread_data["Spread"] - spread_mean) / spread_std

    # Drop NaN values resulting from the rolling window
    spread_data.dropna(inplace=True)

    return spread_data


def generate_signals(
    zscore_data: pd.DataFrame, entry_threshold: float = 2.0, exit_threshold: float = 0.0
) -> pd.DataFrame:
    """Generates trading signals based on Z-score thresholds.

    Args:
        zscore_data: DataFrame containing the Z-score.
        entry_threshold: Z-score threshold to enter a position (positive for short spread, negative for long spread).
        exit_threshold: Z-score threshold to exit a position.

    Returns:
        DataFrame with a 'Signal' column.
    """
    zscore_data["Signal"] = 0  # 0: No Signal, 1: Long Spread, -1: Short Spread

    # Iterate through the data to generate signals (simulating real-time)
    # Note: Vectorization is faster but iteration is clearer for state-dependent logic
    position = 0  # 0: Flat, 1: Long, -1: Short

    signals = []

    for z in zscore_data["Z_Score"]:
        if position == 0:
            if z < -entry_threshold:
                position = 1  # Long Spread (Buy s1, Sell s2)
            elif z > entry_threshold:
                position = -1  # Short Spread (Sell s1, Buy s2)
        elif position == 1:
            if z >= -exit_threshold:
                position = 0  # Exit Long
        elif position == -1:
            if z <= exit_threshold:
                position = 0  # Exit Short
        signals.append(position)

    zscore_data["Signal"] = signals
    return zscore_data


def run_arbitrage_strategy(
    symbols: list[str],
    period: str,
    interval: str,
    config: StrategyConfig = None,
    data: tuple[pd.DataFrame, pd.DataFrame] = (None, None),
):
    """Fetches data, calculates statistics, and generates trading signals.

    Args:
        symbols: A list containing exactly two stock symbols.
        period: The time period to fetch data for (e.g., '1d', '1mo', '1y').
        interval: The data interval (e.g., '1m', '1h', '1d').
        config: Strategy configuration object containing window sizes and thresholds.
            If None, a default StrategyConfig is used.
        data: Optional tuple of (data1, data2) DataFrames to avoid re-fetching data.
            Useful for optimization loops.

    Returns:
        A tuple containing (data1, data2, signals_data).
        - data1: DataFrame for the first symbol.
        - data2: DataFrame for the second symbol.
        - signals_data: DataFrame with columns ['Spread', 'Z_Score', 'Signal', 'HedgeRatio'].
        Returns (None, None, None) if an error occurs.
    """
    if len(symbols) != 2:
        raise ValueError("Exactly two symbols are required for the arbitrage strategy.")

    if config is None:
        config = StrategyConfig()

    data1, data2 = data
    try:
        if data1 is None or data2 is None:
            data1 = fetch_market_data(symbols[0], period, interval)
            data2 = fetch_market_data(symbols[1], period, interval)

        correlation_value = calculate_correlation(data1, data2, symbols[0], symbols[1])
        print(f"\nCorrelation Value: {correlation_value}")

        p_value = test_cointegration(data1, data2, symbols[0], symbols[1])
        print(f"\nCointegration Test P-value: {p_value}")

        if p_value < 0.05:
            print("The two symbols are likely cointegrated.")
        else:
            print("The two symbols are not cointegrated.")

        # Always calculate spread and Z-score for analysis/plotting purposes
        spread_data = calculate_spread(
            data1, data2, symbols[0], symbols[1], window=config.spread_window
        )
        zscore_data = calculate_zscore(spread_data, window=config.zscore_window)
        signals_data = generate_signals(
            zscore_data,
            entry_threshold=config.entry_threshold,
            exit_threshold=config.exit_threshold,
        )

        if p_value < 0.05:
            print("\nSpread, Z-Score, and Signals:")
            print(signals_data[["Datetime", "Spread", "Z_Score", "Signal"]].tail())

        return data1, data2, signals_data

    except ValueError as e:
        print(f"Error fetching data: {e}")
        return None, None, None
