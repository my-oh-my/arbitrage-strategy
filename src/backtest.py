"""Module for backtesting the arbitrage strategy."""

import numpy as np
import pandas as pd


def calculate_pnl(
    data1: pd.DataFrame, data2: pd.DataFrame, signals_data: pd.DataFrame
) -> dict:
    """Calculates PnL and performance metrics for the strategy.

    Args:
        data1: DataFrame for the first symbol (must contain 'Close').
        data2: DataFrame for the second symbol (must contain 'Close').
        signals_data: DataFrame containing 'Signal' and 'HedgeRatio'.

    Returns:
        A dictionary containing performance metrics:
        - 'Total Return': Cumulative PnL.
        - 'Sharpe Ratio': Annualized Sharpe Ratio.
        - 'Max Drawdown': Maximum drawdown percentage (based on cumulative PnL).
    """
    # Ensure indices match
    common_index = signals_data.index
    s1_close = data1.loc[common_index, "Close"]
    s2_close = data2.loc[common_index, "Close"]
    hedge_ratio = signals_data["HedgeRatio"]
    position = (
        signals_data["Signal"].shift(1).fillna(0)
    )  # Position held from previous day

    # Calculate daily price changes
    s1_diff = s1_close.diff()
    s2_diff = s2_close.diff()

    # Calculate daily PnL
    # PnL = Position * (Change_s1 - HedgeRatio * Change_s2)
    daily_pnl = position * (s1_diff - hedge_ratio * s2_diff)
    cumulative_pnl = daily_pnl.cumsum()

    return _calculate_metrics(daily_pnl, cumulative_pnl)


def _calculate_metrics(daily_pnl: pd.Series, cumulative_pnl: pd.Series) -> dict:
    """Calculates performance metrics from PnL data."""
    total_return = cumulative_pnl.iloc[-1] if not cumulative_pnl.empty else 0.0

    # Sharpe Ratio (assuming 0 risk-free rate, annualized)
    if daily_pnl.std() != 0:
        sharpe_ratio = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Max Drawdown
    if not cumulative_pnl.empty:
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0.0

    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
    }
