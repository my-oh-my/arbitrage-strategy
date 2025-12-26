"""Module for optimizing strategy hyperparameters."""

import itertools
import pandas as pd
from src.arbitrage import run_arbitrage_strategy, StrategyConfig
from src.backtest import calculate_pnl
from src.data_fetcher import fetch_market_data


class GridSearchOptimizer:
    """Optimizes strategy hyperparameters using Grid Search.

    Attributes:
        symbols: List of symbols to optimize for.
        period: Data period.
        interval: Data interval.
        search_params: Dictionary to hold search space.
        spread_windows: Search space for spread window.
        zscore_windows: Search space for zscore window.
        entry_thresholds: Search space for entry thresholds.
        exit_thresholds: Search space for exit thresholds.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        symbols: list[str],
        period: str,
        interval: str,
        search_params: dict = None,
    ):
        self.symbols = symbols
        self.period = period
        self.interval = interval
        self.search_params = search_params or {}

        # Default search space
        self.spread_windows = self.search_params.get(
            "spread_windows", [10, 20, 30, 40, 50]
        )
        self.zscore_windows = self.search_params.get(
            "zscore_windows", [10, 15, 20, 25, 30]
        )
        self.entry_thresholds = self.search_params.get(
            "entry_thresholds", [1.5, 2.0, 2.5, 3.0]
        )
        self.exit_thresholds = self.search_params.get(
            "exit_thresholds", [0.0, 0.5, -0.5]
        )

    def get_best_result(self, results_df: pd.DataFrame) -> pd.Series:
        """Returns the best result from the optimization results."""
        if results_df.empty:
            return None
        return results_df.iloc[0]

    def _evaluate_combination(self, params, data1, data2):
        """Evaluates a single combination of hyperparameters."""
        sw, zw, entry, exit_thresh = params
        config = StrategyConfig(
            spread_window=sw,
            zscore_window=zw,
            entry_threshold=entry,
            exit_threshold=exit_thresh,
        )

        try:
            data1_res, data2_res, signals_data = run_arbitrage_strategy(
                self.symbols,
                self.period,
                self.interval,
                config=config,
                data=(data1, data2),
            )

            if signals_data is None or signals_data.empty:
                return None

            metrics = calculate_pnl(data1_res, data2_res, signals_data)

            return {
                "Spread Window": sw,
                "Z-Score Window": zw,
                "Entry Threshold": entry,
                "Exit Threshold": exit_thresh,
                "Total Return": metrics["Total Return"],
                "Sharpe Ratio": metrics["Sharpe Ratio"],
                "Max Drawdown": metrics["Max Drawdown"],
            }
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(
                f"Error optimizing combination {sw}, {zw}, {entry}, {exit_thresh}: {e}"
            )
            return None

    def optimize(self) -> pd.DataFrame:
        """Runs the optimization process.

        Returns:
            DataFrame containing results for all combinations, sorted by Sharpe Ratio.
        """
        results = []
        combinations = list(
            itertools.product(
                self.spread_windows,
                self.zscore_windows,
                self.entry_thresholds,
                self.exit_thresholds,
            )
        )

        print(f"Starting optimization with {len(combinations)} combinations...")

        # Fetch data once
        print("Fetching market data...")
        try:
            data1 = fetch_market_data(self.symbols[0], self.period, self.interval)
            data2 = fetch_market_data(self.symbols[1], self.period, self.interval)
        except ValueError as e:
            print(f"Failed to fetch data: {e}")
            return pd.DataFrame()

        for i, params in enumerate(combinations):
            # Skip invalid combinations if any logic dictates (e.g. exit > entry)
            # For now, we assume all combinations are valid to test.

            result = self._evaluate_combination(params, data1, data2)
            if result:
                results.append(result)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(combinations)} combinations...")

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df.sort_values(by="Sharpe Ratio", ascending=False, inplace=True)

        return results_df
