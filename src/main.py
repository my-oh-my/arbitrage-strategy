"""Main entry point for the application."""

import argparse
from src.arbitrage import run_arbitrage_strategy
from src.plotting import plot_dashboard
from src.optimizer import GridSearchOptimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arbitrage Strategy")
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of stock symbols to fetch data for.",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1d",
        help="Time period to fetch data for (e.g., '1d', '1mo', '1y').",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Data interval (e.g., '1m', '1h', '1d').",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the results using Plotly.",
    )
    parser.add_argument(
        "--spread-window",
        type=int,
        default=30,
        help="Rolling window size for spread calculation (default: 30).",
    )
    parser.add_argument(
        "--zscore-window",
        type=int,
        default=20,
        help="Rolling window size for Z-score calculation (default: 20).",
    )
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=2.0,
        help="Z-score threshold to enter a position (default: 2.0).",
    )
    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=0.0,
        help="Z-score threshold to exit a position (default: 0.0).",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization using Grid Search.",
    )
    args = parser.parse_args()

    if args.optimize:
        print(f"Running optimization for {args.symbols}...")
        optimizer = GridSearchOptimizer(
            args.symbols,
            args.period,
            args.interval,
        )
        results = optimizer.optimize()

        if not results.empty:
            print("\nTop 5 Parameter Combinations:")
            print(results.head(5))

            best_params = results.iloc[0]
            print("\nBest Parameters:")
            print(best_params)

            # Ask if user wants to run with best params?
            # For CLI, we just exit after optimization.
        else:
            print("Optimization failed to produce results.")

    else:
        from src.arbitrage import StrategyConfig

        config = StrategyConfig(
            spread_window=args.spread_window,
            zscore_window=args.zscore_window,
            entry_threshold=args.entry_threshold,
            exit_threshold=args.exit_threshold,
        )

        data1, data2, zscore_data = run_arbitrage_strategy(
            args.symbols,
            args.period,
            args.interval,
            config=config,
        )

        if args.plot:
            if data1 is not None and data2 is not None and zscore_data is not None:
                plot_dashboard(
                    data1, data2, zscore_data, args.symbols[0], args.symbols[1]
                )
            elif data1 is not None and data2 is not None:
                # Fallback if zscore_data is somehow None (shouldn't happen with current logic)
                print("Warning: Z-Score data missing, cannot plot dashboard.")
