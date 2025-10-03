"""Main entry point for the application."""

import argparse
from src.arbitrage import run_arbitrage_strategy

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
    args = parser.parse_args()

    run_arbitrage_strategy(args.symbols, args.period, args.interval)
