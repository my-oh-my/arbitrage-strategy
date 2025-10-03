"""Module for running arbitrage strategy."""

from src.data_fetcher import fetch_market_data


def run_arbitrage_strategy(symbols: list[str], period: str, interval: str):
    """Fetches data for the given symbols and prints them."""
    for symbol in symbols:
        try:
            data = fetch_market_data(symbol, period, interval)
            print(f"\n{symbol} Data:")
            print(data)
        except ValueError as e:
            print(f"Error fetching data for {symbol}: {e}")
