# Arbitrage Strategy

This project implements a statistical arbitrage strategy for pairs trading. It includes modules for backtesting, hyperparameter optimization, and interactive visualization.

## Features

-   **Core Strategy**: Rolling OLS for spread calculation, Z-score generation, and threshold-based signal generation.
-   **Backtesting**: Profit & Loss (PnL) calculation with key performance metrics (Sharpe Ratio, Max Drawdown).
-   **Optimization**: Grid Search optimizer to find the best strategy parameters.
-   **Visualization**: Interactive Plotly dashboard for analyzing prices, spreads, z-scores, and signals.

## Setup

1.  **Create a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```

## Usage

### Running the Strategy

To run the arbitrage strategy on two symbols (e.g., PEO.WA and PKO.WA) with default parameters and plotting enabled:

```bash
python -m src.main --symbols PEO.WA PKO.WA --period 1y --interval 1h --plot
```

### Running Optimization

To optimize strategy parameters using Grid Search:

```bash
python -m src.main --symbols PEO.WA PKO.WA --period 1y --interval 1d --optimize
```

### Configuration

You can customize the strategy using the following CLI arguments:

-   `--symbols`: List of two stock symbols to trade.
-   `--period`: Data period to fetch (e.g., `1d`, `1mo`, `1y`).
-   `--interval`: Data interval (e.g., `1m`, `1h`, `1d`).
-   `--spread-window`: Rolling window size for spread calculation (default: 30).
-   `--zscore-window`: Rolling window size for Z-score calculation (default: 20).
-   `--entry-threshold`: Z-score threshold to enter a position (default: 2.0).
-   `--exit-threshold`: Z-score threshold to exit a position (default: 0.0).

## Development

This project uses Black for code formatting and Pylint for linting.

*   **To format the code:**

    ```bash
    black src/ tests/
    ```

*   **To run the linter:**

    ```bash
    pylint src/ tests/
    ```

*   **To run tests:**

    ```bash
    pytest tests/
    ```
