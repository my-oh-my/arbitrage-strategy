"""Module for plotting arbitrage strategy results."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_dual_prices(
    data1: pd.DataFrame, data2: pd.DataFrame, symbol1: str, symbol2: str
):
    """Plots the normalized prices of two symbols."""
    # Normalize prices to start at 100
    norm_data1 = data1["Close"] / data1["Close"].iloc[0] * 100
    norm_data2 = data2["Close"] / data2["Close"].iloc[0] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data1["Datetime"], y=norm_data1, mode="lines", name=symbol1)
    )
    fig.add_trace(
        go.Scatter(x=data2["Datetime"], y=norm_data2, mode="lines", name=symbol2)
    )

    fig.update_layout(
        title=f"Normalized Price Comparison: {symbol1} vs {symbol2}",
        xaxis_title="Date",
        yaxis_title="Normalized Price (Start=100)",
    )
    fig.show()


def plot_zscore_and_signals(zscore_data: pd.DataFrame):
    """Plots the Z-score with entry/exit thresholds and trading signals."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # Z-Score Plot
    fig.add_trace(
        go.Scatter(
            x=zscore_data["Datetime"],
            y=zscore_data["Z_Score"],
            mode="lines",
            name="Z-Score",
        ),
        row=1,
        col=1,
    )

    # Thresholds
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=-2.0, line_dash="dash", line_color="green", row=1, col=1)
    fig.add_hline(y=0.0, line_dash="dash", line_color="black", row=1, col=1)

    # Signals
    # Buy Signals (Long Spread)
    buy_signals = zscore_data[zscore_data["Signal"] == 1]
    fig.add_trace(
        go.Scatter(
            x=buy_signals["Datetime"],
            y=buy_signals["Z_Score"],
            mode="markers",
            marker={"color": "green", "size": 8, "symbol": "triangle-up"},
            name="Long Spread",
        ),
        row=1,
        col=1,
    )

    # Sell Signals (Short Spread)
    sell_signals = zscore_data[zscore_data["Signal"] == -1]
    fig.add_trace(
        go.Scatter(
            x=sell_signals["Datetime"],
            y=sell_signals["Z_Score"],
            mode="markers",
            marker={"color": "red", "size": 8, "symbol": "triangle-down"},
            name="Short Spread",
        ),
        row=1,
        col=1,
    )

    # Spread Plot
    fig.add_trace(
        go.Scatter(
            x=zscore_data["Datetime"],
            y=zscore_data["Spread"],
            mode="lines",
            name="Spread",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Z-Score and Spread Analysis",
        xaxis_title="Date",
        yaxis_title="Value",
        height=800,
    )
    fig.show()


def plot_correlation_scatter(
    data1: pd.DataFrame, data2: pd.DataFrame, symbol1: str, symbol2: str
):
    """Plots a scatter plot of returns to visualize correlation."""
    # Calculate returns locally to avoid modifying the original dataframes in place if they are used elsewhere
    # or just use the passed dataframes if they are copies.
    # To avoid duplication warning with arbitrage.py, we can just calculate it here directly.
    d1_returns = data1["Close"].pct_change()
    d2_returns = data2["Close"].pct_change()

    # Create a temporary dataframe for merging
    temp_data1 = pd.DataFrame({"Datetime": data1["Datetime"], "Return": d1_returns})
    temp_data2 = pd.DataFrame({"Datetime": data2["Datetime"], "Return": d2_returns})

    merged_data = pd.merge(
        temp_data1,
        temp_data2,
        on="Datetime",
        suffixes=(f"_{symbol1}", f"_{symbol2}"),
    ).dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged_data[f"Return_{symbol1}"],
            y=merged_data[f"Return_{symbol2}"],
            mode="markers",
            name="Returns",
        )
    )

    fig.update_layout(
        title=f"Correlation Scatter: {symbol1} vs {symbol2}",
        xaxis_title=f"{symbol1} Returns",
        yaxis_title=f"{symbol2} Returns",
    )
    fig.show()


def plot_dashboard(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    zscore_data: pd.DataFrame,
    symbol1: str,
    symbol2: str,
):
    """Plots a comprehensive dashboard of the arbitrage strategy results."""
    # Normalize prices
    norm_data1 = data1["Close"] / data1["Close"].iloc[0] * 100
    norm_data2 = data2["Close"] / data2["Close"].iloc[0] * 100

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f"Candlestick Prices ({symbol1} vs {symbol2})",
            f"Normalized Prices ({symbol1} vs {symbol2})",
            "Price vs Hedged Price (Spread Basis)",
            "Z-Score & Signals",
            "Spread",
        ),
        row_heights=[0.3, 0.2, 0.2, 0.15, 0.15],
        specs=[[{"secondary_y": True}], [{}], [{}], [{}], [{}]],
    )

    # 1. Candlestick Prices (Dual Axis)
    fig.add_trace(
        go.Candlestick(
            x=data1["Datetime"],
            open=data1["Open"],
            high=data1["High"],
            low=data1["Low"],
            close=data1["Close"],
            name=symbol1,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Candlestick(
            x=data2["Datetime"],
            open=data2["Open"],
            high=data2["High"],
            low=data2["Low"],
            close=data2["Close"],
            name=symbol2,
            increasing_line_color="cyan",
            decreasing_line_color="gray",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # 2. Normalized Prices
    fig.add_trace(
        go.Scatter(
            x=data1["Datetime"], y=norm_data1, mode="lines", name=f"{symbol1} (Norm)"
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data2["Datetime"], y=norm_data2, mode="lines", name=f"{symbol2} (Norm)"
        ),
        row=2,
        col=1,
    )

    # 3. Price vs Hedged Price
    hedged_price2 = zscore_data[f"Close_{symbol2}"] * zscore_data["HedgeRatio"]
    fig.add_trace(
        go.Scatter(
            x=zscore_data["Datetime"],
            y=zscore_data[f"Close_{symbol1}"],
            mode="lines",
            name=symbol1,
            line={"color": "blue"},
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=zscore_data["Datetime"],
            y=hedged_price2,
            mode="lines",
            name=f"Hedged {symbol2}",
            line={"color": "orange"},
        ),
        row=3,
        col=1,
    )

    # 4. Z-Score & Signals
    fig.add_trace(
        go.Scatter(
            x=zscore_data["Datetime"],
            y=zscore_data["Z_Score"],
            mode="lines",
            name="Z-Score",
            line={"color": "blue"},
        ),
        row=4,
        col=1,
    )
    # Thresholds
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=-2.0, line_dash="dash", line_color="green", row=4, col=1)
    fig.add_hline(y=0.0, line_dash="dash", line_color="black", row=4, col=1)

    # Signals
    buy_signals = zscore_data[zscore_data["Signal"] == 1]
    sell_signals = zscore_data[zscore_data["Signal"] == -1]

    fig.add_trace(
        go.Scatter(
            x=buy_signals["Datetime"],
            y=buy_signals["Z_Score"],
            mode="markers",
            marker={"color": "green", "size": 8, "symbol": "triangle-up"},
            name="Long Spread",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sell_signals["Datetime"],
            y=sell_signals["Z_Score"],
            mode="markers",
            marker={"color": "red", "size": 8, "symbol": "triangle-down"},
            name="Short Spread",
        ),
        row=4,
        col=1,
    )

    # 5. Spread
    fig.add_trace(
        go.Scatter(
            x=zscore_data["Datetime"],
            y=zscore_data["Spread"],
            mode="lines",
            name="Spread",
            line={"color": "purple"},
        ),
        row=5,
        col=1,
    )

    # Determine the global date range
    min_date = min(data1["Datetime"].min(), data2["Datetime"].min())
    max_date = max(data1["Datetime"].max(), data2["Datetime"].max())

    fig.update_layout(
        title="Arbitrage Strategy Dashboard",
        height=1600,
        showlegend=True,
        xaxis5={"range": [min_date, max_date]},  # Apply to the bottom x-axis (shared)
        yaxis1_title=f"{symbol1} Price",
        yaxis2_title=f"{symbol2} Price",
    )

    # Update axes labels
    fig.update_yaxes(title_text="Normalized Price", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=3, col=1)
    fig.update_yaxes(title_text="Z-Score", row=4, col=1)
    fig.update_yaxes(title_text="Spread", row=5, col=1)
    fig.update_xaxes(title_text="Date", row=5, col=1)

    # Remove rangeslider from candlestick
    fig.update_layout(xaxis_rangeslider_visible=False)

    fig.show()
