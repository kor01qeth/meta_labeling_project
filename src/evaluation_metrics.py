"""Purpose:
    This module provides a compact summary of key strategy diagnostics, including
    annualized return and volatility, Sharpe ratio, maximum drawdown, daily return
    statistics, profit/loss ratio, and portfolio turnover.
"""

import numpy as np
import pandas as pd

def evaluate_strategy(returns_df, weights_df):
    """
    Evaluate a strategy using daily returns and portfolio weights.

    Parameters
        returns:
            One-column dataframe of strategy returns
        weights:
            Dataframe of strategy weights

    Output
        summary:
            Strategy performance table
    """

    # Conversion and alignment

    strategy_returns = returns_df.iloc[:, 0].astype(float).dropna()
    strategy_weights = weights_df.reindex(strategy_returns.index).astype(float)

    # Calculating cumulative Returns

    wealth = (1.0 + strategy_returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0

    # Annual metrics

    ann_return = (1.0 + strategy_returns).prod() ** (252 / len(strategy_returns)) - 1.0
    ann_vol = strategy_returns.std(ddof=1) * np.sqrt(252)

    # Sharpe

    if ann_vol == 0:
        sharpe = np.nan
    else:
        sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std(ddof=1)

    # Max drawdown

    max_dd = drawdown.min()

    # PnL ratio
    avg_daily_return = strategy_returns.mean()

    avg_positive_daily = strategy_returns[strategy_returns > 0].mean()
    avg_negative_daily = strategy_returns[strategy_returns < 0].mean()

    if pd.notna(avg_positive_daily) and pd.notna(avg_negative_daily) and avg_negative_daily != 0:
        profit_loss_ratio = avg_positive_daily / abs(avg_negative_daily)
    else:
        profit_loss_ratio = np.nan

    # Turnover

    daily_turnover = 0.5 * strategy_weights.diff().abs().sum(axis=1).dropna()
    avg_daily_turnover = daily_turnover.mean()
    ann_turnover = avg_daily_turnover * 252


    # Output

    summary = pd.DataFrame({
        "Metric": [
            "Annualized return",
            "Annualized volatility",
            "Sharpe ratio",
            "Maximum drawdown",
            "Average daily return",
            "Profit/loss ratio",
            "Average daily turnover",
            "Annualized turnover"
        ],
        "Value": [
            ann_return,
            ann_vol,
            sharpe,
            max_dd,
            avg_daily_return,
            profit_loss_ratio,
            avg_daily_turnover,
            ann_turnover
        ]
    })

    return summary
