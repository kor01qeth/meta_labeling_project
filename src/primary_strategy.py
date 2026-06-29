# Primary strategy

def build_primary_momentum_strategy_fast(df_returns, lookback=260):
    """
    Fast cross-sectional momentum primary strategy.

    Params
        df_returns
            Return matrix
        lookback
            Lookback window used for the primary signal

    Output
        pnl
            One-column dataframe of strategy returns
        wt
            Primary strategy weights
        signal
            Cross-sectional momentum signal
    """

    # Lag

    returns_lagged = df_returns.shift(2)
    window = lookback - 1

    # Time-series score

    rolling_sum = returns_lagged.rolling(window).sum()
    rolling_std = returns_lagged.rolling(window).std(ddof=1)

    score = rolling_sum / rolling_std

    # Cross-sectional standardization

    score_mean = score.mean(axis=1)
    score_std = score.std(axis=1, ddof=1)

    signal = score.sub(score_mean, axis=0).div(score_std, axis=0)

    # Dates

    signal = signal.loc[df_returns.index[lookback:]]

    # Weights

    gross = signal.abs().sum(axis=1)
    wt = signal.div(gross, axis=0).fillna(0.0)

    # PnL

    pnl = (wt * df_returns.reindex(wt.index)).sum(axis=1).to_frame("PnL")

    return pnl, wt, signal