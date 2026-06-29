"""
Purpose: 
    This module implements the triple barrier labeling
    method for assigning labels to data points based on 
    the future price movements of stocks. The method uses 
    three barriers: 
    a profit-taking barrier,
    a stop-loss barrier, 
    and a time barrier. 
    
    The labels are assigned based on which barrier 
    is hit first. The labelling has been done in the way 
    that only the profit-taking barrier results in a 1 label 
    upon it being hit first. The labelling has been done 
    to accomodate both long and short positions at the same time. 
    Weights have been used to determine sides of positions.

"""


import numpy as np
import pandas as pd

def simple_triple_barrier_labels(df_returns, wt, horizon, vol_window=20):
    """
    Parameters
        df_returns
            Full return matrix, indexed by date with stocks in columns.
        wt
            Primary-strategy weights, indexed by date with stocks in columns.
        horizon 
            Number of trading days used as the vertical barrier horizon.
        vol_window (default=20)
            Rolling window used to estimate daily volatility.

    Output
        pd.DataFrame
            With columns:
            - t0
            - stock
            - side
            - weight
            - label
    """
    # Using rolling volatility for barrier construction
    ### Updated: Added a shift for rolling vol calculation
    rolling_vol = df_returns.shift(1).rolling(vol_window).std(ddof=1)
    results = []

    valid_dates = wt.index[:len(wt.index) - horizon]

    # Double for loop for going through each day and each stock
    for t0 in valid_dates:
        loc = df_returns.index.get_loc(t0)

        for stock in wt.columns:
            w = wt.loc[t0, stock]
            side = int(np.sign(w))      # Based on weights of primary strategy
            sigma = rolling_vol.loc[t0, stock]

            # Fixing barriers - symmetric
            pt = sigma * np.sqrt(horizon)
            sl = sigma * np.sqrt(horizon)

            # Using cumulative returns
            ### Updated: taking the +1 away from loc : loc + horizon
            future_window = df_returns.iloc[loc : loc + horizon][stock]
            cum_ret = (1 + future_window).cumprod() - 1
            signed_cum_ret = side * cum_ret

            # Note where is the first time the path crosses the barrier
            pt_cross = np.where(signed_cum_ret >= pt)[0]
            sl_cross = np.where(signed_cum_ret <= -sl)[0]

            first_pt = pt_cross[0] if len(pt_cross) > 0 else None
            first_sl = sl_cross[0] if len(sl_cross) > 0 else None

            # Assign label 1 only if pt barrier was hit first
            if first_pt is not None and (first_sl is None or first_pt < first_sl):
                label = 1
            else:
                label = 0

            results.append({
                "t0": t0,
                "stock": stock,
                "side": side,
                "weight": w,
                "label": label
            })

    return pd.DataFrame(results)