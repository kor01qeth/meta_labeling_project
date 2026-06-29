from numba import njit, prange
import pandas as pd
import numpy as np

"""
Purpose:
    Fast implementation of the triple-barrier labeling method.

    This module mirrors the logic of simple_triple_barrier_labels,
    but moves the event-level barrier search into Numba for speed.
"""

import numpy as np
import pandas as pd

from numba import njit, prange


# Numba core

@njit(parallel=True)
def _triple_barrier_numba(returns_array, weights_array, vol_array, event_locs, horizon):
    """
    Fast triple-barrier label calculation.

    Params
        returns_array
            Full return matrix
        weights_array
            Weight matrix aligned to returns_array
        vol_array
            Rolling volatility matrix aligned to returns_array
        event_locs
            Integer locations of event start dates
        horizon
            Vertical barrier horizon

    Output
        labels
            Triple-barrier labels
        sides
            Position sides
    """

    # Dimensions

    K = len(event_locs)
    N = returns_array.shape[1]

    labels = np.zeros((K, N), dtype=np.int8)
    sides = np.zeros((K, N), dtype=np.int8)

    # Events

    for k in prange(K):

        loc = event_locs[k]

        for j in range(N):

            w = weights_array[loc, j]
            sigma = vol_array[loc, j]

            # Side

            if w > 0.0:
                side = 1
            elif w < 0.0:
                side = -1
            else:
                side = 0

            sides[k, j] = side

            # Invalid cases

            if np.isnan(w) or np.isnan(sigma) or side == 0:
                labels[k, j] = 0
                continue

            # Barriers

            pt = sigma * np.sqrt(horizon)
            sl = sigma * np.sqrt(horizon)

            # Path

            cum_value = 1.0
            label = 0

            for h in range(horizon):

                r = returns_array[loc + h, j]

                if np.isnan(r):
                    continue

                cum_value = cum_value * (1.0 + r)
                cum_ret = cum_value - 1.0
                signed_cum_ret = side * cum_ret

                if signed_cum_ret >= pt:
                    label = 1
                    break

                if signed_cum_ret <= -sl:
                    label = 0
                    break

            labels[k, j] = label

    return labels, sides

# Triple barrier

def simple_triple_barrier_labels_fast(df_returns, wt, horizon, vol_window=20):
    """
    Fast triple-barrier labeling using Numba.

    Params
        df_returns
            Full return matrix, indexed by date with stocks in columns.
        wt
            Primary-strategy weights, indexed by date with stocks in columns.
        horizon
            Number of trading days used as the vertical barrier horizon.
        vol_window
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

    # Align

    df_returns = df_returns.reindex(columns=wt.columns)
    wt_aligned = wt.reindex(df_returns.index)

    # Volatility

    rolling_vol = df_returns.shift(1).rolling(vol_window).std(ddof=1)

    # Valid dates

    valid_dates = wt.index[:len(wt.index) - horizon]
    event_locs = df_returns.index.get_indexer(valid_dates)

    if np.any(event_locs < 0):
        raise ValueError("Some weight dates are not present in df_returns.index.")

    # Arrays

    returns_array = df_returns.to_numpy(dtype=np.float64)
    weights_array = wt_aligned.to_numpy(dtype=np.float64)
    vol_array = rolling_vol.to_numpy(dtype=np.float64)

    # Labels

    labels_array, sides_array = _triple_barrier_numba(
        returns_array=returns_array,
        weights_array=weights_array,
        vol_array=vol_array,
        event_locs=event_locs.astype(np.int64),
        horizon=horizon
    )

    # Flatten

    stocks = wt.columns

    t0_flat = np.repeat(valid_dates.to_numpy(), len(stocks))
    stock_flat = np.tile(stocks.to_numpy(), len(valid_dates))

    side_flat = sides_array.reshape(-1)
    label_flat = labels_array.reshape(-1)
    weight_flat = weights_array[event_locs, :].reshape(-1)

    # Output

    labels = pd.DataFrame({
        "t0": t0_flat,
        "stock": stock_flat,
        "side": side_flat,
        "weight": weight_flat,
        "label": label_flat
    })

    return labels