"""
Purpose:
    The below function is used to determine the
    relationship between the strategy's scores today 
    and future returns across the cross-section, 
    considering a mesh of return horizons.
"""

import numpy as np
import pandas as pd

def compute_spearman_ic_curve(signal, df_returns, horizons):
    ic_curve = []

    for h in horizons:
        daily_ic = []

        for t in signal.index:
            loc = df_returns.index.get_loc(t)

            if loc + h > len(df_returns.index):
                continue

            future_window = df_returns.iloc[loc:loc+h, :]
            future_ret = (1 + future_window).prod(axis=0) - 1

            ic_val = signal.loc[t].corr(future_ret, method="spearman")
            daily_ic.append(ic_val)

        ic_curve.append(np.mean(daily_ic))

    return ic_curve
