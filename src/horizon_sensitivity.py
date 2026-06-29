"""
Purpose:
    This module provides diagnostics for understanding why triple-barrier horizon
    choices can change classification metrics differently from realised economic
    metrics. The functions focus on event-level payoffs, confusion groups,
    expectancy, filtering effects, and portfolio exposure summaries.
"""

import numpy as np
import pandas as pd


def attach_payoff_windows(
    df_returns,
    events_df,
    windows=(1, 5, 20, 30, 60, 150),
    date_col="t0",
    asset_col="stock",
    side_col="side",
    weight_col="weight",
):
    """
    Add realised event payoffs over multiple forward windows.

    Params
        df_returns
            Return matrix, indexed by date with stocks in columns.
        events_df
            Event-level dataframe with event dates, stocks, sides, and weights.
        windows
            Forward return windows. A 1-day window uses the return at t0.

    Output
        out
            Copy of events_df with raw, side-adjusted, and weight-adjusted
            payoff columns for each requested window.
    """

    out = events_df.copy()
    windows = tuple(sorted(set(int(window) for window in windows)))

    date_locs = df_returns.index.get_indexer(out[date_col])
    stock_locs = df_returns.columns.get_indexer(out[asset_col])

    valid_base = (date_locs >= 0) & (stock_locs >= 0)
    side = out[side_col].to_numpy(dtype=float)
    weight = out[weight_col].to_numpy(dtype=float)

    returns_array = df_returns.to_numpy(dtype=float)
    safe_returns = np.maximum(returns_array, -0.999999999)
    cum_log = np.vstack([
        np.zeros((1, returns_array.shape[1])),
        np.cumsum(np.log1p(safe_returns), axis=0),
    ])

    for window in windows:
        end_locs = date_locs + window
        valid = valid_base & (end_locs <= len(df_returns))

        realised = np.full(len(out), np.nan)
        realised[valid] = np.expm1(
            cum_log[end_locs[valid], stock_locs[valid]]
            - cum_log[date_locs[valid], stock_locs[valid]]
        )

        out[f"ret_{window}d"] = realised
        out[f"side_payoff_{window}d"] = side * realised
        out[f"weight_payoff_{window}d"] = weight * realised

    return out


def add_confusion_groups(df, target_col="label", pred_col="meta_label"):
    """
    Add TP, FP, TN, and FN confusion groups to an event dataframe.
    """

    out = df.copy()
    y_true = out[target_col].astype(int)
    y_pred = out[pred_col].astype(int)

    out["confusion_group"] = np.select(
        [
            (y_true == 1) & (y_pred == 1),
            (y_true == 0) & (y_pred == 1),
            (y_true == 0) & (y_pred == 0),
            (y_true == 1) & (y_pred == 0),
        ],
        ["TP", "FP", "TN", "FN"],
        default="unknown",
    )
    out["kept"] = y_pred == 1

    return out


def expectancy_summary(df, payoff_col, group_col=None):
    """
    Compute win rate, average win, average loss, and payoff expectancy.
    """

    if group_col is None:
        groups = [(None, df)]
        group_names = []
    else:
        groups = list(df.groupby(group_col, dropna=False))
        group_names = [group_col]

    rows = []

    for group_value, group in groups:
        payoff = group[payoff_col].dropna()
        wins = payoff[payoff > 0.0]
        losses = payoff[payoff < 0.0]

        n = len(payoff)
        win_rate = len(wins) / n if n > 0 else np.nan
        loss_rate = len(losses) / n if n > 0 else np.nan
        avg_win = wins.mean() if len(wins) > 0 else np.nan
        avg_loss = losses.abs().mean() if len(losses) > 0 else np.nan

        expectancy = 0.0
        if pd.notna(win_rate) and pd.notna(avg_win):
            expectancy += win_rate * avg_win
        if pd.notna(loss_rate) and pd.notna(avg_loss):
            expectancy -= loss_rate * avg_loss
        if n == 0:
            expectancy = np.nan

        row = {
            "payoff_col": payoff_col,
            "n": n,
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "avg_win": avg_win,
            "avg_abs_loss": avg_loss,
            "expectancy": expectancy,
            "total_payoff": payoff.sum(),
            "avg_payoff": payoff.mean() if n > 0 else np.nan,
        }

        if group_col is not None:
            row[group_col] = group_value

        rows.append(row)

    columns = group_names + [
        "payoff_col",
        "n",
        "win_count",
        "loss_count",
        "win_rate",
        "loss_rate",
        "avg_win",
        "avg_abs_loss",
        "expectancy",
        "total_payoff",
        "avg_payoff",
    ]

    return pd.DataFrame(rows).reindex(columns=columns)


def confusion_payoff_summary(df, payoff_col, group_col="confusion_group"):
    """
    Summarise payoff and expectancy by confusion-matrix group.
    """

    summary = expectancy_summary(
        df=df,
        payoff_col=payoff_col,
        group_col=group_col,
    )

    total_n = summary["n"].sum()
    total_payoff = summary["total_payoff"].sum()

    summary["share_of_events"] = summary["n"] / total_n if total_n > 0 else np.nan
    summary["share_of_payoff"] = (
        summary["total_payoff"] / total_payoff
        if total_payoff != 0
        else np.nan
    )

    return summary


def filtering_effect_summary(df, payoff_col, pred_col="meta_label", target_col="label"):
    """
    Summarise the economic effect of keeping or filtering trades.

    Filtering effect is defined as the negative of the filtered trade payoff.
    A positive value means the filter avoided a loss; a negative value means it
    missed a winner.
    """

    out = df.copy()
    out["decision"] = np.where(out[pred_col].astype(int) == 1, "kept", "filtered")
    out["label_group"] = np.where(
        out[target_col].astype(int) == 1,
        "positive_label",
        "negative_label",
    )
    out["filtering_effect"] = np.where(
        out["decision"] == "filtered",
        -out[payoff_col],
        0.0,
    )

    rows = []
    for (decision, label_group), group in out.groupby(["decision", "label_group"]):
        payoff = group[payoff_col].dropna()
        filtering_effect = group["filtering_effect"].dropna()

        rows.append({
            "decision": decision,
            "label_group": label_group,
            "n": len(group),
            "avg_payoff": payoff.mean(),
            "total_payoff": payoff.sum(),
            "avg_filtering_effect": filtering_effect.mean(),
            "total_filtering_effect": filtering_effect.sum(),
        })

    return pd.DataFrame(rows)


def side_diagnostics(
    df,
    payoff_col,
    side_col="side",
    pred_col="meta_label",
    target_col="label",
    proba_col="probability",
):
    """
    Summarise filtering and payoff behaviour separately for longs and shorts.
    """

    out = df.copy()
    out["primary_side"] = np.where(out[side_col] > 0, "long", "short")

    rows = []
    for primary_side, group in out.groupby("primary_side"):
        kept = group[pred_col].astype(int) == 1
        payoff = group[payoff_col].dropna()
        kept_payoff = group.loc[kept, payoff_col].dropna()

        gross_abs_weight = group["weight"].abs().sum()
        kept_abs_weight = group.loc[kept, "weight"].abs().sum()

        rows.append({
            "primary_side": primary_side,
            "n_events": len(group),
            "kept_events": int(kept.sum()),
            "kept_rate": kept.mean(),
            "positive_label_rate": group[target_col].mean(),
            "avg_probability": group[proba_col].mean(),
            "avg_abs_weight": group["weight"].abs().mean(),
            "gross_abs_weight": gross_abs_weight,
            "kept_abs_weight": kept_abs_weight,
            "kept_abs_weight_share": (
                kept_abs_weight / gross_abs_weight
                if gross_abs_weight != 0
                else np.nan
            ),
            "avg_payoff": payoff.mean(),
            "avg_kept_payoff": kept_payoff.mean(),
            "total_kept_payoff": kept_payoff.sum(),
        })

    return pd.DataFrame(rows)


def exposure_summary(weights_df):
    """
    Return per-date net, gross, long, and short exposure diagnostics.
    """

    weights = weights_df.astype(float)
    gross_exposure = weights.abs().sum(axis=1)
    long_exposure = weights.clip(lower=0.0).sum(axis=1)
    short_exposure = weights.clip(upper=0.0).sum(axis=1)
    gross_denom = gross_exposure.replace(0.0, np.nan)

    return pd.DataFrame({
        "net_exposure": weights.sum(axis=1),
        "gross_exposure": gross_exposure,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "long_gross_share": long_exposure / gross_denom,
        "short_gross_share": short_exposure.abs() / gross_denom,
    })
