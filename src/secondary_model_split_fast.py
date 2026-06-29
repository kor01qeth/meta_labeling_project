"""
Purpose:
    Fast helpers for split long/short secondary meta-labeling models.

    The module keeps the event-level dataset compatible with
    secondary_model_fast, but fits separate logistic models for primary long
    and primary short candidates. It also provides portfolio-weight helpers for
    gross-exposure and dollar-neutral secondary strategies.
"""

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.secondary_model_fast import build_secondary_dataset_multiwindow_fast, create_filtered_weights_fast


def build_split_secondary_dataset_fast(df_returns, labels, window=150, windows=None):
    """
    Build a fast secondary dataset with an explicit long/short side label.
    """

    if windows is None:
        windows = (window,)

    out = build_secondary_dataset_multiwindow_fast(
        df_returns=df_returns,
        labels=labels,
        windows=windows,
    )
    out["primary_side"] = np.where(out["side"] > 0, "long", "short")

    return out


def fit_split_logistic(train_df, val_df, test_df, feature_cols, target_col="label"):
    """
    Fit separate logistic secondary models for long and short candidates.

    Output
        models
            Dictionary keyed by "long" and "short".
        train_out, val_out, test_out
            Recombined dataframes with a common probability column.
    """

    train_parts = []
    val_parts = []
    test_parts = []
    models = {}

    for primary_side in ["long", "short"]:
        train_side = train_df[train_df["primary_side"] == primary_side].copy()
        val_side = val_df[val_df["primary_side"] == primary_side].copy()
        test_side = test_df[test_df["primary_side"] == primary_side].copy()

        model, train_pred, val_pred, test_pred = _fit_one_side_logistic(
            train_df=train_side,
            val_df=val_side,
            test_df=test_side,
            feature_cols=feature_cols,
            target_col=target_col,
        )

        models[primary_side] = model
        train_parts.append(train_pred)
        val_parts.append(val_pred)
        test_parts.append(test_pred)

    train_out = _recombine_side_predictions(train_parts)
    val_out = _recombine_side_predictions(val_parts)
    test_out = _recombine_side_predictions(test_parts)

    return models, train_out, val_out, test_out


def create_split_filtered_weights_fast(
    primary_weights,
    filtered_df,
    date_col="t0",
    asset_col="stock",
    signal_col="meta_label",
):
    """
    Create filtered weights from split-model decisions.
    """

    return create_filtered_weights_fast(
        primary_weights=primary_weights,
        filtered_df=filtered_df,
        date_col=date_col,
        asset_col=asset_col,
        signal_col=signal_col,
    )


def create_dollar_neutral_filtered_weights(
    primary_weights,
    filtered_df,
    date_col="t0",
    asset_col="stock",
    signal_col="meta_label",
):
    """
    Filter primary weights and separately rescale kept longs and shorts.

    Each non-empty side is rescaled to the original primary side exposure on
    that date. If no trade is kept on one side, that side stays in cash.
    """

    filtered_weights = create_split_filtered_weights_fast(
        primary_weights=primary_weights,
        filtered_df=filtered_df,
        date_col=date_col,
        asset_col=asset_col,
        signal_col=signal_col,
    )

    primary_long = primary_weights.clip(lower=0.0).sum(axis=1)
    primary_short_abs = primary_weights.clip(upper=0.0).abs().sum(axis=1)

    filtered_long = filtered_weights.clip(lower=0.0)
    filtered_short = filtered_weights.clip(upper=0.0)

    filtered_long_exposure = filtered_long.sum(axis=1)
    filtered_short_abs_exposure = filtered_short.abs().sum(axis=1)

    long_scale = primary_long / filtered_long_exposure
    short_scale = primary_short_abs / filtered_short_abs_exposure

    long_scale = long_scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    short_scale = short_scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    dollar_neutral_weights = (
        filtered_long.mul(long_scale, axis=0)
        + filtered_short.mul(short_scale, axis=0)
    )

    return dollar_neutral_weights


def dollar_neutral_side_availability(weights_df):
    """
    Return the share of dates with non-empty long and short filtered sides.
    """

    long_present = weights_df.clip(lower=0.0).sum(axis=1) > 0.0
    short_present = weights_df.clip(upper=0.0).abs().sum(axis=1) > 0.0

    return {
        "long_present_rate": long_present.mean(),
        "short_present_rate": short_present.mean(),
        "both_sides_present_rate": (long_present & short_present).mean(),
        "missing_long_rate": (~long_present).mean(),
        "missing_short_rate": (~short_present).mean(),
    }


def _fit_one_side_logistic(train_df, val_df, test_df, feature_cols, target_col):
    """
    Fit one side model, with a constant-probability fallback if needed.
    """

    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    if len(train_df) == 0:
        for out in [train_out, val_out, test_out]:
            out["probability"] = np.nan
        return None, train_out, val_out, test_out

    y_train = train_df[target_col]
    if y_train.nunique() < 2:
        probability = float(y_train.mean())
        for out in [train_out, val_out, test_out]:
            out["probability"] = probability
        return {"constant_probability": probability}, train_out, val_out, test_out

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    X_train = train_df[feature_cols]
    model.fit(X_train, y_train)

    train_out["probability"] = _predict_probability(model, train_df, feature_cols)
    val_out["probability"] = _predict_probability(model, val_df, feature_cols)
    test_out["probability"] = _predict_probability(model, test_df, feature_cols)

    return model, train_out, val_out, test_out


def _recombine_side_predictions(parts):
    """
    Recombine side-specific predictions while preserving the original order.
    """

    out = pd.concat(parts, axis=0)
    return out.sort_index()


def _predict_probability(model, df, feature_cols):
    """
    Predict probabilities while allowing empty side-specific slices.
    """

    if len(df) == 0:
        return np.array([], dtype=float)

    return model.predict_proba(df[feature_cols])[:, 1]
