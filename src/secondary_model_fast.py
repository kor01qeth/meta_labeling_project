"""
Purpose:
    The functions below are replicas of the ones in secondary_model.py with the 
    addition of 2 changes in the building of the secondary dataset, as well as in 
    the building of filtered weights, where the change has beed added to reduce 
    time complexity of calculations by introducing matrix computation. 
    Furhtermore, in the secondary model, we have taken away certain features to 
    comply with the idea of simplifications.
"""


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score 

# Fast secondary dataset

def build_secondary_dataset_fast(df_returns, labels, window=150):
    """
    Fast event-level secondary model dataset construction.

    Params
        df_returns
            Full return matrix, indexed by date with stocks in columns.
        labels
            Output of the triple-barrier labeling function.
        window
            Lookback window used for return and volatility features.

    Output
        secondary_data
            Event-level dataset with columns:
            - t0
            - stock
            - weight
            - side
            - abs_weight
            - ret_150
            - signed_ret_150
            - vol_150
            - corr_mkt_150
            - label
    """

    # Features

    returns_lagged = df_returns.shift(1)

    ret_window = np.expm1(
        np.log1p(returns_lagged).rolling(window).sum()
    )

    vol_window = returns_lagged.rolling(window).std(ddof=1)

    market_return = returns_lagged.mean(axis=1)
    stock_sum = returns_lagged.rolling(window).sum()
    stock_sq_sum = returns_lagged.pow(2).rolling(window).sum()
    market_sum = market_return.rolling(window).sum()
    market_sq_sum = market_return.pow(2).rolling(window).sum()
    cross_sum = returns_lagged.mul(market_return, axis=0).rolling(window).sum()

    cov_num = cross_sum.sub(stock_sum.mul(market_sum, axis=0).div(window))
    stock_var_num = stock_sq_sum.sub(stock_sum.pow(2).div(window))
    market_var_num = market_sq_sum.sub(market_sum.pow(2).div(window))
    corr_denom = stock_var_num.mul(market_var_num, axis=0).pow(0.5)
    corr_mkt_window = cov_num.div(corr_denom.replace(0.0, np.nan))

    # Locations

    date_locs = df_returns.index.get_indexer(labels["t0"])
    stock_locs = df_returns.columns.get_indexer(labels["stock"])

    valid = (
        (date_locs >= window) &
        (stock_locs >= 0)
    )

    labels_valid = labels.loc[valid].copy()
    date_locs = date_locs[valid]
    stock_locs = stock_locs[valid]

    # Arrays

    ret_values = ret_window.to_numpy()
    vol_values = vol_window.to_numpy()
    corr_values = corr_mkt_window.to_numpy()

    ret_feature = ret_values[date_locs, stock_locs]
    vol_feature = vol_values[date_locs, stock_locs]
    corr_feature = corr_values[date_locs, stock_locs]
    weight_feature = labels_valid["weight"].to_numpy()
    side_feature = np.sign(weight_feature)
    abs_weight_feature = np.abs(weight_feature)
    signed_ret_feature = side_feature * ret_feature

    # Output

    secondary_data = pd.DataFrame({
        "t0": labels_valid["t0"].to_numpy(),
        "stock": labels_valid["stock"].to_numpy(),
        "weight": weight_feature,
        "side": side_feature,
        "abs_weight": abs_weight_feature,
        f"ret_{window}": ret_feature,
        f"signed_ret_{window}": signed_ret_feature,
        f"vol_{window}": vol_feature,
        f"corr_mkt_{window}": corr_feature,
        "label": labels_valid["label"].to_numpy()
    })

    # Previous active simple feature sets, kept above as raw columns so it is easy
    # to switch back in the notebook:
    # feature_cols = ["weight", f"ret_{window}", f"vol_{window}"]
    # feature_cols = ["side", "abs_weight", f"signed_ret_{window}", f"vol_{window}"]

    return secondary_data


def build_secondary_dataset_multiwindow_fast(df_returns, labels, windows=(30, 60, 150)):
    """
    Fast secondary dataset with multiple t0-safe horizon features.

    The rank features are cross-sectional ranks of the current trade weight on
    each event date. They are not rolling stock-level averages, so they describe
    current signal strength without directly encoding persistent stock identity.

    Params
        df_returns
            Full return matrix, indexed by date with stocks in columns.
        labels
            Output of the triple-barrier labeling function.
        windows
            Lookback windows for return and volatility features.

    Output
        secondary_data
            Event-level dataset with multi-horizon return, signed-return,
            volatility, rank, and change features.
    """

    windows = tuple(sorted(windows))
    max_window = max(windows)
    returns_lagged = df_returns.shift(1)

    # Current-date primary-signal ranks. These are cross-sectional, not rolling.
    weight_matrix = labels.pivot(index="t0", columns="stock", values="weight")
    weight_matrix = weight_matrix.reindex(index=df_returns.index, columns=df_returns.columns)
    weight_rank = weight_matrix.rank(axis=1, pct=True)
    abs_weight_rank = weight_matrix.abs().rank(axis=1, pct=True)

    # Locations
    date_locs = df_returns.index.get_indexer(labels["t0"])
    stock_locs = df_returns.columns.get_indexer(labels["stock"])

    valid = (
        (date_locs >= max_window) &
        (stock_locs >= 0)
    )

    labels_valid = labels.loc[valid].copy()
    date_locs = date_locs[valid]
    stock_locs = stock_locs[valid]

    weight_feature = labels_valid["weight"].to_numpy()
    side_feature = np.sign(weight_feature)

    features = {
        "t0": labels_valid["t0"].to_numpy(),
        "stock": labels_valid["stock"].to_numpy(),
        "weight": weight_feature,
        "side": side_feature,
        "abs_weight": np.abs(weight_feature),
        "weight_rank": weight_rank.to_numpy()[date_locs, stock_locs],
        "abs_weight_rank": abs_weight_rank.to_numpy()[date_locs, stock_locs],
    }

    ret_features = {}
    signed_ret_features = {}
    vol_features = {}

    for window in windows:
        ret_window = np.expm1(
            np.log1p(returns_lagged).rolling(window).sum()
        )
        vol_window = returns_lagged.rolling(window).std(ddof=1)

        ret_feature = ret_window.to_numpy()[date_locs, stock_locs]
        signed_ret_feature = side_feature * ret_feature
        vol_feature = vol_window.to_numpy()[date_locs, stock_locs]

        ret_features[window] = ret_feature
        signed_ret_features[window] = signed_ret_feature
        vol_features[window] = vol_feature

        features[f"ret_{window}"] = ret_feature
        features[f"signed_ret_{window}"] = signed_ret_feature
        features[f"vol_{window}"] = vol_feature

    if 30 in windows and 150 in windows:
        features["ret_30_minus_ret_150"] = ret_features[30] - ret_features[150]
        features["signed_ret_30_minus_signed_ret_150"] = (
            signed_ret_features[30] - signed_ret_features[150]
        )
        features["vol_30_div_vol_150"] = vol_features[30] / np.where(
            vol_features[150] == 0.0,
            np.nan,
            vol_features[150]
        )

    features["label"] = labels_valid["label"].to_numpy()

    return pd.DataFrame(features)



def purged_time_split(df, horizon, train_frac=0.50, val_frac=0.25, dates_col="t0"):
    """
    Parameters
        df
            Secondary-model event dataframe
        horizon
            Triple-barrier horizon
        train_frac
            Fraction for training
        val_frac
            Fraction for validation
        date_col:
            Event start date
    Output
        train, val, test
            Purged train, validation and test datasets
        val_start, test_start
            Split dates
    """

    out = df.copy()

    # For purging, have to get the end of barrier, we use trading days, can't just add horizon
    unique_dates = np.sort(out[dates_col].dropna().unique())

    train_end_id = int(len(unique_dates) * train_frac)
    val_end_id = int(len(unique_dates) * (train_frac + val_frac))

    val_start = unique_dates[train_end_id]
    test_start = unique_dates[val_end_id]

    date_to_id = {date: id for id, date in enumerate(unique_dates)}

    # Creating lengthened dates with horizon added
    out["barrier_start"] = out[dates_col].map(date_to_id)
    out["barrier_end"] = np.minimum(out["barrier_start"] + horizon, len(unique_dates) - 1)
    out["end_date"] = out["barrier_end"].map(lambda i: unique_dates[i])

    # Making sure no intersection
    train = out[
        (out[dates_col] < val_start) &
        (out["end_date"] < val_start)
    ].copy()

    val = out[
        (out[dates_col] >= val_start) &
        (out[dates_col] < test_start) &
        (out["end_date"] < test_start)
    ].copy()

    test = out[
        out[dates_col] >= test_start
    ].copy()

    train = train.drop(columns=["barrier_start", "barrier_end", "end_date"])
    val = val.drop(columns=["barrier_start", "barrier_end", "end_date"])
    test = test.drop(columns=["barrier_start", "barrier_end", "end_date"])

    return train, val, test, val_start, test_start



# Model itself with validation added

def fit_logistic(train_df, val_df, test_df, feature_cols, target_col="label"):

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    # Getting the probabilities
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    train_out["probability"] = model.predict_proba(X_train)[:, 1]
    val_out["probability"] = model.predict_proba(X_val)[:, 1]
    test_out["probability"] = model.predict_proba(X_test)[:, 1]

    return model, train_out, val_out, test_out


def fit_random_forest(train_df, val_df, test_df, feature_cols, target_col="label", random_state=0):
    """
    Fit a shallow random-forest secondary model as a nonlinear benchmark.
    """

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=500,
            max_samples=0.25,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state
        ))
    ])

    model.fit(X_train, y_train)

    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    train_out["probability"] = _positive_class_probability(model, X_train)
    val_out["probability"] = _positive_class_probability(model, X_val)
    test_out["probability"] = _positive_class_probability(model, X_test)

    return model, train_out, val_out, test_out


def fit_gradient_boosting(train_df, val_df, test_df, feature_cols, target_col="label", random_state=0):
    """
    Fit a small gradient-boosting secondary model as a nonlinear benchmark.
    """

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=8,
            l2_regularization=0.01,
            random_state=random_state
        ))
    ])

    model.fit(X_train, y_train)

    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    train_out["probability"] = _positive_class_probability(model, X_train)
    val_out["probability"] = _positive_class_probability(model, X_val)
    test_out["probability"] = _positive_class_probability(model, X_test)

    return model, train_out, val_out, test_out


def _positive_class_probability(model, X):
    clf = model.named_steps["clf"]
    classes = clf.classes_

    if 1 not in classes:
        return np.zeros(len(X))

    class_idx = np.where(classes == 1)[0][0]
    return model.predict_proba(X)[:, class_idx]


# Simple sanity check

def roc_auc_table(df, target_col="label", proba_col="probability"):
  
    y_true = df[target_col]
    proba = df[proba_col]

    roc_auc = roc_auc_score(y_true, proba)
    summary = pd.DataFrame({"Metric": ["ROC AUC"],"Value": [roc_auc]})
    
    return summary


# New column for threshold

def apply_probability_threshold(df, threshold, proba_col="probability"):

    out = df.copy()
    out["meta_label"] = (out[proba_col] >= threshold).astype(int)

    return out

# Sanity check for threshould

def threshold_diagnostics(df, target_col="label", pred_col="meta_label"):

    y_true = df[target_col]
    y_pred = df[pred_col]

    trades_total = len(df)
    trades_kept = y_pred.sum()
    trades_removed = trades_total - trades_kept
    fraction_kept = trades_kept / trades_total

    if trades_kept > 0:
        kept_positive_rate = y_true[y_pred == 1].mean()
    else:
        kept_positive_rate = np.nan

    base_positive_rate = y_true.mean()


    summary = pd.DataFrame({"Metric": [
            "Total trades",
            "Trades kept",
            "Trades removed",
            "Fraction kept",
            "Base positive rate",
            "Positive rate among kept trades"],
        "Value": 
            [trades_total,
            trades_kept,
            trades_removed,
            fraction_kept,
            base_positive_rate,
            kept_positive_rate]})
    
    return summary


def create_filtered_weights_fast(primary_weights, filtered_df, date_col="t0", asset_col="stock", signal_col="meta_label"):
    """
    Create meta-filtered weights using vectorized alignment.

    Params
        primary_weights
            Primary strategy weights
        filtered_df
            Event dataframe with meta-label decisions
        date_col
            Event date column
        asset_col
            Asset column
        signal_col
            Meta-label decision column

    Output
        meta_weights
            Filtered strategy weights
    """

    # Signal matrix

    signal_matrix = filtered_df.pivot(
        index=date_col,
        columns=asset_col,
        values=signal_col
    )

    signal_matrix = signal_matrix.reindex(
        index=primary_weights.index,
        columns=primary_weights.columns
    ).fillna(0.0)

    # Weights

    meta_weights = primary_weights * signal_matrix

    return meta_weights

def calculate_strategy_pnl(weights, returns):

    aligned_returns = returns.reindex(weights.index)
    pnl = (weights * aligned_returns).sum(axis=1).to_frame("PnL")

    return pnl
