"""
Purpose:
    The functions below implement the secondary meta-labeling model.
    They build event-level feature datasets from the labeled trades,
    apply time-based train/test splits, and fit a logistic
    regression to predict whether a proposed trade is good
    or bad. The module is used to test how informative the chosen
    features are.
"""


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score 


# Building a new dataframe that can be used for the logistic regression

def build_secondary_dataset(df_returns, labels, vol_window=20):
    
    """
    Parameters
        df_returns
            Full return matrix, indexed by date with stocks in columns.
        labels
            Output of the simple_triple_barrier_labels function.
        vol_window (default=20)
            Rolling window used to estimate daily volatility.
    Output
        pd.DataFrame
            With columns:
            - t0
            - stock
            - weight
            - ret_20
            - ret_60
            - ret_150
            - vol_20
            - vol_60
            - vol_150
            - label
    """

    # Note that for flexibility, a single volatility window 
    # is specified as an input, but as of now, the function computes rolling 
    # volatility for multiple windows. 
    
    vol_20 = df_returns.rolling(vol_window).std(ddof=1)
    vol_60 = df_returns.rolling(60).std(ddof=1)
    vol_150 = df_returns.rolling(150).std(ddof=1)

    rows = []

    for _, row in labels.iterrows():
        t0 = row["t0"]
        stock = row["stock"]
        weight = row["weight"]
        label = row["label"]

        loc = df_returns.index.get_loc(t0)

        # Ensure enough past data for longest return horizon - want to be unified
        if loc < 150:
            continue

        ret_20 = (1 + df_returns.iloc[loc-20:loc][stock]).prod() - 1
        ret_60 = (1 + df_returns.iloc[loc-60:loc][stock]).prod() - 1
        ret_150 = (1 + df_returns.iloc[loc-150:loc][stock]).prod() - 1
        vol20 = vol_20.loc[t0, stock]
        vol60 = vol_60.loc[t0, stock]
        vol150 = vol_150.loc[t0, stock]

        # Note: the below features have been used to test feature importance, but have not been used in the final model. 
        #denom_20 = max(vol20, 1e-8)
        #denom_60 = max(vol60, 1e-8)
        #denom_150 = max(vol150, 1e-8)
        #ret_20_voladj = ret_20 / denom_20
        #ret_60_voladj = ret_60 / denom_60
        #ret_150_voladj = ret_150 / denom_150

        # Some are for bookkeeping only

        rows.append({
            "t0": t0,
            "stock": stock,   
            "weight": weight,
            "ret_20": ret_20,
            "ret_60": ret_60,
            "ret_150": ret_150,
            #"ret_20_adj": ret_20_voladj,
            #"ret_60_adj": ret_60_voladj,
            #"ret_150_adj": ret_150_voladj,
            "vol_20": vol20,
            "vol_60": vol60,
            "vol_150": vol150,
            "label": label
        })

    return pd.DataFrame(rows)



# 70-30 train-test split, TODO later on: use validation (50-25-25) 

def time_split(df, split_frac=0.7):
    
    unique_dates = np.sort(df["t0"].unique())
    split_idx = int(len(unique_dates) * split_frac)
    split_date = unique_dates[split_idx]

    train = df[df["t0"] < split_date].copy()
    test = df[df["t0"] >= split_date].copy()

    return train, test



# Fitting a logistic regression as a baseline model, with some preprocessing.

def fit_logistic_baseline(train_df, test_df, feature_cols, target_col="label"):
   
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Using inputer for NaNs if I have missed any
    # Using scaler for more stable optimisation
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    # Want to get probability for label 1
    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    # Fixed threshold, TODO check for other values
    pred_test = (proba_test >= 0.6).astype(int)

    print(classification_report(y_test, pred_test))

    # If above threshold is not sensible, safeguard
    try:
        auc = roc_auc_score(y_test, proba_test)
        print(f"Test ROC AUC: {auc:.4f}")
    except ValueError:
        print("Chance threshold. AUC ROC can't be computed")

    out_test_df = test_df.copy()
    out_test_df["probability"] = proba_test
    out_test_df["predicted label"] = pred_test

    return model, out_test_df


############
#ADDED PART
############

"""
Purpose:
    The functions below extend the earlier secondary-model workflow.

    They introduce a chronological train-validation-test split with
    purging, so that the triple-barrier horizon does not create leakage
    between datasets.

    The model-fitting step is also separated from thresholding, allowing
    predicted probabilities to be used later for trade filtering, sizing,
    and test-set evaluation.
"""

# Train - val - test split

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

# New weights according to (predicted meta-label) * primary weights

def create_filtered_weights(primary_weights, filtered_df, date_col="t0", asset_col="stock", signal_col="meta_label"):

    meta_weights = primary_weights.copy() * 0.0

    for _, row in filtered_df.iterrows():

        date = row[date_col]
        asset = row[asset_col]

        if date in meta_weights.index and asset in meta_weights.columns:
            meta_weights.loc[date, asset] = primary_weights.loc[date, asset] * row[signal_col]

    return meta_weights


# PnL

def calculate_strategy_pnl(weights, returns):

    aligned_returns = returns.reindex(weights.index)
    pnl = (weights * aligned_returns).sum(axis=1).to_frame("PnL")

    return pnl