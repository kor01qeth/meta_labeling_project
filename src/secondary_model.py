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