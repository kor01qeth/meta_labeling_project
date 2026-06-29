"""
Purpose:
    Helper functions for the first fundamental triple-barrier analysis.

    The module isolates the synthetic data-generating process and the
    triple-barrier label diagnostics from the primary and secondary strategy
    code. The main question is when label 1 behaves like a useful proxy for
    truly positive-drift stocks.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit, prange


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.triple_barrier_lsymmetric import simple_triple_barrier_labels_lsymmetric


# DGP

def build_covariance_matrix(
    n_stocks=500,
    n_factors=10,
    annual_vol=0.10,
    split=(0.6, 0.4),
    seed=0,
):
    """
    Build a factor-plus-specific covariance matrix.

    The construction mirrors the existing fast workflow: a positive market
    factor, additional standardized factors, and stock-specific variances are
    rescaled to match the target annual volatility.
    """

    rng = np.random.default_rng(seed)
    target_daily_vol = annual_vol / np.sqrt(252)
    equal_weights = np.ones((n_stocks, 1)) / n_stocks

    loadings = rng.normal(0.0, 1.0, size=(n_stocks, n_factors - 1))
    loadings = (
        loadings - loadings.mean(axis=1, keepdims=True)
    ) / loadings.std(axis=1, keepdims=True)
    loadings = np.concatenate(
        [rng.uniform(0.5, 1.5, size=(n_stocks, 1)), loadings],
        axis=1,
    )

    factor_cov = np.diag(
        np.concatenate([
            np.ones(1),
            [0.25 * np.exp(-i) for i in range(n_factors - 1)],
        ])
    )
    specific_var = rng.uniform(0.01, 0.25, size=n_stocks)

    factor_scale = np.squeeze(
        split[0]
        * target_daily_vol**2
        / (equal_weights.T @ (loadings @ factor_cov @ loadings.T) @ equal_weights)
    )
    specific_scale = np.squeeze(
        split[1]
        * target_daily_vol**2
        / (equal_weights.T @ np.diag(specific_var) @ equal_weights)
    )

    factor_cov = factor_scale * factor_cov
    cov_factor = loadings @ factor_cov @ loadings.T
    cov_specific = np.diag(specific_scale * specific_var)

    return cov_factor + cov_specific


def average_pairwise_correlation(cov_or_corr):
    """
    Compute the average off-diagonal correlation.
    """

    matrix = np.asarray(cov_or_corr, dtype=float)
    if not np.allclose(np.diag(matrix), 1.0):
        matrix = matrix / np.sqrt(np.outer(np.diag(matrix), np.diag(matrix)))

    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(matrix[mask].mean())


def average_pairwise_correlation_from_returns(returns_array):
    """
    Compute average pairwise realised correlation without forming corr matrix.
    """

    returns_array = np.asarray(returns_array, dtype=float)
    standardized = (
        returns_array - returns_array.mean(axis=0, keepdims=True)
    ) / returns_array.std(axis=0, ddof=1, keepdims=True)

    n_obs, n_assets = standardized.shape
    row_sums = standardized.sum(axis=1)
    corr_sum_all = np.sum(row_sums**2) / (n_obs - 1)
    off_diag_sum = corr_sum_all - n_assets

    return float(off_diag_sum / (n_assets * (n_assets - 1)))


def volatility_scale_diagnostics(returns_array):
    """
    Summarise portfolio and individual-stock volatility from simulated returns.
    """

    returns_array = np.asarray(returns_array, dtype=float)
    individual_ann_vol = returns_array.std(axis=0, ddof=1) * np.sqrt(252)
    equal_weight_ann_vol = returns_array.mean(axis=1).std(ddof=1) * np.sqrt(252)

    out = {
        "equal_weight_ann_vol": float(equal_weight_ann_vol),
        "individual_ann_vol_avg": float(individual_ann_vol.mean()),
        "individual_ann_vol_median": float(np.median(individual_ann_vol)),
        "individual_ann_vol_min": float(individual_ann_vol.min()),
        "individual_ann_vol_max": float(individual_ann_vol.max()),
        "avg_pairwise_corr": average_pairwise_correlation_from_returns(returns_array),
    }

    return out


def run_original_dgp_volatility_diagnostics(
    n_sims=100,
    base_seed=20_000,
    n_stocks=500,
    n_days=10_000,
    n_factors=10,
    annual_vol=0.10,
    split=(0.6, 0.4),
    trend_daily=0.50 / 252,
    n_up=200,
    n_down=100,
    component_seed=0,
    vary_components=False,
    progress=True,
):
    """
    Run repeated volatility diagnostics using the original DGP style.

    By default the covariance components are fixed, matching the simulation
    notebooks where `component_seed = 0` and only sampled returns/trend-stock
    identities change across simulation seeds.
    """

    rows = []
    fixed_cov = None
    fixed_chol = None

    if not vary_components:
        fixed_cov = build_covariance_matrix(
            n_stocks=n_stocks,
            n_factors=n_factors,
            annual_vol=annual_vol,
            split=split,
            seed=component_seed,
        )
        fixed_chol = np.linalg.cholesky(fixed_cov)

    for sim in range(n_sims):
        seed = base_seed + sim
        rng = np.random.default_rng(seed)

        if vary_components:
            cov = build_covariance_matrix(
                n_stocks=n_stocks,
                n_factors=n_factors,
                annual_vol=annual_vol,
                split=split,
                seed=component_seed + sim,
            )
            chol = np.linalg.cholesky(cov)
        else:
            cov = fixed_cov
            chol = fixed_chol

        up_stocks = rng.choice(n_stocks, size=n_up, replace=False)
        remaining = np.setdiff1d(np.arange(n_stocks), up_stocks)
        down_stocks = rng.choice(remaining, size=n_down, replace=False)

        trend = np.zeros(n_stocks)
        trend[up_stocks] = rng.uniform(0.1, 1.0, size=n_up) * trend_daily
        trend[down_stocks] = -rng.uniform(0.1, 1.0, size=n_down) * trend_daily

        sampled_returns = rng.standard_normal((n_days, n_stocks)) @ chol.T + trend
        diagnostics = volatility_scale_diagnostics(sampled_returns)

        row = {
            "sim": sim,
            "seed": seed,
        }
        row.update(diagnostics)
        rows.append(row)

        if progress and (sim + 1) % max(1, n_sims // 10) == 0:
            print(f"completed {sim + 1}/{n_sims}")

    return pd.DataFrame(rows)


def summarize_volatility_diagnostics(diagnostics):
    """
    Summarise repeated DGP volatility diagnostics across simulations.
    """

    metric_cols = [
        "equal_weight_ann_vol",
        "individual_ann_vol_avg",
        "individual_ann_vol_median",
        "individual_ann_vol_min",
        "individual_ann_vol_max",
        "avg_pairwise_corr",
    ]
    metric_cols = [col for col in metric_cols if col in diagnostics.columns]

    return (
        diagnostics[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .T
        .rename_axis("metric")
        .reset_index()
    )



def build_original_dgp_components(
    N=500,
    K=10,
    target_portfolio_annual_vol=0.10,
    split=(0.6, 0.4),
    component_seed=0,
):
    """
    Build DGP components using notation close to the fast workflow.
    """

    rng_components = np.random.default_rng(component_seed)
    target_daily_vol = target_portfolio_annual_vol / np.sqrt(252)

    wi = np.ones((N, 1)) / N

    Z = rng_components.normal(0, 1, size=(N, K - 1))
    Z = (Z - Z.mean(axis=1, keepdims=True)) / Z.std(axis=1, keepdims=True)
    Z = np.concatenate([rng_components.uniform(0.5, 1.5, size=(N, 1)), Z], axis=1)

    G = np.diag(np.concatenate([np.ones(1), [0.25 * np.exp(-i) for i in range(K - 1)]]))
    S = rng_components.uniform(0.01, 0.25, size=N)

    multf = np.squeeze(split[0] * target_daily_vol**2 / (wi.T @ (Z @ G @ Z.T) @ wi))
    mults = np.squeeze(split[1] * target_daily_vol**2 / (wi.T @ np.diag(S) @ wi))

    G = multf * G
    cov_factor = Z @ G @ Z.T
    S = mults * S
    cov_specific = np.diag(S)
    cov = cov_factor + cov_specific

    beta = (np.eye(N) @ cov @ wi) / (wi.T @ cov @ wi)

    return {
        "wi": wi,
        "Z": Z,
        "G": G,
        "S": S,
        "multf": multf,
        "mults": mults,
        "cov_factor": cov_factor,
        "cov_specific": cov_specific,
        "cov": cov,
        "beta": beta,
        "chol": np.linalg.cholesky(cov),
    }


def sample_fixed_up_returns(
    seed,
    components,
    annual_drift,
    N=500,
    T=10_000,
    n_up=100,
    start_date="1986-01-01",
):
    """
    Sample returns where up stocks have fixed drift and all others have zero drift.
    """

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=T, freq="B")
    cols = [f"Stock_{i:03d}" for i in range(N)]

    up_stocks = rng.choice(N, size=n_up, replace=False)
    trend = np.zeros(N)
    trend[up_stocks] = annual_drift / 252

    sampled_returns = rng.standard_normal((T, N)) @ components["chol"].T + trend
    df_returns = pd.DataFrame(sampled_returns, index=dates, columns=cols)

    stock_info = pd.DataFrame({
        "stock": cols,
        "stock_loc": np.arange(N),
        "annual_drift": trend * 252,
        "true_drift_label": (trend > 0.0).astype(int),
    })

    return df_returns, stock_info


def make_long_only_event_weights(df_returns, vol_window=20, horizon=30, event_stride=1):
    """
    Create equal-weight long-only event candidates.
    """

    event_index = df_returns.index[vol_window: len(df_returns.index) - horizon]
    event_index = event_index[::event_stride]

    if len(event_index) == 0:
        raise ValueError("No event dates remain. Check horizon, vol_window, T, and event_stride.")

    return pd.DataFrame(1.0 / df_returns.shape[1], index=event_index, columns=df_returns.columns)


@njit(parallel=True)
def _long_only_barrier_sampled_numba(
    returns_array,
    vol_array,
    event_locs,
    horizon,
    barrier_multiplier,
):
    """
    Fast sampled-event long-only triple-barrier diagnostics.
    """

    n_events = len(event_locs)
    n_stocks = returns_array.shape[1]

    labels = np.zeros((n_events, n_stocks), dtype=np.int8)
    hit_codes = np.zeros((n_events, n_stocks), dtype=np.int8)
    hit_times = np.empty((n_events, n_stocks), dtype=np.float64)

    hit_times[:, :] = np.nan
    sqrt_horizon = np.sqrt(horizon)

    for k in prange(n_events):
        loc = event_locs[k]

        for j in range(n_stocks):
            sigma = vol_array[loc, j]

            if np.isnan(sigma):
                hit_codes[k, j] = 0
                labels[k, j] = 0
                continue

            pt = barrier_multiplier * sigma * sqrt_horizon
            sl = barrier_multiplier * sigma * sqrt_horizon

            cum_value = 1.0
            label = 0
            hit_code = 3
            hit_time = np.nan

            for h in range(horizon):
                daily_return = returns_array[loc + h, j]

                if np.isnan(daily_return):
                    continue

                if daily_return <= -1.0:
                    label = 0
                    hit_code = 2
                    hit_time = h + 1
                    break

                cum_value *= 1.0 + daily_return
                cum_return = cum_value - 1.0

                if cum_return >= pt:
                    label = 1
                    hit_code = 1
                    hit_time = h + 1
                    break
                elif cum_return <= -sl:
                    label = 0
                    hit_code = 2
                    hit_time = h + 1
                    break

            labels[k, j] = label
            hit_codes[k, j] = hit_code
            hit_times[k, j] = hit_time

    return labels, hit_codes, hit_times


def run_long_only_triple_barrier_diagnostic(
    df_returns,
    stock_info,
    horizon,
    vol_window=20,
    barrier_multiplier=1.0,
    event_stride=20,
):
    """
    Run long-only triple-barrier labels and merge with true drift identity.
    """

    rolling_vol = df_returns.shift(1).rolling(vol_window).std(ddof=1)
    weights = make_long_only_event_weights(
        df_returns=df_returns,
        vol_window=vol_window,
        horizon=horizon,
        event_stride=event_stride,
    )
    event_dates = weights.index
    event_locs = df_returns.index.get_indexer(event_dates).astype(np.int64)

    labels_array, hit_codes_array, hit_times_array = _long_only_barrier_sampled_numba(
        returns_array=df_returns.to_numpy(dtype=np.float64),
        vol_array=rolling_vol.to_numpy(dtype=np.float64),
        event_locs=event_locs,
        horizon=int(horizon),
        barrier_multiplier=float(barrier_multiplier),
    )

    stocks = df_returns.columns
    n_stocks = len(stocks)
    hit_map = np.array(["invalid", "pt", "sl", "vertical"], dtype=object)

    labels = pd.DataFrame({
        "t0": np.repeat(event_dates.to_numpy(), n_stocks),
        "stock": np.tile(stocks.to_numpy(), len(event_dates)),
        "side": 1,
        "weight": 1.0 / n_stocks,
        "label": labels_array.reshape(-1),
        "hit_type": hit_map[hit_codes_array.reshape(-1)],
        "hit_time": hit_times_array.reshape(-1),
    })

    return labels.merge(stock_info, on="stock", how="left")


# Diagnostics

def _safe_div(num, den):
    return np.nan if den == 0 else num / den


def summarise_triple_barrier_classification(events):
    """
    Compare true positive-drift identity with triple-barrier label.
    """

    y_true = events["true_drift_label"].astype(int)
    y_label = events["label"].astype(int)

    tp = int(((y_true == 1) & (y_label == 1)).sum())
    fn = int(((y_true == 1) & (y_label == 0)).sum())
    fp = int(((y_true == 0) & (y_label == 1)).sum())
    tn = int(((y_true == 0) & (y_label == 0)).sum())

    tpr = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)

    return {
        "n_events": int(len(events)),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "true_positive_rate": tpr,
        "false_negative_rate": _safe_div(fn, tp + fn),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "true_negative_rate": tnr,
        "precision": _safe_div(tp, tp + fp),
        "label_positive_rate": float(y_label.mean()) if len(y_label) else np.nan,
        "true_drift_event_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "balanced_accuracy": np.nanmean([tpr, tnr]),
        "pt_hit_share": float((events["hit_type"] == "pt").mean()),
        "sl_hit_share": float((events["hit_type"] == "sl").mean()),
        "vertical_hit_share": float((events["hit_type"] == "vertical").mean()),
        "avg_time_to_pt": events.loc[events["hit_type"] == "pt", "hit_time"].mean(),
        "avg_time_to_sl": events.loc[events["hit_type"] == "sl", "hit_time"].mean(),
    }


def run_fundamental1_single(
    seed,
    target_portfolio_annual_vol,
    annual_drift,
    horizon,
    N=500,
    T=10_000,
    K=10,
    n_up=100,
    split=(0.6, 0.4),
    component_seed=0,
    vol_window=20,
    barrier_multiplier=1.0,
    event_stride=20,
):
    """
    Run one fixed-up-stock DGP and long-only triple-barrier diagnostic.
    """

    components = build_original_dgp_components(
        N=N,
        K=K,
        target_portfolio_annual_vol=target_portfolio_annual_vol,
        split=split,
        component_seed=component_seed,
    )
    df_returns, stock_info = sample_fixed_up_returns(
        seed=seed,
        components=components,
        annual_drift=annual_drift,
        N=N,
        T=T,
        n_up=n_up,
    )
    events = run_long_only_triple_barrier_diagnostic(
        df_returns=df_returns,
        stock_info=stock_info,
        horizon=horizon,
        vol_window=vol_window,
        barrier_multiplier=barrier_multiplier,
        event_stride=event_stride,
    )

    return {
        "df_returns": df_returns,
        "stock_info": stock_info,
        "events": events,
        "metrics": summarise_triple_barrier_classification(events),
    }


def run_fundamental1_grid(
    target_portfolio_annual_vol_grid,
    annual_drift_grid,
    horizons,
    n_sims=3,
    base_seed=20_000,
    N=500,
    T=10_000,
    K=10,
    n_up=100,
    split=(0.6, 0.4),
    component_seed=0,
    vol_window=20,
    barrier_multiplier=1.0,
    event_stride=20,
    progress=True,
):
    """
    Run the long-only triple-barrier drift-recovery grid.
    """

    rows = []
    total = len(target_portfolio_annual_vol_grid) * len(annual_drift_grid) * len(horizons) * n_sims
    completed = 0

    for target_portfolio_annual_vol in target_portfolio_annual_vol_grid:
        components = build_original_dgp_components(
            N=N,
            K=K,
            target_portfolio_annual_vol=float(target_portfolio_annual_vol),
            split=split,
            component_seed=component_seed,
        )

        for sim in range(n_sims):
            seed = base_seed + sim
            for annual_drift in annual_drift_grid:
                df_returns, stock_info = sample_fixed_up_returns(
                    seed=seed,
                    components=components,
                    annual_drift=float(annual_drift),
                    N=N,
                    T=T,
                    n_up=n_up,
                )

                for horizon in horizons:
                    events = run_long_only_triple_barrier_diagnostic(
                        df_returns=df_returns,
                        stock_info=stock_info,
                        horizon=int(horizon),
                        vol_window=vol_window,
                        barrier_multiplier=barrier_multiplier,
                        event_stride=event_stride,
                    )

                    row = {
                        "sim": sim,
                        "seed": seed,
                        "horizon": int(horizon),
                        "target_portfolio_annual_vol": float(target_portfolio_annual_vol),
                        "annual_drift": float(annual_drift),
                    }
                    row.update(summarise_triple_barrier_classification(events))
                    rows.append(row)

                    completed += 1
                    if progress and completed % max(1, total // 10) == 0:
                        print(f"completed {completed}/{total}")

    grid_results = pd.DataFrame(rows)
    grid_summary = summarise_fundamental1_grid(grid_results)

    return grid_results, grid_summary


def summarise_fundamental1_grid(grid_results):
    """
    Average fundamental1 grid diagnostics across simulation seeds.
    """

    group_cols = ["horizon", "target_portfolio_annual_vol", "annual_drift"]
    value_cols = [
        col
        for col in grid_results.columns
        if col not in group_cols
        and pd.api.types.is_numeric_dtype(grid_results[col])
        and col not in {"sim", "seed"}
    ]

    return (
        grid_results
        .groupby(group_cols, as_index=False)[value_cols]
        .mean()
        .sort_values(group_cols)
    )

# Plotting

def plot_metric_heatmaps(
    summary,
    metric,
    horizons=None,
    cmap="viridis",
    figsize=None,
    x_col=None,
):
    """
    Plot volatility-drift heatmaps for one metric, faceted by horizon.
    """

    if horizons is None:
        horizons = sorted(summary["horizon"].unique())
    if x_col is None:
        x_col = (
            "target_portfolio_annual_vol"
            if "target_portfolio_annual_vol" in summary.columns
            else "annual_vol"
        )

    n_horizons = len(horizons)
    ncols = min(2, n_horizons)
    nrows = int(np.ceil(n_horizons / ncols))
    if figsize is None:
        figsize = (6 * ncols, 4.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    vmin = summary[metric].min()
    vmax = summary[metric].max()

    for ax, horizon in zip(axes.ravel(), horizons):
        data = summary[summary["horizon"] == horizon]
        pivot = data.pivot(index="annual_drift", columns=x_col, values=metric)
        image = ax.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[
                pivot.columns.min(),
                pivot.columns.max(),
                pivot.index.min(),
                pivot.index.max(),
            ],
        )
        ax.set_title(f"{metric} | horizon {horizon}d")
        ax.set_xlabel("Target portfolio annual volatility")
        ax.set_ylabel("Annual drift")
        fig.colorbar(image, ax=ax, shrink=0.85)

    for ax in axes.ravel()[n_horizons:]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes


def plot_stacked_horizon_3d(
    summary,
    metric,
    horizons=None,
    cmap="viridis",
    figsize=(10, 7),
    x_col=None,
):
    """
    Plot a 3D stacked-horizon view.

    Horizon is shown as a discrete z-axis layer, while metric values are shown
    by color on each volatility-drift plane.
    """

    if horizons is None:
        horizons = sorted(summary["horizon"].unique())
    if x_col is None:
        x_col = (
            "target_portfolio_annual_vol"
            if "target_portfolio_annual_vol" in summary.columns
            else "annual_vol"
        )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    norm = plt.Normalize(summary[metric].min(), summary[metric].max())
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    for horizon in horizons:
        data = summary[summary["horizon"] == horizon]
        pivot = data.pivot(index="annual_drift", columns=x_col, values=metric)
        x_grid, y_grid = np.meshgrid(pivot.columns.values, pivot.index.values)
        z_grid = np.full_like(x_grid, float(horizon), dtype=float)
        colors = scalar_map.to_rgba(pivot.values)

        ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            facecolors=colors,
            rstride=1,
            cstride=1,
            linewidth=0,
            shade=False,
            alpha=0.92,
        )

    ax.set_xlabel("Target portfolio annual volatility")
    ax.set_ylabel("Annual drift")
    ax.set_zlabel("Horizon")
    ax.set_title(f"Stacked horizon view: {metric}")
    fig.colorbar(scalar_map, ax=ax, shrink=0.65, pad=0.12, label=metric)

    return fig, ax


def save_table(df, path):
    """
    Save a dataframe as CSV, creating parent directories when needed.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
