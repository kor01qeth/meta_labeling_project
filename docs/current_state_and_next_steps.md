# Current State And Next Steps

This file is the short operational memory for the project. For interpretation details, see `docs/drawn_conclusions.md`.

## Current State

The project has moved from broad simulation experiments to a cleaner diagnostic structure.

The current active ideas are:

- use side-symmetric triple-barrier labels for new experiments,
- treat older simple-return barrier results as historical,
- separate classification skill from portfolio exposure effects,
- recognise that the current DGP is a high individual-stock-noise setting,
- use fundamental diagnostics before adding more secondary-model complexity.

## Active Files

Main historical workflows:

- `notebooks/main_workflow.ipynb`
- `notebooks/fast_worflow.ipynb`

Current diagnostic notebooks:

- `notebooks/horizon_sensitivity.ipynb`
- `notebooks/split_secondary_testing.ipynb`
- `fundamental_analysis/fundamental_notebooks/fundamental0.ipynb`
- `fundamental_analysis/fundamental_notebooks/fundamental1.ipynb`

Current helper modules:

- `src/primary_strategy.py`
- `src/evaluation_metrics.py`
- `src/secondary_model_fast.py`
- `src/secondary_model_split_fast.py`
- `src/triple_barrier_lsymmetric.py`
- `src/horizon_sensitivity.py`
- `fundamental_analysis/fundamental_helpers/helper1.py`

Historical modules kept for comparison:

- `src/triple_barrier.py`
- `src/triple_barrier_fast.py`
- `src/secondary_model.py`

## Resolved Issues

### 1. Lookahead Risk

Earlier workflow checks focused on making sure decisions follow the `t0` convention:

- secondary features use lagged returns,
- rolling features are based on information available before `t0`,
- purged time splits are used around triple-barrier horizons,
- strategy weights are decided before the return at `t0` is earned.

This remains the intended convention.

### 2. Triple-Barrier Side Bias

The old fast barrier in `src/triple_barrier_fast.py` was not side-symmetric. It could mechanically prefer one side because compounded simple returns are not symmetric for mirrored up/down paths.

Current solution:

- use `src/triple_barrier_lsymmetric.py`,
- evaluate paths as compounded side-adjusted trade returns,
- keep `src/triple_barrier_fast.py` only for historical comparison.

### 3. Missing Dollar Neutrality After Filtering

The primary strategy is close to dollar-neutral because the signal is demeaned before gross normalisation.

The gross-normalised secondary filter did not preserve dollar neutrality. If it kept mostly longs or mostly shorts, it scaled that side to full gross exposure.

Current solution:

- use split long/short secondary diagnostics in `src/secondary_model_split_fast.py`,
- compare gross-normalised and dollar-neutral split portfolios in `notebooks/split_secondary_testing.ipynb`.

### 4. Noisy DGP

The synthetic DGP’s `annual_vol` parameter targets equal-weight portfolio volatility, not average individual-stock volatility.

`fundamental0.ipynb` showed that a 10% portfolio-volatility target can imply individual stock volatility above 100% annually under the current low-correlation universe.

Current conclusion:

- previous labels are not invalid,
- but they describe a very noisy stock-level labeling problem,
- weak ROC AUC is consistent with that DGP.

## Current Conclusions

The cleanest interpretation is:

- old classification/economic mismatches were partly caused by label construction and exposure construction,
- side-symmetric labels fix the mechanical long/short barrier issue,
- split secondary models help preserve dollar neutrality,
- model complexity has not solved weak classification,
- the central remaining problem is label noise under the current DGP and feature set.

## Current Caveats

Use caution when reading older notebook outputs.

`main_workflow.ipynb`, `fast_worflow.ipynb`, and `fast_simulation.ipynb` contain important work, but some cells use:

- old triple-barrier logic,
- gross-only secondary exposure normalisation,
- the noisy portfolio-volatility-calibrated DGP.

Their outputs are useful historically, but should not be cited as the current clean result without checking which labeler and portfolio construction were used.

## Next Steps

Recommended order:

1. Keep fundamental diagnostics as the first source of truth for the DGP and label behaviour.
2. Decide whether the DGP should offer an individual-volatility target in addition to the current portfolio-volatility target.
3. Re-run the triple-barrier drift diagnostic under any revised DGP calibration.
4. Only then revisit secondary-model features, thresholds, and portfolio construction.
5. If the synthetic pipeline becomes coherent, consider a small real-data pilot.

Deferred work:

- convex optimisation for portfolio construction,
- 3D contour plots beyond the fundamental diagnostic,
- real-data universe construction,
- thesis write-up around the corrected label and exposure findings.

