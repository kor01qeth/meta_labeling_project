# Meta-Labeling for Cross-Sectional Trading Strategies

This repository contains synthetic-data experiments for triple-barrier labeling and meta-labeling in cross-sectional trading strategies.

The current project focus is diagnostic: understanding when the labels are meaningful, whether the secondary filter adds genuine information, and whether portfolio construction preserves the intended exposure constraints.

## Current Status

Recent work identified three important caveats in earlier results:

- the original fast triple-barrier calculation was not side-symmetric,
- gross-normalised secondary portfolios did not necessarily remain dollar-neutral,
- the DGP volatility target controlled equal-weight portfolio volatility, not individual stock volatility.

The active corrected labeler is `src/triple_barrier_lsymmetric.py`. Older notebooks remain useful for history, but their outputs should be read with these caveats.

## Main Notebooks

- `notebooks/main_workflow.ipynb`  
  Original end-to-end workflow. Important baseline, but historical.

- `notebooks/fast_worflow.ipynb`  
  Faster version of the original workflow. Important baseline, but historical.

- `notebooks/fast_simulation.ipynb`  
  Repeated simulation and heatmap workflow. Historical outputs may use old barrier and gross-only exposure logic.

- `notebooks/horizon_sensitivity.ipynb`  
  Diagnostics for horizon effects, payoff windows, confusion groups, and exposure drift.

- `notebooks/split_secondary_testing.ipynb`  
  Split long/short secondary-model testing, including dollar-neutral reweighting.

- `notebooks/tables.ipynb`  
  Supporting tables and appendix-style checks.

## Fundamental Analysis

- `fundamental_analysis/fundamental_notebooks/fundamental0.ipynb`  
  Shows that the original DGP targets equal-weight portfolio volatility and can imply very high individual stock volatility.

- `fundamental_analysis/fundamental_notebooks/fundamental1.ipynb`  
  Long-only diagnostic asking when triple-barrier label `1` can identify true positive-drift stocks.

Supporting outputs are stored in:

- `fundamental_analysis/fundamental_figures/`
- `fundamental_analysis/fundamental_tables/`

## Source Modules

- `src/primary_strategy.py`  
  Primary cross-sectional signal and weight construction.

- `src/evaluation_metrics.py`  
  Strategy evaluation metrics.

- `src/triple_barrier_lsymmetric.py`  
  Current side-symmetric triple-barrier labeler.

- `src/triple_barrier_fast.py`  
  Historical fast triple-barrier implementation.

- `src/triple_barrier.py`  
  Historical non-fast triple-barrier implementation.

- `src/secondary_model.py`  
  Historical pooled secondary-model helpers.

- `src/secondary_model_fast.py`  
  Fast pooled secondary-model helpers.

- `src/secondary_model_split_fast.py`  
  Split long/short secondary-model helpers and dollar-neutral reweighting.

- `src/horizon_sensitivity.py`  
  Event-level payoff, confusion, filtering, and exposure diagnostics.

- `src/ic_analysis.py`  
  Information coefficient diagnostics.

- `fundamental_analysis/fundamental_helpers/helper1.py`  
  Helper functions for the fundamental DGP and triple-barrier diagnostics.

## Documentation

- `docs/current_state_and_next_steps.md`  
  Short operational project memory.

- `docs/drawn_conclusions.md`  
  Main conclusions, resolved issues, and current interpretation.

## Setup

Install dependencies with:

```bash
uv sync
```
