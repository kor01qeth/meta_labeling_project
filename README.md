# Meta-Labeling for Cross-Sectional Trading Strategies Code

This repository contains the code for the synthetic-data, triple-barrier labeling, and meta-labeling experiments.

## Structure

- `notebooks/main_workflow.ipynb`  
  Main workflow notebook containing the data generation, primary strategy, IC analysis, triple-barrier labeling, and secondary-model results.

- `notebooks/tables.ipynb`  
  Appendix notebook containing the supporting tables and additional robustness checks.

- `src/ic_analysis.py`  
  Functions for Spearman IC analysis.

- `src/triple_barrier.py`  
  Functions for triple-barrier labeling.

- `src/secondary_model.py`  
  Functions for building the secondary-model dataset, time splitting, and fitting the logistic-regression baseline.

- `figures/`  
  Saved figures such as the Spearman IC plot.

- `results/`  
  Saved result tables.

## Setup

Install dependencies with:

```bash
uv sync