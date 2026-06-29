# Drawn Conclusions

This file keeps the high-level conclusions from the recent cleanup and diagnostic work. It replaces the older detailed project-history notes, which mixed obsolete and current results.

## Main Conclusion

The earlier secondary-model results were difficult to interpret because three issues were active at the same time:

- the original triple-barrier construction mechanically favoured one trade side,
- the gross-normalised secondary portfolio did not preserve dollar neutrality,
- the synthetic DGP produced much noisier individual stock returns than its headline portfolio-volatility parameter suggested.

After isolating these issues, the current interpretation is more cautious: the secondary model has shown only modest predictive value under the current synthetic setup, and much of the previous economic improvement could be explained by exposure effects or label construction rather than clean meta-labeling skill.

## Triple-Barrier Label Asymmetry

### What We Struggled With

The original fast barrier implementation in `src/triple_barrier_fast.py` used compounded simple returns in a way that was not side-symmetric. For mirrored simple-return paths, upward compounding and downward compounding do not have equal magnitudes:

```text
(10%), +(10%) gives (1.10 * 1.10) - 1 = +21.0%
-(10%), -(10%) gives (0.90 * 0.90) - 1 = -19.0%
```

This created a mechanical preference in label outcomes. In practice, long and short candidates could receive different positive-label rates even when the underlying setup should have been symmetric.

### What We Tried

A log-space barrier was tested in `src/triple_barrier_log_fast.py`. It was useful diagnostically, but it was not kept because the simulated returns are generated in simple-return space and the log transformation introduced a different side imbalance under that DGP.

### How We Solved It

The active corrected implementation is `src/triple_barrier_lsymmetric.py`.

It evaluates each primary trade in side-adjusted trade-return space:

```python
trade_r = side * r
cum_value *= 1.0 + trade_r
side_cum_ret = cum_value - 1.0
```

The label now asks whether the primary trade direction produced a sufficiently good compounded return path.

### Conclusion

The side-symmetric construction removed the mechanical long/short label bias in neutral and balanced sanity checks. Historical notebooks using `simple_triple_barrier_labels_fast` should be treated as pre-fix experiments.

## Dollar Neutrality

### What We Struggled With

The primary strategy is built from a demeaned cross-sectional signal and then gross-normalised, so it is close to dollar-neutral by construction.

The secondary portfolio originally filtered trades and then normalised gross exposure. That spends the available capital, but it does not force long exposure and short exposure to balance.

If the filter keeps mostly one side, gross exposure normalisation scales that side up.

### How We Solved It

`src/secondary_model_split_fast.py` and `notebooks/split_secondary_testing.ipynb` were added to test separate long-side and short-side secondary models.

The split workflow can compare:

- primary strategy,
- pooled secondary with gross exposure normalisation,
- split secondary with gross exposure normalisation,
- split secondary with long/short dollar-neutral reweighting.

### Conclusion

Splitting the secondary model is useful architecturally because it makes long and short decisions explicit and gives a clean way to construct a dollar-neutral secondary portfolio. It did not, by itself, make the classification problem strong. The main remaining bottleneck appears to be label noise and limited feature informativeness.

## Label Noise And DGP Scale

### What We Struggled With

The headline synthetic volatility parameter looked like an individual-stock volatility parameter, but the original covariance scaling targets equal-weight portfolio volatility:

```text
sqrt(w' Sigma w)
```

The triple-barrier labels are stock-level labels. This mismatch matters because stock-level paths can be extremely noisy even when the portfolio volatility is only 10% annually.

### What We Found

`fundamental_analysis/fundamental_notebooks/fundamental0.ipynb` isolates this issue.

In the 100-run diagnostic with the original DGP style, the typical results were approximately:

```text
equal-weight annual volatility:      10.0%
average individual annual volatility: 135.7%
median individual annual volatility:  141.3%
average pairwise correlation:          0.4%
```

So the old `annual_vol = 0.10` setting should be interpreted as a low-correlation, high individual-stock-noise regime.

### Conclusion

The earlier weak ROC AUC results are no longer surprising. The labels are generated from very noisy stock-level paths, and the secondary features are trying to predict outcomes in that noisy environment.

## Fundamental Triple-Barrier Diagnostic

`fundamental_analysis/fundamental_notebooks/fundamental1.ipynb` removes the primary strategy and tests a narrower question:

> When does triple-barrier label 1 behave like a useful proxy for true positive drift?

This is not the final meta-labeling workflow. It is a diagnostic that uses long-only candidates to understand the label itself.

The current result is:

- higher drift improves the relationship between label 1 and true drift,
- lower target portfolio volatility improves the relationship,
- longer horizons generally help,
- false positives remain important,
- the label is informative only under sufficiently favourable signal-to-noise conditions.

## Secondary-Model Feature Experiments

Several feature/model experiments were tested:

- simple logistic regression,
- random forest,
- gradient boosting,
- side-aware features,
- correlation features,
- rank features,
- multi-window return and volatility features,
- split long/short secondary models.

The conclusion so far is that more expressive models and small feature additions have not solved the problem. When labels are very noisy, model complexity does not create signal.

Rank features and extra windows were useful checks, but they are not currently part of the clean workflow.

## Problematic Or Historical Code Paths

These files/notebooks are historical or require caution:

- `src/triple_barrier_fast.py`: old simple-return fast barrier; useful for historical comparison, not the current preferred labeler.
- `src/triple_barrier.py`: older non-fast barrier path.
- `notebooks/main_workflow.ipynb`: important baseline workflow, but created before the side-symmetric barrier and dollar-neutral split analysis.
- `notebooks/fast_worflow.ipynb`: important fast baseline, but uses the old fast barrier in its historical cells.
- `notebooks/fast_simulation.ipynb`: useful simulation scaffold, but older outputs mix old barrier logic, gross-only secondary exposure, and noisy DGP assumptions.
- `notebooks/horizon_sensitivity.ipynb`: current label diagnostics with side-symmetric barrier, but still uses the noisy portfolio-volatility-calibrated DGP.
- `notebooks/split_secondary_testing.ipynb`: current split-secondary diagnostics, still under the noisy synthetic DGP.

Obsolete files removed during cleanup:

- `notebooks/reg_testing.ipynb`
- `notebooks/updates.ipynb`
- `notebooks/project_history_detailed.ipynb`
- `src/triple_barrier_log_fast.py`

## Current Thesis Framing

The honest thesis statement at this point is:

> Meta-labeling can only add value if the labels are economically meaningful, side-symmetric, and evaluated under a portfolio construction that preserves the desired exposure constraints. The recent work showed that the earlier setup violated these conditions in subtle ways. After fixing label symmetry and adding dollar-neutral split-secondary construction, the remaining performance is modest, which points toward label noise and DGP calibration as the central issues rather than pure model complexity.

