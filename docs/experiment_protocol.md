# Running and reading the experiments

Settings live in `configs/synthetic.yaml`. Run `python scripts/reproduce_synthetic.py` to reproduce seeds 7, 11, and 19. Each run saves its configuration, data manifest and hash, predictor and both policy checkpoints, optimizer and RNG states, training progress, split class counts, JSONL decisions, metrics, plots, and a Markdown report.

Checkpoints load through PyTorch's restricted weights-only loader. They retain the model and optimizer state, but the CLI doesn't implement exact interrupted-run continuation or guarantee bitwise agreement across platforms.

## Conditions and comparisons

Predictor selection and calibration finish before test evaluation begins. "Clean" keeps the generator's original noise, conflicts, and missing inputs. Three additional conditions are applied separately: 35% independent missingness, unit Gaussian feature noise, and audio-sign reversal on 50% of examples. Each uses experiment seed + 100. Training normalization stays fixed, and test labels never select settings.

One frozen predictor supports every acquisition comparison. The baselines cover each modality alone, all affordable inputs, fixed order, random order, confidence-triggered acquisition with terminal rejection, and a separately trained learned policy without abstention. The proposed policy keeps abstention enabled. The privileged oracle is described in [method.md](method.md).

Some baselines coincide by design. At a given budget, `all` and `fixed` both acquire in order 0,1,2 without early stopping; `all_available_full_budget` explicitly allocates enough to obtain every available input. Random order only changes the outcome when more than one successor is feasible and the budget prevents acquiring them all.

## Reading the metrics

Accuracy and macro F1 describe accepted predictions. Coverage tells you how often the system answers, and selective error measures errors among those answers. Reports also retain correct predictions divided by all examples, mean normalized acquisition cost, modality counts, NLL, and ECE.

For all-case NLL and ECE, even a rejected example contributes its final probability vector. An all-missing record contributes a uniform vector. With zero coverage, accuracy, macro F1, and selective error are null; an absent class receives zero F1. Seed aggregation excludes undefined values and keeps the number of defined seeds visible.

The trade-off plots connect measured budget settings. They aren't an optimized Pareto frontier. The separate risk/coverage plot ranks all final learned probabilities by confidence, including rejected cases, as a post-hoc diagnostic; it neither retrains the policy nor selects a deployed threshold using test data. Calibration likewise uses all final probabilities. Across seeds, reports give means and sample standard deviations. Three runs don't supply confidence intervals.

## Checking that learning happened

The shuffled-label control permutes training labels, leaves validation and test labels true, refits the predictor, and measures held-out all-available accuracy. Its `passed` flag is calculated from the predictions: true-label accuracy must exceed shuffled-label accuracy by 0.1. Validation model selection remains enabled, so the control needn't land at exactly 50%. Treat this as a diagnostic, not a significance test.

The test suite checks acquisition legality, cost sums, missing-input edge cases, hidden-feature invariance, training-only normalization, padding, checkpoint identity, split/group leakage, test-label independence, and undefined metrics. A comparison at different coverage levels doesn't establish a fair accuracy gain. Matched-cost/coverage evaluation, broader severity grids, real data, measured encoder timings, stronger acquisition baselines, and confidence intervals are still follow-up work.
