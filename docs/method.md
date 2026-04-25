# How the model acquires inputs

The implementation combines a supervised classifier with a learned acquisition policy over three cached vectors. It covers the brief's first synthetic milestone. Raw audiovisual encoders and real sentiment evaluation come later.

Each modality has a projection network with two linear layers and ReLU activations. The classifier receives the three projected embeddings together with a mask showing which inputs it has seen. Only observed rows enter a projection; everything else contributes a zero embedding. That boundary is enforced in code, before a network gets to inspect the input.

A separate MLP makes the acquisition decisions. Its inputs are detached observed embeddings, observed and availability masks, costs, and the remaining budget. It returns scores for three acquisition actions, STOP, and ABSTAIN, with illegal acquisitions masked out before selection.

Inference pays for the configured starting modality. If that input is missing or too expensive, it takes the cheapest available alternative within budget; if none qualifies, it abstains at zero cost. Each acquisition adds a log entry containing its incremental cost, total spend, and updated probabilities. Three acquisitions and a terminal action are the longest possible trajectory. Single-modality baselines request only their designated input; the others follow the same starting rule.

## Training the predictor

Training groups are split again: 65% fit the predictor, and 35% supply policy targets. Their labels never overlap. Normalization uses available training features without labels, while predictor fitting uses cross-entropy over random nonempty observed subsets and separate additive noise.

Validation chooses the predictor weights using mean NLL across feasible nonempty subsets. It also supplies predictions for global temperature scaling: choose one of 51 values in [0.5, 3] using pooled subset predictions. Test labels have no role here.

## Teaching the policy what to request

Freeze the calibrated predictor, then enumerate every feasible nonempty subset and configured budget on the policy partition. Working backwards from larger subsets gives the cost of each action:

```
stop_loss(S) = -log p(y | S)
abstain_loss = rho
acquire_loss(m,S) = lambda * cost(m) + V(S union {m})
V(S) = min(feasible terminal and acquisition losses)
```

The cheapest action becomes the supervised target. Money already spent is sunk, so the state value counts only future acquisitions. Training can use labels and hypothetical future features to construct these targets; the deployed policy gets neither. It learns through cross-entropy on enumerated states. For the ablation, a second policy is trained from scratch with abstention removed from both the targets and action masks.

## What the costs mean

The default costs are [0.2, 0.35, 0.45], totaling one simulated unit. Even the first text input is charged. Retrieving fewer cached features or skipping projections doesn't establish savings in raw encoding time, transcription, energy, or production latency; those would need their own measurements.

The initial experiment uses one cost vector, three budgets, enumerated state coverage, and a global temperature. Thresholds are fixed in configuration before test evaluation, and saved test results weren't used to select hyperparameters. Cost-setting and rollout augmentation, subset-specific calibration, target uncertainty, matched-coverage tuning, and real data remain open work.

The labelled oracle gets a deliberately easier task. It chooses the cheapest correct feasible subset when one exists, or the cheapest feasible subset otherwise, using true labels and future observations and ignoring the deployed starting-modality constraint. Read its accuracy as an optimistic reference. It cannot be deployed as this policy or used to claim comparable efficiency.
