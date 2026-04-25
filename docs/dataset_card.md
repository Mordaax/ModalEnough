# Data and feature format

The bundled experiment generates binary labels, 0 and 1, for artificial classes. There are no human sentiment annotations or third-party datasets in these runs.

## Loading features

Use a compressed NPZ with the following fields:

| Field | Format |
|---|---|
| `sample_id` | Unique string per example |
| `group_id` | Speaker, video, or other grouping identifier |
| `split` | `train`, `validation`, or `test` |
| `label` | Nonnegative integer vector; the current trainer accepts 0/1 |
| `available` | `bool[N,3]` |
| `text`, `audio`, `vision` | `float32[N,D]`, with widths set in configuration |

Align records by identifier before loading them. The loader rejects duplicate IDs, inconsistent shapes, groups crossing splits, unknown split names, and nonfinite available features. Missing vectors can be zero-filled, but they still need an explicit availability mask. Pickled object arrays aren't loaded.

For sequences, `masked_pool([N,T,D], lengths)` averages valid timesteps and returns zeros for zero length. It ignores padding, including nonfinite padded values, and rejects noninteger lengths or lengths outside [0,T]. Pool and align sequences before saving the feature matrices; availability remains a separate field.

To train on your own vectors, put a validated file at `run_dir/features.npz`, then run `train-predictor`, `train-policy`, and `evaluate`. Record its provenance and permitted use in a manifest. Use official group-aware splits and map labels explicitly to 0/1. Training needs at least two training groups, available validation examples, and available training records for every modality.

Take care with the generation commands: `make-synthetic` and `reproduce` replace the input with generated data.

## What the generator creates

Defaults produce 1,800 independent examples, split 60/20/20 before drawing corruption or modality quality. Each example is its own group. Shared class signs and independent noise supply the signal, while a hidden context changes which modality is useful. The policy never receives that context directly; ordinary feature columns contain noisy cues correlated with it.

Each modality is missing 12% of the time and has a 6% chance of conflicting sign. The manifest saves the generation settings, seed, class counts, software versions, and exact NPZ SHA-256. Additional training noise and held-out missingness, noise, and conflict conditions are applied separately, so those settings can be inspected independently.

## Moving to CMU-MOSI or MOSEI

No real benchmark features have been downloaded, bundled, or trained here. Start with the [MOSI resource](https://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/), [MOSEI resource](https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/), and [official SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) when choosing files.

That work still needs decisions about redistribution rights, feature versions, citations, official splits, video/speaker overlap, neutral examples, and label mapping. The synthetic results can't stand in for sentiment accuracy; a real-data experiment has to measure that separately.
