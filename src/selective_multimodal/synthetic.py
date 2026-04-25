"""Synthetic labels and cached features, not real sentiment measurements."""
import numpy as np
from .data import FeatureData


def generate(n=1800, dimensions=(12, 10, 8), seed=7):
    if n < 30 or len(dimensions) != 3 or min(dimensions) < 4:
        raise ValueError('Use at least 30 examples and three dimensions >=4')
    rng = np.random.default_rng(seed)
    # Assign independent groups/splits BEFORE drawing quality and corruption.
    permutation = rng.permutation(n)
    split = np.full(n, 'test', dtype='<U10')
    split[permutation[:int(.6*n)]] = 'train'
    split[permutation[int(.6*n):int(.8*n)]] = 'validation'
    label = rng.integers(0, 2, n)
    sign = (2*label-1).astype(np.float32)
    context = rng.integers(0, 3, n)
    available = rng.random((n, 3)) > .12
    features = []
    for m, width in enumerate(dimensions):
        x = rng.normal(0, 1, (n, width)).astype(np.float32)
        strength = np.where(context == m, 2.5, .35)
        conflict = rng.random(n) < .06
        strength = np.where(conflict, -strength, strength)
        x[:, 0] += sign * strength
        x[:, 1] += sign * strength * .7
        # Context is a noisy observed cue; the latent context is never returned.
        x[:, 2:4] += np.stack([context == 1, context == 2], axis=1).astype(np.float32)*1.5
        x[~available[:, m]] = 0
        features.append(x)
    ids = np.array([f'synthetic-{seed}-{i:06d}' for i in range(n)])
    return FeatureData(features, available, label, ids, ids.copy(), split).validate()


def corrupt(data, condition, seed=0):
    rng = np.random.default_rng(seed)
    d = data.take(np.arange(len(data.label)))
    if condition == 'missing':
        d.available &= rng.random(d.available.shape) > .35
    elif condition == 'noise':
        d.features = [x + rng.normal(0, 1., x.shape).astype(np.float32) for x in d.features]
    elif condition == 'conflict':
        rows = rng.random(len(d.label)) < .5
        d.features[1][rows, :2] *= -1
    elif condition != 'clean':
        raise ValueError(condition)
    for m in range(3):
        d.features[m][~d.available[:, m]] = 0
    return d
