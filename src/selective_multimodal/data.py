"""Strict feature interchange, split checks, and training-only preprocessing."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

MODALITIES = ('text', 'audio', 'vision')


@dataclass
class FeatureData:
    features: list[np.ndarray]
    available: np.ndarray
    label: np.ndarray
    sample_id: np.ndarray
    group_id: np.ndarray
    split: np.ndarray

    def validate(self):
        n = len(self.label)
        if n == 0 or len(self.features) != 3:
            raise ValueError('Expected nonempty data and three modalities')
        if self.available.shape != (n, 3) or self.available.dtype != np.bool_:
            raise ValueError('available must be bool[N,3]')
        if self.label.shape != (n,) or not np.issubdtype(self.label.dtype, np.integer) or np.any(self.label < 0):
            raise ValueError('label must be nonnegative integer[N]')
        for name in ('sample_id', 'group_id', 'split'):
            if getattr(self, name).shape != (n,):
                raise ValueError(f'{name} must have N records')
        if len(set(self.sample_id)) != n:
            raise ValueError('duplicate sample identifiers')
        if not set(self.split).issubset({'train', 'validation', 'test'}):
            raise ValueError('Unknown split')
        groups = {}
        for g, s in zip(self.group_id, self.split):
            if g in groups and groups[g] != s:
                raise ValueError('group leakage across splits')
            groups[g] = s
        for m, x in enumerate(self.features):
            if x.ndim != 2 or x.shape[0] != n or x.shape[1] < 1:
                raise ValueError('Features must be aligned matrices [N,D]')
            if not np.isfinite(x[self.available[:, m]]).all():
                raise ValueError('Nonfinite available features')
        return self

    def take(self, indices):
        return FeatureData([x[indices].copy() for x in self.features], self.available[indices].copy(), self.label[indices].copy(), self.sample_id[indices].copy(), self.group_id[indices].copy(), self.split[indices].copy())

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **dict(zip(MODALITIES, self.features)), available=self.available, label=self.label, sample_id=self.sample_id, group_id=self.group_id, split=self.split)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            result = cls([d[k].astype(np.float32) for k in MODALITIES], d['available'], d['label'], d['sample_id'], d['group_id'], d['split'])
        return result.validate()


def masked_pool(features, lengths):
    x, lengths = np.asarray(features), np.asarray(lengths)
    if x.ndim != 3 or lengths.shape != (len(x),) or not np.issubdtype(lengths.dtype, np.integer) or np.any(lengths < 0) or np.any(lengths > x.shape[1]):
        raise ValueError('Expected [N,T,D] and integer lengths within [0,T]')
    valid = np.arange(x.shape[1])[None, :] < lengths[:, None]
    return np.where(valid[:, :, None], x, 0).sum(axis=1) / np.maximum(lengths[:, None], 1)


@dataclass
class Normalizer:
    mean: list[np.ndarray]
    std: list[np.ndarray]

    @classmethod
    def fit(cls, data):
        data.validate()
        mean, std = [], []
        for m, x in enumerate(data.features):
            rows = (data.split == 'train') & data.available[:, m]
            if not rows.any():
                raise ValueError('Each modality needs available training records')
            mean.append(x[rows].mean(axis=0))
            std.append(np.maximum(x[rows].std(axis=0), 1e-5))
        return cls(mean, std)

    def transform(self, data):
        out = data.take(np.arange(len(data.label)))
        for m, x in enumerate(data.features):
            out.features[m] = np.where(data.available[:, m, None], (x - self.mean[m]) / self.std[m], 0).astype(np.float32)
        return out

    def state_dict(self):
        return {'mean': [x.tolist() for x in self.mean], 'std': [x.tolist() for x in self.std]}

    @classmethod
    def from_state(cls, state):
        return cls([np.asarray(x, dtype=np.float32) for x in state['mean']], [np.asarray(x, dtype=np.float32) for x in state['std']])
