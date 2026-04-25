import argparse
import numpy as np
from selective_multimodal.data import FeatureData, MODALITIES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--index', type=int, default=0)
    a = p.parse_args()
    data = FeatureData.load(a.data)
    if not 0 <= a.index < len(data.label):
        p.error('index is outside dataset')
    np.savez_compressed(a.output, **{k: v[a.index] for k,v in zip(MODALITIES,data.features)}, available=data.available[a.index])


if __name__ == '__main__':
    main()
