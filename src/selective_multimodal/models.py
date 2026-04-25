import torch
from torch import nn


class Predictor(nn.Module):
    def __init__(self, dimensions, hidden=24, classes=2):
        super().__init__()
        self.projections = nn.ModuleList([nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()) for d in dimensions])
        self.state_dim = 3*hidden + 3
        self.classifier = nn.Sequential(nn.Linear(self.state_dim, hidden), nn.ReLU(), nn.Linear(hidden, classes))
        self.register_buffer('temperature', torch.tensor(1.))

    def state(self, features, observed):
        # Select rows BEFORE projection: no encoder processes hidden records.
        parts = []
        for m, projection in enumerate(self.projections):
            z = features[m].new_zeros((len(observed), projection[0].out_features))
            rows = observed[:, m].bool()
            if rows.any():
                z[rows] = projection(features[m][rows])
            parts.append(z)
        return torch.cat(parts + [observed.float()], dim=1)

    def forward(self, features, observed):
        return self.classifier(self.state(features, observed)) / self.temperature.clamp_min(.05)
