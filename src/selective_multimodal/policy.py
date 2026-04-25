"""Policy state contains observed embeddings and public resource metadata only."""
import numpy as np
import torch
from torch import nn
from .costs import validate_costs

STOP, ABSTAIN = 3, 4


def legal_actions(observed, available, remaining, costs, allow_abstain=True):
    acquire = available.bool() & ~observed.bool() & (costs[None, :] <= remaining[:, None]+1e-6)
    terminal = torch.ones((len(observed), 2), dtype=torch.bool, device=observed.device)
    terminal[:, 1] = allow_abstain
    return torch.cat([acquire, terminal], dim=1)


class AcquisitionPolicy(nn.Module):
    def __init__(self, state_dim, hidden=48):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim+10, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 5))

    def forward(self, state, observed, available, remaining, costs, allow_abstain=True):
        features = torch.cat([state.detach(), observed.float(), available.float(), remaining[:, None], costs[None, :].expand(len(state), -1)], dim=1)
        return self.net(features).masked_fill(~legal_actions(observed, available, remaining, costs, allow_abstain), -1e9)


@torch.no_grad()
def rollout(model, policy, features, available, costs, budget, initial=0, strategy='learned', threshold=.85, reject_threshold=.6, rng=None):
    costs = validate_costs(costs, budget)
    if initial not in range(3):
        raise ValueError('initial must be 0, 1, or 2')
    available = np.asarray(available, dtype=bool)
    if available.shape != (3,):
        raise ValueError('available must have length 3')
    rng = np.random.default_rng(0) if rng is None else rng
    x = [torch.as_tensor(np.asarray(v), dtype=torch.float32).reshape(1, -1) for v in features]
    observed = torch.zeros(1, 3, dtype=torch.bool)
    av = torch.tensor(available[None, :])
    ct = torch.tensor(costs)
    acquired, actions, trace = [], [], []
    spent = 0.
    if strategy.startswith('single_'):
        initial = int(strategy[-1])
    def acquire(m):
        nonlocal spent
        observed[0, m] = True
        acquired.append(m)
        spent += float(costs[m])
        actions.append(['text', 'audio', 'vision'][m])
        trace.append({'action': actions[-1], 'incremental_cost': float(costs[m]), 'total_cost': spent, 'probabilities': model(x, observed).softmax(-1)[0].tolist()})
    if available[initial] and costs[initial] <= budget+1e-6:
        acquire(initial)
    # If configured initial is unavailable, permit a feasible alternative.
    elif not strategy.startswith('single_'):
        feasible = [m for m in range(3) if available[m] and costs[m] <= budget+1e-6]
        if feasible:
            acquire(min(feasible, key=lambda m: costs[m]))
    if not acquired:
        return {'prediction': None, 'decision': 'abstain', 'cost': 0., 'probabilities': [1/model.classifier[-1].out_features]*model.classifier[-1].out_features, 'acquired': [], 'actions': ['abstain'], 'trace': [], 'stopping_reason': 'no_affordable_evidence'}
    reason, abstain = 'stop', False
    for _ in range(3):
        probabilities = model(x, observed).softmax(-1)[0]
        remaining = torch.tensor([max(0., budget-spent)])
        feasible = torch.where(legal_actions(observed, av, remaining, ct)[0, :3])[0].tolist()
        if strategy.startswith('single_'):
            action = STOP
        elif strategy in ('learned', 'learned_no_abstain'):
            action = int(policy(model.state(x, observed), observed, av, remaining, ct, strategy == 'learned').argmax(-1))
        elif strategy == 'threshold' and float(probabilities.max()) >= threshold:
            action = STOP
        elif feasible:
            action = int(rng.choice(feasible)) if strategy == 'random' else feasible[0]
        else:
            action = STOP
        if action < 3:
            acquire(action)
            continue
        abstain = action == ABSTAIN
        reason = 'policy_abstain' if abstain else ('budget_or_availability' if not feasible else 'stop')
        break
    probabilities = model(x, observed).softmax(-1)[0].tolist()
    if strategy == 'threshold' and max(probabilities) < reject_threshold:
        abstain, reason = True, 'confidence_rejection'
    actions.append('abstain' if abstain else 'stop')
    return {'prediction': None if abstain else int(np.argmax(probabilities)), 'decision': 'abstain' if abstain else 'predict', 'cost': spent, 'probabilities': probabilities, 'acquired': acquired, 'actions': actions, 'trace': trace, 'stopping_reason': reason}
