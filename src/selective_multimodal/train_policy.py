"""Privileged labels/future subsets construct training targets only."""
import numpy as np
import torch
from torch.nn import functional as F
from .policy import AcquisitionPolicy, STOP, ABSTAIN
from .train_predictor import tensors


@torch.no_grad()
def construct_targets(model, data, config, allow_abstain=True):
    x, available, y = tensors(data)
    n = len(y)
    costs = torch.tensor(config['costs'])
    states, stop_losses = {}, {}
    for bits in range(1, 8):
        observed = torch.tensor([[bool(bits & (1 << m)) for m in range(3)]]).expand(n, -1)
        states[bits] = model.state(x, observed)
        logits = model.classifier(states[bits])/model.temperature
        stop_losses[bits] = F.cross_entropy(logits, y, reduction='none')
    packed = []
    for budget in config['budgets']:
        values = {}
        for bits in sorted(range(1, 8), key=int.bit_count, reverse=True):
            obs = torch.tensor([[bool(bits & (1 << m)) for m in range(3)]]).expand(n, -1)
            spent = float((obs[0]*costs).sum())
            valid = ((~obs | available).all(1)) & (spent <= budget+1e-6)
            remaining = torch.full((n,), max(0., budget-spent))
            objective = torch.full((n, 5), float('inf'))
            objective[:, STOP] = stop_losses[bits]
            if allow_abstain:
                objective[:, ABSTAIN] = config['rho']
            for m in range(3):
                if bits & (1 << m) or spent+float(costs[m]) > budget+1e-6:
                    continue
                objective[:, m] = torch.where(available[:, m], config['lambda']*costs[m] + values[bits | (1 << m)], float('inf'))
            values[bits], action = objective.min(1)
            if valid.any():
                packed.append((states[bits][valid], obs[valid], available[valid], remaining[valid], action[valid]))
    if not packed:
        raise ValueError('No affordable policy training states')
    return tuple(torch.cat([row[j] for row in packed]) for j in range(5))


def train(model, data, config, allow_abstain=True):
    torch.manual_seed(config['seed']+int(not allow_abstain))
    model.eval()
    policy = AcquisitionPolicy(model.state_dim, config['policy_hidden'])
    state, observed, available, remaining, target = construct_targets(model, data, config, allow_abstain)
    costs = torch.tensor(config['costs'])
    optimizer = torch.optim.Adam(policy.parameters(), lr=config['learning_rate'])
    counts = torch.bincount(target, minlength=5)
    history = []
    for epoch in range(config['policy_epochs']):
        total = 0.
        for ids in torch.randperm(len(target)).split(config['batch_size']):
            logits = policy(state[ids], observed[ids], available[ids], remaining[ids], costs, allow_abstain)
            loss = F.cross_entropy(logits, target[ids])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach())*len(ids)
        history.append(total/len(target))
    return policy.eval(), {'states': len(target), 'target_counts': counts.tolist(), 'loss': history, 'state_coverage': 'all feasible nonempty subsets at configured budgets; no rollout augmentation', 'allow_abstain': allow_abstain}, optimizer.state_dict()
