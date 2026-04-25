import copy
import numpy as np
import torch
from torch.nn import functional as F
from .models import Predictor


def tensors(data):
    return [torch.tensor(x, dtype=torch.float32) for x in data.features], torch.tensor(data.available), torch.tensor(data.label, dtype=torch.long)


def training_partitions(data, seed):
    indices = np.where(data.split == 'train')[0]
    # Partition GROUPS to preserve disjointness even for externally loaded data.
    groups = np.unique(data.group_id[indices])
    np.random.default_rng(seed).shuffle(groups)
    if len(groups) < 2:
        raise ValueError('At least two training groups required for target separation')
    predictor_groups = set(groups[:max(1, int(.65*len(groups)))])
    a = indices[np.array([g in predictor_groups for g in data.group_id[indices]])]
    b = indices[np.array([g not in predictor_groups for g in data.group_id[indices]])]
    return a, b


def train(data, config):
    torch.manual_seed(config['seed'])
    model = Predictor(config['dimensions'], config['hidden'], config['classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    predictor_idx, policy_idx = training_partitions(data, config['seed'])
    x, available, y = tensors(data)
    eligible = available.any(1)
    train_idx = torch.tensor(predictor_idx)[eligible[predictor_idx]]
    val_idx = torch.tensor(np.where(data.split == 'validation')[0])
    val_idx = val_idx[eligible[val_idx]]
    if not len(train_idx) or not len(val_idx):
        raise ValueError('Need available predictor and validation examples')
    best, best_optimizer, best_epoch, best_loss, history = None, None, 0, float('inf'), []
    for epoch in range(config['predictor_epochs']):
        model.train()
        total = 0.
        order = train_idx[torch.randperm(len(train_idx))]
        for ids in order.split(config['batch_size']):
            mask = (torch.rand(len(ids), 3) < .65) & available[ids]
            for j in torch.where(~mask.any(1))[0]:
                candidates = torch.where(available[ids[j]])[0]
                mask[j, candidates[torch.randint(len(candidates), (1,))]] = True
            noisy = [v[ids] + torch.randn_like(v[ids])*config['train_noise'] for v in x]
            loss = F.cross_entropy(model(noisy, mask), y[ids])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach())*len(ids)
        model.eval()
        with torch.no_grad():
            losses = []
            for bits in range(1, 8):
                mask = available[val_idx] & torch.tensor([bool(bits & (1<<m)) for m in range(3)])
                rows = mask.any(1)
                if rows.any():
                    losses.append(F.cross_entropy(model([v[val_idx][rows] for v in x], mask[rows]), y[val_idx][rows]))
            val_loss = float(torch.stack(losses).mean())
        history.append({'epoch': epoch+1, 'train_loss': total/len(order), 'validation_nll': val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch+1
    model.load_state_dict(best)
    model.eval()
    # A single global temperature fitted across nonempty validation subsets.
    logits, labels = [], []
    with torch.no_grad():
        for bits in range(1, 8):
            mask = available[val_idx] & torch.tensor([bool(bits & (1<<m)) for m in range(3)])
            rows = mask.any(1)
            if rows.any():
                logits.append(model([v[val_idx][rows] for v in x], mask[rows]))
                labels.append(y[val_idx][rows])
    logits, labels = torch.cat(logits), torch.cat(labels)
    temperatures = torch.linspace(.5, 3., 51)
    losses = [float(F.cross_entropy(logits/t, labels)) for t in temperatures]
    model.temperature.fill_(temperatures[int(np.argmin(losses))])
    return model, {'history': history, 'selected_epoch': best_epoch, 'predictor_ids': data.sample_id[predictor_idx].tolist(), 'policy_ids': data.sample_id[policy_idx].tolist(), 'temperature': float(model.temperature), 'calibration_split': 'validation', 'calibration': 'global pooled nonempty subsets'}, best_optimizer
