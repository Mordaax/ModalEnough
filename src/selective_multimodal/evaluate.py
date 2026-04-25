"""Held-out metrics and static reports; rejected cases retain confidence logs."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss
from .policy import rollout


def metrics(labels, decisions, costs, classes):
    accepted = np.array([r['prediction'] is not None for r in decisions])
    probabilities = np.array([r['probabilities'] for r in decisions], dtype=np.float64)
    # Float32 softmax lists become float64 here; restore unit sums at that precision.
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = np.array([r['prediction'] if r['prediction'] is not None else -1 for r in decisions])
    accuracy = float(accuracy_score(labels[accepted], predictions[accepted])) if accepted.any() else None
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    ece = 0.
    calibration = []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        rows = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if rows.any():
            gap = abs(float(correct[rows].mean()) - float(confidence[rows].mean()))
            ece += rows.mean()*gap
            calibration.append({'confidence': float(confidence[rows].mean()), 'accuracy': float(correct[rows].mean()), 'count': int(rows.sum())})
    return {'n': len(labels), 'coverage': float(accepted.mean()), 'accuracy': accuracy, 'selective_error': None if accuracy is None else 1-accuracy, 'macro_f1': float(f1_score(labels[accepted], predictions[accepted], labels=list(range(classes)), average='macro', zero_division=0)) if accepted.any() else None, 'correct_per_total': float(((predictions == labels) & accepted).mean()), 'average_cost': float(np.mean([r['cost'] for r in decisions])), 'normalized_cost': float(np.mean([r['cost'] for r in decisions])/sum(costs)), 'modality_counts': [int(sum(m in r['acquired'] for r in decisions)) for m in range(3)], 'nll_all_cases': float(log_loss(labels, probabilities, labels=list(range(classes)))), 'ece_all_cases': float(ece), 'calibration': calibration}


def evaluate_dataset(model, policy, data, config, strategy, budget, seed=0):
    rng = np.random.default_rng(seed)
    decisions = [rollout(model, policy, [x[i] for x in data.features], data.available[i], config['costs'], budget, initial=config['initial'], strategy=strategy, threshold=config['confidence_threshold'], reject_threshold=config['reject_threshold'], rng=rng) for i in range(len(data.label))]
    for i, d in enumerate(decisions):
        d['sample_id'] = str(data.sample_id[i])
    return metrics(data.label, decisions, config['costs'], config['classes']), decisions


@torch.no_grad()
def oracle_reference(model, data, config, budget):
    """Privileged best correct subset under budget; reports its own cost/coverage.

    Uses labels to choose subsets. Never a deployable policy or fair competitor.
    """
    decisions = []
    for i, y in enumerate(data.label):
        candidates = []
        x = [torch.tensor(v[i:i+1]) for v in data.features]
        for bits in range(1, 8):
            mask = np.array([bool(bits & (1 << m)) for m in range(3)])
            cost = float(np.dot(mask, config['costs']))
            if np.any(mask & ~data.available[i]) or cost > budget+1e-6:
                continue
            p = model(x, torch.tensor(mask[None, :])).softmax(-1)[0].numpy()
            candidates.append((int(p.argmax()) != y, cost, -float(p[y]), mask, p))
        if not candidates:
            decisions.append({'prediction': None, 'cost': 0., 'probabilities': [.5, .5], 'acquired': []})
        else:
            _, cost, _, mask, p = min(candidates, key=lambda v: v[:3])
            decisions.append({'prediction': int(p.argmax()), 'cost': cost, 'probabilities': p.tolist(), 'acquired': np.where(mask)[0].tolist()})
    return metrics(data.label, decisions, config['costs'], config['classes'])


def write_report(out, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    rows = ['# Synthetic test results', '', 'These predictions come from generated feature vectors. Acquisition costs are simulated; real sentiment accuracy and raw encoder savings still need separate measurements.', '', '| Condition | Strategy | Budget | Accuracy (accepted) | Coverage | Normalized cost |', '|---|---|---:|---:|---:|---:|']
    for r in results:
        m = r['metrics']
        acc = 'undefined' if m['accuracy'] is None else f"{m['accuracy']:.3f}"
        rows.append(f"| {r['condition']} | {r['strategy']} | {r['budget']:.2f} | {acc} | {m['coverage']:.3f} | {m['normalized_cost']:.3f} |")
    (out/'report.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    clean = [r for r in results if r['condition'] == 'clean' and r['strategy'] != 'oracle_reference']
    for strategy in sorted({r['strategy'] for r in clean}):
        rows_s = [r for r in clean if r['strategy'] == strategy]
        axes[0].plot([r['metrics']['normalized_cost'] for r in rows_s], [r['metrics']['accuracy'] if r['metrics']['accuracy'] is not None else np.nan for r in rows_s], 'o-', label=strategy)
        axes[1].plot([r['metrics']['coverage'] for r in rows_s], [r['metrics']['selective_error'] if r['metrics']['selective_error'] is not None else np.nan for r in rows_s], 'o-')
    axes[0].set(xlabel='Normalized simulated acquisition cost', ylabel='Accepted accuracy', title='Clean synthetic: budget sweep')
    axes[1].set(xlabel='Coverage', ylabel='Selective error', title='Observed operating points (budget sweep)')
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out/'tradeoffs.png', dpi=160)
    plt.close(fig)
    chosen = next(r for r in clean if r['strategy'] == 'learned' and r['budget'] == max(x['budget'] for x in clean))
    bins = chosen['metrics']['calibration']
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.plot([b['confidence'] for b in bins], [b['accuracy'] for b in bins], 'o-')
    ax.set(xlabel='Confidence', ylabel='Empirical accuracy', title='Learned policy: all final predictions')
    fig.tight_layout(); fig.savefig(out/'calibration.png', dpi=160); plt.close(fig)
