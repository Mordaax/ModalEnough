"""Reproducible standard-library CLI. Paths in config are relative to CWD."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from .checkpoint import load_checkpoint, save_checkpoint
from .costs import validate_costs
from .data import FeatureData, Normalizer, MODALITIES
from .evaluate import evaluate_dataset, oracle_reference, write_report
from .models import Predictor
from .policy import AcquisitionPolicy, rollout
from .synthetic import generate, corrupt
from .train_predictor import train as fit_predictor
from .train_policy import train as fit_policy


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(c):
    required = ['seed','samples','dimensions','classes','hidden','policy_hidden','costs','initial','budgets','lambda','rho','predictor_epochs','policy_epochs','batch_size','learning_rate','train_noise','confidence_threshold','reject_threshold','run_dir']
    missing = set(required)-set(c)
    if missing:
        raise ValueError(f'Missing configuration fields: {sorted(missing)}')
    if len(c['dimensions']) != 3 or any(not isinstance(d, int) or d < 1 for d in c['dimensions']):
        raise ValueError('Three positive feature dimensions required')
    if c['classes'] != 2:
        raise ValueError('Initial release implements binary classification')
    if c['initial'] not in range(3):
        raise ValueError('initial must be 0, 1 or 2')
    if not c['budgets']:
        raise ValueError('At least one budget required')
    for b in c['budgets']:
        validate_costs(c['costs'], b)
    for k in ('hidden','policy_hidden','predictor_epochs','policy_epochs','batch_size'):
        if not isinstance(c[k], int) or c[k] < 1:
            raise ValueError(f'{k} must be a positive integer')
    for k in ('lambda','rho','train_noise','learning_rate'):
        if not np.isfinite(c[k]) or c[k] < 0 or (k == 'learning_rate' and c[k] == 0):
            raise ValueError(f'Invalid {k}')
    for k in ('confidence_threshold','reject_threshold'):
        if not 0 <= c[k] <= 1:
            raise ValueError(f'{k} must lie in [0,1]')
    return c


def load_config(path):
    return validate_config(yaml.safe_load(Path(path).read_text(encoding='utf-8')))


def paths(config):
    out = Path(config['run_dir'])
    out.mkdir(parents=True, exist_ok=True)
    return out, out/'features.npz'


def make_synthetic(config):
    out, path = paths(config)
    data = generate(config['samples'], config['dimensions'], config['seed'])
    data.save(path)
    write_json(out/'config.json', config)
    versions = {}
    for name in ('torch','numpy','scikit-learn','matplotlib','PyYAML'):
        versions[name] = importlib.metadata.version(name)
    manifest = {'seed': config['seed'], 'dataset_sha256': file_hash(path), 'data_kind': 'controlled synthetic features', 'class_mapping': {'0': 'synthetic negative class', '1': 'synthetic positive class'}, 'generation': {'availability_probability': .88, 'conflict_probability': .06, 'signal_strength': [2.5,.35], 'context_cue': 'noisy feature columns 2:4; latent context not returned'}, 'splits': {s: {'samples': int((data.split == s).sum()), 'class_counts': np.bincount(data.label[data.split == s], minlength=config['classes']).tolist()} for s in ('train','validation','test')}, 'versions': versions, 'python': sys.version, 'platform': platform.platform(), 'cost_model': 'simulated cached feature acquisition; initial cost charged', 'created_date': '2026-10-24'}
    write_json(out/'manifest.json', manifest)
    return data


def train_predictor(config):
    out, path = paths(config)
    data = FeatureData.load(path)
    if [x.shape[1] for x in data.features] != config['dimensions'] or np.any(data.label >= config['classes']):
        raise ValueError('Data dimensions or labels disagree with configuration')
    norm = Normalizer.fit(data)
    normalized = norm.transform(data)
    model, progress, optimizer = fit_predictor(normalized, config)
    progress['dataset_sha256'] = file_hash(path)
    policy = AcquisitionPolicy(model.state_dim, config['policy_hidden'])
    save_checkpoint(out/'predictor.pt', model, policy, config, norm.state_dict(), progress, {'predictor': optimizer})
    write_json(out/'predictor_progress.json', progress)
    return model


def train_policy(config):
    out, path = paths(config)
    model, _, state = load_checkpoint(out/'predictor.pt')
    if file_hash(path) != state['progress']['dataset_sha256']:
        raise ValueError('Dataset hash differs from predictor training data')
    if config != state['config']:
        raise ValueError('Configuration changed after predictor training; retrain predictor')
    raw = FeatureData.load(path)
    data = Normalizer.from_state(state['preprocessing']).transform(raw)
    policy_ids = set(state['progress']['policy_ids'])
    policy_data = data.take(np.array([i for i, sid in enumerate(data.sample_id) if sid in policy_ids]))
    for allow in (True, False):
        policy, progress, optimizer = fit_policy(model, policy_data, config, allow)
        full_progress = {**state['progress'], 'policy_training': progress}
        save_checkpoint(out/('model.pt' if allow else 'model_no_abstain.pt'), model, policy, config, state['preprocessing'], full_progress, {**state['optimizer'], 'policy': optimizer})
        write_json(out/('policy_progress.json' if allow else 'policy_no_abstain_progress.json'), progress)


def validate_checkpoint_pair(primary, ablation):
    if primary['config'] != ablation['config'] or primary['preprocessing'] != ablation['preprocessing'] or primary['progress']['dataset_sha256'] != ablation['progress']['dataset_sha256']:
        raise ValueError('Ablation checkpoint config/data/preprocessing do not match primary checkpoint')
    if primary['model'].keys() != ablation['model'].keys() or any(not torch.equal(value, ablation['model'][key]) for key, value in primary['model'].items()):
        raise ValueError('Ablation checkpoint predictor differs from primary checkpoint')


def evaluate(config):
    out, path = paths(config)
    model, policy, state = load_checkpoint(out/'model.pt')
    if config != state['config'] or file_hash(path) != state['progress']['dataset_sha256']:
        raise ValueError('Evaluation config/data differ from checkpoint provenance')
    _, no_abstain, ablation_state = load_checkpoint(out/'model_no_abstain.pt')
    validate_checkpoint_pair(state, ablation_state)
    raw = FeatureData.load(path)
    test = raw.take(raw.split == 'test')
    norm = Normalizer.from_state(state['preprocessing'])
    results, log = [], []
    strategies = ['single_0','single_1','single_2','all','fixed','random','threshold','learned_no_abstain','learned']
    for condition in ('clean','missing','noise','conflict'):
        # Corruption is applied on raw held-out features, then fixed training normalization.
        d = norm.transform(corrupt(test, condition, config['seed']+100))
        for budget in config['budgets']:
            for strategy in strategies:
                p = no_abstain if strategy == 'learned_no_abstain' else policy
                m, decisions = evaluate_dataset(model, p, d, config, strategy, budget, config['seed'])
                results.append({'condition': condition, 'strategy': strategy, 'budget': budget, 'metrics': m})
                log.extend({'condition': condition, 'strategy': strategy, 'budget': budget, **decision} for decision in decisions)
            results.append({'condition': condition, 'strategy': 'oracle_reference', 'budget': budget, 'metrics': oracle_reference(model, d, config, budget)})
    # A true full-available reference has a budget sufficient for every input.
    d = norm.transform(test)
    all_metrics, _ = evaluate_dataset(model, policy, d, config, 'all', sum(config['costs']), config['seed'])
    results.append({'condition': 'clean', 'strategy': 'all_available_full_budget', 'budget': sum(config['costs']), 'metrics': all_metrics})
    write_json(out/'metrics.json', results)
    with (out/'decisions.jsonl').open('w', encoding='utf-8') as f:
        for decision in log:
            f.write(json.dumps(decision, allow_nan=False)+'\n')
    write_report(out, results)
    # A fixed acquisition system without early termination equals all-affordable.
    write_json(out/'evaluation_notes.json', {'fixed_and_all': 'At each budget, all and fixed use the same order; all_available_full_budget is the unbudgeted all-available reference.', 'thresholds': 'Fixed in configuration before test evaluation; not tuned on test labels.', 'oracle': 'Uses test labels and future features solely as an explicitly privileged reference.', 'risk_coverage': 'Budget sweep shows observed operating points; dedicated confidence ranking curve is reported separately.'})
    # Confidence-ranking risk/coverage of final learned probabilities; does not retune policy.
    _, decisions = evaluate_dataset(model, policy, d, config, 'learned', max(config['budgets']), config['seed'])
    probs = np.array([r['probabilities'] for r in decisions])
    order = np.argsort(-probs.max(1), kind='stable')
    errors = (probs.argmax(1) != d.label)[order]
    curve = [{'coverage': float(k/len(errors)), 'risk': float(errors[:k].mean())} for k in range(1,len(errors)+1)]
    write_json(out/'risk_coverage.json', curve)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot([r['coverage'] for r in curve], [r['risk'] for r in curve])
    ax.set(xlabel='Post-hoc confidence ranking coverage', ylabel='Risk', title='Clean learned final predictions (including rejected)')
    fig.tight_layout(); fig.savefig(out/'risk_coverage.png', dpi=160); plt.close(fig)
    return results


def learning_control(config):
    """Shuffle training labels, preserve true validation/test labels and features."""
    out, path = paths(config)
    raw = FeatureData.load(path)
    norm = Normalizer.fit(raw)
    data = norm.transform(raw)
    ids = np.where(data.split == 'train')[0]
    data.label[ids] = np.random.default_rng(config['seed']+999).permutation(data.label[ids])
    shuffled_model, progress, _ = fit_predictor(data, config)
    dummy = AcquisitionPolicy(shuffled_model.state_dim, config['policy_hidden'])
    m, _ = evaluate_dataset(shuffled_model, dummy, data.take(data.split == 'test'), config, 'all', sum(config['costs']), config['seed'])
    model, policy, state = load_checkpoint(out/'model.pt')
    actual, _ = evaluate_dataset(model, policy, norm.transform(raw.take(raw.split == 'test')), config, 'all', sum(config['costs']), config['seed'])
    check = {'control': 'training labels shuffled; validation/test labels preserved; validation selection remains enabled', 'true_label_accuracy': actual['accuracy'], 'shuffled_label_accuracy': m['accuracy'], 'chance_reference': .5, 'passed': actual['accuracy'] is not None and m['accuracy'] is not None and actual['accuracy'] > m['accuracy']+.1, 'interpretation': 'Preliminary held-out learning diagnostic; no guaranteed outcome'}
    write_json(out/'learning_check.json', check)
    return check


def reproduce(config):
    torch.set_num_threads(1)
    make_synthetic(config)
    train_predictor(config)
    train_policy(config)
    results = evaluate(config)
    check = learning_control(config)
    print(json.dumps({'run_dir': config['run_dir'], 'comparisons': len(results), 'learning_check': check}, indent=2))


def predict(args):
    model, policy, state = load_checkpoint(args.checkpoint)
    c = state['config']
    with np.load(args.input, allow_pickle=False) as d:
        available = np.asarray(d['available'])
        if available.dtype != np.bool_ or available.shape != (3,):
            raise ValueError('Prediction available must be bool[3]')
        raw = [np.asarray(d[k], dtype=np.float32) for k in MODALITIES]
    if [x.shape for x in raw] != [(d,) for d in c['dimensions']]:
        raise ValueError('Prediction feature dimensions disagree with checkpoint')
    norm = Normalizer.from_state(state['preprocessing'])
    features = []
    for m, v in enumerate(raw):
        if available[m] and not np.isfinite(v).all():
            raise ValueError('Nonfinite available features')
        features.append((v-norm.mean[m])/norm.std[m] if available[m] else np.zeros_like(v))
    result = rollout(model, policy, features, available, c['costs'], args.budget if args.budget is not None else max(c['budgets']), c['initial'])
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('make-synthetic','train-predictor','train-policy','evaluate','reproduce','learning-control'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
    p = sub.add_parser('predict')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--budget', type=float)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == 'predict':
        predict(args)
    else:
        config = load_config(args.config)
        {'make-synthetic': make_synthetic, 'train-predictor': train_predictor, 'train-policy': train_policy, 'evaluate': evaluate, 'reproduce': reproduce, 'learning-control': learning_control}[args.command](config)


if __name__ == '__main__':
    main()
