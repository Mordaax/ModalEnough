import numpy as np
import torch
from selective_multimodal.synthetic import generate
from selective_multimodal.data import Normalizer
from selective_multimodal.train_predictor import training_partitions, train
from selective_multimodal.train_policy import construct_targets, train as train_policy
from selective_multimodal.models import Predictor
from selective_multimodal.cli import validate_config, validate_checkpoint_pair


def config():
    return dict(seed=1, samples=100, dimensions=[4,5,6], classes=2, hidden=8, policy_hidden=12, costs=[.2,.35,.45], initial=0, budgets=[.2,.55,1.], **{'lambda': .35}, rho=.5, predictor_epochs=2, policy_epochs=2, batch_size=32, learning_rate=.003, train_noise=.1, confidence_threshold=.85, reject_threshold=.6, run_dir='runs/test')


def test_partitions_disjoint():
    d = generate(100, [4,5,6], 1)
    a, b = training_partitions(d, 1)
    assert not set(d.group_id[a]) & set(d.group_id[b])
    assert set(a) | set(b) == set(np.where(d.split == 'train')[0])


def test_test_labels_do_not_change_training_or_calibration():
    torch.set_num_threads(1)
    d = generate(100, [4,5,6], 1)
    d = Normalizer.fit(d).transform(d)
    m, p, _ = train(d, config())
    d.label[d.split == 'test'] = 1-d.label[d.split == 'test']
    m2, p2, _ = train(d, config())
    for a,b in zip(m.state_dict().values(), m2.state_dict().values()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_policy_targets_and_fit():
    torch.set_num_threads(1)
    c = config()
    d = generate(100, c['dimensions'], 1)
    model = Predictor(c['dimensions'], c['hidden'], 2).eval()
    state, obs, av, remaining, target = construct_targets(model, d, c)
    for i, a in enumerate(target.tolist()):
        if a < 3:
            assert av[i,a] and not obs[i,a] and c['costs'][a] <= remaining[i]+1e-6
    p, progress, _ = train_policy(model, d, c)
    assert np.isfinite(progress['loss']).all()


def test_config_rejects_invalid_costs():
    c = config()
    c['costs'] = [-1, .3, .4]
    import pytest
    with pytest.raises(ValueError):
        validate_config(c)


def test_stale_ablation_checkpoint_rejected():
    import copy
    import pytest
    a = {'config': config(), 'progress': {'dataset_sha256': 'same'}, 'preprocessing': {'mean': [[0.]], 'std': [[1.]]}, 'model': {'weight': torch.tensor([1.])}}
    validate_checkpoint_pair(a, copy.deepcopy(a))
    for field in ('config', 'progress', 'preprocessing', 'model'):
        b = copy.deepcopy(a)
        if field == 'config':
            b[field]['seed'] = 99
        elif field == 'progress':
            b[field]['dataset_sha256'] = 'stale'
        elif field == 'preprocessing':
            b[field]['mean'][0][0] = 7.
        else:
            b[field]['weight'][0] = 8.
        with pytest.raises(ValueError, match='checkpoint'):
            validate_checkpoint_pair(a, b)


def test_dp_targets_match_hand_calculated_objective():
    from selective_multimodal.train_policy import construct_targets
    from selective_multimodal.policy import ABSTAIN, STOP
    # A test-only predictor: text alone assigns p(y)=.4; adding audio gives .95.
    class KnownPredictor:
        temperature = torch.tensor(1.)
        def state(self, features, mask):
            return mask.float()
        def classifier(self, state):
            p = torch.where(state[:,1].bool(), .95, .4)
            return torch.stack([p.log(), (1-p).log()], 1)
    d = generate(30, [4,5,6], 1).take(np.array([0]))
    d.label[:] = 0
    d.available[:] = [True, True, False]
    c = config()
    c['budgets'] = [.2, .55]
    c['lambda'] = .35
    c['rho'] = .5
    _, obs, _, remaining, actions = construct_targets(KnownPredictor(), d, c)
    # At .2 text cannot expand: abstain .5 beats -log(.4)=.916.
    # At .55 audio: .35*.35 -log(.95)=.174 beats abstain .5.
    rows = torch.where(obs[:,0] & ~obs[:,1] & ~obs[:,2])[0]
    assert actions[rows].tolist() == [ABSTAIN, 1]
    d.available[:,1] = False
    _, _, _, _, actions = construct_targets(KnownPredictor(), d, c)
    assert actions.tolist() == [ABSTAIN, ABSTAIN]


def test_checkpoint_optimizer_matches_selected_epoch():
    import math
    torch.set_num_threads(1)
    c = config()
    c.update(predictor_epochs=15, learning_rate=.1)
    d = generate(100, c['dimensions'], 1)
    model, progress, optimizer = train(d, c)
    best_epoch = 1+int(np.argmin([r['validation_nll'] for r in progress['history']]))
    assert progress['selected_epoch'] == best_epoch
    ids = np.where(np.isin(d.sample_id, progress['predictor_ids']) & d.available.any(1))[0]
    steps = best_epoch*math.ceil(len(ids)/c['batch_size'])
    assert all(int(s['step']) == steps for s in optimizer['state'].values())
