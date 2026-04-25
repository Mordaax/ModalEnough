import numpy as np
import pytest
import torch

from selective_multimodal.data import FeatureData, Normalizer, masked_pool
from selective_multimodal.synthetic import generate
from selective_multimodal.models import Predictor
from selective_multimodal.policy import AcquisitionPolicy, legal_actions, rollout
from selective_multimodal.evaluate import metrics
from selective_multimodal.checkpoint import save_checkpoint, load_checkpoint


def test_synthetic_reproducible_and_disjoint():
    d = generate(180, [12, 10, 8], 4)
    e = generate(180, [12, 10, 8], 4)
    assert np.array_equal(d.features[0], e.features[0])
    d.validate()
    assert set(d.split) == {'train', 'validation', 'test'}
    for s in set(d.split):
        assert not set(d.group_id[d.split == s]) & set(d.group_id[d.split != s])


def test_normalizer_train_only_and_missing():
    d = generate(100, [4, 5, 6], 1)
    n = Normalizer.fit(d)
    means = [x.copy() for x in n.mean]
    for x in d.features:
        x[d.split != 'train'] += 1000
    n2 = Normalizer.fit(d)
    for a, b in zip(means, n2.mean):
        np.testing.assert_array_equal(a, b)
    z = n.transform(d)
    for m, x in enumerate(z.features):
        assert np.all(x[~d.available[:, m]] == 0)


def test_pool_ignores_padding():
    x = np.array([[[1., 3.], [3., 5.], [np.nan, np.nan]]])
    np.testing.assert_array_equal(masked_pool(x, np.array([2])), [[2., 4.]])
    with pytest.raises(ValueError):
        masked_pool(x, np.array([4]))


def test_invalid_data_rejected():
    d = generate(100, [4, 5, 6], 1)
    d.sample_id[1] = d.sample_id[0]
    with pytest.raises(ValueError, match='duplicate'):
        d.validate()


def test_information_boundary():
    torch.manual_seed(3)
    model = Predictor([4, 5, 6], 8, 2).eval()
    policy = AcquisitionPolicy(model.state_dim, 16).eval()
    x = [torch.randn(5, k) for k in [4, 5, 6]]
    observed = torch.tensor([[True, False, False]] * 5)
    available = torch.ones(5, 3, dtype=torch.bool)
    remaining = torch.ones(5)
    costs = torch.tensor([.2, .3, .5])
    a = policy(model.state(x, observed), observed, available, remaining, costs)
    x[1][:] = float('nan')
    x[2][:] = 1e9
    b = policy(model.state(x, observed), observed, available, remaining, costs)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_legal_actions():
    observed = torch.tensor([[True, False, False]])
    available = torch.tensor([[True, True, False]])
    mask = legal_actions(observed, available, torch.tensor([.1]), torch.tensor([.2, .3, .5]))
    assert mask.tolist() == [[False, False, False, True, True]]


@pytest.mark.parametrize('budget', [0., .1, .2, .55, 1.])
@pytest.mark.parametrize('available', [[False]*3, [True, False, False], [True]*3])
def test_rollout_safety(budget, available):
    model = Predictor([4, 5, 6], 8, 2).eval()
    policy = AcquisitionPolicy(model.state_dim, 16).eval()
    result = rollout(model, policy, [np.ones(4), np.ones(5), np.ones(6)], np.array(available), [.2, .35, .45], budget)
    assert result['cost'] <= budget + 1e-6
    assert len(result['acquired']) == len(set(result['acquired']))
    assert len(result['actions']) <= 4
    assert all(available[m] for m in result['acquired'])
    assert result['cost'] == pytest.approx(sum([.2, .35, .45][m] for m in result['acquired']))
    if not any(available) or budget < .2:
        assert result['decision'] == 'abstain'


def test_metrics_zero_coverage():
    r = metrics(np.array([0, 0]), [{'prediction': None, 'cost': 0., 'probabilities': [.5, .5], 'acquired': []}]*2, [1., 1., 1.], 2)
    assert r['coverage'] == 0
    assert r['selective_error'] is None
    assert r['macro_f1'] is None


def test_metrics_float32_probabilities_no_warnings():
    import warnings
    p = np.array([.9, .10000005], dtype=np.float32).tolist()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = metrics(np.array([1]), [{'prediction': 1, 'cost': .2, 'probabilities': p, 'acquired': [0]}], [.2,.3,.5], 2)
    assert np.isfinite(result['nll_all_cases'])


def test_checkpoint_roundtrip(tmp_path):
    model = Predictor([4, 5, 6], 8, 2).eval()
    policy = AcquisitionPolicy(model.state_dim, 16).eval()
    cfg = {'dimensions': [4, 5, 6], 'hidden': 8, 'classes': 2, 'policy_hidden': 16}
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model, policy, cfg, None, {'epoch': 0})
    m, p, checkpoint = load_checkpoint(path)
    x = [torch.randn(2, k) for k in cfg['dimensions']]
    mask = torch.ones(2, 3, dtype=torch.bool)
    torch.testing.assert_close(model(x, mask), m(x, mask), rtol=0, atol=0)
