from pathlib import Path
import torch
from .models import Predictor
from .policy import AcquisitionPolicy


def save_checkpoint(path, model, policy, config, preprocessing, progress, optimizer=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format_version': 1, 'model': model.state_dict(), 'policy': policy.state_dict(), 'config': config, 'preprocessing': preprocessing, 'progress': progress, 'optimizer': optimizer, 'torch_rng': torch.get_rng_state()}, path)


def load_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format_version'] != 1:
        raise ValueError('Unknown checkpoint format')
    c = state['config']
    model = Predictor(c['dimensions'], c['hidden'], c['classes'])
    policy = AcquisitionPolicy(model.state_dim, c['policy_hidden'])
    model.load_state_dict(state['model'])
    policy.load_state_dict(state['policy'])
    return model.eval(), policy.eval(), state
