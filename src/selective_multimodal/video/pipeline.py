"""Budgeted real-media inference. Synthetic acquisition weights are never reused."""
from dataclasses import dataclass
import math
from pathlib import Path
import time
from .media import probe

LABELS = ('negative', 'positive', 'neutral')
COSTS = {'text': .2, 'audio': .35, 'vision': .45}


@dataclass(frozen=True)
class VideoOptions:
    budget: float = 1.
    confidence: float = .85
    strategy: str = 'adaptive'
    enabled: tuple[str, ...] = ('text', 'audio', 'vision')

    def __post_init__(self):
        if not math.isfinite(self.budget) or not 0 <= self.budget <= 1:
            raise ValueError('Budget must be between 0 and 1.')
        if not math.isfinite(self.confidence) or not .5 <= self.confidence <= .99:
            raise ValueError('Decision threshold must be between 0.5 and 0.99.')
        if self.strategy not in ('adaptive', 'all'):
            raise ValueError('Unknown acquisition strategy.')
        if not set(self.enabled).issubset(COSTS) or len(set(self.enabled)) != len(self.enabled):
            raise ValueError('Unknown or duplicate modality.')


def validate_scores(scores):
    if set(scores) != set(LABELS):
        raise ValueError('Model did not return all three sentiment scores.')
    scores = {key: float(value) for key, value in scores.items()}
    if any(not math.isfinite(v) or v < 0 for v in scores.values()) or sum(scores.values()) <= 0:
        raise ValueError('Model returned invalid sentiment scores.')
    total = sum(scores.values())
    return {key: value/total for key, value in scores.items()}


class VideoPipeline:
    def __init__(self, backend):
        self.backend = backend

    def run(self, path, directory, options=None, on_event=None, media_info=None):
        options = options or VideoOptions()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        info = media_info or probe(path)
        available = {'text': info.has_audio, 'audio': info.has_audio, 'vision': info.has_video}
        steps, acquired, observed, errors = [], [], {}, []
        scores = {label: 1/3 for label in LABELS}
        spent, linguistic = 0., False
        scoring_inputs = []
        terminal = 'insufficient_evidence'

        def emit(event):
            if on_event:
                on_event(event)

        emit({'type': 'metadata', 'media': info.to_dict()})
        for modality in COSTS:
            if modality not in options.enabled or not available[modality] or spent+COSTS[modality] > options.budget+1e-9:
                continue
            emit({'type': 'acquiring', 'modality': modality, 'message': {'text': 'Transcribing speech and classifying the transcript', 'audio': 'Running native audio understanding', 'vision': 'Sampling and analysing video frames'}[modality]})
            step_start = time.perf_counter()
            spent = round(spent+COSTS[modality], 8)
            acquired.append(modality)
            step = {'modality': modality, 'incremental_cost': COSTS[modality], 'total_cost': spent}
            try:
                result = self.backend.acquire(modality, Path(path), info, directory, dict(observed))
                candidate = validate_scores(result['scores'])
                declared = result.get('scoring_inputs', [modality])
                if set(declared)-set(observed)-{modality}:
                    raise ValueError('Scorer claimed an unacquired input.')
                observed[modality] = result
                scores = candidate
                scoring_inputs = list(declared)
                if modality == 'text' and result.get('evidence', {}).get('transcript', '').strip():
                    linguistic = True
                linguistic = linguistic or bool(result.get('linguistic_evidence', False))
                step.update(status='complete', scores=scores, scoring_inputs=scoring_inputs, model=result['model'], evidence=result.get('evidence', {}), details=result.get('details', ''))
            except Exception as exc:
                # A failed acquisition still consumed work and is never retried for free.
                message = str(exc)[:300]
                errors.append({'modality': modality, 'message': message})
                step.update(status='failed', error=message)
            step['elapsed_seconds'] = round(time.perf_counter()-step_start, 3)
            steps.append(step)
            emit({'type': 'step', 'step': step})
            if options.strategy == 'adaptive' and linguistic and max(scores.values()) >= options.confidence:
                terminal = 'sufficient_evidence'
                break

        confidence = max(scores.values())
        prediction = max(scores, key=scores.get) if linguistic and confidence >= options.confidence else None
        if prediction and terminal != 'sufficient_evidence':
            terminal = 'all_requested_inputs_processed'
        if not linguistic:
            terminal = 'no_linguistic_evidence'
        if not acquired:
            terminal = 'no_affordable_input'
        result = {'prediction': prediction, 'decision': 'predict' if prediction else 'abstain', 'scores': scores, 'score_note': 'Relative model scores; not calibrated probabilities of correctness.', 'acquired': acquired, 'used_inputs': scoring_inputs, 'successful_acquisitions': list(observed), 'cost': spent, 'budget': options.budget, 'threshold': options.confidence, 'policy': 'confidence_baseline', 'strategy': options.strategy, 'steps': steps, 'errors': errors, 'stopping_reason': terminal, 'elapsed_seconds': round(time.perf_counter()-started, 3), 'media': info.to_dict(), 'runtime': self.backend.runtime(), 'cost_note': 'Configured effort units, not money. Text includes audio decoding and speech recognition; audio is an additional understanding pass.'}
        emit({'type': 'finished', 'result': result})
        return result
