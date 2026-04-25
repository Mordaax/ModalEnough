"""Real-media controller tests use an explicit test backend, never production fallbacks."""
from pathlib import Path
import numpy as np
import pytest
from selective_multimodal.video.pipeline import VideoPipeline, VideoOptions
from selective_multimodal.video.media import MediaInfo, MediaError, probe, extract_audio, sample_frames
from selective_multimodal.video.backends import parse_claude_response, chunk_token_ids


class RecordingBackend:
    name = 'test-only'
    def __init__(self, text=(.1,.8,.1), audio=(.05,.9,.05), vision=(.05,.9,.05)):
        self.calls = []
        self.values = dict(text=text, audio=audio, vision=vision)
    def acquire(self, modality, media, info, directory, observed):
        self.calls.append(modality)
        return {'scores': dict(zip(('negative','positive','neutral'),self.values[modality])), 'model': 'test-fixture', 'evidence': {'transcript': 'A review'} if modality=='text' else {}, 'details': 'Observed review evidence'}
    def runtime(self):
        return {'backend':'test-only'}


def info(audio=True):
    return MediaInfo(duration=10., has_audio=audio, has_video=True, width=320, height=240)


def run(tmp_path, backend, **options):
    return VideoPipeline(backend).run(tmp_path/'clip.mp4',tmp_path,VideoOptions(**options),media_info=info())


def test_confident_text_does_not_run_hidden_encoders(tmp_path):
    backend=RecordingBackend(text=(.01,.98,.01))
    result=run(tmp_path,backend)
    assert backend.calls==['text']
    assert result['prediction']=='positive'
    assert result['acquired']==['text']
    assert result['cost']==pytest.approx(.2)
    assert result['policy']=='confidence_baseline'


def test_uncertain_text_acquires_audio_and_stops(tmp_path):
    backend=RecordingBackend(text=(.4,.5,.1),audio=(.02,.96,.02))
    result=run(tmp_path,backend,confidence=.8)
    assert backend.calls==['text','audio']
    assert result['cost']==pytest.approx(.55)
    assert result['prediction']=='positive'
    assert len(result['steps'])==2


def test_unaffordable_inputs_are_never_run(tmp_path):
    backend=RecordingBackend(text=(.4,.5,.1))
    result=run(tmp_path,backend,budget=.2)
    assert backend.calls==['text']
    assert result['prediction'] is None
    assert result['stopping_reason']=='insufficient_evidence'


def test_zero_budget_abstains_without_inference(tmp_path):
    backend=RecordingBackend()
    result=run(tmp_path,backend,budget=0.)
    assert not backend.calls
    assert result['cost']==0.
    assert result['prediction'] is None


def test_silent_video_never_runs_audio_or_asr(tmp_path):
    backend=RecordingBackend(vision=(.01,.98,.01))
    result=VideoPipeline(backend).run(tmp_path/'clip.mp4',tmp_path,VideoOptions(),media_info=info(False))
    assert backend.calls==['vision']
    assert result['prediction'] is None  # Frames alone cannot establish the reviewer's opinion.
    assert result['stopping_reason']=='no_linguistic_evidence'


def test_all_strategy_calls_each_encoder_once(tmp_path):
    backend=RecordingBackend()
    result=run(tmp_path,backend,strategy='all')
    assert backend.calls==['text','audio','vision']
    assert result['cost']==pytest.approx(1.)


def test_failed_encoder_is_visible_and_charged(tmp_path):
    class Failed(RecordingBackend):
        def acquire(self,modality,*args):
            self.calls.append(modality)
            raise RuntimeError('model unavailable')
    backend=Failed()
    result=run(tmp_path,backend)
    assert result['prediction'] is None
    assert len(result['errors'])==3
    assert result['cost']==pytest.approx(1.)
    assert backend.calls==['text','audio','vision']


@pytest.mark.parametrize('kwargs',[{'budget':-1},{'confidence':1.1},{'strategy':'fake'},{'budget':float('nan')}])
def test_bad_options(kwargs):
    with pytest.raises(ValueError): VideoOptions(**kwargs)


def test_claude_response_validation():
    value=parse_claude_response('```json\n{"scores":{"negative":0.1,"positive":0.8,"neutral":0.1},"explanation":"Clear recommendation"}\n```')
    assert value['scores']['positive']==.8
    with pytest.raises(ValueError): parse_claude_response('{"scores":{"negative":-1,"positive":2,"neutral":0}}')
    with pytest.raises(ValueError): parse_claude_response('{"scores":{"negative":0,"positive":0,"neutral":0}}')


def test_token_chunking_covers_every_token():
    pieces=chunk_token_ids(list(range(1000)),200)
    assert [v for piece in pieces for v in piece]==list(range(1000))
    assert max(map(len,pieces))<=200


def test_transcriber_receives_decoded_samples_not_a_version_sensitive_av_path(tmp_path):
    from types import SimpleNamespace
    import soundfile as sf
    from selective_multimodal.video.backends import LocalBackend
    path=tmp_path/'speech.wav'
    sf.write(path,np.ones(16000,dtype=np.float32)*.01,16000)
    class Speech:
        def transcribe(self,audio,**kwargs):
            assert isinstance(audio,np.ndarray)
            return iter([SimpleNamespace(start=0.,end=1.,text='I loved this camera.',no_speech_prob=.01)]),None
    backend=LocalBackend(device='cpu')
    backend._speech=Speech()
    text,segments=backend._transcribe(path)
    assert text=='I loved this camera.'


def test_text_sentiment_chunks_have_a_batch_dimension():
    import torch
    from types import SimpleNamespace
    from selective_multimodal.video.backends import LocalBackend
    class Tokenizer:
        def encode(self,*args,**kwargs): return [1,2,3]
        def prepare_for_model(self,tokens,prepend_batch_axis=False,**kwargs):
            x=torch.tensor(tokens)
            return {'input_ids':x[None,:] if prepend_batch_axis else x}
    class Model:
        config=SimpleNamespace(id2label={0:'negative',1:'neutral',2:'positive'})
        def __call__(self,input_ids):
            assert input_ids.ndim==2
            return SimpleNamespace(logits=torch.tensor([[0.,1.,4.]]))
    backend=LocalBackend(device='cpu')
    backend._tokenizer=Tokenizer()
    backend._text=Model()
    assert backend._text_scores('A lovely review')['positive']>.9


def test_visual_stage_retains_acquired_audio_but_never_reads_hidden_audio(tmp_path,monkeypatch):
    import soundfile as sf
    from PIL import Image
    import selective_multimodal.video.backends as module
    image=tmp_path/'frame.jpg'
    Image.new('RGB',(32,32),'blue').save(image)
    sf.write(tmp_path/'speech.wav',np.ones(16000,dtype=np.float32)*.1,16000)
    monkeypatch.setattr(module,'sample_frames',lambda *a,**k:[{'path':str(image),'timestamp':.5}])
    backend=module.LocalBackend(device='cpu')
    received=[]
    def score(transcript='',images=None,audio=None):
        received.append(audio)
        return {'positive':.9,'negative':.05,'neutral':.05}
    backend._omni_scores=score
    text={'text':{'evidence':{'transcript':'I recommend this camera.'}}}
    backend.acquire('vision',tmp_path/'video.mp4',info(),tmp_path,text)
    assert received[-1] is None
    observed={**text,'audio':{'evidence':{'windows':[{'start':0.,'end':1.}]}}}
    result=backend.acquire('vision',tmp_path/'video.mp4',info(),tmp_path,observed)
    assert isinstance(received[-1],np.ndarray)
    assert result['scoring_inputs']==['text','audio','vision']


def test_model_prefetch_keeps_required_speaker_resource(tmp_path,monkeypatch):
    import fnmatch
    import huggingface_hub
    from selective_multimodal.video.backends import prefetch
    def download(repo_id,revision,local_dir,ignore_patterns):
        local_dir=Path(local_dir)
        local_dir.mkdir(parents=True,exist_ok=True)
        for name in ['config.json','model.safetensors','spk_dict.pt']:
            if not any(fnmatch.fnmatch(name,pattern) for pattern in ignore_patterns):
                (local_dir/name).write_bytes(b'fixture')
    monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    prefetch(tmp_path)
    assert (tmp_path/'qwen-omni'/'spk_dict.pt').is_file()


def test_reject_remote_media():
    with pytest.raises(MediaError): probe('https://example.com/video.mp4')


@pytest.fixture
def real_clip(tmp_path):
    import shutil,subprocess
    binary=shutil.which('ffmpeg')
    if not binary: pytest.skip('ffmpeg not installed')
    path=tmp_path/'fixture.mp4'
    subprocess.run([binary,'-v','error','-f','lavfi','-i','color=c=blue:s=160x120:d=1','-f','lavfi','-i','sine=frequency=440:duration=1','-c:v','libx264','-c:a','aac','-shortest','-y',str(path)],check=True)
    return path


def test_real_media_decode_and_bounded_frames(real_clip,tmp_path):
    details=probe(real_clip)
    assert details.has_audio and details.has_video
    assert .9<details.duration<1.2
    audio=extract_audio(real_clip,tmp_path/'sound.wav')
    assert audio.exists()
    frames=sample_frames(real_clip,tmp_path/'frames',details.duration,count=3)
    assert len(frames)==3
    assert all(Path(f['path']).exists() for f in frames)
