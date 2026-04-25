"""Bounded local-media decoding. No model runs during metadata inspection."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess


class MediaError(ValueError):
    pass


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    has_audio: bool
    has_video: bool
    width: int = 0
    height: int = 0

    def to_dict(self):
        return asdict(self)


def binary(name):
    location = shutil.which(name)
    if not location:
        raise MediaError(f'{name} is required. Install FFmpeg and add its bin directory to PATH.')
    return location


def _local(path):
    path = Path(path)
    if not path.is_file():
        raise MediaError('A local video file is required.')
    return path.resolve()


def _run(command, timeout=45):
    try:
        return subprocess.run(command, capture_output=True, timeout=timeout, check=True)
    except subprocess.TimeoutExpired as exc:
        raise MediaError('Media decoding exceeded its time limit.') from exc
    except subprocess.CalledProcessError as exc:
        raise MediaError('The media could not be decoded. Try an MP4 with H.264 video and AAC audio.') from exc


def probe(path, max_duration=180.):
    path = _local(path)
    result = _run([binary('ffprobe'), '-v', 'error', '-protocol_whitelist', 'file,pipe', '-format_whitelist', 'mov,matroska,webm,avi', '-show_streams', '-show_format', '-of', 'json', str(path)], timeout=15)
    try:
        metadata = json.loads(result.stdout)
        duration = float(metadata['format']['duration'])
        video = next((s for s in metadata['streams'] if s['codec_type'] == 'video'), None)
        audio = any(s['codec_type'] == 'audio' for s in metadata['streams'])
        if not 0 < duration <= max_duration:
            raise MediaError(f'Use a video between 0 and {max_duration:g} seconds long.')
        if not video:
            raise MediaError('No video stream was found.')
        width, height = int(video.get('width', 0)), int(video.get('height', 0))
        if not 0 < width <= 7680 or not 0 < height <= 4320:
            raise MediaError('Video dimensions are invalid or exceed 8K.')
        return MediaInfo(duration, audio, True, width, height)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise MediaError('Missing or invalid media metadata.') from exc


def extract_audio(path, output):
    path, output = _local(path), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _run([binary('ffmpeg'), '-v', 'error', '-nostdin', '-protocol_whitelist', 'file,pipe', '-format_whitelist', 'mov,matroska,webm,avi', '-i', str(path), '-vn', '-t', '180', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', '-y', str(output)], timeout=60)
    return output


def sample_frames(path, directory, duration, count=4):
    path, directory = _local(path), Path(directory)
    if not 1 <= count <= 8 or not 0 < duration <= 180:
        raise MediaError('Invalid frame sampling request.')
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(count):
        timestamp = duration*(index+.5)/count
        output = directory/f'frame_{index:02d}.jpg'
        _run([binary('ffmpeg'), '-v', 'error', '-nostdin', '-ss', f'{timestamp:.4f}', '-protocol_whitelist', 'file,pipe', '-format_whitelist', 'mov,matroska,webm,avi', '-i', str(path), '-frames:v', '1', '-vf', 'scale=512:-2', '-q:v', '3', '-y', str(output)], timeout=20)
        if not output.is_file():
            raise MediaError('Could not extract a requested video frame.')
        frames.append({'path': str(output), 'timestamp': round(timestamp, 3)})
    return frames
