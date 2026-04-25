import argparse
import json
from pathlib import Path
import uuid
from .backends import LocalBackend, ClaudeBackend, prefetch
from .pipeline import VideoPipeline, VideoOptions


def main(argv=None):
    parser = argparse.ArgumentParser(description='ModalEnough real video sentiment analysis')
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('download-models')
    download.add_argument('--text-only', action='store_true')
    predict = commands.add_parser('predict')
    predict.add_argument('video', type=Path)
    predict.add_argument('--budget', type=float, default=1.)
    predict.add_argument('--confidence', type=float, default=.85)
    predict.add_argument('--all', action='store_true')
    predict.add_argument('--backend', choices=['local','claude'], default='local')
    predict.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    predict.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'download-models':
        prefetch(include_multimodal=not args.text_only)
    else:
        backend = (LocalBackend if args.backend == 'local' else ClaudeBackend)(device=args.device)
        directory = Path('runs/video-cli')/uuid.uuid4().hex
        result = VideoPipeline(backend).run(args.video, directory, VideoOptions(budget=args.budget, confidence=args.confidence, strategy='all' if args.all else 'adaptive'))
        serialized = json.dumps(result, indent=2, allow_nan=False)+'\n'
        (directory/'result.json').write_text(serialized, encoding='utf-8')
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding='utf-8')
        print(serialized)


if __name__ == '__main__':
    main()
