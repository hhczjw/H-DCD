"""Extract MOSEI features using the existing MOSI Data2Vec/OpenFace3 pipeline.

Cache each utterance for resuming; package features in manifest split order.
Audio keeps Data2Vec's full 6-second (96,000 samples) output sequence.
Vision samples 50 frames, matching MOSI split_vision_openface3.pkl.
"""
import argparse
import csv
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = '/media/zjw/951FB31A9E1EB7E0/dateSet/MSA-DataSets/CMU-MOSEI'
DEFAULT_OUT = str(ROOT / 'features')


def sample_id(row):
    return f"{row['video_id']}_{int(row['clip_id'])}"


def save_array(path, array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f'Invalid feature at {path}: {array.shape}')
    with path.with_suffix('.tmp').open('wb') as f:
        np.save(f, array, allow_pickle=False)
    os.replace(path.with_suffix('.tmp'), path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modality', choices=['audio', 'vision'], required=True)
    p.add_argument('--data-root', type=Path, default=Path(DEFAULT_DATA))
    p.add_argument('--output-dir', type=Path, default=Path(DEFAULT_OUT))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--audio-model', default=str(ROOT / 'pretrained/data2vec-audio-base'))
    p.add_argument('--openface-weights', default=str(ROOT / 'weights'))
    p.add_argument('--limit', type=int, default=0, help='smoke subset; never writes production output')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--allow-zero-fallback', action='store_true', help='explicitly allow failed samples as zeros; failures are logged')
    args = p.parse_args()
    with (args.data_root / 'label.csv').open() as f:
        rows = list(csv.DictReader(f))
    ids = [sample_id(r) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate sample IDs in label.csv')
    if set(r['mode'] for r in rows) != {'train', 'valid', 'test'}:
        raise ValueError('Unexpected split names')
    rows = rows[:args.limit] if args.limit else rows
    subdir, suffix = ('wav', '.wav') if args.modality == 'audio' else ('Raw', '.mp4')
    paths = {sample_id(r): args.data_root / subdir / r['video_id'] / (str(int(r['clip_id'])) + suffix) for r in rows}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    print(json.dumps({'samples': len(rows), 'missing_media': len(missing), 'examples': missing[:10]}, indent=2))
    if args.check_only:
        return
    if missing and not args.allow_zero_fallback:
        raise FileNotFoundError(f'{len(missing)} media files missing; first: {missing[0]}')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    cache = args.output_dir / 'mosei_cache' / args.modality
    cache.mkdir(parents=True, exist_ok=True)
    failures_path = cache / 'failures.json'
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else {}
    device = torch.device(args.device)
    if args.modality == 'audio':
        from transformers import Data2VecAudioModel, Wav2Vec2FeatureExtractor
        from extract_audio_data2vec import load_and_process_audio, extract_data2vec_features
        model = Data2VecAudioModel.from_pretrained(args.audio_model, local_files_only=True).to(device).eval()
        extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                            do_normalize=True, return_attention_mask=True)
        frames = (96000 - 400) // 320 + 1
        for sid, path in tqdm(paths.items(), desc='MOSEI Data2Vec'):
            dest = cache / (sid + '.npy')
            if dest.exists():
                continue
            try:
                values, mask = load_and_process_audio(str(path), extractor, max_length=96000)
                features = extract_data2vec_features(model, values, mask, 0, device)
            except Exception as exc:
                failures[sid] = str(exc)
                failures_path.write_text(json.dumps(failures, indent=2))
                if not args.allow_zero_fallback:
                    raise
                features = np.zeros((frames, model.config.hidden_size), dtype=np.float32)
            save_array(dest, features)
    else:
        from extract_video_features import run_openface
        pending = [str(path) for sid, path in paths.items() if not (cache / (sid + '.npy')).exists() and path.is_file()]
        options = SimpleNamespace(video_dir=str(args.data_root / 'Raw'), video_paths=pending,
                                  num_frames=50, openface_weights_dir=args.openface_weights,
                                  on_feature=lambda sid, feat: save_array(cache / (sid + '.npy'), feat))
        if pending:
            run_openface(options, device)
        for sid, path in paths.items():
            if not path.is_file():
                failures[sid] = 'missing video'
                save_array(cache / (sid + '.npy'), np.zeros((50, 18), dtype=np.float32))
        failures_path.write_text(json.dumps(failures, indent=2))
    if failures and not args.allow_zero_fallback:
        raise RuntimeError('Cached failures found; inspect failures.json before packaging')
    output = {}
    for split in ('train', 'valid', 'test'):
        split_ids = [sample_id(r) for r in rows if r['mode'] == split]
        if not split_ids:
            continue
        features = [np.load(cache / (sid + '.npy'), allow_pickle=False) for sid in split_ids]
        output[split] = {args.modality: np.stack(features), args.modality + '_ids': split_ids}
        print(split, output[split][args.modality].shape)
    filename = 'mosei_audio_data2vec_full.pkl' if args.modality == 'audio' else 'mosei_vision_openface3.pkl'
    if args.limit:
        filename = 'smoke_' + filename
    target = args.output_dir / filename
    with target.with_suffix('.tmp').open('wb') as f:
        pickle.dump(output, f, protocol=4)
    os.replace(target.with_suffix('.tmp'), target)
    print('Saved', target)


if __name__ == '__main__':
    main()
