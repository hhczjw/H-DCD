"""Audit MOSEI p0B through the same dataset/collation path as MOSI.

Use --full-check to load every split and collate a batch before training.
An ID-only audit does not certify training readiness.
"""
import argparse
import csv
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs import load_config

SPLITS = ('train', 'valid', 'test')
PIPELINE_KEYS = (
    'use_bert', 'bert_pretrained', 'bert_finetune', 'bert_learning_rate',
    'need_data_aligned', 'need_normalized', 'use_context',
    'use_bssm_gate', 'use_gcmn_gate', 'ism_full_frame', 'multi_task',
    'sub_loss_lambda', 'ism_depth', 'ism_d_state', 'ism_mixer_type',
    'd_model', 'num_layers', 'd_state', 'expand', 'headdim', 'ngroups',
    'rope_fraction', 'is_mimo', 'chunk_size', 'dropout', 'pool_type',
    'learning_rate', 'batch_size', 'epochs', 'early_stop', 'loss_type',
    'grad_clip', 'warmup_ratio', 'weight_decay', 'KeyEval',
    'secondary_metric', 'tertiary_metric', 'quaternary_metric', 'seq_lens',
)


def audit(config_path, full_check=False):
    args = load_config('MOSEI', str(config_path))
    for key in ('featurePath', 'feature_A', 'feature_V'):
        path = Path(getattr(args, key))
        setattr(args, key, str(path if path.is_absolute() else ROOT / path))
    report = {
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'source_experiment': 'MOSI_p0B_seed2024',
        'config_path': str(config_path),
        'splits': {}, 'missing': [], 'errors': [],
        'full_check_requested': full_check,
        'base_pickle_checked': False, 'manifest_checked': False,
        'full_loader_check_passed': False, 'ready_for_training': False,
    }
    log = (ROOT / 'logs/MSA_MOSI_20260918_233755.log').read_text()
    recorded, _ = json.JSONDecoder().raw_decode(log.split('Args: ', 1)[1])
    mismatches = [key for key in PIPELINE_KEYS
                  if getattr(args, key, None) != recorded.get(key)]
    report['mosi_pipeline_mismatches'] = mismatches
    if mismatches:
        report['errors'].append('MOSI p0B settings differ: ' + ', '.join(mismatches))
    # Check extracted features before loading the ~13 GB base pickle.
    for key in ('feature_A', 'feature_V'):
        if not Path(getattr(args, key)).is_file():
            report['missing'].append(getattr(args, key))
    if report['missing']:
        return report
    report['base_pickle_checked'] = True
    if not Path(args.featurePath).is_file():
        report['missing'].append(args.featurePath)
    manifest = Path(args.featurePath).parent.parent / 'label.csv'
    if not manifest.is_file():
        report['missing'].append(str(manifest))
        return report
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    report['manifest_checked'] = True
    ids = {split: [f"{r['video_id']}_{int(r['clip_id'])}"
                   for r in rows if r['mode'] == split] for split in SPLITS}
    for split in SPLITS:
        report['splits'][split] = {
            'samples': len(ids[split]), 'source': 'label.csv', 'loader_checked': False,
        }
        if not ids[split] or len(ids[split]) != len(set(ids[split])):
            report['errors'].append(f'{split}: empty split or duplicate manifest IDs')
    if set(r['mode'] for r in rows) != set(SPLITS):
        report['errors'].append('Unexpected manifest split names')
    if args.train_samples != len(ids['train']):
        report['errors'].append('Configured train_samples differs from manifest')
    if not full_check or report['missing'] or report['errors']:
        return report
    from dataset.data_loader import MMDataset, _collate_fn
    from torch.utils.data import DataLoader
    import numpy as np
    expected_dims = list(args.feature_dims)
    for split in SPLITS:
        try:
            ds = MMDataset(args, mode=split)
            normalized_ids = [str(sid).replace('$_$', '_') for sid in ds.ids]
            if len(ds) != len(ids[split]) or set(normalized_ids) != set(ids[split]):
                raise ValueError('Base pickle sample IDs differ from manifest')
            if [ds.audio.shape[-1], ds.vision.shape[-1]] != expected_dims[1:]:
                raise ValueError('Loaded audio/vision dimensions differ from p0B')
            if not np.isfinite(ds.labels['M']).all():
                raise ValueError('Non-finite regression labels')
            batch = next(iter(DataLoader(ds, batch_size=args.batch_size,
                                         num_workers=0, collate_fn=_collate_fn)))
            report['splits'][split].update(
                loader_checked=True,
                text_shape=list(ds.text.shape), audio_shape=list(ds.audio.shape),
                vision_shape=list(ds.vision.shape),
                batch_audio_shape=list(batch['audio'].shape),
            )
            del batch, ds
            gc.collect()
        except Exception as exc:
            report['errors'].append(f'{split}: {type(exc).__name__}: {exc}')
            break
    report['full_loader_check_passed'] = all(
        report['splits'][split]['loader_checked'] for split in SPLITS)
    report['ready_for_training'] = report['full_loader_check_passed'] and not report['errors']
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-path', type=Path, default=ROOT / 'configs/mosei_p0B.json')
    p.add_argument('--full-check', action='store_true')
    p.add_argument('--report-path', type=Path)
    cli = p.parse_args()
    report = audit(cli.config_path, cli.full_check)
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if cli.report_path:
        cli.report_path.write_text(output + '\n')
    if cli.full_check and not report['ready_for_training']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
