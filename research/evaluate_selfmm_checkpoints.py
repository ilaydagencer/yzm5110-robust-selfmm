"""Compare two frozen Self-MM checkpoints under four input conditions."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import sentencepiece  # Must precede torch/MMSA on Windows.
import torch
import MMSA
from torch.utils.data import DataLoader
from MMSA.config import get_config_regression
from MMSA.data_loader import MMDataset
from MMSA.models import AMIO
from MMSA.utils import MetricsTop, setup_seed

CONDITIONS = {'K0': (False, False), 'K1': (True, False),
              'K2': (False, True), 'K3': (True, True)}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def mask_inputs(audio, vision, condition):
    drop_a, drop_v = CONDITIONS[condition]
    return (torch.zeros_like(audio) if drop_a else audio,
            torch.zeros_like(vision) if drop_v else vision)


def write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--split', choices=['valid', 'test'], default='test')
    parser.add_argument('--repo-dir', type=Path, default=Path('C:/Users/ilayda/YZM5110/MMSA'))
    parser.add_argument('--output-dir', type=Path)
    cli = parser.parse_args()
    repo = cli.repo_dir.resolve()
    project = repo.parent
    output = cli.output_dir or project / 'runs' / ('selfmm_checkpoint_comparison_' + cli.split + '_seed1111')
    specs = {
        'baseline': (project / 'local_configs/selfmm_mosi_batch4.json',
                     project / 'runs/selfmm_seed1111/saved_models/self_mm-mosi.pth'),
        'dropout': (project / 'local_configs/selfmm_mosi_dropout_a20_v20_batch4_seed1111.json',
                    project / 'runs/selfmm_dropout_a20_v20_seed1111/saved_models/self_mm-mosi.pth')}
    require(Path(MMSA.__file__).resolve().is_relative_to(repo / 'src'), 'Yanlis MMSA kurulumu')
    require(not output.exists(), f'Cikti klasoru zaten var; uzerine yazilmaz: {output}')
    configs, arguments = {}, {}
    for name, (config_path, checkpoint) in specs.items():
        require(config_path.is_file() and checkpoint.is_file(), f'Dosya eksik: {name}')
        configs[name] = json.loads(config_path.read_text(encoding='utf-8'))
        args = get_config_regression('self_mm', 'mosi', str(config_path))
        require(args.batch_size == 4 and not args.need_data_aligned
                and not args.get('data_missing', False), 'Beklenmeyen deney ayarlari')
        require(Path(args.featurePath).is_file(), 'Veri dosyasi bulunamadi')
        args.update(train_mode='regression', device=torch.device('cuda:0'),
                    custom_feature=None, feature_T=None, feature_A=None, feature_V=None,
                    data_missing=False)
        arguments[name] = args
        print(name.upper(), 'CHECKPOINT:', checkpoint)
    require(configs['baseline'] == configs['dropout'], 'Iki egitim konfigurasyonu farkli')
    a, v = torch.ones(4, 3, 5), torch.ones(4, 2, 20)
    for condition, (da, dv) in CONDITIONS.items():
        am, vm = mask_inputs(a, v, condition)
        require(torch.equal(am, torch.zeros_like(a) if da else a)
                and torch.equal(vm, torch.zeros_like(v) if dv else v), 'Maske testi basarisiz')
        require(bool((a == 1).all()) and bool((v == 1).all()), 'Asil girdiler degisti')
    print('SPLIT:', cli.split, 'BATCH: 4; SEED: 1111; CUDA:', torch.cuda.is_available())
    print('CIKTI:', output)
    if cli.check:
        print('KONTROL_BASARILI: model/veri yuklenmedi, egitim ve dosya yazma yapilmadi')
        return
    require(torch.cuda.is_available(), 'CUDA bulunamadi')
    setup_seed(1111)
    torch.cuda.set_device(0)
    files = [p for pair in specs.values() for p in pair]
    hashes = {str(p): digest(p) for p in files}
    # Only requested split is instantiated; ordering is fixed for all eight passes.
    dataset = MMDataset(copy.deepcopy(arguments['baseline']), mode=cli.split)
    require(len(dataset) == {'valid': 229, 'test': 686}[cli.split], 'Beklenmeyen ornek sayisi')
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
    output.mkdir(parents=True, exist_ok=False)
    metric_fn = MetricsTop('regression').getMetics('mosi')
    rows, predictions = [], []
    reference_indices, reference_labels = None, None
    for name, (_, checkpoint) in specs.items():
        setup_seed(1111)
        model = AMIO(copy.deepcopy(arguments[name]))
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        model.to('cuda:0').eval()
        model.requires_grad_(False)
        for condition in CONDITIONS:
            pred_parts, label_parts, indices = [], [], []
            with torch.inference_mode():
                for batch in loader:
                    require(not model.training and not model.Model.training, 'Model eval modunda degil')
                    text = batch['text'].to('cuda:0')
                    audio, vision = mask_inputs(batch['audio'].to('cuda:0'),
                                                batch['vision'].to('cuda:0'), condition)
                    # Text and original sequence lengths are never masked.
                    pred = model(text, (audio, batch['audio_lengths']),
                                 (vision, batch['vision_lengths']))['M'].reshape(-1).cpu()
                    labels = batch['labels']['M'].reshape(-1).cpu()
                    require(bool(torch.isfinite(pred).all()) and bool(torch.isfinite(labels).all()),
                            'Gecersiz tahmin/etiket')
                    pred_parts.append(pred)
                    label_parts.append(labels)
                    indices.extend(batch['index'].tolist())
            pred, labels = torch.cat(pred_parts), torch.cat(label_parts)
            if reference_indices is None:
                reference_indices, reference_labels = indices, labels.clone()
            require(indices == reference_indices and torch.equal(labels, reference_labels),
                    'Modeller/kosullar arasinda ornek sirasi veya etiket farkli')
            metrics = {k: float(value) for k, value in metric_fn(pred, labels).items()}
            # Global sample MAE, independent of last batch size; not trainer batch-mean Loss.
            metrics['MAE_exact'] = float((pred-labels).abs().mean())
            row = dict(model=name, split=cli.split, condition=condition, n=len(labels), **metrics)
            rows.append(row)
            predictions.extend(dict(model=name, condition=condition, index=i,
                                    label=float(y), prediction=float(p))
                               for i, y, p in zip(indices, labels, pred))
            print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    require(all(digest(Path(p)) == h for p, h in hashes.items()), 'Config/checkpoint degisti')
    metric_keys = list(metrics)
    comparisons = []
    for condition in CONDITIONS:
        b, d = [next(r for r in rows if r['model'] == name and r['condition'] == condition)
                for name in ('baseline', 'dropout')]
        comparisons.append(dict(condition=condition, **{k+'_delta': d[k]-b[k] for k in metric_keys}))
    degradation = []
    for name in specs:
        clean = next(r for r in rows if r['model'] == name and r['condition'] == 'K0')
        for row in [r for r in rows if r['model'] == name]:
            degradation.append(dict(model=name, condition=row['condition'],
                                    MAE_increase=row['MAE_exact']-clean['MAE_exact'],
                                    Non0_acc_drop=clean['Non0_acc_2']-row['Non0_acc_2']))
    write_csv(output / 'metrics.csv', rows)
    write_csv(output / 'comparison.csv', comparisons)
    write_csv(output / 'degradation.csv', degradation)
    write_csv(output / 'predictions.csv', predictions)
    (output / 'manifest.json').write_text(json.dumps(dict(
        status='completed', split=cli.split, seed=1111, n=len(dataset), batch_size=4,
        conditions=CONDITIONS, protected_hashes=hashes, protected_files_unchanged=True,
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
        note='Raw metrics, not multiplied by 100. Input zeroing only; no fusion masking.'), indent=2), encoding='utf-8')
    print('DEGERLENDIRME_TAMAMLANDI:', output)


if __name__ == '__main__':
    main()
