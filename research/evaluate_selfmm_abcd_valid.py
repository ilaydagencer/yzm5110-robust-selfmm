"""Frozen-checkpoint A/B/C/D validation; fixed tau, no training or test access.

A/C use the baseline checkpoint; B/D use the input-dropout checkpoint.
C/D apply 3*softmax(-deviation/tau) to encoder representations before fusion.
The unmodified forward is used twice for unweighted/weighted inference; an
additional all-one control pass checks identity on every validation batch.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

import sentencepiece  # Windows: MUST precede torch and MMSA.
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
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def weights_from_outputs(outputs, tau):
    require(math.isfinite(tau) and tau > 0, 'Tau pozitif ve sonlu olmali')
    y = torch.stack([outputs[k].reshape(-1).detach() for k in ('T', 'A', 'V')], dim=1)
    require(bool(torch.isfinite(y).all()), 'Tek modlu tahmin gecersiz')
    deviations = (y - (y.sum(1, keepdim=True)-y)/2).abs()
    w = 3 * torch.softmax(-deviations/tau, dim=1)
    require(bool(torch.isfinite(w).all()) and bool((w >= 0).all())
            and torch.allclose(w.sum(1), torch.full_like(w[:, 0], 3), atol=1e-5),
            'Agirlik sonluluk/toplam kontrolu basarisiz')
    return w, deviations


@contextmanager
def fusion_weights(layer, weights, dims):
    require(layer.in_features == sum(dims), 'Fusion boyutlari uyumsuz')

    def hook(module, inputs):
        x = inputs[0]
        require(x.ndim == 2 and weights.shape == (x.shape[0], 3), 'Hook batch/boyut hatasi')
        chunks = torch.split(x, dims, dim=-1)
        return (torch.cat([h * weights[:, i:i+1] for i, h in enumerate(chunks)], dim=-1),) + inputs[1:]

    handle = layer.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def masked(audio, vision, condition):
    a, v = CONDITIONS[condition]
    return torch.zeros_like(audio) if a else audio, torch.zeros_like(vision) if v else vision


def self_check():
    # fork_rng ensures these CPU tests do not disturb model RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1111)
        layer = torch.nn.Linear(6, 2)
        x = torch.randn(4, 6)
        original = layer(x)
        hooks = len(layer._forward_pre_hooks)
        with fusion_weights(layer, torch.ones(4, 3), (1, 2, 3)):
            require(torch.equal(original, layer(x)), 'Esit agirlik identity testi basarisiz')
        require(len(layer._forward_pre_hooks) == hooks, 'Hook kaldirilmadi')
        try:
            with fusion_weights(layer, torch.ones(4, 3), (1, 2, 3)):
                raise ValueError('intentional test')
        except ValueError:
            pass
        require(len(layer._forward_pre_hooks) == hooks, 'Hata sonrasi hook kaldirilmadi')
        with fusion_weights(layer, torch.tensor([[1., 2., 0.]]).repeat(4, 1), (1, 2, 3)):
            expected = torch.cat([x[:, :1], x[:, 1:3]*2, x[:, 3:]*0], dim=1)
            require(torch.equal(layer(x), torch.nn.functional.linear(expected, layer.weight, layer.bias)),
                    'Modalite sirasi/agirlik uygulama testi basarisiz')
        equal = {k: torch.ones(4, 1) for k in ('T', 'A', 'V')}
        w, _ = weights_from_outputs(equal, 1)
        require(torch.equal(w, torch.ones(4, 3)), 'Esit tahmin esit agirlik testi basarisiz')
        outlier = dict(T=torch.tensor([[0.]]), A=torch.tensor([[0.]]), V=torch.tensor([[2.]]))
        w, _ = weights_from_outputs(outlier, 1)
        require(w[0, 2] < w[0, 0] and w[0, 0] == w[0, 1], 'Sapma yonu testi basarisiz')
        a, v = torch.ones(4, 2, 5), torch.ones(4, 3, 20)
        for condition, (da, dv) in CONDITIONS.items():
            am, vm = masked(a, v, condition)
            require(torch.equal(am, torch.zeros_like(a) if da else a)
                    and torch.equal(vm, torch.zeros_like(v) if dv else v), 'Maskeleme testi basarisiz')
        require(bool((a == 1).all()) and bool((v == 1).all()), 'Asil girdi degisti')
    print('BIRIM_KONTROLLERI_BASARILI: agirlik, modalite sirasi, identity, hook temizligi, maske')


def write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--repo-dir', type=Path, default=Path('C:/Users/ilayda/YZM5110/MMSA'))
    parser.add_argument('--output-dir', type=Path)
    cli = parser.parse_args()
    require(math.isfinite(cli.tau) and cli.tau > 0, 'Tau pozitif ve sonlu olmali')
    repo = cli.repo_dir.resolve()
    project = repo.parent
    tau_name = format(cli.tau, '.12g').replace('.', 'p')
    output = cli.output_dir or project / 'runs' / f'selfmm_abcd_valid_seed1111_tau{tau_name}'
    require(not output.exists(), f'Cikti klasoru zaten var; uzerine yazilmaz: {output}')
    require(Path(MMSA.__file__).resolve().is_relative_to(repo / 'src'), 'Yanlis MMSA kurulumu')
    specs = {
        'baseline': (project / 'local_configs/selfmm_mosi_batch4.json',
                     project / 'runs/selfmm_seed1111/saved_models/self_mm-mosi.pth', 'A', 'C'),
        'dropout': (project / 'local_configs/selfmm_mosi_dropout_a20_v20_batch4_seed1111.json',
                    project / 'runs/selfmm_dropout_a20_v20_seed1111/saved_models/self_mm-mosi.pth', 'B', 'D')}
    configs, arguments = {}, {}
    for name, (config, checkpoint, plain, weighted) in specs.items():
        require(config.is_file() and checkpoint.is_file(), f'Config/checkpoint eksik: {name}')
        configs[name] = json.loads(config.read_text(encoding='utf-8'))
        args = get_config_regression('self_mm', 'mosi', str(config))
        require(args.batch_size == 4 and not args.need_data_aligned
                and not args.need_model_aligned and not args.get('data_missing', False),
                'Beklenmeyen config: batch/alignment/data_missing')
        require(Path(args.featurePath).is_file(), f'Veri bulunamadi: {args.featurePath}')
        require(args.post_fusion_dropout == 0, 'Bu protokol sifir fusion dropout bekliyor')
        args.update(train_mode='regression', device=torch.device('cuda:0'), custom_feature=None,
                    feature_T=None, feature_A=None, feature_V=None, data_missing=False)
        arguments[name] = args
        print(f'{plain}/{weighted} CHECKPOINT: {checkpoint}')
    require(configs['baseline'] == configs['dropout'], 'Egitim configleri eslesmiyor')
    self_check()
    print('SPLIT: valid; BATCH: 4; SEED: 1111; TAU:', cli.tau)
    print('CUDA:', torch.cuda.is_available(), 'CIKTI:', output)
    if cli.check:
        print('KONTROL_BASARILI: model/veri yuklenmedi, egitim ve dosya yazma yapilmadi')
        return
    require(torch.cuda.is_available(), 'CUDA bulunamadi')
    setup_seed(1111)
    torch.cuda.set_device(0)
    protected = [Path(__file__).resolve(),
                 repo / 'src/MMSA/models/multiTask/SELF_MM.py',
                 repo / 'src/MMSA/config/config_regression.json']
    protected += [p for config, checkpoint, _, _ in specs.values() for p in (config, checkpoint)]
    hashes = {str(p): digest(p) for p in protected}
    dataset = MMDataset(copy.deepcopy(arguments['baseline']), mode='valid')
    require(len(dataset) == 229, f'229 validation ornegi bekleniyor: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', split='valid', seed=1111, batch_size=4, n=229,
                    tau=cli.tau, weighting='3*softmax(-abs(y_m-mean(y_other))/tau)',
                    condition_masks=CONDITIONS, checkpoint_paths={k: str(v[1]) for k, v in specs.items()},
                    protected_hashes=hashes, torch_version=torch.__version__,
                    gpu=torch.cuda.get_device_name(0), inference_only=True,
                    selection='Fixed-tau diagnostic; no hyperparameter selection or test evaluation.',
                    note='Uniform identity pass is a control, not a latency benchmark.')

    def save_manifest():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    save_manifest()
    rows, sample_rows, weight_rows, controls = [], [], [], []
    metrics_fn = MetricsTop('regression').getMetics('mosi')
    reference_labels = reference_indices = None
    try:
        for name, (_, checkpoint, plain, weighted) in specs.items():
            setup_seed(1111)
            args = copy.deepcopy(arguments[name])
            model = AMIO(args)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            del state
            model.to(args.device).eval().requires_grad_(False)
            dims = (args.text_out, args.audio_out, args.video_out)
            layer = model.Model.post_fusion_layer_1
            require(layer.in_features == sum(dims), 'Gercek fusion boyutlari uyusmuyor')
            for condition in CONDITIONS:
                pp, wp, ys, ws, indices = [], [], [], [], []
                max_identity = 0.0
                hooks_before = len(layer._forward_pre_hooks)
                with torch.inference_mode():
                    for batch in loader:
                        require(all(not m.training for m in model.modules()), 'Model eval modunda degil')
                        text = batch['text'].to(args.device)
                        audio, vision = masked(batch['audio'].to(args.device), batch['vision'].to(args.device), condition)

                        def forward():
                            return model(text, (audio, batch['audio_lengths']), (vision, batch['vision_lengths']))

                        base = forward()
                        weights, deviations = weights_from_outputs(base, cli.tau)
                        with fusion_weights(layer, torch.ones_like(weights), dims):
                            uniform = forward()
                        for key in ('M', 'T', 'A', 'V'):
                            err = float((uniform[key]-base[key]).abs().max())
                            max_identity = max(max_identity, err)
                            require(torch.allclose(uniform[key], base[key], atol=1e-6, rtol=1e-6),
                                    f'Esit agirlik kontrolu basarisiz: {name}/{condition}/{key}')
                        with fusion_weights(layer, weights, dims):
                            changed = forward()
                        require(len(layer._forward_pre_hooks) == hooks_before, 'Hook sizintisi')
                        for key in ('T', 'A', 'V'):
                            require(torch.allclose(changed[key], base[key], atol=1e-6, rtol=1e-6),
                                    'Agirliklandirma tek modlu basligi degistirdi')
                        p, q = base['M'].reshape(-1).cpu(), changed['M'].reshape(-1).cpu()
                        labels = batch['labels']['M'].reshape(-1).cpu()
                        require(all(bool(torch.isfinite(t).all()) for t in (p, q, labels)), 'Gecersiz sayi')
                        batch_indices = batch['index'].tolist()
                        pp.append(p); wp.append(q); ys.append(labels); ws.append(weights.cpu())
                        indices.extend(batch_indices)
                        for j, index in enumerate(batch_indices):
                            sample_rows.append(dict(checkpoint=name, condition=condition, index=index,
                                label=float(labels[j]), plain_method=plain, weighted_method=weighted,
                                plain_prediction=float(p[j]), weighted_prediction=float(q[j]),
                                y_text=float(base['T'][j].item()), y_audio=float(base['A'][j].item()),
                                y_vision=float(base['V'][j].item()), w_text=float(weights[j, 0]),
                                w_audio=float(weights[j, 1]), w_vision=float(weights[j, 2]),
                                deviation_text=float(deviations[j, 0]), deviation_audio=float(deviations[j, 1]),
                                deviation_vision=float(deviations[j, 2])))
                p, q, labels, weights = torch.cat(pp), torch.cat(wp), torch.cat(ys), torch.cat(ws)
                if reference_indices is None:
                    reference_indices, reference_labels = indices, labels.clone()
                require(indices == reference_indices and torch.equal(labels, reference_labels), 'Ornek/etiket eslesmiyor')
                for method, pred in ((plain, p), (weighted, q)):
                    metrics = {k: float(v) for k, v in metrics_fn(pred, labels).items()}
                    row = dict(method=method, checkpoint=name, split='valid', condition=condition,
                               tau=cli.tau, n=len(labels), **metrics, MAE_exact=float((pred-labels).abs().mean()))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                entry = dict(method=weighted, condition=condition, tau=cli.tau, n=len(labels))
                for i, modality in enumerate(('text', 'audio', 'vision')):
                    entry.update({f'w_{modality}_mean': float(weights[:, i].mean()),
                                  f'w_{modality}_std': float(weights[:, i].std(unbiased=False)),
                                  f'w_{modality}_min': float(weights[:, i].min()),
                                  f'w_{modality}_max': float(weights[:, i].max())})
                weight_rows.append(entry)
                controls.append(dict(checkpoint=name, condition=condition, n=len(labels),
                                     max_uniform_error=max_identity, hooks_restored=True))
            del model
            torch.cuda.empty_cache()
        comparisons = []
        for condition in CONDITIONS:
            lookup = {r['method']: r for r in rows if r['condition'] == condition}
            for first, second in (('A', 'B'), ('A', 'C'), ('B', 'D'), ('A', 'D')):
                comparisons.append(dict(condition=condition, comparison=f'{second}-minus-{first}',
                    Non0_acc_delta=lookup[second]['Non0_acc_2']-lookup[first]['Non0_acc_2'],
                    Non0_F1_delta=lookup[second]['Non0_F1_score']-lookup[first]['Non0_F1_score'],
                    MAE_exact_delta=lookup[second]['MAE_exact']-lookup[first]['MAE_exact'],
                    Corr_delta=lookup[second]['Corr']-lookup[first]['Corr']))
        write_csv(output / 'metrics.csv', rows)
        write_csv(output / 'comparisons.csv', comparisons)
        write_csv(output / 'predictions_and_weights.csv', sample_rows)
        write_csv(output / 'weight_summary.csv', weight_rows)
        write_csv(output / 'controls.csv', controls)
        manifest['status'] = 'completed'
    except BaseException:
        manifest['status'] = 'failed_or_interrupted'
        raise
    finally:
        manifest['protected_files_unchanged'] = all(digest(Path(p)) == h for p, h in hashes.items())
        save_manifest()
    require(manifest['protected_files_unchanged'], 'Korunan dosya degisti')
    print('ABCD_VALIDATION_TAMAMLANDI:', output)


if __name__ == '__main__':
    main()
