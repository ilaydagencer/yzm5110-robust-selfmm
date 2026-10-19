"""Replay seed-1111 K0 test inference from explicit assets, without old run manifests.

This is a frozen checkpoint replay, not training or a latency benchmark.
--check hashes assets and exercises CPU controls; it loads no model or dataset.
Both modes are offline. Existing output directories are never overwritten.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
# Set before importing transformers/MMSA. Only explicitly supplied local BERT assets.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

METRICS = ('MAE', 'Non0_acc_2', 'Non0_F1_score', 'Corr')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def effective_config(config):
    """Mirror MMSA's self_mm/mosi merge; location is provided separately by CLI."""
    common = config['self_mm']['commonParams']
    require(common['need_data_aligned'] is False, 'Unaligned config gerekli')
    values = dict(model_name='self_mm', dataset_name='mosi')
    values.update(config['datasetCommonParams']['mosi']['unaligned'])
    values.update(common)
    values.update(config['self_mm']['datasetParams']['mosi'])
    values.pop('featurePath')
    return values


def config_digest(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_new_output(output, repo, package, bert, checkpoints):
    require(not os.path.lexists(output), f'Cikti zaten var; uzerine yazilmaz: {output}')
    require(output.parent.is_dir(), f'Cikti ust klasoru bulunamadi: {output.parent}')
    target = output.resolve()
    for protected in (repo, package, bert, *(p.parent for p in checkpoints.values())):
        require(not target.is_relative_to(protected.resolve()), 'Cikti repo/asset klasorlerinin disinda olmali')


def check_bert(directory):
    require(directory.is_dir(), f'Yerel BERT klasoru bulunamadi: {directory}')
    for name in ('config.json', 'vocab.txt'):
        require((directory / name).is_file(), f'BERT dosyasi eksik: {name}')
    require(any((directory / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin')),
            'BERT agirlik dosyasi eksik: model.safetensors veya pytorch_model.bin')
    values = load_json(directory / 'config.json')
    expected = dict(model_type='bert', hidden_size=768, num_hidden_layers=12,
                    num_attention_heads=12, intermediate_size=3072, vocab_size=30522,
                    type_vocab_size=2, max_position_embeddings=512, layer_norm_eps=1e-12,
                    hidden_act='gelu', pad_token_id=0)
    require(all(values.get(k) == v for k, v in expected.items()), 'bert-base-uncased mimarisi uyusmuyor')
    vocab = (directory / 'vocab.txt').read_text(encoding='utf-8').splitlines()
    require(len(vocab) == 30522 and vocab[0] == '[PAD]' and vocab[101:103] == ['[CLS]', '[SEP]'],
            'BERT vocab uyusmuyor')
    return [p for p in directory.iterdir() if p.is_file()]


def read_reference(package):
    path = package / 'evaluation_test/per_seed_metrics.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['training_seed'] == '1111' and r['condition'] == 'K0']
    require(len(rows) == 4 and {r['method'] for r in rows} == set('ABCD'), 'Paket K0 referansi eksik/tekrar')
    result = {}
    for row in rows:
        require(int(row['n']) == 686 and int(row['non0_n']) == 656, 'Referans ornek sayisi uyumsuz')
        values = {k: float(row[k]) for k in METRICS}
        require(all(math.isfinite(v) for v in values.values()), 'Referans metrik sonlu degil')
        result[row['method']] = values
    return result


def prepare(cli):
    global np, torch, MMDataset, AMIO, setup_seed, DataLoader, helper
    repo = cli.repo_dir.resolve()
    package = (cli.package_dir or repo / 'research/results_package_v1').resolve()
    config = cli.config.resolve()
    dependencies = cli.dependencies.resolve()
    data = cli.data_file.resolve()
    bert = cli.bert_dir.resolve()
    checkpoints = dict(baseline=cli.baseline_checkpoint.resolve(), dropout=cli.dropout_checkpoint.resolve())
    output = cli.output_dir.absolute()
    check_new_output(output, repo, package, bert, checkpoints)
    frozen = load_json(dependencies)
    require(frozen['schema_version'] == 1 and frozen['training_seed'] == 1111
            and frozen['condition'] == 'K0' and frozen['split'] == 'test'
            and frozen['n'] == 686 and frozen['non0_n'] == 656 and frozen['batch_size'] == 4
            and frozen['tau'] == 1 and frozen['metric_absolute_tolerance'] == 1e-5,
            'Dondurulmus K0 protokolu uyumsuz')
    settings = effective_config(load_json(config))
    require(config_digest(settings) == frozen['effective_config_sha256'], 'Model/config ayarlari referanstan farkli')
    # Check source before importing model code. Git CRLF/LF conversion is harmless.
    sources = []
    for relative, expected in frozen['source_sha256_lf'].items():
        p = (repo / relative).resolve()
        require(p.is_relative_to(repo) and p.is_file(), f'Kaynak dosyasi eksik: {relative}')
        require(source_digest(p) == expected, f'Kaynak kodu referanstan farkli: {relative}')
        sources.append(p)
    verifier_path = repo / 'research/verify_results_package.py'
    verifier = load_module(verifier_path, 'portable_package_verifier')
    verifier.verify(package)
    provenance = load_json(package / 'provenance.json')
    require(provenance['data_sha256'] == frozen['data_sha256']
            and provenance['checkpoint_sha256'] == frozen['checkpoint_sha256'], 'Paket/asset kimlikleri uyumsuz')
    print('ASSET_HASH_KONTROLU: veri ve iki checkpoint bayt olarak okunuyor; model/veri nesnesi kurulmaz', flush=True)
    require(data.is_file() and digest(data) == frozen['data_sha256'], 'Veri eksik veya SHA256 farkli')
    for name, p in checkpoints.items():
        require(p.is_file() and digest(p) == frozen['checkpoint_sha256'][name], f'Checkpoint eksik/farkli: {name}')
    bert_files = check_bert(bert)
    # Ordering is mandatory on this Windows environment.
    import sentencepiece
    import numpy as np
    import torch
    import transformers
    import MMSA
    from MMSA.config import get_config_regression
    from MMSA.data_loader import MMDataset
    from MMSA.models import AMIO
    from MMSA.utils import setup_seed
    from torch.utils.data import DataLoader
    require(Path(MMSA.__file__).resolve() == (repo / 'src/MMSA/__init__.py').resolve(), 'Yanlis yerel MMSA kurulumu')
    versions = dict(torch=torch.__version__, numpy=np.__version__, transformers=transformers.__version__)
    require(versions == frozen['versions'], f'Kutuphane surumleri farkli: {versions}')
    args = get_config_regression('self_mm', 'mosi', str(config))
    imported_settings = dict(args)
    imported_settings.pop('featurePath')
    require(imported_settings == settings, 'MMSA config birlestirmesi beklenenden farkli')
    args.update(featurePath=str(data), pretrained=str(bert), train_mode='regression',
                device=torch.device('cuda:0'), custom_feature=None, feature_T=None,
                feature_A=None, feature_V=None, data_missing=False)
    helper = load_module(repo / 'research/evaluate_selfmm_abcd_valid.py', 'portable_weighting_helper')
    helper.self_check()
    protected = set(sources + bert_files + [config, dependencies, data, *checkpoints.values(),
                    Path(__file__).resolve(), verifier_path])
    protected.update(p for p in package.rglob('*') if p.is_file())
    # Protect the original personal framework config too, although this runner never uses it.
    protected.add(repo / 'src/MMSA/config/config_regression.json')
    hashes = {p: digest(p) for p in sorted(protected)}
    plan = dict(repo=repo, package=package, output=output.resolve(), data=data, bert=bert,
                checkpoints=checkpoints, config=config, dependencies=dependencies, frozen=frozen,
                args=args, settings=settings, reference=read_reference(package), hashes=hashes,
                versions=versions, cuda_available=torch.cuda.is_available())
    print('BAGIMLILIKLER_DOGRULANDI: config, kaynak kodu, paket, veri, checkpoint, yerel BERT')
    print('ESKI_RUN_MANIFESTI_GEREKLI: False; SEED: 1111; SPLIT: test; KOSUL: K0; BATCH: 4; TAU: 1')
    print('CUDA:', plan['cuda_available'], 'CIKTI:', plan['output'])
    return plan


def prediction_metrics(pred, truth):
    from sklearn.metrics import accuracy_score, f1_score
    p, y = pred.numpy(), truth.numpy()
    mask = y != 0
    values = dict(MAE=float((pred - truth).abs().mean()),
                  Non0_acc_2=float(accuracy_score(y[mask] > 0, p[mask] > 0)),
                  Non0_F1_score=float(f1_score(y[mask] > 0, p[mask] > 0, average='weighted')),
                  Corr=float(np.corrcoef(p, y)[0, 1]))
    require(all(math.isfinite(v) for v in values.values()), 'Sonlu olmayan metrik')
    return values


def metric_checks(method, measured, reference, tolerance):
    return [dict(method=method, metric=k, reference=reference[k], current=measured[k],
                 absolute_error=abs(measured[k] - reference[k]), tolerance=tolerance,
                 passed=math.isfinite(measured[k]) and abs(measured[k] - reference[k]) <= tolerance)
            for k in METRICS]


def write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest_inputs(hashes, output):
    # Relocatable descriptive paths. Replays use CLI assets, never these old paths.
    entries = []
    for path, sha in hashes.items():
        try:
            location = os.path.relpath(path, output).replace('\\', '/')
        except ValueError:  # Windows: a different drive cannot have a relative path.
            location = str(path)
        entries.append(dict(path=location, sha256=sha))
    return entries


def unchanged(hashes):
    return [str(p) for p, expected in hashes.items() if not p.is_file() or digest(p) != expected]


def evaluate(plan):
    require(plan['cuda_available'], 'Gercek model kontrolu icin CUDA gerekli; --check CPU ile yapilabilir')
    output = plan['output']
    check_new_output(output, plan['repo'], plan['package'], plan['bert'], plan['checkpoints'])
    require(not unchanged(plan['hashes']), 'Kontrolden sonra bir girdi degisti')
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    manifest = dict(status='running', schema_version=1, training_seed=1111, split='test', condition='K0',
                    n=686, non0_n=656, batch_size=4, tau=1, inference_only=True, training_performed=False,
                    historical_run_manifests_required=False, offline=True, versions=plan['versions'],
                    python_version=platform.python_version(), config_settings=plan['settings'],
                    inputs=manifest_inputs(plan['hashes'], output),
                    input_path_base='directory containing this manifest; different-drive paths may be absolute',
                    reference_scope='16 aggregate metric comparisons; no historical per-sample predictions included',
                    test_history='Test results were already observed. Fixed replay only; no tuning or seed selection.',
                    timing_scope='total replay including controls, loading and hashing; not a latency benchmark',
                    created_at_utc=datetime.now(timezone.utc).isoformat())
    def save():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save()
    rows, samples, controls, checks = [], [], [], []
    try:
        torch.cuda.set_device(0)
        manifest.update(gpu=torch.cuda.get_device_name(0), dtype='float32', autocast=False)
        dataset_args = copy.deepcopy(plan['args'])
        dataset = MMDataset(dataset_args, mode='test')
        require(len(dataset) == 686 and dataset_args.feature_dims == [768, 5, 20], 'Veri boyutlari uyumsuz')
        require(int(np.count_nonzero(dataset.labels['M'])) == 656, 'Non0 etiket sayisi uyumsuz')
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
        shared_fingerprint = None
        for name, plain, weighted in (('baseline', 'A', 'C'), ('dropout', 'B', 'D')):
            print(f'MODEL_CALISIYOR: {plain}/{weighted}; {name}; 686 test ornegi', flush=True)
            setup_seed(1111)
            args = copy.deepcopy(plan['args'])
            model = AMIO(args)
            state = torch.load(plan['checkpoints'][name], map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            del state
            model.to(args.device).eval().requires_grad_(False)
            layer = model.Model.post_fusion_layer_1
            dims = (args.text_out, args.audio_out, args.video_out)
            hooks_before = len(layer._forward_pre_hooks)
            fingerprint = hashlib.sha256()
            predictions, weighted_predictions, labels, indices = [], [], [], []
            max_identity = 0.0
            with torch.inference_mode():
                for batch in loader:
                    require(all(not m.training for m in model.modules())
                            and all(not p.requires_grad for p in model.parameters()), 'Model eval/dondurulmus degil')
                    tensors = {k: batch[k] for k in ('text', 'audio', 'vision', 'audio_lengths', 'vision_lengths')}
                    tensors['label'] = batch['labels']['M']
                    for key, value in tensors.items():
                        require(bool(torch.isfinite(value).all()), f'Sonlu olmayan girdi: {key}')
                        fingerprint.update(key.encode())
                        fingerprint.update(value.numpy().tobytes())
                    text, audio, vision = (batch[k].to(args.device) for k in ('text', 'audio', 'vision'))
                    audio_lens, vision_lens = batch['audio_lengths'], batch['vision_lengths']
                    inputs = (text, audio, vision, audio_lens, vision_lens)
                    saved = tuple(t.clone() for t in inputs)
                    def forward():
                        return model(text, (audio, audio_lens), (vision, vision_lens))
                    base = forward()
                    weights, _ = helper.weights_from_outputs(base, 1.0)
                    with helper.fusion_weights(layer, torch.ones_like(weights), dims):
                        uniform = forward()
                    for key in ('M', 'T', 'A', 'V'):
                        error = float((base[key] - uniform[key]).abs().max())
                        max_identity = max(max_identity, error)
                        require(torch.allclose(base[key], uniform[key], atol=1e-6, rtol=1e-6), 'Identity kontrolu basarisiz')
                    with helper.fusion_weights(layer, weights, dims):
                        changed = forward()
                    require(len(layer._forward_pre_hooks) == hooks_before, 'Gecici hook kaldirilmadi')
                    for key in ('T', 'A', 'V'):
                        require(torch.allclose(base[key], changed[key], atol=1e-6, rtol=1e-6), 'Tek modlu baslik degisti')
                    require(all(torch.equal(a, b) for a, b in zip(inputs, saved)), 'Girdi/uzunluk degisti')
                    pred, alt = base['M'].reshape(-1).cpu(), changed['M'].reshape(-1).cpu()
                    truth = batch['labels']['M'].reshape(-1).cpu()
                    require(all(bool(torch.isfinite(v).all()) for v in (pred, alt, truth)), 'Gecersiz tahmin/etiket')
                    for j, index in enumerate(batch['index'].tolist()):
                        samples.append(dict(checkpoint=name, condition='K0', index=index, label=float(truth[j]),
                            plain_method=plain, weighted_method=weighted, plain_prediction=float(pred[j]),
                            weighted_prediction=float(alt[j]), w_text=float(weights[j, 0]),
                            w_audio=float(weights[j, 1]), w_vision=float(weights[j, 2])))
                    indices.extend(batch['index'].tolist())
                    predictions.append(pred); weighted_predictions.append(alt); labels.append(truth)
            require(indices == list(range(686)), 'Ornek sirasi/tekrar/eksik kayit')
            input_sha = fingerprint.hexdigest()
            if shared_fingerprint is None:
                shared_fingerprint = input_sha
            require(input_sha == shared_fingerprint, 'Checkpoint girdileri/etiketleri eslesmedi')
            truths = torch.cat(labels)
            for method, values in ((plain, torch.cat(predictions)), (weighted, torch.cat(weighted_predictions))):
                measured = prediction_metrics(values, truths)
                rows.append(dict(training_seed=1111, method=method, condition='K0', n=686, non0_n=656, **measured))
                checks.extend(metric_checks(method, measured, plan['reference'][method], 1e-5))
                print(f"{method}: MAE={measured['MAE']:.6f}; Non0_acc_2={measured['Non0_acc_2']:.6f}", flush=True)
            controls.append(dict(checkpoint=name, n=686, max_uniform_error=max_identity,
                                 hooks_restored=True, inputs_unchanged=True, input_sha256=input_sha))
            del model, layer, base, uniform, changed, weights, saved, inputs, text, audio, vision
            torch.cuda.empty_cache()
        require(len(rows) == 4 and len(samples) == 1372 and len(checks) == 16, 'Cikti satir sayisi')
        tables = {'metrics.csv': rows, 'reference_checks.csv': checks, 'controls.csv': controls,
                  'predictions_and_weights.csv': samples}
        for name, records in tables.items():
            write_csv(output / name, records)
        matched = all(r['passed'] for r in checks)
        manifest.update(metric_rows=4, prediction_rows=1372, control_rows=2, reference_check_rows=16,
                        reference_metrics_matched=matched, shared_inputs_verified=True, hooks_restored=True,
                        inputs_unchanged=True, output_hashes={name: digest(output / name) for name in tables})
        require(matched, 'Referans metrik eslesmedi; reference_checks.csv incelenmeli; toleransi degistirmeyin')
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        changed_files = unchanged(plan['hashes'])
        manifest.update(changed_inputs=changed_files, protected_files_unchanged=not changed_files,
                        data_file_unchanged=str(plan['data']) not in changed_files,
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        if changed_files:
            manifest['status'] = 'integrity_failed'
        manifest['total_replay_seconds'] = time.perf_counter() - started
        save()
    require(manifest['status'] == 'completed', 'Calisma/butunluk kontrolu basarisiz')
    print('TASINABILIR_K0_TAMAMLANDI:', output)
    print('KAPSAM: seed1111/K0/test; 16 metrik eslesmesi; tam egitim/yeni ortam kurulumu testi degil')


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--repo-dir', type=Path, default=here.parent)
    parser.add_argument('--config', type=Path, default=here / 'configs/selfmm_mosi_portable_k0.json')
    parser.add_argument('--dependencies', type=Path, default=here / 'configs/selfmm_portable_k0_dependencies.json')
    parser.add_argument('--package-dir', type=Path)
    for name in ('data-file', 'baseline-checkpoint', 'dropout-checkpoint', 'bert-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    plan = prepare(cli)
    if cli.check:
        require(not unchanged(plan['hashes']), 'Kontrol sirasinda kaynak degisti')
        print('KONTROL_BASARILI: model/veri yuklenmedi, indirme/egitim/dosya yazma yapilmadi')
        print('Gercek tahmin ve metrik eslesmesi ancak --check olmadan calistirinca sinanir.')
        return
    evaluate(plan)


if __name__ == '__main__':
    main()
