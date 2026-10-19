"""Offline seed-1111 K0-K7 A/B/C/D replay from explicit local assets.

Reuses the verified K0 runner's setup, config, metrics and weighting helpers.
Computes K7 statistics from TRAIN arrays only. No training or old run manifests.
--check reads hashes and runs synthetic CPU controls, never loads real data/model.
"""
import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
CONDITIONS = tuple(f'K{i}' for i in range(8))
METRICS = ('MAE', 'Non0_acc_2', 'Non0_F1_score', 'Corr')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_digest(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def read_reference(package):
    with (package / 'evaluation_test/per_seed_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['training_seed'] == '1111']
    reference = {}
    for row in rows:
        key = row['method'], row['condition']
        require(key not in reference, 'Tekrarlanan referans metrik')
        require(int(row['n']) == 686 and int(row['non0_n']) == 656, 'Referans ornek sayisi uyumsuz')
        values = {k: float(row[k]) for k in METRICS}
        require(all(math.isfinite(v) for v in values.values()), 'Sonlu olmayan referans')
        reference[key] = values
    require(set(reference) == {(m, c) for m in 'ABCD' for c in CONDITIONS}, '32 referans satiri gerekli')
    return reference


def statistics_from_train(block, expected_n=1284):
    """Only the explicitly passed TRAIN block is accessible here; no labels used."""
    np = core.np
    stats, cleanup = {}, {}
    for modality, dimension in (('audio', 5), ('vision', 20)):
        values = np.asarray(block[modality], dtype=np.float32).copy()
        lengths = np.asarray(block[modality + '_lengths'])
        require(values.ndim == 3 and values.shape[0] == expected_n and values.shape[2] == dimension,
                f'Train boyutu uyumsuz: {modality}')
        require(lengths.shape == (expected_n,), f'Train uzunluk sayisi uyumsuz: {modality}')
        count = int(np.isneginf(values).sum()) if modality == 'audio' else 0
        if modality == 'audio':
            values[np.isneginf(values)] = 0  # Exactly the existing MMSA cleanup.
        stats[modality] = corruptions.training_statistics(values, lengths)
        cleanup[modality + '_negative_inf_replaced'] = count
    return dict(stats_source='train_only_valid_frames', ddof=0, train_n=expected_n,
                preprocessing='float32; audio negative infinity replaced by zero as in MMSA',
                modalities=stats, cleanup=cleanup)


def load_train_statistics(path):
    # prepare() has already matched the trusted dataset's exact SHA256.
    # The pickle contains all splits. Only its TRAIN block contributes statistics.
    with path.open('rb') as stream:
        raw = pickle.load(stream)
    train = raw['train']
    del raw
    result = statistics_from_train(train)
    del train
    gc.collect()
    return result


def statistics_checks(stats, protocol):
    np = core.np
    rows = []
    for modality, dimension in (('audio', 5), ('vision', 20)):
        current = stats['modalities'][modality]
        expected = protocol['expected_train_statistics'][modality]
        row = dict(modality=modality, current_valid_frames=current['valid_frames'],
                   reference_valid_frames=expected['valid_frames'])
        passed = current['valid_frames'] == expected['valid_frames']
        for key in ('mean', 'std'):
            a, b = np.asarray(current[key]), np.asarray(expected[key])
            require(a.shape == b.shape == (dimension,) and np.isfinite(a).all() and np.isfinite(b).all(),
                    'Train istatistik boyutu/sonlulugu uyumsuz')
            if key == 'std':
                require((a >= 0).all() and (b >= 0).all(), 'Negatif std')
            error = float(np.max(np.abs(a - b)))
            row[key + '_max_absolute_error'] = error
            passed = passed and error <= protocol['statistics_absolute_tolerance']
        row.update(tolerance=protocol['statistics_absolute_tolerance'], passed=bool(passed))
        rows.append(row)
    return rows


def synthetic_check():
    np = core.np
    x = np.arange(240, dtype=np.float32).reshape(80, 3) + 1
    saved = x.copy()
    masks = {}
    for modality in ('audio', 'vision'):
        for condition in CONDITIONS:
            kwargs = dict(split='test', seed=1111)
            a, mask = corruptions.corrupt_modality(x, 67, modality, condition, [0., 1., 2.], 17, **kwargs)
            b, mask2 = corruptions.corrupt_modality(x, 67, modality, condition, [0., 1., 2.], 17, **kwargs)
            require(np.array_equal(a, b) and np.array_equal(mask, mask2), 'Bozulma seed tekrari basarisiz')
            require(np.array_equal(a[67:], x[67:]) and not mask[67:].any(), 'Padding degisti')
            if condition == 'K0':
                require(np.array_equal(a, x), 'K0 identity basarisiz')
            full = (modality == 'audio' and condition in ('K1', 'K3')) or (modality == 'vision' and condition in ('K2', 'K3'))
            if full:
                require((a[:67] == 0).all() and mask[:67].all(), 'Tam eksiklik basarisiz')
            elif condition in ('K1', 'K2'):
                require(np.array_equal(a, x), 'Mevcut modalite degisti')
            if condition in corruptions.PARTIAL_RATES:
                masks[modality, condition] = mask
                require((a[mask] == 0).all() and np.array_equal(a[~mask], x[~mask]), 'Kismi maske basarisiz')
            if condition == 'K7':
                valid, _ = corruptions.corrupt_modality(x, 67, modality, condition, [0., 1., 2.], 17, split='valid', seed=1111)
                require(not np.array_equal(a, valid), 'Test/valid RNG alanlari ayni')
                require(np.array_equal(a[:, 0], x[:, 0]), 'Sifir std ozelligi degisti')
                other, _ = corruptions.corrupt_modality(x, 67, modality, condition, [0., 1., 2.], 17, split='test', seed=1112)
                require(not np.array_equal(a, other), 'Farkli seed ayni gurultuyu verdi')
        require((~masks[modality, 'K4'] | masks[modality, 'K5']).all()
                and (~masks[modality, 'K5'] | masks[modality, 'K6']).all(), 'Maskeler ic ice degil')
    require(not np.array_equal(masks['audio', 'K5'], masks['vision', 'K5']), 'Modalite RNG ayrimi basarisiz')
    require(np.array_equal(x, saved), 'Sentetik kaynak degisti')
    train = np.array([[[1., 2.], [3., 4.], [999., 999.]], [[5., 6.], [999., 999.], [999., 999.]]], dtype=np.float32)
    stats = corruptions.training_statistics(train, [2, 1])
    require(stats['valid_frames'] == 3 and np.allclose(stats['std'], np.array([[1., 2.], [3., 4.], [5., 6.]]).std(0)),
            'Train std/padding dislama testi basarisiz')
    print('K0_K7_SENTETIK_KONTROLLER_BASARILI: maske, seed, modalite, test/valid, padding, sifir std, train std')


def prepare(cli):
    global core, corruptions
    repo = cli.repo_dir.resolve()
    protocol_path = cli.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding='utf-8-sig'))
    expected = dict(schema_version=1, training_seed=1111, corruption_seed=1111, split='test',
                    conditions=list(CONDITIONS), n=686, non0_n=656, batch_size=4, tau=1,
                    partial_rates={'K4': .3, 'K5': .5, 'K6': .7}, noise_factor=.5,
                    statistics_source='train_only_valid_frames', train_n=1284, statistics_ddof=0,
                    statistics_absolute_tolerance=1e-12, metric_absolute_tolerance=1e-5,
                    historical_run_manifests_required=False)
    require(all(protocol.get(k) == v for k, v in expected.items()), 'K0-K7 protokolu degismis')
    required_sources = {'research/evaluate_selfmm_portable_k0.py', 'research/selfmm_corruptions.py'}
    require(set(protocol['source_sha256_lf']) == required_sources, 'Ek kaynak listesi uyumsuz')
    for name, expected_sha in protocol['source_sha256_lf'].items():
        require(source_digest(repo / name) == expected_sha, f'Kaynak referanstan farkli: {name}')
    require(set(protocol['expected_test_input_sha256']) == set(CONDITIONS), 'Sekiz girdi referansi gerekli')
    core = load_module(repo / 'research/evaluate_selfmm_portable_k0.py', 'portable_core_k0')
    cli.config = cli.config or repo / 'research/configs/selfmm_mosi_portable_k0.json'
    cli.dependencies = cli.dependencies or repo / 'research/configs/selfmm_portable_k0_dependencies.json'
    print('TEMEL_K0_BAGIMLILIK_KONTROLU: ayni config, veri, checkpoint ve BERT')
    plan = core.prepare(cli)  # No actual dataset/checkpoint/model loading here.
    corruptions = load_module(repo / 'research/selfmm_corruptions.py', 'portable_corruptions')
    require(corruptions.CONDITIONS == CONDITIONS and corruptions.PARTIAL_RATES == protocol['partial_rates']
            and corruptions.NOISE_FACTOR == .5, 'Bozulma modulu protokolle uyumsuz')
    # Validate embedded reference shapes without opening the real dataset.
    statistics_checks({'modalities': protocol['expected_train_statistics']}, protocol)
    for value in [*protocol['expected_test_input_sha256'].values(), protocol['expected_test_labels_float32_sha256']]:
        require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                'Referans SHA256 bicimi uyumsuz')
    synthetic_check()
    plan.update(protocol=protocol, reference=read_reference(plan['package']))
    for path in (protocol_path, Path(__file__).resolve(), repo / 'research/selfmm_corruptions.py',
                 repo / 'research/evaluate_selfmm_portable_k0.py'):
        plan['hashes'][path] = core.digest(path)
    print('KAPSAM: seed1111/test/K0-K7; 32 metrik satiri; 128 metrik kontrolu; bozulma seed=1111')
    print('TRAIN_STD: gercek calismada train verisinden yeniden hesaplanacak; eski run manifesti okunmaz')
    return plan


def prepare_batch(batch, condition, stats, fingerprint, device, totals):
    """Byte order matches the frozen K0-K7 input fingerprints, including batch tail."""
    np, torch = core.np, core.torch
    features = {}
    for modality in ('audio', 'vision'):
        original = batch[modality].numpy()
        saved = original.copy()
        altered = []
        scale = np.asarray(stats['modalities'][modality]['std'], dtype=np.float64)
        summary = totals.setdefault(modality, dict(valid_frames=0, masked_frames=0, z_square=0., z_count=0))
        for j, index in enumerate(batch['index'].tolist()):
            length = int(batch[modality + '_lengths'][j])
            out, mask = corruptions.corrupt_modality(original[j], length, modality, condition, scale, index,
                                                    split='test', seed=1111)
            require(np.array_equal(out[length:], original[j, length:], equal_nan=True), 'Padding degisti')
            summary['valid_frames'] += length
            summary['masked_frames'] += int(mask[:length].sum())
            if condition == 'K7' and (scale > 0).any():
                positive = scale > 0
                z = (out[:length, positive].astype(np.float64) - original[j, :length][:, positive].astype(np.float64)) / scale[positive]
                summary['z_square'] += float((z * z).sum())
                summary['z_count'] += z.size
            altered.append(out)
        require(np.array_equal(original, saved, equal_nan=True), 'Kaynak batch yerinde degisti')
        values = np.stack(altered)
        fingerprint.update(values.tobytes())
        fingerprint.update(batch[modality + '_lengths'].numpy().tobytes())
        features[modality] = torch.from_numpy(values).to(device)
    fingerprint.update(batch['text'].numpy().tobytes())
    fingerprint.update(batch['index'].numpy().tobytes())
    return features


def label_digest(labels):
    return hashlib.sha256(core.np.asarray(labels, dtype='<f4').reshape(-1).tobytes()).hexdigest()


def evaluate(plan):
    torch, np = core.torch, core.np
    require(plan['cuda_available'], 'Gercek degerlendirme icin CUDA gerekli')
    output = plan['output']
    core.check_new_output(output, plan['repo'], plan['package'], plan['bert'], plan['checkpoints'])
    require(not core.unchanged(plan['hashes']), 'Kontrolden sonra bir kaynak degisti')
    output.mkdir(exist_ok=False)
    start = time.perf_counter()
    protocol = plan['protocol']
    manifest = dict(status='running', schema_version=1, training_seed=1111, corruption_seed=1111,
        split='test', conditions=list(CONDITIONS), n=686, non0_n=656, batch_size=4, tau=1,
        inference_only=True, training_performed=False, offline=True, historical_run_manifests_required=False,
        versions=plan['versions'], python_version=platform.python_version(), config_settings=plan['settings'],
        noise_statistics_source='recomputed_train_only_valid_frames', statistics_ddof=0,
        noise_factor=.5, partial_rates=corruptions.PARTIAL_RATES,
        inputs=core.manifest_inputs(plan['hashes'], output),
        input_path_base='directory containing this manifest; different-drive paths may be absolute',
        reference_scope='128 aggregate metric checks, 16 input fingerprint checks, train-statistics and label checks',
        historical_per_sample_predictions_compared=False,
        test_history='Test already observed; frozen protocol replay, no tuning or seed selection.',
        data_loading_note='Pickle contains all splits. Only TRAIN arrays compute noise scales; only TEST is evaluated.',
        timing_scope='total replay including train statistics, loading, identity controls and hashes; not latency benchmark',
        created_at_utc=datetime.now(timezone.utc).isoformat())
    def save_manifest():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save_manifest()
    rows, samples, controls, checks, input_checks, corruption_rows = [], [], [], [], [], []
    fingerprints = {}
    try:
        print('TRAIN_STD_HESAPLANIYOR: yalniz train; gercek zaman adimlari; ddof=0', flush=True)
        stats = load_train_statistics(plan['data'])
        stats['source_sha256'] = plan['frozen']['data_sha256']
        stats_rows = statistics_checks(stats, protocol)
        (output / 'train_feature_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
        core.write_csv(output / 'train_stats_checks.csv', stats_rows)
        require(all(r['passed'] for r in stats_rows), 'Train istatistik referansi eslesmedi')
        manifest['train_statistics_recomputed_and_matched'] = True
        print('TRAIN_STD_DOGRULANDI: audio/vision; iki referans kontrolu gecti', flush=True)
        torch.cuda.set_device(0)
        manifest.update(gpu=torch.cuda.get_device_name(0), dtype='float32', autocast=False)
        dataset_args = copy.deepcopy(plan['args'])
        dataset = core.MMDataset(dataset_args, mode='test')
        require(len(dataset) == 686 and dataset_args.feature_dims == [768, 5, 20], 'Test boyutlari uyumsuz')
        require(int(np.count_nonzero(dataset.labels['M'])) == 656, 'Non0 sayisi uyumsuz')
        require(label_digest(dataset.labels['M']) == protocol['expected_test_labels_float32_sha256'], 'Test etiketi/sirasi uyumsuz')
        manifest['reference_test_labels_matched'] = True
        loader = core.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
        for name, plain, weighted in (('baseline', 'A', 'C'), ('dropout', 'B', 'D')):
            core.setup_seed(1111)
            args = copy.deepcopy(plan['args'])
            model = core.AMIO(args)
            state = torch.load(plan['checkpoints'][name], map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            del state
            model.to(args.device).eval().requires_grad_(False)
            layer = model.Model.post_fusion_layer_1
            dims = (args.text_out, args.audio_out, args.video_out)
            hooks_before = len(layer._forward_pre_hooks)
            for condition in CONDITIONS:
                print(f'MODEL_CALISIYOR: {plain}/{weighted}; {condition}; 686 test ornegi', flush=True)
                predictions, weighted_predictions, truths, indices = [], [], [], []
                fingerprint = hashlib.sha256()
                totals = {}
                max_identity = 0.
                with torch.inference_mode():
                    for batch in loader:
                        require(all(not m.training for m in model.modules())
                                and all(not p.requires_grad for p in model.parameters()), 'Model eval/dondurulmus degil')
                        cpu_text = batch['text'].clone()
                        lengths = {m: batch[m + '_lengths'].clone() for m in ('audio', 'vision')}
                        features = prepare_batch(batch, condition, stats, fingerprint, args.device, totals)
                        text = batch['text'].to(args.device)
                        inputs = (text, features['audio'], features['vision'], batch['audio_lengths'], batch['vision_lengths'])
                        saved = tuple(t.clone() for t in inputs)
                        def forward():
                            return model(text, (features['audio'], batch['audio_lengths']),
                                         (features['vision'], batch['vision_lengths']))
                        base = forward()
                        weights, _ = core.helper.weights_from_outputs(base, 1.)
                        with core.helper.fusion_weights(layer, torch.ones_like(weights), dims):
                            uniform = forward()
                        for key in ('M', 'T', 'A', 'V'):
                            error = float((base[key] - uniform[key]).abs().max())
                            max_identity = max(max_identity, error)
                            require(torch.allclose(base[key], uniform[key], atol=1e-6, rtol=1e-6), 'Identity kontrolu basarisiz')
                        with core.helper.fusion_weights(layer, weights, dims):
                            changed = forward()
                        require(len(layer._forward_pre_hooks) == hooks_before, 'Hook kaldirilmadi')
                        for key in ('T', 'A', 'V'):
                            require(torch.allclose(base[key], changed[key], atol=1e-6, rtol=1e-6), 'Tek modlu baslik degisti')
                        require(all(torch.equal(a, b) for a, b in zip(inputs, saved)), 'Model girdiyi degistirdi')
                        require(torch.equal(batch['text'], cpu_text) and all(torch.equal(batch[m + '_lengths'], lengths[m])
                                for m in lengths), 'Metin/uzunluk degisti')
                        pred, alt = base['M'].reshape(-1).cpu(), changed['M'].reshape(-1).cpu()
                        labels = batch['labels']['M'].reshape(-1).cpu()
                        require(all(bool(torch.isfinite(v).all()) for v in (pred, alt, labels)), 'Gecersiz tahmin/etiket')
                        for j, index in enumerate(batch['index'].tolist()):
                            samples.append(dict(training_seed=1111, corruption_seed=1111, checkpoint=name,
                                condition=condition, index=index, label=float(labels[j]), plain_method=plain,
                                weighted_method=weighted, plain_prediction=float(pred[j]), weighted_prediction=float(alt[j]),
                                w_text=float(weights[j, 0]), w_audio=float(weights[j, 1]), w_vision=float(weights[j, 2])))
                        indices.extend(batch['index'].tolist())
                        predictions.append(pred); weighted_predictions.append(alt); truths.append(labels)
                require(indices == list(range(686)), 'Eksik/tekrar/sirasi farkli ornek')
                labels = torch.cat(truths)
                require(label_digest(labels.numpy()) == protocol['expected_test_labels_float32_sha256'], 'Batch etiket sirasi uyumsuz')
                input_sha = fingerprint.hexdigest()
                if name == 'baseline':
                    fingerprints[condition] = input_sha
                shared = input_sha == fingerprints[condition]
                reference_sha = protocol['expected_test_input_sha256'][condition]
                matched = input_sha == reference_sha
                input_checks.append(dict(checkpoint=name, condition=condition, current_sha256=input_sha,
                                         reference_sha256=reference_sha, shared_inputs=shared, passed=matched and shared))
                # Continue to export all diagnostics even if a reference fingerprint differs.
                for method, values in ((plain, torch.cat(predictions)), (weighted, torch.cat(weighted_predictions))):
                    measured = core.prediction_metrics(values, labels)
                    rows.append(dict(training_seed=1111, method=method, condition=condition, n=686, non0_n=656, **measured))
                    for check in core.metric_checks(method, measured, plan['reference'][method, condition], 1e-5):
                        checks.append(dict(condition=condition, **check))
                    print(f"{method}/{condition}: MAE={measured['MAE']:.6f}; ACC={measured['Non0_acc_2']:.6f}", flush=True)
                controls.append(dict(checkpoint=name, condition=condition, n=686, max_uniform_error=max_identity,
                    hooks_restored=True, text_unchanged=True, lengths_unchanged=True, padding_unchanged=True,
                    source_arrays_unchanged=True, input_sha256=input_sha))
                if name == 'baseline':
                    for modality, total in totals.items():
                        full = (modality == 'audio' and condition in ('K1', 'K3')) or (modality == 'vision' and condition in ('K2', 'K3'))
                        corruption_rows.append(dict(modality=modality, condition=condition, n=686,
                            valid_frames=total['valid_frames'], masked_frames=total['masked_frames'],
                            masked_rate=total['masked_frames'] / total['valid_frames'],
                            target_mask_rate=corruptions.PARTIAL_RATES.get(condition, 1. if full else 0.),
                            noise_rms_div_train_std=float(np.sqrt(total['z_square'] / total['z_count'])) if total['z_count'] else 0.))
            del model, layer, base, uniform, changed, weights, saved, inputs, text, features
            torch.cuda.empty_cache()
        require(len(rows) == 32 and len(samples) == 10976 and len(checks) == 128
                and len(controls) == len(input_checks) == len(corruption_rows) == 16, 'Cikti satir sayisi uyumsuz')
        tables = {'metrics.csv': rows, 'reference_checks.csv': checks, 'controls.csv': controls,
                  'predictions_and_weights.csv': samples, 'reference_input_checks.csv': input_checks,
                  'corruption_summary.csv': corruption_rows}
        for filename, records in tables.items():
            core.write_csv(output / filename, records)
        matched = all(r['passed'] for r in checks)
        inputs_matched = all(r['passed'] for r in input_checks)
        outputs = list(tables) + ['train_feature_stats.json', 'train_stats_checks.csv']
        manifest.update(metric_rows=32, prediction_rows=10976, reference_check_rows=128, control_rows=16,
            reference_input_check_rows=16, train_stats_check_rows=2, corruption_summary_rows=16,
            reference_metrics_matched=matched, reference_corrupted_inputs_matched=inputs_matched,
            shared_inputs_verified=all(r['shared_inputs'] for r in input_checks), hooks_restored=True,
            text_unchanged=True, lengths_unchanged=True, padding_unchanged=True, source_arrays_unchanged=True,
            output_hashes={name: core.digest(output / name) for name in outputs})
        require(inputs_matched and matched, 'Referans girdi/metrik eslesmedi; kontrol CSVlerini inceleyin, toleransi degistirmeyin')
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        changed_files = core.unchanged(plan['hashes'])
        manifest.update(changed_inputs=changed_files, protected_files_unchanged=not changed_files,
                        data_file_unchanged=str(plan['data']) not in changed_files,
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        if changed_files:
            manifest['status'] = 'integrity_failed'
        manifest['total_replay_seconds'] = time.perf_counter() - start
        save_manifest()
    require(manifest['status'] == 'completed', 'Calisma/butunluk kontrolu basarisiz')
    print('TASINABILIR_K0_K7_TAMAMLANDI:', output)
    print('KAPSAM: seed1111/test; 128 metrik, 16 girdi, 2 train-istatistik kontrolu; yeni egitim yok')


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--repo-dir', type=Path, default=here.parent)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--dependencies', type=Path, help='Existing portable K0 dependency file')
    parser.add_argument('--protocol', type=Path, default=here / 'configs/selfmm_portable_k0_k7_dependencies.json')
    parser.add_argument('--package-dir', type=Path)
    for name in ('data-file', 'baseline-checkpoint', 'dropout-checkpoint', 'bert-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    plan = prepare(cli)
    if cli.check:
        require(not core.unchanged(plan['hashes']), 'Kontrol sirasinda kaynak degisti')
        print('KONTROL_BASARILI: K0-K7; gercek model/veri acilmadi, indirme/egitim/dosya yazma yapilmadi')
        print('Train std, gercek bozulmus girdiler ve 128 metrik gercek calismada dogrulanacak.')
        return
    evaluate(plan)


if __name__ == '__main__':
    main()
