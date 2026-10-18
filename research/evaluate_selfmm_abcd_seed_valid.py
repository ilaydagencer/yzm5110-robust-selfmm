"""A/B/C/D validation for matched training repeats (1112-1115), K0-K7.

Training seed varies; corruption seed=1111 and tau=1 remain fixed. Reuses the
verified seed-1111 helpers and compares INPUT fingerprints, not predictions,
across training seeds. --check loads no model/dataset and writes no files.
Copy beside the existing research helpers before running without --repo-dir.
"""
import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import sentencepiece  # Windows: MUST precede torch and MMSA.
import numpy as np
import torch
import MMSA
from torch.utils.data import DataLoader
from MMSA.config import get_config_regression
from MMSA.data_loader import MMDataset
from MMSA.models import AMIO
from MMSA.utils import MetricsTop, setup_seed

TAU = 1.0
CORRUPTION_SEED = 1111
TRAINING_SEEDS = (1112, 1113, 1114, 1115)
N = 229


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return Path(path).read_text(encoding='utf-8')


def load_json(path):
    return json.loads(read_json(path))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load_helpers(repo):
    # Explicitly use the target repo even if this evaluator is in outputs/.
    global corruptions, weights_from_outputs, fusion_weights, write_csv
    directory = repo / 'research'
    sys.path.insert(0, str(directory))
    modules = {}
    for name in ('evaluate_selfmm_abcd_valid', 'selfmm_corruptions', 'check_selfmm_corruptions'):
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve() == (directory / f'{name}.py').resolve(),
                f'Yanlis yardimci modul: {name}')
        modules[name] = module
    base = modules['evaluate_selfmm_abcd_valid']
    corruptions = modules['selfmm_corruptions']
    weights_from_outputs = base.weights_from_outputs
    fusion_weights = base.fusion_weights
    write_csv = base.write_csv
    return modules


def prepare(repo, seed):
    require(seed in TRAINING_SEEDS, 'Egitim seed 1112-1115 olmali; 1111 korunuyor')
    repo = Path(repo).resolve()
    project = repo.parent
    output = project / 'runs' / f'selfmm_abcd_corrupt_valid_seed{seed}_tau1_v1'
    require(not os.path.lexists(output), f'Cikti zaten var; uzerine yazilmaz: {output}')
    require(Path(MMSA.__file__).resolve().is_relative_to(repo / 'src'), 'Yanlis MMSA kurulumu')
    modules = load_helpers(repo)
    protocol_path = project / 'runs/selfmm_corruption_protocol_valid_seed1111_v1/manifest.json'
    stats_path = protocol_path.with_name('train_feature_stats.json')
    reference_path = project / 'runs/selfmm_abcd_corrupt_valid_seed1111_tau1_v1/manifest.json'
    labels_path = project / 'runs/selfmm_abcd_valid_seed1111_tau1/predictions_and_weights.csv'
    protocol, stats, reference = map(load_json, (protocol_path, stats_path, reference_path))
    require(protocol['status'] == 'completed' and protocol['split'] == 'valid'
            and protocol['protocol_version'] == 1 and protocol['n'] == N
            and protocol['seed'] == CORRUPTION_SEED and protocol['noise_factor'] == 0.5
            and protocol['partial_rates'] == corruptions.PARTIAL_RATES, 'Protokol uyumsuz')
    require(corruptions.NOISE_FACTOR == 0.5
            and corruptions.CONDITIONS == tuple(f'K{i}' for i in range(8)), 'Bozulma modulu uyumsuz')
    for flag in ('text_unchanged', 'lengths_unchanged', 'padding_unchanged', 'source_unchanged',
                 'masks_nested', 'modality_rng_independent', 'model_order_independent'):
        require(protocol.get(flag) is True, f'Protokol kontrolu eksik: {flag}')
    require(stats['source_sha256'] == protocol['source_sha256']
            and stats['stats_source'] == 'train_only_valid_frames' and stats['ddof'] == 0,
            'Gurultu istatistikleri train-only olmali')
    require(reference['status'] == 'completed' and reference['split'] == 'valid'
            and reference['n'] == N and reference['batch_size'] == 4
            and reference['training_seed'] == 1111 and reference['corruption_seed'] == CORRUPTION_SEED
            and reference['tau'] == TAU and reference['protocol_version'] == 1
            and reference['conditions'] == list(corruptions.CONDITIONS)
            and reference['data_sha256'] == protocol['source_sha256'], 'Seed 1111 referansi uyumsuz')
    for flag in ('protected_files_unchanged', 'data_file_unchanged', 'shared_corrupted_inputs',
                 'prior_K0_K3_reproduced'):
        require(reference.get(flag) is True, f'Referans kontrolu eksik: {flag}')
    require(digest(protocol_path) == reference['protocol_manifest_sha256']
            and digest(stats_path) == reference['train_stats_sha256'], 'Protokol/istatistik dosyasi degismis')
    expected_inputs = reference['corrupted_input_sha256']
    require(set(expected_inputs) == set(corruptions.CONDITIONS)
            and all(len(h) == 64 for h in expected_inputs.values()), 'Girdi ozetleri eksik')

    protected = {Path(__file__).resolve(), protocol_path, stats_path, reference_path, labels_path}
    for module in modules.values():
        path = Path(module.__file__).resolve()
        require(digest(path) == reference['protected_hashes'][str(path)], f'Yardimci kod degismis: {path}')
        if path.name in protocol['code_hashes']:
            require(digest(path) == protocol['code_hashes'][path.name], 'Protokol kodu degismis')
        protected.add(path)
    # Preserve all earlier reference files, including seed-1111 checkpoints.
    for filename, expected in reference['protected_hashes'].items():
        path = Path(filename)
        require(digest(path) == expected, f'Seed 1111 referansi degismis: {path}')
        protected.add(path)

    labels = {}
    for row in read_csv(labels_path):
        if row['checkpoint'] == 'baseline' and row['condition'] == 'K0':
            index, value = int(row['index']), float(row['label'])
            require(index not in labels and np.isfinite(value), 'Referans etiket tekrar/gecersiz')
            labels[index] = value
    require(set(labels) == set(range(N)), 'Referans etiketleri eksik')
    specs, arguments, configs, training_records = {}, {}, {}, {}
    for name, plain, weighted in (('baseline', 'A', 'C'), ('dropout', 'B', 'D')):
        suffix = 'baseline' if name == 'baseline' else 'dropout_a20_v20'
        run = project / 'runs' / f'selfmm_{suffix}_seed{seed}'
        record_path = run / 'experiment.json'
        config = project / 'local_configs' / f'selfmm_mosi_{suffix}_batch4_seed{seed}.json'
        checkpoint = run / 'saved_models/self_mm-mosi.pth'
        record = load_json(record_path)
        require(record['status'] == 'completed' and record['variant'] == name
                and record['seed'] == seed and record['batch_size'] == 4, f'Egitim kaydi uyumsuz: {name}')
        for flag in ('protected_files_unchanged', 'data_file_unchanged', 'local_config_unchanged', 'forward_restored'):
            require(record.get(flag) is True, f'Egitim kontrolu eksik: {name}/{flag}')
        require(Path(record['checkpoint_path']).resolve() == checkpoint.resolve()
                and digest(checkpoint) == record['checkpoint_sha256'], f'Checkpoint degismis: {name}')
        require(Path(record['local_config']).resolve() == config.resolve()
                and digest(config) == record['local_config_sha256'], f'Config degismis: {name}')
        require(record['data_sha256'] == protocol['source_sha256'], 'Egitim verisi protokolden farkli')
        require(record['dropout']['enabled'] is (name == 'dropout'), 'Dropout yontemi uyumsuz')
        if name == 'dropout':
            require(record['dropout']['probability'] == 0.2 and record['dropout']['generator_seed'] == seed
                    and record['dropout']['text_unchanged'] is True
                    and record['dropout_counts']['eval_dropped'] == 0, 'Dropout protokolu uyumsuz')
        # Source hashes establish what code produced each checkpoint, even with local edits.
        for filename, expected in record['protected_hashes'].items():
            path = Path(filename)
            require(digest(path) == expected, f'Egitimde korunan dosya degismis: {path}')
            protected.add(path)
        configs[name] = load_json(config)
        args = get_config_regression('self_mm', 'mosi', str(config))
        require(dict(args) == record['resolved_model_config'], 'Cozumlenmis egitim ayarlari degismis')
        require(args.batch_size == 4 and args.use_bert and args.use_finetune
                and not args.need_data_aligned and not args.need_model_aligned
                and not args.get('data_missing', False) and args.post_fusion_dropout == 0
                and args.KeyEval == 'Loss', 'Model ayarlari uyumsuz')
        args.update(train_mode='regression', device=torch.device('cuda:0'), custom_feature=None,
                    feature_T=None, feature_A=None, feature_V=None, data_missing=False)
        for modality, dimension in (('audio', args.feature_dims[1]), ('vision', args.feature_dims[2])):
            scale = np.asarray(stats['modalities'][modality]['std'])
            require(scale.shape == (dimension,) and np.isfinite(scale).all() and (scale >= 0).all(),
                    'Train std boyutu/degeri uyumsuz')
        specs[name] = (config, checkpoint, plain, weighted)
        arguments[name] = args
        training_records[name] = dict(manifest=str(record_path), manifest_sha256=digest(record_path),
                                     checkpoint_sha256=record['checkpoint_sha256'], git_head=record['git_head'])
        protected.update((record_path, config, checkpoint))
        print(f'{plain}/{weighted} CHECKPOINT: {checkpoint}')
    require(configs['baseline'] == configs['dropout']
            == load_json(project / 'local_configs/selfmm_mosi_batch4.json'), 'Eslenmis configler farkli')
    data_path = Path(arguments['baseline'].featurePath).resolve()
    require(data_path == (project / 'data/MOSI/Processed/unaligned_50.pkl').resolve()
            and data_path.is_file(), 'MOSI veri yolu uyumsuz')
    modules['evaluate_selfmm_abcd_valid'].self_check()
    modules['check_selfmm_corruptions'].synthetic_check()
    print(f'SPLIT: valid; N: {N}; BATCH: 4; TAU: {TAU}; EGITIM SEED: {seed}; BOZULMA SEED: {CORRUPTION_SEED}')
    print('KOSULLAR: K0-K7; CIKTI:', output)
    return dict(seed=seed, output=output, specs=specs, arguments=arguments, stats=stats,
                labels=labels, expected_inputs=expected_inputs, data_path=data_path,
                data_sha256=protocol['source_sha256'], training_records=training_records,
                reference_manifest_sha256=digest(reference_path),
                protocol_manifest_sha256=digest(protocol_path), train_stats_sha256=digest(stats_path),
                hashes={str(p): digest(p) for p in sorted(protected, key=str)})


def prepare_batch(batch, condition, stats, fingerprint, device):
    features = {}
    for modality in ('audio', 'vision'):
        original = batch[modality].numpy()
        altered = []
        for j, index in enumerate(batch['index'].tolist()):
            length = int(batch[modality + '_lengths'][j])
            out, _ = corruptions.corrupt_modality(original[j], length, modality, condition,
                stats['modalities'][modality]['std'], index, split='valid', seed=CORRUPTION_SEED)
            require(np.array_equal(out[length:], original[j, length:], equal_nan=True), 'Padding degisti')
            altered.append(out)
        values = np.stack(altered)
        # Keep the seed-1111 evaluator's exact fingerprint order and dtypes.
        fingerprint.update(values.tobytes())
        fingerprint.update(batch[modality + '_lengths'].numpy().tobytes())
        features[modality] = torch.from_numpy(values).to(device)
    fingerprint.update(batch['text'].numpy().tobytes())
    fingerprint.update(batch['index'].numpy().tobytes())
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--repo-dir', type=Path, default=Path(__file__).resolve().parents[1])
    cli = parser.parse_args()
    plan = prepare(cli.repo_dir, cli.seed)
    if cli.check:
        print('KONTROL_BASARILI: model/veri yuklenmedi, dosya yazilmadi; metadata/kod/checkpoint ozetleri eslesti')
        return
    evaluate(plan)


def evaluate(plan):
    seed, output, specs = plan['seed'], plan['output'], plan['specs']
    arguments, stats, reference_labels = plan['arguments'], plan['stats'], plan['labels']
    data_path, hashes = plan['data_path'], plan['hashes']
    require(not os.path.lexists(output), f'Cikti zaten var; uzerine yazilmaz: {output}')
    require(torch.cuda.is_available(), 'CUDA bulunamadi')
    require(digest(data_path) == plan['data_sha256'], 'Veri dosyasi protokolden farkli')
    require(all(digest(Path(p)) == h for p, h in hashes.items()), 'Kontrolden sonra korunan dosya degisti')
    setup_seed(seed)
    torch.cuda.set_device(0)
    dataset = MMDataset(copy.deepcopy(arguments['baseline']), mode='valid')
    require(len(dataset) == N, 'Validation ornek sayisi uyumsuz')
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', created_at_utc=datetime.now(timezone.utc).isoformat(),
        split='valid', n=N, batch_size=4, training_seed=seed, corruption_seed=CORRUPTION_SEED,
        tau=TAU, conditions=list(corruptions.CONDITIONS), protocol_version=1,
        protocol_manifest_sha256=plan['protocol_manifest_sha256'], train_stats_sha256=plan['train_stats_sha256'],
        reference_seed1111_manifest_sha256=plan['reference_manifest_sha256'],
        data_sha256=plan['data_sha256'], protected_hashes=hashes, training_records=plan['training_records'],
        torch_version=torch.__version__, numpy_version=np.__version__, gpu=torch.cuda.get_device_name(0),
        inference_only=True, hyperparameter_tuning=False, test_evaluation=False,
        selection='Fixed tau=1 and corruption seed=1111; all conditions reported, no tuning.',
        reference_control='Match seed-1111 INPUT hashes and labels; predictions may differ across training seeds.',
        data_loading_note='MMSA reads a pickle containing all splits; only the valid dataset is evaluated.',
        note='Three forward passes include identity control; not a latency benchmark.')
    def save_manifest():
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save_manifest()
    rows=[]; sample_rows=[]; weights_rows=[]; controls=[]; reference_checks=[]
    input_digests={}
    metric_fn=MetricsTop('regression').getMetics('mosi')
    try:
        for name,(_,checkpoint,plain,weighted) in specs.items():
            setup_seed(seed)
            args=copy.deepcopy(arguments[name])
            model=AMIO(args)
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True)
            del state
            model.to(args.device).eval().requires_grad_(False)
            layer=model.Model.post_fusion_layer_1
            dims=(args.text_out,args.audio_out,args.video_out)
            for condition in corruptions.CONDITIONS:
                predictions=[]; weighted_predictions=[]; truths=[]; all_weights=[]; indices=[]
                identity_error=0.
                fingerprint=hashlib.sha256()
                hooks_before=len(layer._forward_pre_hooks)
                with torch.inference_mode():
                    for batch in loader:
                        require(all(not module.training for module in model.modules()),'Model eval modunda degil')
                        batch_indices=batch['index'].tolist()
                        features=prepare_batch(batch,condition,stats,fingerprint,args.device)
                        text=batch['text'].to(args.device)
                        def forward():
                            return model(text,(features['audio'],batch['audio_lengths']),
                                         (features['vision'],batch['vision_lengths']))
                        base=forward()
                        w,deviations=weights_from_outputs(base,TAU)
                        with fusion_weights(layer,torch.ones_like(w),dims):uniform=forward()
                        for key in ('M','T','A','V'):
                            identity_error=max(identity_error,float((uniform[key]-base[key]).abs().max()))
                            require(torch.allclose(uniform[key],base[key],atol=1e-6,rtol=1e-6),'Identity kontrolu basarisiz')
                        with fusion_weights(layer,w,dims):changed=forward()
                        require(len(layer._forward_pre_hooks)==hooks_before,'Hook sizintisi')
                        for key in ('T','A','V'):
                            require(torch.allclose(base[key],changed[key],atol=1e-6,rtol=1e-6),'Tek modlu baslik degisti')
                        p=base['M'].reshape(-1).cpu();q=changed['M'].reshape(-1).cpu()
                        labels=batch['labels']['M'].reshape(-1).cpu()
                        require(all(bool(torch.isfinite(t).all()) for t in (p,q,labels)),'Gecersiz sayi')
                        for j,index in enumerate(batch_indices):
                            require(abs(float(labels[j])-reference_labels[index])<1e-7,'Etiket/sira degisti')
                            sample_rows.append(dict(training_seed=seed,corruption_seed=CORRUPTION_SEED,checkpoint=name,condition=condition,index=index,
                                label=float(labels[j]),plain_method=plain,weighted_method=weighted,
                                plain_prediction=float(p[j]),weighted_prediction=float(q[j]),
                                y_text=float(base['T'][j].item()),y_audio=float(base['A'][j].item()),
                                y_vision=float(base['V'][j].item()),w_text=float(w[j,0]),
                                w_audio=float(w[j,1]),w_vision=float(w[j,2]),
                                deviation_text=float(deviations[j,0]),deviation_audio=float(deviations[j,1]),
                                deviation_vision=float(deviations[j,2])))
                        predictions.append(p);weighted_predictions.append(q);truths.append(labels)
                        all_weights.append(w.cpu());indices.extend(batch_indices)
                require(indices==list(range(229)),'Validation sirasi beklenenden farkli')
                checksum=fingerprint.hexdigest()
                if name=='baseline':input_digests[condition]=checksum
                require(checksum==input_digests[condition],'Checkpointler ayni bozuk girdiyi almadi')
                expected=plan['expected_inputs'][condition]
                require(checksum==expected, f'Seed 1111 ile ayni girdi degil: {name}/{condition}')
                reference_checks.append(dict(training_seed=seed,corruption_seed=CORRUPTION_SEED,
                    checkpoint=name,condition=condition,actual_sha256=checksum,
                    reference_seed1111_sha256=expected,passed=True))
                p=torch.cat(predictions);q=torch.cat(weighted_predictions);labels=torch.cat(truths);w=torch.cat(all_weights)
                for method,pred in ((plain,p),(weighted,q)):
                    metric={k:float(v) for k,v in metric_fn(pred,labels).items()}
                    row=dict(training_seed=seed,corruption_seed=CORRUPTION_SEED,method=method,checkpoint=name,split='valid',condition=condition,tau=TAU,
                             n=229,**metric,MAE_exact=float((pred-labels).abs().mean()))
                    rows.append(row);print(json.dumps(row),flush=True)
                entry=dict(training_seed=seed,corruption_seed=CORRUPTION_SEED,method=weighted,condition=condition,tau=TAU,n=229)
                for i,modality in enumerate(('text','audio','vision')):
                    entry.update({f'w_{modality}_mean':float(w[:,i].mean()),
                                  f'w_{modality}_std':float(w[:,i].std(unbiased=False)),
                                  f'w_{modality}_min':float(w[:,i].min()),f'w_{modality}_max':float(w[:,i].max())})
                weights_rows.append(entry)
                controls.append(dict(training_seed=seed,corruption_seed=CORRUPTION_SEED,checkpoint=name,condition=condition,n=229,max_uniform_error=identity_error,
                                     hooks_restored=True,corrupted_input_sha256=checksum))
            del model
            torch.cuda.empty_cache()
        comparisons=[];degradation=[]
        for condition in corruptions.CONDITIONS:
            lookup={r['method']:r for r in rows if r['condition']==condition}
            for first,second in (('A','B'),('A','C'),('B','D'),('A','D')):
                comparisons.append(dict(training_seed=seed,corruption_seed=CORRUPTION_SEED,condition=condition,comparison=f'{second}-minus-{first}',
                    Non0_acc_delta=lookup[second]['Non0_acc_2']-lookup[first]['Non0_acc_2'],
                    Non0_F1_delta=lookup[second]['Non0_F1_score']-lookup[first]['Non0_F1_score'],
                    MAE_exact_delta=lookup[second]['MAE_exact']-lookup[first]['MAE_exact'],
                    Corr_delta=lookup[second]['Corr']-lookup[first]['Corr']))
        for method in ('A','B','C','D'):
            clean=next(r for r in rows if r['method']==method and r['condition']=='K0')
            for row in (r for r in rows if r['method']==method):
                degradation.append(dict(training_seed=seed,corruption_seed=CORRUPTION_SEED,method=method,condition=row['condition'],
                    MAE_increase=row['MAE_exact']-clean['MAE_exact'],
                    Non0_acc_drop=clean['Non0_acc_2']-row['Non0_acc_2'],
                    Corr_drop=clean['Corr']-row['Corr']))
        for filename,records in (('metrics.csv',rows),('comparisons.csv',comparisons),
                ('degradation.csv',degradation),('predictions_and_weights.csv',sample_rows),
                ('weight_summary.csv',weights_rows),('controls.csv',controls),('reference_input_checks.csv',reference_checks)):
            write_csv(output/filename,records)
        manifest.update(status='completed', shared_corrupted_inputs=True, same_corruptions_as_seed1111=True,
                        reference_labels_matched=True, corrupted_input_sha256=input_digests,
                        metric_rows=len(rows), prediction_rows=len(sample_rows), control_rows=len(controls),
                        reference_input_checks=len(reference_checks))
    except BaseException as exc:
        manifest.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['changed_protected_files']=[p for p,h in hashes.items()
                                            if not Path(p).is_file() or digest(Path(p))!=h]
        manifest['protected_files_unchanged']=not manifest['changed_protected_files']
        manifest['data_file_unchanged']=data_path.is_file() and digest(data_path)==plan['data_sha256']
        if not manifest['protected_files_unchanged'] or not manifest['data_file_unchanged']:
            manifest['status']='integrity_failed'
        manifest['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        save_manifest()
    require(manifest['protected_files_unchanged'] and manifest['data_file_unchanged'], 'Korunan dosya degisti')
    print('ABCD_SEED_VALIDATION_TAMAMLANDI:',output)


if __name__ == '__main__':
    main()
