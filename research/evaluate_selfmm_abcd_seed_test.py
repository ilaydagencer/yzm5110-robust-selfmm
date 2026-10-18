"""A/B/C/D TEST evaluation for training seeds 1112-1115, frozen K0-K7 protocol.

Run one explicitly selected seed at a time. Training seed changes; corruption
seed=1111 and tau=1 stay fixed. Reuses the verified seed-1111 TEST input builder
and compares input fingerprints, never predictions across different seeds.
--check constructs no model/dataset and writes no files. No training or tuning.
"""
import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
import platform
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

TRAINING_SEEDS = (1112, 1113, 1114, 1115)
CORRUPTION_SEED = 1111
SPLIT = 'test'
N = 686
NON0_N = 656
TAU = 1.0
BATCH = 4
# Saved training metrics are rounded to four decimals. Allow one last-place
# unit plus floating-point tolerance. Loss is batch-weighted and NOT compared.
CLEAN_METRIC_TOLERANCE = 1.1e-4
CLEAN_METRICS = ('MAE', 'Non0_acc_2', 'Non0_F1_score', 'Corr')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load_helpers(repo, reference):
    global corruptions, weights_from_outputs, fusion_weights, write_csv, prepare_batch
    directory = repo / 'research'
    sys.path.insert(0, str(directory))
    modules = {}
    for name in ('evaluate_selfmm_abcd_valid', 'selfmm_corruptions',
                 'check_selfmm_corruptions', 'evaluate_selfmm_abcd_test'):
        path = (directory / f'{name}.py').resolve()
        require(path.is_file() and digest(path) == reference['protected_hashes'].get(str(path)),
                f'Dondurulmus yardimci kod degismis: {path.name}')
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve() == path, f'Yanlis yardimci modul: {name}')
        modules[name] = module
    base = modules['evaluate_selfmm_abcd_valid']
    test = modules['evaluate_selfmm_abcd_test']
    test.load_helpers(repo)
    corruptions = modules['selfmm_corruptions']
    weights_from_outputs, fusion_weights, write_csv = base.weights_from_outputs, base.fusion_weights, base.write_csv
    prepare_batch = test.prepare_batch
    require(test.SPLIT == SPLIT and test.CORRUPTION_SEED == CORRUPTION_SEED
            and test.BATCH == BATCH and test.N == N and test.NON0_N == NON0_N and test.TAU == TAU,
            'Seed 1111 test yardimcisinin sabitleri uyumsuz')
    return modules


def read_reference_labels(path):
    rows = read_csv(path)
    records = {}
    for row in rows:
        key = row['checkpoint'], row['condition'], int(row['index'])
        require(key not in records, 'Test referansinda tekrar var')
        label = float(row['label'])
        require(np.isfinite(label) and row['split'] == SPLIT
                and int(row['training_seed']) == 1111 and int(row['corruption_seed']) == CORRUPTION_SEED,
                'Test referansinin etiketi veya seed kaydi uyumsuz')
        records[key] = label
    expected = {(name, f'K{k}', i) for name in ('baseline', 'dropout') for k in range(8) for i in range(N)}
    require(set(records) == expected, 'Test referansi 10976 benzersiz satir icermeli')
    labels = {i: records['baseline', 'K0', i] for i in range(N)}
    require(sum(y != 0 for y in labels.values()) == NON0_N, 'Test Non0 sayisi 656 olmali')
    require(all(y == labels[k[2]] for k, y in records.items()), 'Referans test etiketleri eslesmiyor')
    return labels


def prepare(repo, seed, output_dir=None):
    require(seed in TRAINING_SEEDS, 'Egitim seed 1112-1115 olmali; seed 1111 korunuyor')
    repo = Path(repo).resolve()
    project = repo.parent
    raw_output = Path(output_dir) if output_dir else project/'runs'/f'selfmm_abcd_corrupt_test_seed{seed}_tau1_v1'
    require(not os.path.lexists(raw_output), f'Cikti zaten var; uzerine yazilmaz: {raw_output}')
    output = raw_output.resolve()
    require(output.parent == (project/'runs').resolve(), 'Cikti, runs altinda yeni bir dogrudan alt klasor olmali')
    require(Path(MMSA.__file__).resolve().is_relative_to(repo/'src'), 'Yanlis MMSA kurulumu')
    reference_path = project/'runs/selfmm_abcd_corrupt_test_seed1111_tau1_v1/manifest.json'
    protocol_path = project/'runs/selfmm_corruption_protocol_valid_seed1111_v1/manifest.json'
    stats_path = protocol_path.with_name('train_feature_stats.json')
    validation_path = project/'runs'/f'selfmm_abcd_corrupt_valid_seed{seed}_tau1_v1/manifest.json'
    reference, protocol, stats, valid = map(load_json,(reference_path,protocol_path,stats_path,validation_path))
    modules = load_helpers(repo, reference)
    require(reference['status'] == 'completed' and reference['split'] == SPLIT and reference['n'] == N
            and reference['non0_n'] == NON0_N and reference['training_seed'] == 1111
            and reference['corruption_seed'] == CORRUPTION_SEED and reference['corruption_rng_split'] == SPLIT
            and reference['batch_size'] == BATCH and reference['tau'] == TAU
            and reference['protocol_version'] == 1 and reference['conditions'] == list(corruptions.CONDITIONS),
            'Seed 1111 test referansi uyumsuz')
    for flag in ('protected_files_unchanged','data_file_unchanged','shared_corrupted_inputs',
                 'reference_test_labels_matched','prior_AB_K0_K3_test_reproduced',
                 'text_unchanged','lengths_unchanged','padding_unchanged'):
        require(reference.get(flag) is True, f'Test referans kontrolu eksik: {flag}')
    require(protocol['status'] == 'completed' and protocol['split'] == 'valid' and protocol['n'] == 229
            and protocol['protocol_version'] == 1 and protocol['seed'] == CORRUPTION_SEED
            and protocol['noise_factor'] == reference['noise_factor'] == corruptions.NOISE_FACTOR == 0.5
            and protocol['partial_rates'] == reference['partial_rates'] == corruptions.PARTIAL_RATES
            and corruptions.CONDITIONS == tuple(f'K{i}' for i in range(8)), 'Dondurulmus bozulma protokolu uyumsuz')
    for flag in ('text_unchanged','lengths_unchanged','padding_unchanged','source_unchanged',
                 'masks_nested','modality_rng_independent','model_order_independent'):
        require(protocol.get(flag) is True, f'Protokol kontrolu eksik: {flag}')
    require(stats['stats_source'] == reference['noise_statistics_source'] == 'train_only_valid_frames'
            and stats['ddof'] == 0 and stats['source_sha256'] == protocol['source_sha256'] == reference['data_sha256'],
            'Train-only gurultu istatistikleri uyumsuz')
    require(valid['status'] == 'completed' and valid['split'] == 'valid' and valid['n'] == 229
            and valid['training_seed'] == seed and valid['corruption_seed'] == CORRUPTION_SEED
            and valid['batch_size'] == BATCH and valid['tau'] == TAU and valid['protocol_version'] == 1
            and valid['conditions'] == list(corruptions.CONDITIONS) and valid['data_sha256'] == reference['data_sha256'],
            'Bu seed icin validation kaydi uyumsuz')
    for flag in ('protected_files_unchanged','data_file_unchanged','shared_corrupted_inputs',
                 'same_corruptions_as_seed1111','reference_labels_matched'):
        require(valid.get(flag) is True, f'Validation kontrolu eksik: {flag}')
    require(digest(protocol_path) == valid['protocol_manifest_sha256'] == reference['protocol_manifest_sha256']
            and digest(stats_path) == valid['train_stats_sha256'] == reference['train_stats_sha256'],
            'Protokol veya train std degismis')
    require(np.__version__ == valid['numpy_version'] == reference['numpy_version'] == protocol['numpy_version']
            and torch.__version__ == valid['torch_version'] == reference['torch_version'],
            'NumPy/Torch surumu farkli; proje venv kullanilmali')
    expected_inputs = reference['corrupted_input_sha256']
    require(set(expected_inputs) == set(corruptions.CONDITIONS)
            and all(isinstance(h,str) and len(h)==64 for h in expected_inputs.values()), 'Test girdi ozetleri eksik')
    protected = {Path(__file__).resolve(),reference_path,protocol_path,stats_path,validation_path}
    expected_hashes = {}
    def remember_hash(path, expected):
        path = Path(path).resolve()
        require(path not in expected_hashes or expected_hashes[path] == expected, f'Kaynak ozetleri celisiyor: {path}')
        expected_hashes[path] = expected
        protected.add(path)
    for manifest in (reference,valid):
        for filename,expected in manifest['protected_hashes'].items():remember_hash(filename,expected)
    expected_counts = {'metrics.csv':32,'comparisons.csv':32,'degradation.csv':32,
        'predictions_and_weights.csv':10976,'weight_summary.csv':16,'controls.csv':16,
        'reproduction_checks.csv':8,'corruption_summary.csv':16}
    require(reference['row_counts'] == expected_counts and set(reference['output_hashes']) == set(expected_counts),
            'Seed 1111 test ciktilari eksik')
    for filename,count in expected_counts.items():
        path = reference_path.parent/filename
        remember_hash(path,reference['output_hashes'][filename])
        require(len(read_csv(path)) == count, f'Test referans satir sayisi uyumsuz: {filename}')
    checks = read_csv(reference_path.with_name('controls.csv'))
    require({(r['checkpoint'],r['condition']) for r in checks}
            == {(m,c) for m in ('baseline','dropout') for c in corruptions.CONDITIONS}, 'Test girdi kontrolleri eksik')
    for row in checks:
        require(row['corrupted_input_sha256'] == expected_inputs[row['condition']]
                and int(row['n']) == N and float(row['max_uniform_error']) <= 1e-6
                and all(row[f] == 'True' for f in ('hooks_restored','text_unchanged','lengths_unchanged','padding_unchanged')),
                'Test girdi kontrolu basarisiz')
    labels = read_reference_labels(reference_path.with_name('predictions_and_weights.csv'))
    specs, arguments, configs, training_records, clean_metrics = {},{},{},{},{}
    for name,plain,weighted in (('baseline','A','C'),('dropout','B','D')):
        suffix = 'baseline' if name == 'baseline' else 'dropout_a20_v20'
        run = project/'runs'/f'selfmm_{suffix}_seed{seed}'
        record_path = run/'experiment.json'
        config = project/'local_configs'/f'selfmm_mosi_{suffix}_batch4_seed{seed}.json'
        checkpoint = run/'saved_models/self_mm-mosi.pth'
        record = load_json(record_path)
        require(record['status'] == 'completed' and record['variant'] == name
                and record['seed'] == seed and record['batch_size'] == BATCH
                and record['automatic_final_test'] is True and record['hyperparameter_tuning'] is False,
                f'Egitim kaydi uyumsuz: {name}')
        for flag in ('protected_files_unchanged','data_file_unchanged','local_config_unchanged','forward_restored'):
            require(record.get(flag) is True, f'Egitim kontrolu eksik: {name}/{flag}')
        require(Path(record['checkpoint_path']).resolve() == checkpoint.resolve()
                and Path(record['local_config']).resolve() == config.resolve(), 'Egitim config/checkpoint yolu farkli')
        remember_hash(checkpoint,record['checkpoint_sha256'])
        remember_hash(config,record['local_config_sha256'])
        vr = valid['training_records'][name]
        require(Path(vr['manifest']).resolve() == record_path.resolve()
                and vr['manifest_sha256'] == digest(record_path)
                and vr['checkpoint_sha256'] == record['checkpoint_sha256'], 'Validation ve egitim checkpoint kayitlari farkli')
        require(record['data_sha256'] == protocol['source_sha256'] and record['torch_version'] == torch.__version__
                and record['packages']['numpy'] == np.__version__, 'Egitim veri/surum kaydi uyumsuz')
        require(record['dropout']['enabled'] is (name == 'dropout'), 'Dropout yontemi uyumsuz')
        if name == 'dropout':
            require(record['dropout']['probability'] == 0.2 and record['dropout']['generator_seed'] == seed
                    and record['dropout']['text_unchanged'] is True and record['dropout_counts']['eval_dropped'] == 0,
                    'Train-only dropout protokolu uyumsuz')
        for filename,expected in record['protected_hashes'].items():remember_hash(filename,expected)
        configs[name] = load_json(config)
        args = get_config_regression('self_mm','mosi',str(config))
        require(dict(args) == record['resolved_model_config'], 'Cozumlenmis egitim ayarlari degismis')
        require(args.batch_size == BATCH and args.use_bert and args.use_finetune
                and not args.need_data_aligned and not args.need_model_aligned
                and not args.get('data_missing',False) and args.post_fusion_dropout == 0
                and args.KeyEval == 'Loss' and (args.text_out,args.audio_out,args.video_out) == (768,16,32),
                'Self-MM ayarlari uyumsuz')
        for modality,dimension in (('audio',args.feature_dims[1]),('vision',args.feature_dims[2])):
            scale = np.asarray(stats['modalities'][modality]['std'])
            require(scale.shape == (dimension,) and np.isfinite(scale).all() and (scale >= 0).all(), 'Train std uyumsuz')
        args.update(train_mode='regression',device=torch.device('cuda:0'),custom_feature=None,
                    feature_T=None,feature_A=None,feature_V=None,data_missing=False)
        specs[name] = (config,checkpoint,plain,weighted)
        arguments[name] = args
        clean_metrics[name] = {k:float(record['final_test_metrics'][k]) for k in CLEAN_METRICS}
        require(all(np.isfinite(v) for v in clean_metrics[name].values()), 'Kayitli temiz test metrigi gecersiz')
        training_records[name] = dict(manifest=str(record_path),manifest_sha256=digest(record_path),
            checkpoint_sha256=record['checkpoint_sha256'],git_head=record['git_head'])
        protected.update((record_path,config,checkpoint))
        print(f'{plain}/{weighted} CHECKPOINT: {checkpoint}')
    require(configs['baseline'] == configs['dropout'] == load_json(project/'local_configs/selfmm_mosi_batch4.json'),
            'Eslenmis configler farkli')
    data_path = Path(arguments['baseline'].featurePath).resolve()
    require(data_path == Path(arguments['dropout'].featurePath).resolve()
            == (project/'data/MOSI/Processed/unaligned_50.pkl').resolve() and data_path.is_file(), 'MOSI veri yolu uyumsuz')
    protected.update(p for p in (repo/'src/MMSA').rglob('*') if p.is_file() and p.suffix in ('.py','.json'))
    hashes = {str(p):digest(p) for p in sorted(protected,key=str)}
    for path,expected in expected_hashes.items():
        require(hashes[str(path)] == expected, f'Korunan kaynak dosyasi degismis: {path}')
    modules['evaluate_selfmm_abcd_valid'].self_check()
    modules['check_selfmm_corruptions'].synthetic_check()
    modules['evaluate_selfmm_abcd_test'].test_split_check()
    print('MMSA:',MMSA.__file__)
    print(f'SPLIT: test; N: {N}; Non0: {NON0_N}; BATCH: {BATCH}; TAU: {TAU}; EGITIM SEED: {seed}; BOZULMA SEED: {CORRUPTION_SEED}')
    print('REFERANS: seed 1111 test girdileri; KOSULLAR: K0-K7; EGITIM/TUNING: yok; CUDA:',torch.cuda.is_available())
    print('CIKTI:',output)
    return dict(seed=seed,output=output,specs=specs,arguments=arguments,stats=stats,
        labels=labels,expected_inputs=expected_inputs,clean_metrics=clean_metrics,training_records=training_records,
        hashes=hashes,data_path=data_path,data_sha256=protocol['source_sha256'],
        protocol_manifest_sha256=digest(protocol_path),train_stats_sha256=digest(stats_path),
        validation_manifest_sha256=digest(validation_path),reference_test_manifest_sha256=digest(reference_path))


def evaluate(plan):
    seed = plan['seed']
    output, data_path, hashes = plan['output'], plan['data_path'], plan['hashes']
    require(not os.path.lexists(output), f'Cikti zaten var; uzerine yazilmaz: {output}')
    require(torch.cuda.is_available(), 'CUDA bulunamadi')
    require(digest(data_path) == plan['data_sha256'], 'Veri dosyasi protokolden farkli')
    require(all(Path(p).is_file() and digest(p) == h for p,h in hashes.items()), 'Kontrolden sonra korunan dosya degisti')
    setup_seed(seed)
    torch.cuda.set_device(0)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running',created_at_utc=datetime.now(timezone.utc).isoformat(),
        split=SPLIT,n=N,non0_n=NON0_N,batch_size=BATCH,training_seed=seed,corruption_seed=CORRUPTION_SEED,
        tau=TAU,conditions=list(corruptions.CONDITIONS),protocol_version=1,
        partial_rates=corruptions.PARTIAL_RATES,noise_factor=corruptions.NOISE_FACTOR,
        corruption_rng_split=SPLIT,noise_statistics_source='train_only_valid_frames',
        protocol_manifest_sha256=plan['protocol_manifest_sha256'],train_stats_sha256=plan['train_stats_sha256'],
        validation_manifest_sha256=plan['validation_manifest_sha256'],
        reference_test_seed=1111,reference_test_manifest_sha256=plan['reference_test_manifest_sha256'],
        training_records=plan['training_records'],clean_metric_tolerance=CLEAN_METRIC_TOLERANCE,
        data_sha256=plan['data_sha256'],protected_hashes=hashes,
        checkpoint_paths={name:str(spec[1]) for name,spec in plan['specs'].items()},
        python_version=platform.python_version(),torch_version=torch.__version__,numpy_version=np.__version__,
        gpu=torch.cuda.get_device_name(0),inference_only=True,training_performed=False,
        hyperparameter_tuning=False,test_evaluation=True,
        weighting='3*softmax(-abs(y_m-mean(y_other))/tau)',
        selection='Frozen validation protocol. Report all four methods and eight conditions; no test-based tuning.',
        test_history='Seed-1111 full K0-K7 test and automatic training-end clean test were already observed. No test-based tuning.',
        data_loading_note='MMSA reads a pickle containing all splits; only mode=test is instantiated/evaluated. Train noise scales are read from saved JSON.',
        note='Three forward passes include an all-one identity control; elapsed execution is not an inference latency benchmark.')
    def save_manifest():
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save_manifest()
    rows=[]; samples=[]; weights_rows=[]; controls=[]; reference_checks=[]; clean_checks=[]; corruption_rows=[]
    input_digests={}
    try:
        dataset=MMDataset(copy.deepcopy(plan['arguments']['baseline']),mode=SPLIT)
        require(len(dataset)==N,'Test ornek sayisi 686 olmali')
        loader=DataLoader(dataset,batch_size=BATCH,shuffle=False,num_workers=0,drop_last=False)
        metric_fn=MetricsTop('regression').getMetics('mosi')
        for name,(_,checkpoint,plain,weighted) in plan['specs'].items():
            setup_seed(seed)
            args=copy.deepcopy(plan['arguments'][name])
            model=AMIO(args)
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True)
            del state
            model.to(args.device).eval().requires_grad_(False)
            layer=model.Model.post_fusion_layer_1
            dims=(args.text_out,args.audio_out,args.video_out)
            for condition in corruptions.CONDITIONS:
                predictions=[]; weighted_predictions=[]; truths=[]; all_weights=[]; indices=[]
                fingerprint=hashlib.sha256();totals={}
                identity_error=0.
                hooks_before=len(layer._forward_pre_hooks)
                with torch.inference_mode():
                    for batch in loader:
                        require(all(not m.training for m in model.modules())
                                and all(not param.requires_grad for param in model.parameters()), 'Model dondurulmamis')
                        text_saved=batch['text'].clone()
                        lengths_saved={m:batch[m+'_lengths'].clone() for m in ('audio','vision')}
                        features=prepare_batch(batch,condition,plan['stats'],fingerprint,args.device,totals)
                        text=batch['text'].to(args.device)
                        def forward():
                            return model(text,(features['audio'],batch['audio_lengths']),
                                         (features['vision'],batch['vision_lengths']))
                        base=forward()
                        w,deviations=weights_from_outputs(base,TAU)
                        with fusion_weights(layer,torch.ones_like(w),dims): uniform=forward()
                        for key in ('M','T','A','V'):
                            identity_error=max(identity_error,float((uniform[key]-base[key]).abs().max()))
                            require(torch.allclose(uniform[key],base[key],atol=1e-6,rtol=1e-6),'Esit agirlik identity kontrolu basarisiz')
                        with fusion_weights(layer,w,dims): changed=forward()
                        require(len(layer._forward_pre_hooks)==hooks_before,'Hook kaldirilmadi')
                        for key in ('T','A','V'):
                            require(torch.allclose(base[key],changed[key],atol=1e-6,rtol=1e-6),'Tek modlu baslik degisti')
                        require(torch.equal(batch['text'],text_saved)
                                and all(torch.equal(batch[m+'_lengths'],lengths_saved[m]) for m in lengths_saved),
                                'Metin veya uzunluk degisti')
                        pred=base['M'].reshape(-1).cpu();changed_pred=changed['M'].reshape(-1).cpu()
                        labels=batch['labels']['M'].reshape(-1).cpu()
                        require(all(bool(torch.isfinite(t).all()) for t in (pred,changed_pred,labels)),'Gecersiz tahmin/etiket')
                        batch_indices=batch['index'].tolist()
                        for j,index in enumerate(batch_indices):
                            require(abs(float(labels[j])-plan['labels'][index])<1e-7,'Test etiket/sira eslesmedi')
                            samples.append(dict(training_seed=seed,corruption_seed=CORRUPTION_SEED,split=SPLIT,
                                checkpoint=name,condition=condition,index=index,label=float(labels[j]),
                                plain_method=plain,weighted_method=weighted,plain_prediction=float(pred[j]),
                                weighted_prediction=float(changed_pred[j]),y_text=float(base['T'][j].item()),
                                y_audio=float(base['A'][j].item()),y_vision=float(base['V'][j].item()),
                                w_text=float(w[j,0]),w_audio=float(w[j,1]),w_vision=float(w[j,2]),
                                deviation_text=float(deviations[j,0]),deviation_audio=float(deviations[j,1]),
                                deviation_vision=float(deviations[j,2])))
                        predictions.append(pred);weighted_predictions.append(changed_pred)
                        truths.append(labels);all_weights.append(w.cpu());indices.extend(batch_indices)
                require(indices==list(range(N)),'Test sirasinda eksik/tekrar ornek')
                checksum=fingerprint.hexdigest()
                if name=='baseline':input_digests[condition]=checksum
                require(checksum==input_digests[condition],'Iki checkpoint ayni bozulmus girdileri almadi')
                require(checksum==plan['expected_inputs'][condition],
                        f'Seed 1111 test girdileri ile ayni degil: {name}/{condition}')
                reference_checks.append(dict(checkpoint=name,condition=condition,n=N,reference_training_seed=1111,
                    reference_input_sha256=plan['expected_inputs'][condition],actual_input_sha256=checksum,passed=True))
                pred=torch.cat(predictions);changed_pred=torch.cat(weighted_predictions)
                labels=torch.cat(truths);w=torch.cat(all_weights)
                require(int((labels!=0).sum())==NON0_N,'Test Non0 sayisi degisti')
                for method,values in ((plain,pred),(weighted,changed_pred)):
                    metric={k:float(v) for k,v in metric_fn(values,labels).items()}
                    require(all(np.isfinite(v) for v in metric.values()),'Sonlu olmayan metrik')
                    row=dict(training_seed=seed,corruption_seed=CORRUPTION_SEED,method=method,
                        checkpoint=name,split=SPLIT,condition=condition,tau=TAU,n=N,non0_n=NON0_N,
                        **metric,MAE_exact=float((values-labels).abs().mean()))
                    if method==plain and condition=='K0':
                        for key in CLEAN_METRICS:
                            expected=plan['clean_metrics'][name][key]
                            error=abs(metric[key]-expected)
                            require(error<=CLEAN_METRIC_TOLERANCE,
                                f'Egitim sonu K0 test metrigi eslesmiyor: {name}/{key}; yeni={metric[key]}, kayit={expected}')
                            clean_checks.append(dict(checkpoint=name,method=method,condition='K0',metric=key,
                                recorded_training_test=expected,current_test=metric[key],absolute_error=error,
                                tolerance=CLEAN_METRIC_TOLERANCE,passed=True))
                    rows.append(row);print(json.dumps(row),flush=True)
                entry=dict(method=weighted,condition=condition,tau=TAU,n=N)
                for i,modality in enumerate(('text','audio','vision')):
                    entry.update({f'w_{modality}_mean':float(w[:,i].mean()),f'w_{modality}_std':float(w[:,i].std(unbiased=False)),
                                  f'w_{modality}_min':float(w[:,i].min()),f'w_{modality}_max':float(w[:,i].max())})
                weights_rows.append(entry)
                controls.append(dict(checkpoint=name,condition=condition,n=N,max_uniform_error=identity_error,
                    hooks_restored=True,text_unchanged=True,lengths_unchanged=True,padding_unchanged=True,
                    corrupted_input_sha256=checksum))
                if name=='baseline':
                    for modality,t in totals.items():
                        full=(modality=='audio' and condition in ('K1','K3')) or (modality=='vision' and condition in ('K2','K3'))
                        corruption_rows.append(dict(modality=modality,condition=condition,n=N,
                            valid_frames=t['valid_frames'],masked_frames=t['masked_frames'],
                            masked_rate=t['masked_frames']/t['valid_frames'],
                            target_mask_rate=corruptions.PARTIAL_RATES.get(condition,1. if full else 0.),
                            noise_rms_div_train_std=float(np.sqrt(t['z_square']/t['z_count'])) if t['z_count'] else 0.))
            del model
            torch.cuda.empty_cache()
        require(len(rows)==32 and len(samples)==10976 and len(controls)==16
                and len(reference_checks)==16 and len(clean_checks)==8 and len(corruption_rows)==16,'Cikti satir sayisi uyumsuz')
        comparisons=[];degradation=[]
        lookup={(r['method'],r['condition']):r for r in rows}
        for condition in corruptions.CONDITIONS:
            for first,second in (('A','B'),('A','C'),('B','D'),('A','D')):
                a,b=lookup[first,condition],lookup[second,condition]
                comparisons.append(dict(condition=condition,comparison=f'{second}-minus-{first}',
                    MAE_exact_delta=b['MAE_exact']-a['MAE_exact'],Non0_acc_delta=b['Non0_acc_2']-a['Non0_acc_2'],
                    Non0_F1_delta=b['Non0_F1_score']-a['Non0_F1_score'],Corr_delta=b['Corr']-a['Corr']))
        for method in 'ABCD':
            clean=lookup[method,'K0']
            for condition in corruptions.CONDITIONS:
                row=lookup[method,condition]
                degradation.append(dict(method=method,condition=condition,MAE_increase=row['MAE_exact']-clean['MAE_exact'],
                    Non0_acc_drop=clean['Non0_acc_2']-row['Non0_acc_2'],Non0_F1_drop=clean['Non0_F1_score']-row['Non0_F1_score'],
                    Corr_drop=clean['Corr']-row['Corr']))
        tables={'metrics.csv':rows,'comparisons.csv':comparisons,'degradation.csv':degradation,
                'predictions_and_weights.csv':samples,'weight_summary.csv':weights_rows,'controls.csv':controls,
                'reference_input_checks.csv':reference_checks,'clean_test_checks.csv':clean_checks,
                'corruption_summary.csv':corruption_rows}
        for filename,records in tables.items():write_csv(output/filename,records)
        manifest.update(status='completed',shared_corrupted_inputs=True,reference_test_labels_matched=True,
            same_corruptions_as_seed1111=True,clean_K0_training_metrics_reproduced=True,text_unchanged=True,lengths_unchanged=True,padding_unchanged=True,
            corrupted_input_sha256=input_digests,metric_rows=len(rows),prediction_rows=len(samples),
            control_rows=len(controls),reference_input_checks=len(reference_checks),
            clean_test_check_rows=len(clean_checks),corruption_summary_rows=len(corruption_rows),
            row_counts={k:len(v) for k,v in tables.items()},output_hashes={name:digest(output/name) for name in tables})
    except BaseException as exc:
        manifest.update(status='failed_or_interrupted',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['changed_protected_files']=[p for p,h in hashes.items() if not Path(p).is_file() or digest(p)!=h]
        manifest['protected_files_unchanged']=not manifest['changed_protected_files']
        manifest['data_file_unchanged']=data_path.is_file() and digest(data_path)==plan['data_sha256']
        if not manifest['protected_files_unchanged'] or not manifest['data_file_unchanged']:
            manifest['status']='integrity_failed'
        manifest['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        save_manifest()
    require(manifest['status']=='completed','Degerlendirme/butunluk kontrolu tamamlanamadi')
    print('ABCD_SEED_TEST_TAMAMLANDI:',output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True,choices=TRAINING_SEEDS)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--repo-dir',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir',type=Path,help='Optional new direct child of project/runs; never overwritten.')
    cli=parser.parse_args()
    plan=prepare(cli.repo_dir,cli.seed,cli.output_dir)
    if cli.check:
        print('KONTROL_BASARILI: model/veri yuklenmedi; dosya yazilmadi; egitim, validation ve seed 1111 test referansi dogrulandi')
        return
    evaluate(plan)


if __name__=='__main__':
    main()
