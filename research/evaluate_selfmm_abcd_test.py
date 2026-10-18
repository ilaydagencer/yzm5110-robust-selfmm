"""Seed 1111 A/B/C/D test evaluation, K0-K7, frozen tau=1 protocol.

No training or tuning. Reuses the verified validation helpers and TRAIN-only
noise scales. Test corruptions use split='test'; validation input fingerprints
are deliberately NOT reused. Prior A/B K0-K3 TEST predictions are regression
controls, not selection criteria. --check loads no model/dataset and writes
no files. Copy beside the research helpers, then run with the project venv.
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

SEED = 1111
CORRUPTION_SEED = 1111
SPLIT = 'test'
N = 686
NON0_N = 656
TAU = 1.0
BATCH = 4
TOLERANCE = 1e-5
DEFAULT_OUTPUT = 'selfmm_abcd_corrupt_test_seed1111_tau1_v1'


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


def load_helpers(repo):
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
    weights_from_outputs, fusion_weights, write_csv = base.weights_from_outputs, base.fusion_weights, base.write_csv
    return modules


def read_test_reference(path):
    reference = {}
    for row in read_csv(path):
        key = row['model'], row['condition'], int(row['index'])
        require(key not in reference, 'Eski test kaydinda tekrar var')
        value = {k: float(row[k]) for k in ('label', 'prediction')}
        require(all(np.isfinite(v) for v in value.values()), 'Eski test kaydinda gecersiz sayi')
        reference[key] = value
    expected = {(name, f'K{k}', i) for name in ('baseline', 'dropout') for k in range(4) for i in range(N)}
    require(set(reference) == expected, 'Eski test kaydi 5488 benzersiz tahmin icermeli')
    labels = {i: reference['baseline', 'K0', i]['label'] for i in range(N)}
    require(sum(y != 0 for y in labels.values()) == NON0_N, 'Test Non0 sayisi 656 olmali')
    require(all(r['label'] == labels[k[2]] for k, r in reference.items()), 'Eski test etiketleri eslesmiyor')
    return reference, labels


def prepare(repo, output_dir=None):
    repo = Path(repo).resolve()
    project = repo.parent
    raw_output = Path(output_dir) if output_dir else project / 'runs' / DEFAULT_OUTPUT
    require(not os.path.lexists(raw_output), f'Cikti zaten var; uzerine yazilmaz: {raw_output}')
    output = raw_output.resolve()
    require(output.parent == (project / 'runs').resolve(), 'Cikti, runs altinda yeni bir dogrudan alt klasor olmali')
    require(Path(MMSA.__file__).resolve().is_relative_to(repo / 'src'), 'Yanlis MMSA kurulumu')
    modules = load_helpers(repo)
    protocol_path = project / 'runs/selfmm_corruption_protocol_valid_seed1111_v1/manifest.json'
    stats_path = protocol_path.with_name('train_feature_stats.json')
    validation_path = project / 'runs/selfmm_abcd_corrupt_valid_seed1111_tau1_v1/manifest.json'
    old_test_path = project / 'runs/selfmm_checkpoint_comparison_test_seed1111/manifest.json'
    old_predictions_path = old_test_path.with_name('predictions.csv')
    protocol, stats, valid, old = map(load_json, (protocol_path, stats_path, validation_path, old_test_path))
    require(protocol['status'] == 'completed' and protocol['split'] == 'valid'
            and protocol['protocol_version'] == 1 and protocol['n'] == 229
            and protocol['seed'] == CORRUPTION_SEED and protocol['noise_factor'] == 0.5
            and protocol['partial_rates'] == corruptions.PARTIAL_RATES, 'Dondurulmus protokol uyumsuz')
    require(corruptions.CONDITIONS == tuple(f'K{i}' for i in range(8)) and corruptions.NOISE_FACTOR == 0.5,
            'Bozulma modulu uyumsuz')
    for flag in ('text_unchanged', 'lengths_unchanged', 'padding_unchanged', 'source_unchanged',
                 'masks_nested', 'modality_rng_independent', 'model_order_independent'):
        require(protocol.get(flag) is True, f'Protokol kontrolu eksik: {flag}')
    require(stats['stats_source'] == 'train_only_valid_frames' and stats['ddof'] == 0
            and stats['source_sha256'] == protocol['source_sha256'], 'Train-only istatistik kaydi uyumsuz')
    require(valid['status'] == 'completed' and valid['split'] == 'valid' and valid['n'] == 229
            and valid['batch_size'] == BATCH and valid['training_seed'] == SEED
            and valid['corruption_seed'] == CORRUPTION_SEED and valid['tau'] == TAU
            and valid['protocol_version'] == 1 and valid['conditions'] == list(corruptions.CONDITIONS)
            and valid['data_sha256'] == protocol['source_sha256'], 'Validation protokolu uyumsuz')
    for flag in ('protected_files_unchanged', 'data_file_unchanged', 'shared_corrupted_inputs', 'prior_K0_K3_reproduced'):
        require(valid.get(flag) is True, f'Validation kontrolu eksik: {flag}')
    require(digest(protocol_path) == valid['protocol_manifest_sha256']
            and digest(stats_path) == valid['train_stats_sha256'], 'Protokol veya train std degismis')
    require(np.__version__ == valid['numpy_version'] == protocol['numpy_version'], 'NumPy surumu farkli; proje venv kullanilmali')
    require(old['status'] == 'completed' and old['split'] == SPLIT and old['seed'] == SEED
            and old['n'] == N and old['batch_size'] == BATCH and old['protected_files_unchanged'] is True
            and old['conditions'] == {'K0':[False,False], 'K1':[True,False], 'K2':[False,True], 'K3':[True,True]},
            'Eski K0-K3 test kaydi uyumsuz')
    require(torch.__version__ == old['torch_version'] == valid['torch_version'], 'Torch surumu farkli; proje venv kullanilmali')

    protected = {Path(__file__).resolve(), protocol_path, stats_path, validation_path, old_test_path, old_predictions_path}
    expected_hashes = {}
    for manifest in (valid, old):
        for filename, expected in manifest['protected_hashes'].items():
            path = Path(filename).resolve()
            require(path not in expected_hashes or expected_hashes[path] == expected, 'Referans hashleri celisiyor')
            expected_hashes[path] = expected
            protected.add(path)
    for path, expected in expected_hashes.items():
        require(path.is_file() and digest(path) == expected, f'Korunan referans dosyasi degismis: {path}')
    for module in modules.values():
        path = Path(module.__file__).resolve()
        require(path in expected_hashes, f'Yardimci kod referansta yok: {path.name}')
        if path.name in protocol['code_hashes']:
            require(digest(path) == protocol['code_hashes'][path.name], 'Bozulma kodu degismis')
        protected.add(path)
    # Also guard local framework sources during this run. These are current
    # source hashes, not an assertion that all were recorded at seed-1111 training.
    protected.update(p for p in (repo / 'src/MMSA').rglob('*') if p.is_file() and p.suffix in ('.py', '.json'))
    specs = {
        'baseline': (project/'local_configs/selfmm_mosi_batch4.json',
                     project/'runs/selfmm_seed1111/saved_models/self_mm-mosi.pth', 'A', 'C'),
        'dropout': (project/'local_configs/selfmm_mosi_dropout_a20_v20_batch4_seed1111.json',
                    project/'runs/selfmm_dropout_a20_v20_seed1111/saved_models/self_mm-mosi.pth', 'B', 'D')}
    arguments, configs = {}, {}
    for name, (config, checkpoint, plain, weighted) in specs.items():
        require(config.resolve() in expected_hashes and checkpoint.resolve() in expected_hashes, 'Config/checkpoint referansi eksik')
        configs[name] = load_json(config)
        args = get_config_regression('self_mm', 'mosi', str(config))
        require(args.batch_size == BATCH and args.use_bert and args.use_finetune
                and not args.need_data_aligned and not args.need_model_aligned
                and not args.get('data_missing', False) and args.post_fusion_dropout == 0
                and args.KeyEval == 'Loss' and (args.text_out, args.audio_out, args.video_out) == (768,16,32),
                'Self-MM ayarlari uyumsuz')
        for modality, dimension in (('audio', args.feature_dims[1]), ('vision', args.feature_dims[2])):
            scale = np.asarray(stats['modalities'][modality]['std'])
            require(scale.shape == (dimension,) and np.isfinite(scale).all() and (scale >= 0).all(), 'Train std uyumsuz')
        args.update(train_mode='regression', device=torch.device('cuda:0'), custom_feature=None,
                    feature_T=None, feature_A=None, feature_V=None, data_missing=False)
        arguments[name] = args
        print(f'{plain}/{weighted} CHECKPOINT: {checkpoint}')
    require(configs['baseline'] == configs['dropout'], 'A/B eslenmis configler farkli')
    data_path = Path(arguments['baseline'].featurePath).resolve()
    require(data_path == Path(arguments['dropout'].featurePath).resolve()
            == (project/'data/MOSI/Processed/unaligned_50.pkl').resolve() and data_path.is_file(), 'MOSI veri yolu uyumsuz')
    reference, labels = read_test_reference(old_predictions_path)
    modules['evaluate_selfmm_abcd_valid'].self_check()
    modules['check_selfmm_corruptions'].synthetic_check()
    test_split_check()
    hashes = {str(p): digest(p) for p in sorted(protected, key=str)}
    print('MMSA:', MMSA.__file__)
    print(f'SPLIT: test; N: {N}; Non0: {NON0_N}; BATCH: {BATCH}; TAU: {TAU}; EGITIM/BOZULMA SEED: 1111/1111')
    print('KOSULLAR: K0-K7; EGITIM/TUNING: yok; CUDA:', torch.cuda.is_available())
    print('CIKTI:', output)
    return dict(output=output, specs=specs, arguments=arguments, stats=stats,
                reference=reference, labels=labels, hashes=hashes, data_path=data_path,
                data_sha256=protocol['source_sha256'], protocol_manifest_sha256=digest(protocol_path),
                train_stats_sha256=digest(stats_path), validation_manifest_sha256=digest(validation_path),
                prior_test_manifest_sha256=digest(old_test_path), prior_test_predictions_sha256=digest(old_predictions_path))


def test_split_check():
    # Synthetic arrays only; no real dataset or model is constructed.
    x = np.arange(240, dtype=np.float32).reshape(80, 3)
    scales = [0., 1., 2.]
    masks = {}
    for modality in ('audio', 'vision'):
        for condition in corruptions.CONDITIONS:
            a, mask = corruptions.corrupt_modality(x, 67, modality, condition, scales, 17, split=SPLIT, seed=CORRUPTION_SEED)
            b, mask2 = corruptions.corrupt_modality(x, 67, modality, condition, scales, 17, split=SPLIT, seed=CORRUPTION_SEED)
            require(np.array_equal(a, b) and np.array_equal(mask, mask2), 'Test bozulmasi tekrarlanamadi')
            require(np.array_equal(a[67:], x[67:]) and not mask[67:].any(), 'Test padding degisti')
            if condition in corruptions.PARTIAL_RATES: masks[modality, condition] = mask
            if condition == 'K7':
                valid, _ = corruptions.corrupt_modality(x,67,modality,condition,scales,17,split='valid',seed=CORRUPTION_SEED)
                require(not np.array_equal(a, valid), 'Test ve valid RNG alanlari ayrilmadi')
                require(np.array_equal(a[:,0],x[:,0]), 'Sifir std ozelligi degisti')
        require((~masks[modality,'K4'] | masks[modality,'K5']).all()
                and (~masks[modality,'K5'] | masks[modality,'K6']).all(), 'Test maskeleri ic ice degil')
    require(np.array_equal(x, np.arange(240,dtype=np.float32).reshape(80,3)), 'Sentetik kaynak degisti')
    print('TEST_SPLIT_KONTROLU_BASARILI: test RNG, tekrar, ic ice maskeler, padding, sifir std')


def prepare_batch(batch, condition, stats, fingerprint, device, totals):
    features = {}
    for modality in ('audio', 'vision'):
        original = batch[modality].numpy()
        saved = original.copy()
        altered = []
        scale = np.asarray(stats['modalities'][modality]['std'], dtype=np.float64)
        summary = totals.setdefault(modality, dict(valid_frames=0, masked_frames=0, z_square=0., z_count=0))
        for j, index in enumerate(batch['index'].tolist()):
            length = int(batch[modality+'_lengths'][j])
            out, mask = corruptions.corrupt_modality(original[j],length,modality,condition,scale,index,
                                                    split=SPLIT,seed=CORRUPTION_SEED)
            require(np.array_equal(out[length:],original[j,length:],equal_nan=True), 'Padding degisti')
            summary['valid_frames'] += length
            summary['masked_frames'] += int(mask[:length].sum())
            if condition == 'K7' and (scale > 0).any():
                positive = scale > 0
                z = (out[:length,positive].astype(np.float64)-original[j,:length][:,positive].astype(np.float64)) / scale[positive]
                summary['z_square'] += float((z*z).sum())
                summary['z_count'] += z.size
            altered.append(out)
        require(np.array_equal(original,saved,equal_nan=True), 'Kaynak batch yerinde degistirildi')
        values = np.stack(altered)
        fingerprint.update(values.tobytes())
        fingerprint.update(batch[modality+'_lengths'].numpy().tobytes())
        features[modality] = torch.from_numpy(values).to(device)
    fingerprint.update(batch['text'].numpy().tobytes())
    fingerprint.update(batch['index'].numpy().tobytes())
    return features


def evaluate(plan):
    output, data_path, hashes = plan['output'], plan['data_path'], plan['hashes']
    require(not os.path.lexists(output), f'Cikti zaten var; uzerine yazilmaz: {output}')
    require(torch.cuda.is_available(), 'CUDA bulunamadi')
    require(digest(data_path) == plan['data_sha256'], 'Veri dosyasi protokolden farkli')
    require(all(Path(p).is_file() and digest(p) == h for p,h in hashes.items()), 'Kontrolden sonra korunan dosya degisti')
    setup_seed(SEED)
    torch.cuda.set_device(0)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running',created_at_utc=datetime.now(timezone.utc).isoformat(),
        split=SPLIT,n=N,non0_n=NON0_N,batch_size=BATCH,training_seed=SEED,corruption_seed=CORRUPTION_SEED,
        tau=TAU,conditions=list(corruptions.CONDITIONS),protocol_version=1,
        partial_rates=corruptions.PARTIAL_RATES,noise_factor=corruptions.NOISE_FACTOR,
        corruption_rng_split=SPLIT,noise_statistics_source='train_only_valid_frames',
        protocol_manifest_sha256=plan['protocol_manifest_sha256'],train_stats_sha256=plan['train_stats_sha256'],
        validation_manifest_sha256=plan['validation_manifest_sha256'],
        prior_test_manifest_sha256=plan['prior_test_manifest_sha256'],
        prior_test_predictions_sha256=plan['prior_test_predictions_sha256'],
        data_sha256=plan['data_sha256'],protected_hashes=hashes,
        checkpoint_paths={name:str(spec[1]) for name,spec in plan['specs'].items()},
        python_version=platform.python_version(),torch_version=torch.__version__,numpy_version=np.__version__,
        gpu=torch.cuda.get_device_name(0),inference_only=True,training_performed=False,
        hyperparameter_tuning=False,test_evaluation=True,
        weighting='3*softmax(-abs(y_m-mean(y_other))/tau)',
        selection='Frozen validation protocol. Report all four methods and eight conditions; no test-based tuning.',
        test_history='Seed-1111 A/B K0-K3 test and automatic training-end test were already observed.',
        data_loading_note='MMSA reads a pickle containing all splits; only mode=test is instantiated/evaluated. Train noise scales are read from saved JSON.',
        note='Three forward passes include an all-one identity control; elapsed execution is not an inference latency benchmark.')
    def save_manifest():
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save_manifest()
    rows=[]; samples=[]; weights_rows=[]; controls=[]; reproduction=[]; corruption_rows=[]
    input_digests={}
    try:
        dataset=MMDataset(copy.deepcopy(plan['arguments']['baseline']),mode=SPLIT)
        require(len(dataset)==N,'Test ornek sayisi 686 olmali')
        loader=DataLoader(dataset,batch_size=BATCH,shuffle=False,num_workers=0,drop_last=False)
        metric_fn=MetricsTop('regression').getMetics('mosi')
        for name,(_,checkpoint,plain,weighted) in plan['specs'].items():
            setup_seed(SEED)
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
                identity_error=0.; prior_error=0.
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
                            if condition in ('K0','K1','K2','K3'):
                                prior_error=max(prior_error,abs(float(pred[j])-plan['reference'][name,condition,index]['prediction']))
                            samples.append(dict(training_seed=SEED,corruption_seed=CORRUPTION_SEED,split=SPLIT,
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
                if condition in ('K0','K1','K2','K3'):
                    require(prior_error<=TOLERANCE,f'Eski A/B test tahminleri eslesmiyor: {name}/{condition}/{prior_error}')
                    reproduction.append(dict(checkpoint=name,method=plain,condition=condition,n=N,
                        max_prediction_error=prior_error,tolerance=TOLERANCE,passed=True))
                pred=torch.cat(predictions);changed_pred=torch.cat(weighted_predictions)
                labels=torch.cat(truths);w=torch.cat(all_weights)
                require(int((labels!=0).sum())==NON0_N,'Test Non0 sayisi degisti')
                for method,values in ((plain,pred),(weighted,changed_pred)):
                    metric={k:float(v) for k,v in metric_fn(values,labels).items()}
                    require(all(np.isfinite(v) for v in metric.values()),'Sonlu olmayan metrik')
                    row=dict(training_seed=SEED,corruption_seed=CORRUPTION_SEED,method=method,
                        checkpoint=name,split=SPLIT,condition=condition,tau=TAU,n=N,non0_n=NON0_N,
                        **metric,MAE_exact=float((values-labels).abs().mean()))
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
                and len(reproduction)==8 and len(corruption_rows)==16,'Cikti satir sayisi uyumsuz')
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
                'reproduction_checks.csv':reproduction,'corruption_summary.csv':corruption_rows}
        for filename,records in tables.items():write_csv(output/filename,records)
        manifest.update(status='completed',shared_corrupted_inputs=True,reference_test_labels_matched=True,
            prior_AB_K0_K3_test_reproduced=True,text_unchanged=True,lengths_unchanged=True,padding_unchanged=True,
            corrupted_input_sha256=input_digests,metric_rows=len(rows),prediction_rows=len(samples),
            control_rows=len(controls),reproduction_check_rows=len(reproduction),corruption_summary_rows=len(corruption_rows),
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
    print('ABCD_TEST_TAMAMLANDI:',output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--repo-dir',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir',type=Path,help='Optional new direct child of project/runs; never overwritten.')
    cli=parser.parse_args()
    plan=prepare(cli.repo_dir,cli.output_dir)
    if cli.check:
        print('KONTROL_BASARILI: model/veri yuklenmedi; dosya yazilmadi; protokol, checkpoint ve eski test kayitlari dogrulandi')
        return
    evaluate(plan)


if __name__=='__main__':
    main()
