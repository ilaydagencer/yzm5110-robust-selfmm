"""A/B/C/D, K0-K7, fixed tau=1, validation only. No training or tuning.

Uses verified TRAIN-only noise statistics and shared corruption functions.
Requires the earlier K0-K3 validation run as a numerical regression control.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import sentencepiece  # Windows: before torch and MMSA.
import numpy as np
import torch
import MMSA
from torch.utils.data import DataLoader
from MMSA.config import get_config_regression
from MMSA.data_loader import MMDataset
from MMSA.models import AMIO
from MMSA.utils import MetricsTop, setup_seed

from evaluate_selfmm_abcd_valid import require, digest, weights_from_outputs, fusion_weights, self_check, write_csv
import selfmm_corruptions as corruptions
from check_selfmm_corruptions import synthetic_check

TAU = 1.0
SEED = 1111


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--repo-dir', type=Path, default=Path('C:/Users/ilayda/YZM5110/MMSA'))
    parser.add_argument('--output-dir', type=Path)
    cli = parser.parse_args()
    repo = cli.repo_dir.resolve()
    project = repo.parent
    protocol_dir = project/'runs/selfmm_corruption_protocol_valid_seed1111_v1'
    reference_dir = project/'runs/selfmm_abcd_valid_seed1111_tau1'
    output = cli.output_dir or project/'runs/selfmm_abcd_corrupt_valid_seed1111_tau1_v1'
    require(not output.exists(), f'Cikti klasoru zaten var; uzerine yazilmaz: {output}')
    require(Path(MMSA.__file__).resolve().is_relative_to(repo/'src'), 'Yanlis MMSA kurulumu')
    protocol_path = protocol_dir/'manifest.json'
    stats_path = protocol_dir/'train_feature_stats.json'
    reference_manifest_path = reference_dir/'manifest.json'
    reference_predictions_path = reference_dir/'predictions_and_weights.csv'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    stats = json.loads(stats_path.read_text(encoding='utf-8'))
    ref_manifest = json.loads(reference_manifest_path.read_text(encoding='utf-8'))
    require(protocol['status']=='completed' and protocol['split']=='valid'
            and protocol['protocol_version']==1 and protocol['n']==229
            and protocol['seed']==SEED and protocol['noise_factor']==0.5
            and protocol['partial_rates']==corruptions.PARTIAL_RATES, 'Protokol uyumsuz')
    require(corruptions.NOISE_FACTOR==0.5 and corruptions.CONDITIONS==tuple(f'K{i}' for i in range(8)),
            'Bozulma modulu protokolle uyumsuz')
    for flag in ('text_unchanged','lengths_unchanged','padding_unchanged','source_unchanged'):
        require(protocol.get(flag) is True, f'Protokol koruma kontrolu eksik: {flag}')
    require(stats['source_sha256']==protocol['source_sha256']
            and stats['stats_source']=='train_only_valid_frames' and stats['ddof']==0,
            'Gurultu istatistikleri uyumsuz')
    require(ref_manifest['status']=='completed' and ref_manifest['split']=='valid'
            and ref_manifest['seed']==SEED and ref_manifest['tau']==TAU
            and ref_manifest.get('protected_files_unchanged') is True, 'Onceki validation kaydi uyumsuz')
    for module_path in (Path(corruptions.__file__), Path(__file__).with_name('check_selfmm_corruptions.py')):
        require(digest(module_path)==protocol['code_hashes'][module_path.name],
                f'Kontrol edilmis bozulma kodu degismis: {module_path.name}')
    specs = {
        'baseline': (project/'local_configs/selfmm_mosi_batch4.json',
                     project/'runs/selfmm_seed1111/saved_models/self_mm-mosi.pth', 'A', 'C'),
        'dropout': (project/'local_configs/selfmm_mosi_dropout_a20_v20_batch4_seed1111.json',
                    project/'runs/selfmm_dropout_a20_v20_seed1111/saved_models/self_mm-mosi.pth', 'B', 'D')}
    configs, arguments = {}, {}
    protected = [Path(__file__).resolve(),Path(corruptions.__file__).resolve(),
                 Path(__file__).with_name('check_selfmm_corruptions.py').resolve(),
                 Path(__file__).with_name('evaluate_selfmm_abcd_valid.py').resolve(),
                 protocol_path, stats_path, reference_manifest_path, reference_predictions_path,
                 repo/'src/MMSA/models/multiTask/SELF_MM.py',repo/'src/MMSA/config/config_regression.json']
    for name,(config,checkpoint,plain,weighted) in specs.items():
        configs[name] = json.loads(config.read_text(encoding='utf-8'))
        require(checkpoint.is_file(), f'Checkpoint eksik: {checkpoint}')
        require(digest(checkpoint)==ref_manifest['protected_hashes'][str(checkpoint)],
                'Checkpoint onceki validation sonrasinda degismis')
        args = get_config_regression('self_mm','mosi',str(config))
        require(args.batch_size==4 and not args.need_data_aligned and not args.need_model_aligned
                and not args.get('data_missing',False) and args.post_fusion_dropout==0,
                'Model config uyumsuz')
        args.update(train_mode='regression',device=torch.device('cuda:0'),custom_feature=None,
                    feature_T=None,feature_A=None,feature_V=None,data_missing=False)
        for modality,dimension in (('audio',args.feature_dims[1]),('vision',args.feature_dims[2])):
            scale = np.asarray(stats['modalities'][modality]['std'])
            require(scale.shape==(dimension,) and np.isfinite(scale).all() and (scale>=0).all(),
                    'Std boyutu/degeri uyumsuz')
        arguments[name] = args
        protected.extend((config,checkpoint))
        print(f'{plain}/{weighted} CHECKPOINT: {checkpoint}')
    require(configs['baseline']==configs['dropout'],'Configler eslesmiyor')
    data_path = Path(arguments['baseline'].featurePath)
    require(data_path.is_file(),'Veri bulunamadi')
    reference = {}
    for row in read_csv(reference_predictions_path):
        key = row['checkpoint'],row['condition'],int(row['index'])
        require(key not in reference,'Onceki tahminlerde tekrar')
        reference[key] = {k:float(row[k]) for k in ('label','plain_prediction','weighted_prediction')}
    expected = {(name,f'K{k}',i) for name in specs for k in range(4) for i in range(229)}
    require(set(reference)==expected,'Onceki validation ornekleri eksik/fazla')
    self_check()
    synthetic_check()
    print('SPLIT: valid; N: 229; BATCH: 4; TAU: 1; EGITIM SEED: 1111; BOZULMA SEED: 1111')
    print('KOSULLAR: K0-K7; CIKTI:',output)
    if cli.check:
        print('KONTROL_BASARILI: model/veri yuklenmedi, dosya yazilmadi; metadata/kod/checkpoint ozetleri eslesti')
        return
    require(torch.cuda.is_available(),'CUDA bulunamadi')
    require(digest(data_path)==protocol['source_sha256'],'Veri dosyasi protokolden farkli')
    setup_seed(SEED)
    torch.cuda.set_device(0)
    dataset = MMDataset(copy.deepcopy(arguments['baseline']),mode='valid')
    require(len(dataset)==229,'Validation ornek sayisi uyumsuz')
    loader = DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0,drop_last=False)
    hashes = {str(p):digest(p) for p in protected}
    output.mkdir(parents=True,exist_ok=False)
    manifest = dict(status='running',split='valid',n=229,batch_size=4,training_seed=SEED,
                    corruption_seed=SEED,tau=TAU,conditions=list(corruptions.CONDITIONS),
                    protocol_version=1,protocol_manifest_sha256=digest(protocol_path),
                    train_stats_sha256=digest(stats_path),data_sha256=protocol['source_sha256'],
                    protected_hashes=hashes,torch_version=torch.__version__,numpy_version=np.__version__,
                    gpu=torch.cuda.get_device_name(0),inference_only=True,
                    selection='Fixed tau=1 diagnostic; no tuning and no test evaluation.',
                    note='Three forward passes include identity control; not a latency benchmark.')
    def save_manifest():
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save_manifest()
    rows=[]; sample_rows=[]; weights_rows=[]; controls=[]; reproduction=[]
    input_digests={}
    metric_fn=MetricsTop('regression').getMetics('mosi')
    try:
        for name,(_,checkpoint,plain,weighted) in specs.items():
            setup_seed(SEED)
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
                identity_error=0.; old_error=0.
                fingerprint=hashlib.sha256()
                hooks_before=len(layer._forward_pre_hooks)
                with torch.inference_mode():
                    for batch in loader:
                        require(all(not module.training for module in model.modules()),'Model eval modunda degil')
                        batch_indices=batch['index'].tolist()
                        features={}
                        for modality in ('audio','vision'):
                            original=batch[modality].numpy()
                            altered=[]
                            for j,index in enumerate(batch_indices):
                                length=int(batch[modality+'_lengths'][j])
                                out,_=corruptions.corrupt_modality(original[j],length,modality,condition,
                                    stats['modalities'][modality]['std'],index,split='valid',seed=SEED)
                                require(np.array_equal(out[length:],original[j,length:],equal_nan=True),'Padding degisti')
                                altered.append(out)
                            values=np.stack(altered)
                            fingerprint.update(values.tobytes())
                            fingerprint.update(batch[modality+'_lengths'].numpy().tobytes())
                            features[modality]=torch.from_numpy(values).to(args.device)
                        fingerprint.update(batch['text'].numpy().tobytes())
                        fingerprint.update(batch['index'].numpy().tobytes())
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
                            # Labels/order are checked against the old VALIDATION run for every condition.
                            ref=reference[name,'K0',index]
                            require(abs(float(labels[j])-ref['label'])<1e-7,'Etiket/sira degisti')
                            if condition in ('K0','K1','K2','K3'):
                                ref=reference[name,condition,index]
                                old_error=max(old_error,abs(float(p[j])-ref['plain_prediction']),
                                              abs(float(q[j])-ref['weighted_prediction']))
                            sample_rows.append(dict(checkpoint=name,condition=condition,index=index,
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
                if condition in ('K0','K1','K2','K3'):
                    require(old_error<=1e-5,f'Onceki K0-K3 tahminleri eslesmiyor: {name}/{condition}/{old_error}')
                    reproduction.append(dict(checkpoint=name,condition=condition,max_prediction_error=old_error,tolerance=1e-5,passed=True))
                p=torch.cat(predictions);q=torch.cat(weighted_predictions);labels=torch.cat(truths);w=torch.cat(all_weights)
                for method,pred in ((plain,p),(weighted,q)):
                    metric={k:float(v) for k,v in metric_fn(pred,labels).items()}
                    row=dict(method=method,checkpoint=name,split='valid',condition=condition,tau=TAU,
                             n=229,**metric,MAE_exact=float((pred-labels).abs().mean()))
                    rows.append(row);print(json.dumps(row),flush=True)
                entry=dict(method=weighted,condition=condition,tau=TAU,n=229)
                for i,modality in enumerate(('text','audio','vision')):
                    entry.update({f'w_{modality}_mean':float(w[:,i].mean()),
                                  f'w_{modality}_std':float(w[:,i].std(unbiased=False)),
                                  f'w_{modality}_min':float(w[:,i].min()),f'w_{modality}_max':float(w[:,i].max())})
                weights_rows.append(entry)
                controls.append(dict(checkpoint=name,condition=condition,n=229,max_uniform_error=identity_error,
                                     hooks_restored=True,corrupted_input_sha256=checksum))
            del model
            torch.cuda.empty_cache()
        comparisons=[];degradation=[]
        for condition in corruptions.CONDITIONS:
            lookup={r['method']:r for r in rows if r['condition']==condition}
            for first,second in (('A','B'),('A','C'),('B','D'),('A','D')):
                comparisons.append(dict(condition=condition,comparison=f'{second}-minus-{first}',
                    Non0_acc_delta=lookup[second]['Non0_acc_2']-lookup[first]['Non0_acc_2'],
                    MAE_exact_delta=lookup[second]['MAE_exact']-lookup[first]['MAE_exact'],
                    Corr_delta=lookup[second]['Corr']-lookup[first]['Corr']))
        for method in ('A','B','C','D'):
            clean=next(r for r in rows if r['method']==method and r['condition']=='K0')
            for row in (r for r in rows if r['method']==method):
                degradation.append(dict(method=method,condition=row['condition'],
                    MAE_increase=row['MAE_exact']-clean['MAE_exact'],
                    Non0_acc_drop=clean['Non0_acc_2']-row['Non0_acc_2'],
                    Corr_drop=clean['Corr']-row['Corr']))
        for filename,records in (('metrics.csv',rows),('comparisons.csv',comparisons),
                ('degradation.csv',degradation),('predictions_and_weights.csv',sample_rows),
                ('weight_summary.csv',weights_rows),('controls.csv',controls),('reproduction_checks.csv',reproduction)):
            write_csv(output/filename,records)
        manifest.update(status='completed',shared_corrupted_inputs=True,prior_K0_K3_reproduced=True,
                        corrupted_input_sha256=input_digests)
    except BaseException:
        manifest['status']='failed_or_interrupted'
        raise
    finally:
        manifest['protected_files_unchanged']=all(digest(Path(p))==h for p,h in hashes.items())
        manifest['data_file_unchanged']=digest(data_path)==protocol['source_sha256']
        save_manifest()
    require(manifest['protected_files_unchanged'] and manifest['data_file_unchanged'],'Korunan dosya degisti')
    print('ABCD_CORRUPT_VALIDATION_TAMAMLANDI:',output)


if __name__=='__main__':main()
