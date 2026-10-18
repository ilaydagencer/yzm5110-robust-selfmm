"""Check K0-K7 protocol; --check uses synthetic data only.

Normal mode reads the user's trusted MOSI pickle, computes TRAIN statistics,
and verifies corruptions on VALIDATION without constructing a model.
"""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from selfmm_corruptions import CONDITIONS, PARTIAL_RATES, NOISE_FACTOR, corrupt_modality, training_statistics


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def synthetic_check():
    x=np.arange(24,dtype=np.float32).reshape(8,3)+1
    before=x.copy()
    std=np.array([0.,1.,2.])
    masks={}
    state=np.random.get_state()
    for condition in CONDITIONS:
        for modality in ('audio','vision'):
            out,mask=corrupt_modality(x,5,modality,condition,std,9)
            twin,mask2=corrupt_modality(x,5,modality,condition,std,9)
            require(np.array_equal(out,twin) and np.array_equal(mask,mask2),'Seed repeatability failed')
            require(np.array_equal(out[5:],x[5:]) and not mask[5:].any(),'Padding changed')
            require(np.array_equal(x,before),'Input modified in place')
            if condition in PARTIAL_RATES:
                require(np.array_equal(out[~mask],x[~mask]) and (out[mask]==0).all(),'Partial masking failed')
                masks[modality,condition]=mask
            if condition=='K0':require(np.array_equal(out,x),'K0 identity failed')
            expected_drop=(modality=='audio' and condition in ('K1','K3')) or (modality=='vision' and condition in ('K2','K3'))
            if expected_drop:require((out[:5]==0).all() and mask[:5].all(),'Full dropout failed')
            if condition=='K7':require(np.array_equal(out[:5,0],x[:5,0]),'Zero-variance feature changed')
    for modality in ('audio','vision'):
        require((~masks[modality,'K4']|masks[modality,'K5']).all()
                and (~masks[modality,'K5']|masks[modality,'K6']).all(),'Masks not nested')
    a,_=corrupt_modality(x,5,'audio','K7',std,9)
    b,_=corrupt_modality(x,5,'audio','K7',std,9,seed=1112)
    require(not np.array_equal(a,b),'Different seeds gave identical noise')
    # Check training statistics exclude padding and match a direct calculation.
    train=np.array([[[1.,2.],[3.,4.],[999.,999.]],[[5.,6.],[999.,999.],[999.,999.]]],dtype=np.float32)
    stat=training_statistics(train,[2,1])
    values=np.array([[1.,2.],[3.,4.],[5.,6.]])
    require(stat['valid_frames']==3 and np.allclose(stat['std'],values.std(0)),'Training std calculation failed')
    after=np.random.get_state()
    require(state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:],'Global RNG changed')
    print('SENTETIK_KONTROLLER_BASARILI: K0, maskeler, padding, seed, sifir std, train std, RNG')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--project-dir',type=Path,default=Path('C:/Users/ilayda/YZM5110'))
    parser.add_argument('--output-dir',type=Path)
    cli=parser.parse_args()
    source=cli.project_dir/'data/MOSI/Processed/unaligned_50.pkl'
    output=cli.output_dir or cli.project_dir/'runs/selfmm_corruption_protocol_valid_seed1111_v1'
    require(source.is_file(),f'Data file missing: {source}')
    require(not output.exists(),f'Cikti zaten var; uzerine yazilmaz: {output}')
    synthetic_check()
    print('PROTOKOL: K4/K5/K6 = Bernoulli %30/%50/%70; K7 = 0.5 * train feature std')
    print('CIKTI:',output)
    if cli.check:
        print('KONTROL_BASARILI: gercek veri/model yuklenmedi, dosya yazilmadi')
        return
    original_hash=digest(source)
    # This is the user's known/trusted pickle. It contains all splits; only
    # train and valid arrays are used. No test metric or test statistic is read.
    with source.open('rb') as f:data=pickle.load(f)
    train,valid=data['train'],data['valid']
    del data
    require(len(train['audio'])==1284 and len(valid['audio'])==229,'Unexpected MOSI split size')
    stats={}; cleanup={}
    arrays={}
    for split,block in (('train',train),('valid',valid)):
        for modality in ('audio','vision'):
            x=np.asarray(block[modality],dtype=np.float32).copy()
            count=int(np.isneginf(x).sum()) if modality=='audio' else 0
            if modality=='audio':x[np.isneginf(x)]=0  # Same cleanup as MMSA loader.
            arrays[split,modality]=x
            cleanup[f'{split}_{modality}_negative_inf_replaced']=count
    for modality in ('audio','vision'):
        stats[modality]=training_statistics(arrays['train',modality],train[modality+'_lengths'])
    text=np.asarray(valid['text_bert'])
    text_hash=hashlib.sha256(text.tobytes()).hexdigest()
    length_copies={m:np.asarray(valid[m+'_lengths']).copy() for m in ('audio','vision')}
    summaries=[]; per_sample=[]
    for modality in ('audio','vision'):
        features=arrays['valid',modality]
        scale=np.asarray(stats[modality]['std'])
        for condition in CONDITIONS:
            selected=frames=0
            square=0.; standardized_square=0.; standardized_count=0
            for index,x in enumerate(features):
                length=int(length_copies[modality][index])
                out,mask=corrupt_modality(x,length,modality,condition,scale,index)
                twin,twin_mask=corrupt_modality(x,length,modality,condition,scale,index)
                require(np.array_equal(out,twin) and np.array_equal(mask,twin_mask),'Repeatability failed')
                # Simulate a later call in a different evaluation order.
                corrupt_modality(x,length,modality,'K7',scale,(index+1)%len(features))
                third,_=corrupt_modality(x,length,modality,condition,scale,index)
                require(np.array_equal(out,third),'Evaluation order affected corruption')
                require(np.array_equal(out[length:],x[length:],equal_nan=True),'Padding modified')
                delta=out[:length].astype(np.float64)-x[:length].astype(np.float64)
                square+=float((delta**2).sum())
                positive=scale>0
                if condition=='K7' and positive.any():
                    z=delta[:,positive]/scale[positive]
                    standardized_square+=float((z**2).sum())
                    standardized_count+=z.size
                count=int(mask[:length].sum()); selected+=count;frames+=length
                per_sample.append(dict(modality=modality,condition=condition,index=index,length=length,
                                       masked_frames=count,masked_rate=count/length,
                                       delta_rms=float(np.sqrt((delta**2).mean()))))
            normalized_rms=np.sqrt(standardized_square/standardized_count) if standardized_count else 0.
            summaries.append(dict(modality=modality,condition=condition,samples=229,valid_frames=frames,
                masked_frames=selected,masked_rate=selected/frames,
                target_mask_rate=PARTIAL_RATES.get(condition,1. if (modality=='audio' and condition in ('K1','K3')) or (modality=='vision' and condition in ('K2','K3')) else 0.),
                delta_rms=float(np.sqrt(square/(frames*features.shape[2]))),
                noise_rms_div_train_std=float(normalized_rms)))
            print(json.dumps(summaries[-1]),flush=True)
    require(hashlib.sha256(text.tobytes()).hexdigest()==text_hash,'Text changed')
    require(all(np.array_equal(valid[m+'_lengths'],length_copies[m]) for m in ('audio','vision')),'Lengths changed')
    require(digest(source)==original_hash,'Data file changed')
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in (('corruption_summary.csv',summaries),('corruption_per_sample.csv',per_sample)):
        with (output/name).open('x',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'train_feature_stats.json').write_text(json.dumps(dict(
        source_sha256=original_hash,stats_source='train_only_valid_frames',ddof=0,
        preprocessing='float32; audio negative infinity replaced by zero as in MMSA',
        cleanup=cleanup,modalities=stats),indent=2),encoding='utf-8')
    manifest=dict(status='completed',protocol_version=1,split='valid',n=229,seed=1111,
        noise_factor=NOISE_FACTOR,partial_rates=PARTIAL_RATES,partial_type='independent Bernoulli per valid time step; all features of selected frame zeroed',
        masks_nested=True,modality_rng_independent=True,model_order_independent=True,
        text_unchanged=True,lengths_unchanged=True,padding_unchanged=True,source_unchanged=True,
        source_sha256=original_hash,numpy_version=np.__version__,zero_std='no noise added',
        note='Pickle contains all splits; only train/valid arrays used. No model, training, metric or hyperparameter selection.',
        code_hashes={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('selfmm_corruptions.py'))})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('GERCEK_VERI_KONTROLU_TAMAMLANDI:',output)


if __name__=='__main__':main()
