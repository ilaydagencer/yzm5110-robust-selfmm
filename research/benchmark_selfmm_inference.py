"""Benchmark existing Self-MM A/B/C/D checkpoints on clean test inputs (K0).

From MMSA: python -B research/benchmark_selfmm_inference.py --seed 1111 --check
Then omit --check. No training, tuning, downloads, or changes to old results.
Default: 6 timing repetitions, 10 warm-up batches, all 686 examples, batch 4.
One checkpoint is resident at a time. Timings use synchronized wall time;
data loading/transfers, correctness checks and warm-up are outside the timer.
"""
import argparse
import contextlib
import copy
import gc
import hashlib
import importlib
import io
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import sentencepiece  # Windows: MUST precede torch and MMSA.
import numpy as np
import torch

VERSION = 1
N = 686
BATCH = 4
TAU = 1.0
TOLERANCE = 1e-5
MIB = 1024**2
METHODS = ('A','B','C','D')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def order_for(round_number, plain, weighted):
    return (plain,weighted) if round_number % 2 == 1 else (weighted,plain)


def infer(model, inputs, weighted, layer, dims, weight_fn, fusion_context):
    """Exactly one original forward or two fresh forwards plus weighting.

    Keep the first output alive until the second forward finishes, matching
    the existing two-pass wrapper. Never cache predictions across calls.
    Helper validation/synchronization and temporary hook costs are included.
    """
    def forward():
        return model(inputs['text'], (inputs['audio'],inputs['audio_lengths']),
                     (inputs['vision'],inputs['vision_lengths']))
    if not weighted:
        return forward()['M']
    first = forward()
    weights, deviations = weight_fn(first,TAU)
    with fusion_context(layer,weights,dims):
        result = forward()['M']
    return result


def timed_call(call, synchronize, clock=time.perf_counter):
    synchronize()                 # Prior transfers finish before the timer.
    start = clock()
    result = call()
    synchronize()                 # Wait for completion, not just launch.
    return result, (clock()-start)*1000.0


def self_check(base):
    """CPU-only: forward counts, fresh weights, cleanup, timer and pairing."""
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion = torch.nn.Linear(6,1,bias=False)
            with torch.no_grad():self.fusion.weight.fill_(1)
            self.calls=0
        def forward(self,text,audio,vision):
            self.calls+=1
            return dict(M=self.fusion(text),T=text[:,:1],A=text[:,1:2],V=text[:,2:3])
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        model=Toy().eval().requires_grad_(False)
        inputs=dict(text=torch.tensor([[0.,0.,2.,1.,2.,3.],[1.,2.,3.,4.,5.,6.]]),
                    audio=torch.zeros(2,1,1),vision=torch.zeros(2,1,1),
                    audio_lengths=torch.ones(2,dtype=torch.long),vision_lengths=torch.ones(2,dtype=torch.long))
        saved=inputs['text'].clone();dims=(1,2,3)
        original=infer(model,inputs,False,model.fusion,dims,base.weights_from_outputs,base.fusion_weights)
        require(model.calls==1,'A/B bir forward olmali')
        model.calls=0
        altered=infer(model,inputs,True,model.fusion,dims,base.weights_from_outputs,base.fusion_weights)
        require(model.calls==2,'C/D tam iki forward olmali')
        out=dict(T=inputs['text'][:,:1],A=inputs['text'][:,1:2],V=inputs['text'][:,2:3])
        w,_=base.weights_from_outputs(out,1)
        x=torch.cat([part*w[:,i:i+1] for i,part in enumerate(torch.split(inputs['text'],dims,dim=1))],dim=1)
        require(torch.allclose(altered,model.fusion(x)) and not torch.equal(original,altered),'Agirlikli ciktinin anlami degisti')
        require(torch.equal(saved,inputs['text']) and not model.fusion._forward_pre_hooks,'Girdi/hook korunmadi')
        def fail(*args):raise ValueError('intentional cleanup test')
        handle=model.fusion.register_forward_hook(fail)
        try:
            with base.fusion_weights(model.fusion,w,dims):model.fusion(inputs['text'])
        except ValueError:pass
        finally:handle.remove()
        require(not model.fusion._forward_pre_hooks,'Hata sonrasi hook kaldi')
    events=[];ticks=iter((10.,10.002))
    result,elapsed=timed_call(lambda:events.append('forward') or 7,
                             lambda:events.append('sync'),lambda:next(ticks))
    require(events==['sync','forward','sync'] and result==7 and abs(elapsed-2)<1e-8,'Zamanlayici siniri hatali')
    require([order_for(i,'A','C')[0] for i in range(1,7)]==['A','C','A','C','A','C'],'Sira dengesi hatali')
    sizes=[4]*171+[2]
    require(sum(sizes)==N and len(sizes)==172,'Son batch hesabi hatali')
    require(abs(statistics.stdev([1,2,3,4,5])-math.sqrt(2.5))<1e-12,'Orneklem SS hatali')
    print('BENCHMARK_BIRIM_KONTROLLERI_BASARILI: 1/2 forward, agirlik, hook, zaman siniri, sira, son batch, SS')


def prepare(cli):
    repo=cli.repo_dir.resolve();research=repo/'research'
    require(repo.is_dir() and research.is_dir(),'Repo/research bulunamadi; --repo-dir kullanin')
    require(2 <= cli.repeats <= 20 and cli.repeats % 2 == 0,'Tekrar sayisi 2-20 arasinda cift olmali')
    require(2 <= cli.warmup_batches <= 100,'Isinma batch sayisi 2-100 olmali')
    output=cli.output_dir or repo.parent/'runs'/f'selfmm_inference_k0_seed{cli.seed}_batch4_r{cli.repeats}_v1'
    require(not os.path.lexists(output),'Cikti zaten var; uzerine yazilmaz: '+str(output))
    reference_path=repo.parent/'runs'/f'selfmm_abcd_corrupt_test_seed{cli.seed}_tau1_v1/manifest.json'
    reference=json.loads(reference_path.read_text(encoding='utf-8'))
    require(reference['status']=='completed' and reference['split']=='test' and reference['training_seed']==cli.seed
            and reference['n']==N and reference['non0_n']==656 and reference['batch_size']==BATCH
            and reference['tau']==TAU and reference['inference_only'] is True,
            'Tamamlanmis ayni seed test referansi gerekli')
    sys.path.insert(0,str(research))
    name='evaluate_selfmm_abcd_test' if cli.seed==1111 else 'evaluate_selfmm_abcd_seed_test'
    helper_path=(research/(name+'.py')).resolve()
    require(digest(helper_path)==reference['protected_hashes'].get(str(helper_path)),'Dondurulmus evaluator degismis')
    evaluator=importlib.import_module(name)
    require(Path(evaluator.__file__).resolve()==helper_path,'Yanlis evaluator yuklendi')
    capture=io.StringIO()
    try:
        with contextlib.redirect_stdout(capture):
            plan=evaluator.prepare(repo,output_dir=output) if cli.seed==1111 else evaluator.prepare(repo,cli.seed,output_dir=output)
    except Exception:
        print(capture.getvalue());raise
    base=importlib.import_module('evaluate_selfmm_abcd_valid')
    test=importlib.import_module('evaluate_selfmm_abcd_test')
    require(Path(base.__file__).resolve()==(research/'evaluate_selfmm_abcd_valid.py').resolve(),'Yanlis agirlik yardimcisi')
    require(reference['data_sha256']==plan['data_sha256'] and reference['torch_version']==torch.__version__
            and reference['numpy_version']==np.__version__,'Referans veri/surum uyusmazligi')
    for flag in ('protected_files_unchanged','data_file_unchanged','shared_corrupted_inputs',
                 'text_unchanged','lengths_unchanged','padding_unchanged'):
        require(reference.get(flag) is True,'Test referans kontrolu basarisiz: '+flag)
    predictions_path=reference_path.with_name('predictions_and_weights.csv')
    require(digest(predictions_path)==reference['output_hashes']['predictions_and_weights.csv'],'Referans tahmin dosyasi degismis')
    rows=evaluator.read_csv(predictions_path)
    require(len(rows)==10976,'Test referans satir sayisi uyusmuyor')
    records={}
    for row in rows:
        if row['condition']!='K0':continue
        cp,index=row['checkpoint'],int(row['index'])
        require((cp,index) not in records and cp in ('baseline','dropout') and 0<=index<N,'Tekrarli/gecersiz referans')
        require(row['split']=='test' and int(row['training_seed'])==cli.seed,'Referans split/seed farkli')
        expected=('A','C') if cp=='baseline' else ('B','D')
        require((row['plain_method'],row['weighted_method'])==expected,'Referans yontem farkli')
        records[cp,index]={k:float(row[k]) for k in ('label','plain_prediction','weighted_prediction')}
        require(all(math.isfinite(x) for x in records[cp,index].values()),'Referans sayisi sonlu degil')
    require(set(records)=={(cp,i) for cp in ('baseline','dropout') for i in range(N)},'Referans indeksleri eksik')
    for cp,(_,checkpoint,_,_) in plan['specs'].items():
        require(Path(reference['checkpoint_paths'][cp]).resolve()==checkpoint.resolve(),'Referans checkpoint yolu farkli')
        require(digest(checkpoint)==reference['protected_hashes'].get(str(checkpoint.resolve())),'Referans checkpoint ozeti farkli')
    for i in range(N):
        require(records['baseline',i]['label']==records['dropout',i]['label']==plan['labels'][i],'Referans etiketleri farkli')
    for path in (Path(__file__).resolve(),reference_path,predictions_path):plan['hashes'][str(path)]=digest(path)
    plan.update(seed=cli.seed,repeats=cli.repeats,warmup_batches=cli.warmup_batches,
                records=records,expected_k0=reference['corrupted_input_sha256']['K0'],
                reference_manifest_sha256=digest(reference_path),base=base,test=test,evaluator=evaluator)
    self_check(base)
    print('REFERANS_KONTROLLERI_BASARILI: yerel MMSA, config, checkpoint, kaynaklar ve K0 tahminleri')
    print(f'BENCHMARK: seed={cli.seed}; K0; test=686; batch=4; son batch=2; tekrar={cli.repeats}; isinma={cli.warmup_batches}')
    print('SURE: senkronize forward + C/D agirlik/hook; yukleme, veri aktarimi ve kontrol gecisleri haric')
    print('BELLEK: tek checkpoint + batch + ara tensorler; PyTorch allocated/reserved MiB')
    print('CUDA:',torch.cuda.is_available(),'CIKTI:',plan['output'])
    return plan


def fingerprint(batches):
    h=hashlib.sha256()
    for batch in batches:
        for modality in ('audio','vision'):
            h.update(batch[modality].numpy().tobytes())
            h.update(batch[modality+'_lengths'].numpy().tobytes())
        h.update(batch['text'].numpy().tobytes());h.update(batch['index'].numpy().tobytes())
    return h.hexdigest()


def to_device(batch,device):
    return {k:batch[k].to(device) if k in ('text','audio','vision') else batch[k]
            for k in ('text','audio','vision','audio_lengths','vision_lengths')}


def check_prediction(prediction,batch,method,cp,plan):
    actual=prediction.reshape(-1).detach().cpu().numpy()
    field='weighted_prediction' if method in ('C','D') else 'plain_prediction'
    expected=np.array([plan['records'][cp,int(i)][field] for i in batch['index']],dtype=np.float64)
    require(actual.shape==expected.shape and np.isfinite(actual).all(),'Tahmin boyutu/sonlulugu hatali')
    error=float(np.max(np.abs(actual-expected)))
    require(error<=TOLERANCE,f'Eski K0 tahmini tekrar uretilemedi: {method}; hata={error}')
    return error


def correctness(model,batches,cp,plain,weighted,plan,device):
    layer=model.Model.post_fusion_layer_1;args=plan['arguments'][cp]
    dims=(args.text_out,args.audio_out,args.video_out);base=plan['base'];errors={plain:0.,weighted:0.}
    identity_error=0.;hooks=len(layer._forward_pre_hooks)
    for i,batch in enumerate(batches):
        inputs=to_device(batch,device)
        for method in (plain,weighted):
            prediction=infer(model,inputs,method==weighted,layer,dims,base.weights_from_outputs,base.fusion_weights)
            errors[method]=max(errors[method],check_prediction(prediction,batch,method,cp,plan))
            del prediction
        # Uniform controls are deliberately outside timed repetitions.
        if i in (0,len(batches)-1):
            original=infer(model,inputs,False,layer,dims,base.weights_from_outputs,base.fusion_weights)
            with base.fusion_weights(layer,torch.ones(len(batch['index']),3,device=device),dims):
                uniform=infer(model,inputs,False,layer,dims,base.weights_from_outputs,base.fusion_weights)
            identity_error=max(identity_error,float((original-uniform).abs().max()))
            require(torch.allclose(original,uniform,atol=1e-6,rtol=1e-6),'Esit agirlik kontrolu basarisiz')
            del original,uniform
        require(len(layer._forward_pre_hooks)==hooks,'Kontrolden sonra hook kaldi')
        del inputs
    return [dict(method=m,checkpoint=cp,n=N,max_prediction_error=errors[m],tolerance=TOLERANCE,
                 uniform_control_max_error=identity_error,uniform_batch_sizes='4;2',hooks_restored=True,passed=True)
            for m in (plain,weighted)]


def measure_round(model,batches,cp,method,round_number,position,plan,device):
    base=plan['base'];args=plan['arguments'][cp];layer=model.Model.post_fusion_layer_1
    dims=(args.text_out,args.audio_out,args.video_out);weighted=method in ('C','D')
    hooks=len(layer._forward_pre_hooks)
    require(all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),'Model eval/frozen degil')
    gc.collect();torch.cuda.synchronize(device);torch.cuda.empty_cache()
    # A clean allocator cache per method/repetition, followed by method-specific
    # warm-up. Include full and tail batch shapes. No empty_cache inside timing.
    warm_indices=[i%(len(batches)-1) for i in range(plan['warmup_batches']-1)]+[len(batches)-1]
    for index in warm_indices:
        inputs=to_device(batches[index],device)
        prediction=infer(model,inputs,weighted,layer,dims,base.weights_from_outputs,base.fusion_weights)
        del prediction,inputs
    torch.cuda.synchronize(device)
    resident=torch.cuda.memory_allocated(device)/MIB
    reserved_before=torch.cuda.memory_reserved(device)/MIB
    torch.cuda.reset_peak_memory_stats(device)
    rows=[];max_error=0.
    for index,batch in enumerate(batches):
        inputs=to_device(batch,device)  # Outside timed call; transfers excluded.
        prediction,elapsed=timed_call(
            lambda:infer(model,inputs,weighted,layer,dims,base.weights_from_outputs,base.fusion_weights),
            lambda:torch.cuda.synchronize(device))
        require(math.isfinite(elapsed) and elapsed>0,'Gecersiz sure')
        # CPU copy/reproduction check are outside the timer. No GPU outputs are
        # retained between batches, so memory does not accumulate with N.
        max_error=max(max_error,check_prediction(prediction,batch,method,cp,plan))
        rows.append(dict(method=method,checkpoint=cp,round=round_number,order_position=position,
                         batch_index=index,batch_n=len(batch['index']),latency_ms=elapsed))
        del prediction,inputs
        require(len(layer._forward_pre_hooks)==hooks,'Olcumden sonra hook kaldi')
    torch.cuda.synchronize(device)
    total=math.fsum(r['latency_ms'] for r in rows)
    require(len(rows)==172 and sum(r['batch_n'] for r in rows)==N,'Olcumde ornek/batch kaybi')
    peak=torch.cuda.max_memory_allocated(device)/MIB
    full=[r['latency_ms'] for r in rows if r['batch_n']==BATCH]
    summary=dict(method=method,checkpoint=cp,round=round_number,order_position=position,n=N,
        batches=len(rows),full_batches=len(full),tail_batch_n=rows[-1]['batch_n'],
        forward_passes_per_batch=2 if weighted else 1,warmup_batches=plan['warmup_batches'],
        timed_total_ms=total,ms_per_sample=total/N,samples_per_second=1000*N/total,
        full_batch_ms_mean=statistics.mean(full),tail_batch_ms=rows[-1]['latency_ms'],
        resident_allocated_mib=resident,warm_reserved_mib=reserved_before,
        peak_allocated_mib=peak,peak_extra_allocated_mib=peak-resident,
        peak_reserved_mib=torch.cuda.max_memory_reserved(device)/MIB,
        end_allocated_mib=torch.cuda.memory_allocated(device)/MIB,
        max_prediction_error=max_error,hooks_restored=True)
    require(peak>=resident and summary['tail_batch_n']==2,'Bellek/son batch kontrolu basarisiz')
    return rows,summary


def aggregate(rounds,batch_rows,repeats):
    require(len(rounds)==4*repeats,'Tur sayisi eksik')
    lookup={(r['method'],r['round']):r for r in rounds}
    require(len(lookup)==len(rounds) and set(lookup)=={(m,r) for m in METHODS for r in range(1,repeats+1)},'Tekrarli/eksik tur')
    summaries=[]
    for method in METHODS:
        rows=[lookup[method,r] for r in range(1,repeats+1)]
        row=dict(method=method,condition='K0',n=N,repeats=repeats,batch_size=BATCH)
        for key in ('ms_per_sample','samples_per_second','full_batch_ms_mean','peak_allocated_mib','peak_extra_allocated_mib','peak_reserved_mib'):
            values=[r[key] for r in rows];row[key+'_mean']=statistics.mean(values);row[key+'_sd']=statistics.stdev(values)
        row['peak_allocated_mib_max']=max(r['peak_allocated_mib'] for r in rows)
        row['peak_reserved_mib_max']=max(r['peak_reserved_mib'] for r in rows)
        full=[r['latency_ms'] for r in batch_rows if r['method']==method and r['batch_n']==4]
        require(len(full)==171*repeats,'Tam batch sayisi eksik')
        row['full_batch_ms_p50']=float(np.percentile(full,50))
        row['full_batch_ms_p95']=float(np.percentile(full,95))
        summaries.append(row)
    paired=[];paired_summary=[]
    for weighted,plain in (('C','A'),('D','B')):
        local=[]
        for r in range(1,repeats+1):
            a,b=lookup[plain,r],lookup[weighted,r]
            row=dict(comparison=f'{weighted}-vs-{plain}',round=r,
                ms_per_sample_delta=b['ms_per_sample']-a['ms_per_sample'],
                latency_ratio=b['ms_per_sample']/a['ms_per_sample'],
                peak_allocated_mib_delta=b['peak_allocated_mib']-a['peak_allocated_mib'])
            local.append(row)
        paired.extend(local)
        row=dict(comparison=f'{weighted}-vs-{plain}',repeats=repeats)
        for key in ('ms_per_sample_delta','latency_ratio','peak_allocated_mib_delta'):
            row[key+'_mean']=statistics.mean(r[key] for r in local);row[key+'_sd']=statistics.stdev(r[key] for r in local)
        paired_summary.append(row)
    return summaries,paired,paired_summary


NOTES = '''Self-MM inference benchmark v1: existing checkpoints, K0 only.
This measures the current FP32 eager implementation, not an optimized fused
algorithm and not end-to-end application latency. A/B use one original forward;
C/D use two fresh forwards, weight computation and temporary fusion hooks.
Existing helper validation/CPU synchronization overhead is intentionally included.
The first output remains alive during the second pass. No extra identity pass
is timed. No training-mode dropout is active. Parameters require no gradients.

All 686 test samples: 171 batches of 4 plus one batch of 2. CPU inputs are cached;
one batch is copied to the GPU before its timer. Synchronize before the start and
after completion; perf_counter measures wall time including CPU launch/dispatch.
Loading, preprocessing, transfer, warm-up, CSV writing and correctness checks are
excluded. Repeated input transfers/checks still occur between timed batches.
ms_per_sample is summed batch latency / 686, an amortized cost at batch 4 with
a tail, NOT latency for a batch-size-1 request. Throughput uses the timed scope.
Full-batch p50/p95 pool only batches of size 4. Mean/SD summaries use timing
repetitions; these are NOT independent training seeds or confidence intervals.

Only one checkpoint is resident. Pair order alternates A/C then C/A (and B/D
then D/B). Each method/repetition clears unused cache BEFORE method-specific
warm-up including sizes 4 and 2; then peak counters reset. No cache clearing
inside timed inference. Memory peaks include model, batch inputs and intermediate
tensors over the measurement window; input transfers are in the memory window
but outside latency timing. Extra allocated = peak minus resident allocated.
Reserved includes PyTorch's allocator cache. These are process PyTorch CUDA
figures, NOT all GPU use/driver memory or other applications' allocations.

Same K0 predictions are checked against the recorded evaluation before and during
timing (absolute tolerance 1e-5). Uniform controls run outside timing on batch
sizes 4 and 2. CPU input fingerprints and protected source/checkpoint/data hashes
are checked. Existing outputs are never overwritten. Offline cache only.
One training seed, one device, one condition, one process; checkpoint pair order
is fixed. Power, temperature, OS scheduling and other GPU work can affect times.
Do not generalize these resource measurements to all conditions/devices/seeds.
Do not use test timing to choose new model hyperparameters.

PyTorch timing/memory API reference for the installed major/minor version:
https://docs.pytorch.org/docs/2.3/generated/torch.cuda.synchronize.html
https://docs.pytorch.org/docs/2.3/notes/cuda.html#cuda-memory-management
'''


def benchmark(plan):
    import MMSA
    from torch.utils.data import DataLoader
    from MMSA.data_loader import MMDataset
    from MMSA.models import AMIO
    from MMSA.utils import setup_seed
    output=plan['output'];evaluator=plan['evaluator']
    require(torch.cuda.is_available(),'CUDA gerekli; CPU olcumu bu protokole dahil degil')
    require(not os.path.lexists(output),'Cikti zaten var; uzerine yazilmaz')
    require(all(digest(p)==h for p,h in plan['hashes'].items()),'Korunan dosya kontrolden sonra degisti')
    require(digest(plan['data_path'])==plan['data_sha256'],'Veri dosyasi degismis')
    setup_seed(plan['seed']);torch.cuda.set_device(0);device=torch.device('cuda:0')
    output.mkdir(exist_ok=False)
    manifest=dict(status='running',benchmark_version=VERSION,created_at_utc=datetime.now(timezone.utc).isoformat(),
        training_seed=plan['seed'],condition='K0',split='test',n=N,batch_size=BATCH,tail_batch_size=2,
        tau=TAU,repeats=plan['repeats'],warmup_batches=plan['warmup_batches'],sd_ddof=1,
        inference_only=True,training_performed=False,hyperparameter_tuning=False,
        timing_scope='Synchronized wall time: original eager forward(s), weight helper and hooks; GPU inputs ready',
        memory_scope='PyTorch CUDA model plus one input batch and temporaries; warmed allocator; not total GPU use',
        timed_forward_passes={'A':1,'B':1,'C':2,'D':2},timed_identity_passes=0,
        pair_order='Alternating within each checkpoint; baseline checkpoint before dropout',
        amortized_sample_time=True,precision='FP32, no autocast, no compile',
        data_sha256=plan['data_sha256'],reference_manifest_sha256=plan['reference_manifest_sha256'],
        protected_hashes=plan['hashes'],checkpoint_paths={name:str(spec[1]) for name,spec in plan['specs'].items()},
        python_version=platform.python_version(),platform=platform.platform(),torch_version=torch.__version__,
        numpy_version=np.__version__,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(device),gpu_total_mib=torch.cuda.get_device_properties(device).total_memory/MIB,
        torch_num_threads=torch.get_num_threads(),cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,offline=True)
    def save_manifest():
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    save_manifest()
    try:
        dataset=MMDataset(copy.deepcopy(plan['arguments']['baseline']),mode='test')
        require(len(dataset)==N,'Test ornek sayisi uyusmuyor')
        loader=DataLoader(dataset,batch_size=BATCH,shuffle=False,num_workers=0,drop_last=False)
        batches=[];fp=hashlib.sha256();totals={}
        for batch in loader:
            features=plan['test'].prepare_batch(batch,'K0',plan['stats'],fp,torch.device('cpu'),totals)
            cached={k:batch[k] for k in ('text','audio_lengths','vision_lengths','index')}
            cached.update(features)
            for i,y in zip(batch['index'].tolist(),batch['labels']['M'].reshape(-1).tolist()):
                require(y==plan['records']['baseline',i]['label'],'Gercek veri etiketi farkli')
            batches.append(cached)
        require([int(i) for b in batches for i in b['index']]==list(range(N)),'Veri sirasinda eksik/tekrar var')
        require(len(batches)==172 and len(batches[-1]['index'])==2,'Son batch kayboldu')
        require(fp.hexdigest()==fingerprint(batches)==plan['expected_k0'],'K0 girdileri referansla eslesmiyor')
        del dataset,loader,batch,features,cached
        rows=[];rounds=[];checks=[]
        for cp,(_,checkpoint,plain,weighted) in plan['specs'].items():
            setup_seed(plan['seed'])
            args=copy.deepcopy(plan['arguments'][cp])
            model=AMIO(args)
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=True);del state
            model.to(device).eval().requires_grad_(False)
            require(all(p.dtype==torch.float32 for p in model.parameters()),'FP32 model bekleniyor')
            with torch.inference_mode():
                checks.extend(correctness(model,batches,cp,plain,weighted,plan,device))
                print(f'TAHMIN_KONTROLU_BASARILI: {plain}/{weighted}; 686 ornek; K0 referansi; tolerans={TOLERANCE}')
                for repeat in range(1,plan['repeats']+1):
                    for position,method in enumerate(order_for(repeat,plain,weighted),start=1):
                        measured,summary=measure_round(model,batches,cp,method,repeat,position,plan,device)
                        rows.extend(measured);rounds.append(summary)
                        print(f"TUR_TAMAMLANDI: {method}; {repeat}/{plan['repeats']}; ms/ornek={summary['ms_per_sample']:.4f}; peak_MiB={summary['peak_allocated_mib']:.2f}",flush=True)
            require(not model.Model.post_fusion_layer_1._forward_pre_hooks,'Son durumda hook var')
            del model,args
            gc.collect();torch.cuda.synchronize(device);torch.cuda.empty_cache()
        require(fingerprint(batches)==plan['expected_k0'],'CPU girdileri olcumde degisti')
        summaries,paired,paired_summary=aggregate(rounds,rows,plan['repeats'])
        require(all(digest(p)==h for p,h in plan['hashes'].items()),'Korunan dosya olcumde degisti')
        require(digest(plan['data_path'])==plan['data_sha256'],'Veri dosyasi olcumde degisti')
        tables={'batch_timings.csv':rows,'rounds.csv':rounds,'summary.csv':summaries,
                'paired_rounds.csv':paired,'paired_summary.csv':paired_summary,'correctness_checks.csv':checks}
        for name,values in tables.items():plan['base'].write_csv(output/name,values)
        with (output/'measurement_notes.txt').open('x',encoding='utf-8') as stream:stream.write(NOTES)
        manifest.update(status='completed',finished_at_utc=datetime.now(timezone.utc).isoformat(),
            protected_files_unchanged=True,data_file_unchanged=True,inputs_unchanged=True,
            k0_input_sha256=plan['expected_k0'],reference_predictions_matched=True,
            hooks_restored=True,correctness_rows=len(checks),round_rows=len(rounds),
            timed_batch_rows=len(rows),row_counts={k:len(v) for k,v in tables.items()},
            max_prediction_error=max(r['max_prediction_error'] for r in rounds),
            output_hashes={p.name:digest(p) for p in sorted(output.iterdir()) if p.name!='manifest.json'})
        save_manifest()
        for row in summaries:
            print(f"{row['method']} ms/ornek={row['ms_per_sample_mean']:.4f}+/-{row['ms_per_sample_sd']:.4f}; peak_allocated_MiB={row['peak_allocated_mib_max']:.2f}")
        print('BENCHMARK_TAMAMLANDI:',output)
    except BaseException as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}',finished_at_utc=datetime.now(timezone.utc).isoformat())
        save_manifest();raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-dir',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--seed',type=int,choices=range(1111,1116),default=1111)
    parser.add_argument('--repeats',type=int,default=6)
    parser.add_argument('--warmup-batches',type=int,default=10)
    parser.add_argument('--output-dir',type=Path,help='NEW direct child of project runs; never overwrite')
    parser.add_argument('--check',action='store_true',help='No Self-MM model/dataset/download/measurement or file writes')
    cli=parser.parse_args(argv)
    try:
        plan=prepare(cli)
        if cli.check:
            print('KONTROL_BASARILI: Self-MM/veri yuklenmedi, indirme/olcum/egitim/dosya yazma yapilmadi')
            return 0
        benchmark(plan)
        return 0
    except (OSError,RuntimeError,ValueError,KeyError,TypeError) as error:
        print(f'BENCHMARK_DURDURULDU: {error}',file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
