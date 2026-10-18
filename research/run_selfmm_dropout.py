"""Self-MM / MOSI: train-only, independent full-modality input dropout.

Run --check first. No MMSA source files or baseline checkpoints are modified.
"""
import argparse
import ast
import csv
import hashlib
import inspect
import json
import time
from contextlib import contextmanager
from pathlib import Path

# Windows DLL/import order: this MUST precede torch and MMSA.
import sentencepiece
import torch
import MMSA
from MMSA import MMSA_run
from MMSA.config import get_config_regression
from MMSA.models.multiTask.SELF_MM import SELF_MM

SEED = 1111
PROBABILITY = 0.20


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class InputDropout:
    def __init__(self, seed=SEED, probability=PROBABILITY):
        require(0 <= probability <= 1, 'Gecersiz dropout olasiligi')
        self.p = probability
        # Dedicated CPU generator: do not consume model/DataLoader RNG state.
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.counts = dict(train_samples=0, audio_dropped=0,
                           vision_dropped=0, both_dropped=0,
                           eval_samples=0, eval_dropped=0)

    def transform(self, model, text, audio, video):
        require(isinstance(audio, tuple) and len(audio) == 2,
                'Ses (tensor, uzunluklar) olmali')
        require(isinstance(video, tuple) and len(video) == 2,
                'Goruntu (tensor, uzunluklar) olmali')
        a, al = audio
        v, vl = video
        require(a.ndim == v.ndim == 3 and a.shape[0] == v.shape[0] == text.shape[0],
                'Beklenen girdi bicimi: [batch, zaman, ozellik]')
        n = a.shape[0]
        if not model.training:
            self.counts['eval_samples'] += n
            return text, audio, video
        # Each row and each modality gets an independent Bernoulli draw.
        masks = torch.rand((n, 2), generator=self.generator) < self.p
        ma, mv = masks[:, 0], masks[:, 1]
        self.counts['train_samples'] += n
        self.counts['audio_dropped'] += int(ma.sum())
        self.counts['vision_dropped'] += int(mv.sum())
        self.counts['both_dropped'] += int((ma & mv).sum())
        # Out-of-place masking; keep sequence lengths and text unchanged.
        # No inverted-dropout scaling. masked_fill also clears nonfinite entries.
        a_out = a.masked_fill(ma.to(a.device).view(n, 1, 1), 0)
        v_out = v.masked_fill(mv.to(v.device).view(n, 1, 1), 0)
        return text, (a_out, al), (v_out, vl)

    def report(self):
        n = self.counts['train_samples']
        return {**self.counts, 'audio_rate': self.counts['audio_dropped'] / n if n else 0,
                'vision_rate': self.counts['vision_dropped'] / n if n else 0,
                'both_rate': self.counts['both_dropped'] / n if n else 0}


@contextmanager
def patched_forward(cls, dropout):
    original = cls.forward
    require(list(inspect.signature(original).parameters) == ['self', 'text', 'audio', 'video'],
            'Self-MM forward imzasi degismis; kaynaklari kontrol edin')

    def forward(self, text, audio, video):
        text, audio, video = dropout.transform(self, text, audio, video)
        return original(self, text, audio, video)

    cls.forward = forward
    try:
        yield
    finally:
        cls.forward = original


def check_dropout():
    # Exercise the SAME wrapper without constructing BERT or loading the dataset.
    class Probe(torch.nn.Module):
        def forward(self, text, audio, video):
            return text, audio, video

    for p in (0.0, 0.2, 1.0):
        d = InputDropout(probability=p)
        model = Probe()
        original = Probe.forward
        t = torch.ones(10000, 3, 2)
        a = torch.ones(10000, 2, 5, requires_grad=True)
        v = torch.ones(10000, 3, 20)
        lengths = torch.full((10000,), 2, dtype=torch.long)
        ap, vp = (a, lengths), (v, lengths)
        with patched_forward(Probe, d):
            model.eval()
            rng_before = d.generator.get_state().clone()
            te, ae, ve = model(t, ap, vp)
            require(te is t and ae is ap and ve is vp, 'Eval girdisi degisti')
            require(torch.equal(rng_before, d.generator.get_state()), 'Eval RNG tuketti')
            model.train()
            tt, at, vt = model(t, ap, vp)
            require(tt is t and at[1] is lengths and vt[1] is lengths,
                    'Metin/uzunluklar degisti')
            require(bool((a == 1).all()) and bool((v == 1).all()), 'Asil girdi degisti')
            for output in (at[0], vt[0]):
                rows = output.flatten(1)
                require(bool(((rows == 0).all(1) | (rows == 1).all(1)).all()),
                        'Tam modalite yerine kismi maskeleme')
            at[0].sum().backward()
            require(torch.equal(a.grad, at[0].detach()), 'Gradyan maskesi hatali')
            twin = InputDropout(probability=p)
            _, ar, vr = twin.transform(model, t, ap, vp)
            require(torch.equal(ar[0], at[0]) and torch.equal(vr[0], vt[0]),
                    'Seed tekrarlanabilir degil')
        require(Probe.forward is original, 'Patch geri alinmadi')
        r = d.report()
        require(r['eval_dropped'] == 0, 'Eval dropout tespit edildi')
        if p == 0.2:
            require(abs(r['audio_rate'] - p) < .02 and abs(r['vision_rate'] - p) < .02
                    and abs(r['both_rate'] - p*p) < .015, 'Bagimsizlik/oran kontrolu basarisiz')
        else:
            require(r['audio_rate'] == p and r['vision_rate'] == p, 'Sinir testi basarisiz')
    print('DROPOUT_KONTROLU_BASARILI: train/eval, metin, uzunluk, seed, gradyan, patch')


def compare_results(baseline, experiment):
    # MMSA_run multiplies ALL CSV metrics (including MAE and Corr) by 100.
    def read(path):
        with path.open(encoding='utf-8', newline='') as stream:
            rows = [r for r in csv.DictReader(stream) if r['Model'] == 'self_mm']
        require(len(rows) == 1, f'Tek self_mm satiri bekleniyor: {path}')
        return {k: ast.literal_eval(v)[0] / 100 for k, v in rows[0].items() if k != 'Model'}
    b, e = read(baseline), read(experiment)
    result = {k: {'baseline': b[k], 'dropout': e[k], 'delta': e[k]-b[k]}
              for k in b.keys() & e.keys()}
    print('KARSILASTIRMA (ham olcek; delta=dropout-baseline):')
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--repo-dir', type=Path,
                        default=Path('C:/Users/ilayda/YZM5110/MMSA'))
    cli = parser.parse_args()
    repo = cli.repo_dir.resolve()
    project = repo.parent
    source = repo / 'src/MMSA/config/config_regression.json'
    baseline_config = project / 'local_configs/selfmm_mosi_batch4.json'
    local = project / 'local_configs/selfmm_mosi_dropout_a20_v20_batch4_seed1111.json'
    run = project / 'runs/selfmm_dropout_a20_v20_seed1111'
    require(Path(MMSA.__file__).resolve().is_relative_to(repo / 'src'),
            f'Yanlis MMSA kurulumu: {MMSA.__file__}')
    for path in (source, baseline_config, project / 'data/MOSI/Processed/unaligned_50.pkl'):
        require(path.is_file(), f'Dosya bulunamadi: {path}')
    config = json.loads(source.read_text(encoding='utf-8'))
    config['datasetCommonParams']['dataset_root_dir'] = (project / 'data').as_posix()
    config['self_mm']['datasetParams']['mosi']['batch_size'] = 4
    require(config == json.loads(baseline_config.read_text(encoding='utf-8')),
            'Baseline ayarlari beklenenden farkli; karsilastirmadan once inceleyin')
    # Resolve baseline arguments with the actual repository config loader.
    args = get_config_regression('self_mm', 'mosi', str(baseline_config))
    require(args.batch_size == 4 and not args.need_data_aligned and not args.get('data_missing', False),
            'Batch/alignment/data_missing ayarlari uyumsuz')
    require(list(inspect.signature(SELF_MM.forward).parameters) == ['self', 'text', 'audio', 'video'],
            'Gercek forward imzasi uyumsuz')
    print('MMSA:', MMSA.__file__)
    print('CONFIG:', local)
    print('RUN:', run)
    print('BATCH: 4; SEED: 1111; AUDIO: 0.20; VISION: 0.20; TEXT: unchanged')
    print('BERT_FINETUNE:', args.use_finetune, 'CUDA:', torch.cuda.is_available())
    check_dropout()
    if cli.check:
        require(not run.exists(), f'Deney klasoru zaten var; uzerine yazilmaz: {run}')
        require(not local.exists(), f'Yerel config zaten var; uzerine yazilmaz: {local}')
        print('KONTROL_BASARILI: egitim/model indirme/veri yukleme/dosya yazma yapilmadi')
        return
    require(torch.cuda.is_available(), 'CUDA bulunamadi; venv/GPU kurulumunu kontrol edin')
    require(not local.exists(), f'Yerel config zaten var; uzerine yazilmaz: {local}')
    # Atomic reservation: even interrupted runs cannot silently be overwritten.
    run.mkdir(parents=True, exist_ok=False)
    local.parent.mkdir(parents=True, exist_ok=True)
    with local.open('x', encoding='utf-8') as stream:
        json.dump(config, stream, indent=2)
    d = InputDropout()
    protected = [source, baseline_config, repo / 'research/run_selfmm_smoke.py']
    protected += list((project / 'runs/selfmm_seed1111/saved_models').glob('*.pth'))
    before = {str(p): sha256(p) for p in protected}
    metadata = dict(seed=SEED, audio_probability=.2, vision_probability=.2,
                    torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
                    protected_hashes=before, status='running')
    (run / 'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    try:
        with patched_forward(SELF_MM, d):
            MMSA_run('self_mm', 'mosi', config_file=str(local),
                     seeds=[SEED], gpu_ids=[0], num_workers=0,
                     model_save_dir=str(run / 'saved_models'),
                     res_save_dir=str(run / 'results'), log_dir=str(run / 'logs'))
        require(d.counts['train_samples'] > 0 and d.counts['eval_samples'] > 0,
                'Train/eval forward gozlenmedi')
        require(d.counts['eval_dropped'] == 0, 'Eval dropout korumasi basarisiz')
        baseline_csv = project / 'runs/selfmm_seed1111/results/normal/mosi.csv'
        if baseline_csv.is_file():
            metadata['comparison'] = compare_results(baseline_csv, run / 'results/normal/mosi.csv')
        metadata['status'] = 'completed'
    finally:
        metadata.update(dropout_counts=d.report(), elapsed_seconds=time.perf_counter()-start,
                        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated(0)/1024**2,
                        peak_gpu_reserved_mb=torch.cuda.max_memory_reserved(0)/1024**2,
                        protected_files_unchanged=all(sha256(Path(p)) == h for p, h in before.items()))
        if metadata['status'] == 'running':
            metadata['status'] = 'failed_or_interrupted'
        (run / 'experiment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
