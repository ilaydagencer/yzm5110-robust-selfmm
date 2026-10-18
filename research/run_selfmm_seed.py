"""Run ONE matched Self-MM/MOSI training repeat, preserving seed 1111.

Copy into research beside run_selfmm_dropout.py. Start with:
  python -B research/run_selfmm_seed.py --variant baseline --seed 1112 --check

--check constructs no model, loads no dataset, and writes no files.
Normal execution uses MMSA_run, including its automatic final TEST evaluation.
Checkpoint selection remains validation Loss; no tuning is performed.
"""
import argparse
import ast
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Also suppress import caches when the user omits -B.
sys.dont_write_bytecode = True
# Windows DLL requirement: MUST precede torch and MMSA.
import sentencepiece
import torch
import MMSA
from MMSA import MMSA_run
from MMSA.config import get_config_regression
from MMSA.models.multiTask.SELF_MM import SELF_MM

SEEDS = (1112, 1113, 1114, 1115)
PROBABILITY = 0.20
EXPECTED_DATA_SHA256 = '78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require_unused(*paths):
    for path in paths:
        require(not os.path.lexists(path), f'Hedef zaten var; uzerine yazilmaz: {path}')


def load_dropout_helpers(repo):
    # Always load the actual repo helper, even when this runner is in outputs/.
    path = repo / 'research/run_selfmm_dropout.py'
    require(path.is_file(), f'Dropout yardimci betigi eksik: {path}')
    spec = importlib.util.spec_from_file_location('selfmm_seed_dropout_helpers', path)
    require(spec is not None and spec.loader is not None, 'Dropout modulu yuklenemedi')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.PROBABILITY == PROBABILITY, 'Referans dropout olasiligi degismis')
    return module


@dataclass
class RunPlan:
    repo: Path
    variant: str
    seed: int
    config: dict
    arguments: dict
    local_config: Path
    run_dir: Path
    data_path: Path
    helpers: object
    protected: list


def prepare(repo, variant, seed):
    require(variant in ('baseline', 'dropout') and seed in SEEDS,
            'Yontem baseline/dropout, seed 1112-1115 olmali; 1111 korunuyor')
    repo = Path(repo).resolve()
    project = repo.parent
    require(Path(MMSA.__file__).resolve().is_relative_to(repo / 'src'),
            f'Yanlis MMSA kurulumu: {MMSA.__file__}; --repo-dir yolunu kontrol edin')
    baseline = project / 'local_configs/selfmm_mosi_batch4.json'
    dropout = project / 'local_configs/selfmm_mosi_dropout_a20_v20_batch4_seed1111.json'
    source = repo / 'src/MMSA/config/config_regression.json'
    for path in (baseline, dropout, source):
        require(path.is_file(), f'Referans config eksik: {path}')
    config = json.loads(baseline.read_text(encoding='utf-8'))
    require(config == json.loads(dropout.read_text(encoding='utf-8')),
            'Seed 1111 baseline/dropout configleri eslesmiyor')
    expected = json.loads(source.read_text(encoding='utf-8'))
    expected['datasetCommonParams']['dataset_root_dir'] = (project / 'data').as_posix()
    expected['self_mm']['datasetParams']['mosi']['batch_size'] = 4
    require(config == expected, 'Kaynak config ile seed 1111 ayarlari uyusmuyor')
    args = get_config_regression('self_mm', 'mosi', str(baseline))
    require(args.batch_size == 4 and args.use_bert and args.use_finetune
            and not args.need_data_aligned and not args.need_model_aligned
            and not args.get('data_missing', False) and args.KeyEval == 'Loss'
            and args.update_epochs == 4 and args.early_stop == 8,
            'Batch/BERT/alignment/early-stopping ayarlari beklenenden farkli')
    data_path = Path(args.featurePath).resolve()
    require(data_path == (project / 'data/MOSI/Processed/unaligned_50.pkl').resolve()
            and data_path.is_file(), f'Beklenen MOSI verisi bulunamadi: {data_path}')
    require(list(inspect.signature(SELF_MM.forward).parameters)
            == ['self', 'text', 'audio', 'video'], 'Self-MM forward imzasi degismis')
    label = 'baseline' if variant == 'baseline' else 'dropout_a20_v20'
    local = project / 'local_configs' / f'selfmm_mosi_{label}_batch4_seed{seed}.json'
    run = project / 'runs' / f'selfmm_{label}_seed{seed}'
    require_unused(local, run)
    protected = [baseline, dropout, Path(__file__).resolve(),
                 repo / 'research/run_selfmm_smoke.py', repo / 'research/run_selfmm_dropout.py']
    # Record source content as well as Git HEAD: HEAD alone omits local edits.
    protected += [p for p in (repo / 'src/MMSA').rglob('*')
                  if p.is_file() and p.suffix in ('.py', '.json')]
    for name in ('selfmm_seed1111', 'selfmm_dropout_a20_v20_seed1111'):
        previous = project / 'runs' / name
        require((previous / 'saved_models/self_mm-mosi.pth').is_file(),
                f'Seed 1111 checkpoint eksik: {previous}')
        protected.extend(p for p in previous.rglob('*') if p.is_file())
    protected = sorted(set(protected), key=str)
    require(all(p.is_file() for p in protected), 'Korunacak referans dosyalardan biri eksik')
    return RunPlan(repo, variant, seed, copy.deepcopy(config), dict(args), local,
                   run, data_path, load_dropout_helpers(repo), protected)


def self_check(helpers, seed):
    # Tests must not advance the model's RNG before MMSA seeds its training.
    before = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        helpers.check_dropout()

        class Probe(torch.nn.Module):
            def forward(self, text, audio, video):
                return text, audio, video

        model = Probe().train()
        text = torch.ones(2048, 3, 2)
        lengths = torch.ones(2048, dtype=torch.long)
        audio = (torch.ones(2048, 2, 5), lengths)
        vision = (torch.ones(2048, 3, 20), lengths)
        first = helpers.InputDropout(seed=seed, probability=PROBABILITY)
        twin = helpers.InputDropout(seed=seed, probability=PROBABILITY)
        other = helpers.InputDropout(seed=seed + 1, probability=PROBABILITY)
        _, a, v = first.transform(model, text, audio, vision)
        _, a2, v2 = twin.transform(model, text, audio, vision)
        _, a3, v3 = other.transform(model, text, audio, vision)
        require(torch.equal(a[0], a2[0]) and torch.equal(v[0], v2[0]),
                'Secilen seed ayni maskeleri uretmiyor')
        require(not torch.equal(a[0], a3[0]) and not torch.equal(v[0], v3[0]),
                'Farkli seed maske cekilisini degistirmedi')
        original = Probe.forward
        try:
            with helpers.patched_forward(Probe, first):
                raise ValueError('intentional restore test')
        except ValueError:
            pass
        require(Probe.forward is original, 'Hata sonrasi forward geri yuklenmedi')
    require(torch.equal(before, torch.random.get_rng_state()), 'Kontroller global RNG durumunu degistirdi')
    print(f'SEED_KONTROLU_BASARILI: {seed}, ayni/farkli seed, RNG ve hata sonrasi patch')


def git_revision(repo):
    git = shutil.which('git')
    require(git is not None, 'Git bulunamadi; kaynak revizyonu kaydedilemiyor')
    result = subprocess.run([git, f'--git-dir={repo / ".git"}', 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, timeout=15, check=True)
    revision = result.stdout.strip()
    require(len(revision) == 40 and all(c in '0123456789abcdef' for c in revision),
            'Git HEAD okunamadi')
    return revision


def package_versions():
    versions = {}
    for name in ('MMSA', 'torch', 'numpy', 'transformers', 'sentencepiece', 'pandas', 'scikit-learn'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def changed_files(hashes):
    changed = []
    for name, expected in hashes.items():
        try:
            if sha256(name) != expected:
                changed.append(name)
        except OSError:
            changed.append(name)
    return changed


def read_final_test_metrics(path):
    # MMSA's CSV multiplies every metric, including MAE/Corr/Loss, by 100.
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 1 and rows[0].get('Model') == 'self_mm',
            'Yeni sonuc CSV dosyasinda tek self_mm satiri bekleniyor')
    metrics = {}
    for key, value in rows[0].items():
        if key == 'Model':
            continue
        mean, std = ast.literal_eval(value)
        require(math.isfinite(mean) and math.isfinite(std) and std == 0,
                f'Tek seed sonuc bicimi gecersiz: {key}')
        metrics[key] = mean / 100
    require({'Non0_acc_2', 'Non0_F1_score', 'MAE', 'Corr'} <= metrics.keys(), 'Sonuc metrikleri eksik')
    return metrics


def execute(plan, revision):
    require_unused(plan.local_config, plan.run_dir)
    require(torch.cuda.is_available(), 'CUDA bulunamadi; egitim baslatilmadi')
    torch.cuda.set_device(0)
    before = {str(p): sha256(p) for p in plan.protected}
    require(sha256(plan.data_path) == EXPECTED_DATA_SHA256, 'MOSI veri ozeti referanstan farkli')
    options = dict(config_file=str(plan.local_config), seeds=[plan.seed], is_tune=False,
                   gpu_ids=[0], num_workers=0, model_save_dir=str(plan.run_dir / 'saved_models'),
                   res_save_dir=str(plan.run_dir / 'results'), log_dir=str(plan.run_dir / 'logs'))
    metadata = dict(status='preparing', variant=plan.variant, seed=plan.seed, batch_size=4,
        created_at_utc=utc_now(), git_head=revision, protected_hashes=before,
        source_note='Git HEAD plus actual file hashes; no claim that the worktree is clean.',
        data_path=str(plan.data_path), data_sha256=EXPECTED_DATA_SHA256,
        resolved_model_config=plan.arguments, local_config=str(plan.local_config),
        python_version=sys.version, platform=platform.platform(), packages=package_versions(),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), mmsa_call=dict(model_name='self_mm', dataset_name='mosi', **options),
        dropout=dict(enabled=plan.variant == 'dropout', probability=PROBABILITY if plan.variant == 'dropout' else 0,
                     generator_seed=plan.seed if plan.variant == 'dropout' else None,
                     location='full audio/vision inputs; independent per sample; train only', text_unchanged=True),
        checkpoint_selection='Validation Loss from original trainer (mean of batch L1 losses).',
        automatic_final_test=True, hyperparameter_tuning=False,
        selection_note='Final test is logged by MMSA; use validation for subsequent method decisions.',
        timing_scope='Total MMSA_run call: loading, training, validation, final test and cleanup; not inference latency.',
        csv_note='MMSA CSV scales all metrics by 100; normalized metrics retain CSV rounding. Single-seed std=0 is not uncertainty.')
    # Reserve the entire run first. Never reuse even an interrupted run folder.
    plan.run_dir.mkdir(parents=True, exist_ok=False)
    record = plan.run_dir / 'experiment.json'
    dropout = None
    original_forward = SELF_MM.forward
    expected_config_hash = None
    elapsed = None
    peak_allocated = peak_reserved = None
    memory_warning = None
    try:
        plan.local_config.parent.mkdir(parents=True, exist_ok=True)
        with plan.local_config.open('x', encoding='utf-8') as stream:
            json.dump(plan.config, stream, indent=2)
        expected_config_hash = sha256(plan.local_config)
        require(dict(get_config_regression('self_mm', 'mosi', str(plan.local_config))) == plan.arguments,
                'Yeni config cozumlemesi seed 1111 ile ayni degil')
        if plan.variant == 'dropout':
            dropout = plan.helpers.InputDropout(seed=plan.seed, probability=PROBABILITY)
        metadata.update(status='running', mmsa_started_at_utc=utc_now(), local_config_sha256=expected_config_hash)
        record.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        context = plan.helpers.patched_forward(SELF_MM, dropout) if dropout is not None else nullcontext()
        start = time.perf_counter()
        try:
            with context:
                MMSA_run('self_mm', 'mosi', **options)
        finally:
            try:
                torch.cuda.synchronize(0)
            except RuntimeError as error:
                memory_warning = str(error)
            elapsed = time.perf_counter() - start
            try:
                peak_allocated = torch.cuda.max_memory_allocated(0) / 1024 ** 2
                peak_reserved = torch.cuda.max_memory_reserved(0) / 1024 ** 2
            except RuntimeError as error:
                memory_warning = str(error)
        if dropout is not None:
            require(dropout.counts['train_samples'] > 0 and dropout.counts['eval_samples'] > 0,
                    'Dropout wrapper icinde train/eval forward gozlenmedi')
        checkpoint = plan.run_dir / 'saved_models/self_mm-mosi.pth'
        require(checkpoint.is_file() and checkpoint.stat().st_size > 0, 'Yeni checkpoint olusmadi')
        metadata.update(checkpoint_path=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                        final_test_metrics=read_final_test_metrics(plan.run_dir / 'results/normal/mosi.csv'),
                        status='completed')
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error=f'{type(error).__name__}: {error}')
        raise
    finally:
        changed = changed_files(before)
        data_changed = changed_files({str(plan.data_path): EXPECTED_DATA_SHA256})
        config_ok = expected_config_hash is not None and not changed_files({str(plan.local_config): expected_config_hash})
        restored = SELF_MM.forward is original_forward
        if metadata['status'] == 'completed' and (changed or data_changed or not config_ok or not restored):
            metadata['status'] = 'failed_integrity_check'
        metadata.update(finished_at_utc=utc_now(), mmsa_total_seconds=elapsed,
                        peak_gpu_allocated_mib=peak_allocated, peak_gpu_reserved_mib=peak_reserved,
                        resource_measurement_warning=memory_warning,
                        protected_files_unchanged=not changed, changed_protected_files=changed,
                        data_file_unchanged=not data_changed, local_config_unchanged=config_ok,
                        forward_restored=restored,
                        dropout_counts=dropout.report() if dropout is not None else None)
        record.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print('DENEY_DURUMU:', metadata['status'], 'KAYIT:', record, flush=True)
    require(metadata['status'] == 'completed', 'Butunluk kontrolu basarisiz; experiment.json kaydini inceleyin')
    print('MMSA_TOPLAM_SURE_SANIYE:', round(elapsed, 2))
    print('PEAK_GPU_ALLOCATED_MIB:', peak_allocated, 'PEAK_GPU_RESERVED_MIB:', peak_reserved)
    print('SEED_EGITIMI_TAMAMLANDI:', plan.variant, plan.seed, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('baseline', 'dropout'), required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--repo-dir', type=Path, default=Path(__file__).resolve().parents[1])
    cli = parser.parse_args(argv)
    plan = prepare(cli.repo_dir, cli.variant, cli.seed)
    revision = git_revision(plan.repo)
    self_check(plan.helpers, plan.seed)
    print('MMSA:', MMSA.__file__)
    print('GIT_HEAD:', revision)
    print('YONTEM:', plan.variant, 'SEED:', plan.seed, 'BATCH: 4; BERT_FINETUNE: True')
    print('CONFIG:', plan.local_config)
    print('RUN:', plan.run_dir)
    print('CUDA:', torch.cuda.is_available(), 'DROPOUT:', 'A=.20 V=.20 train only' if plan.variant == 'dropout' else 'kapali')
    print('NOT: MMSA_run egitim sonunda otomatik TEST olcumu yapar; tuning kapali.')
    if cli.check:
        print('KONTROL_BASARILI: model kurulmadan/veri yuklenmeden/dosya yazilmadan tamamlandi')
        return
    execute(plan, revision)


if __name__ == '__main__':
    main()
