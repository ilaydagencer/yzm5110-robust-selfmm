"""Create a small source snapshot for a new venv; stdlib only, no downloads."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT_FILES = ('pyproject.toml', 'setup.cfg', 'MANIFEST.in', 'README.md', 'LICENSE')
RESEARCH_FILES = (
    'evaluate_selfmm_portable_k0.py', 'evaluate_selfmm_portable_k0_k7.py',
    'evaluate_selfmm_portable_seed_test.py', 'evaluate_selfmm_abcd_valid.py',
    'selfmm_corruptions.py', 'verify_results_package.py',
    'prepare_selfmm_clean_copy.py', 'check_selfmm_clean_environment.py',
    'requirements-research.txt', 'environment-observed.txt',
    'README.md', 'PORTABLE_MODEL.md', 'PORTABLE_K0_K7.md',
    'PORTABLE_SEED_TEST.md', 'CLEAN_ENVIRONMENT.md', '.gitattributes',
)
CONFIGS = ('selfmm_mosi_portable_k0.json', 'selfmm_portable_k0_dependencies.json',
           'selfmm_portable_k0_k7_dependencies.json', 'selfmm_portable_seed_dependencies.json')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source, target):
    source, target = source.resolve(), target.resolve()
    require(not target.exists(), 'Hedef zaten var; uzerine yazilmaz.')
    require(not target.is_relative_to(source.parent) and not source.is_relative_to(target),
            'Temiz hedef mevcut proje disinda, ayri bir klasor olmali.')
    require(target.parent.is_dir(), 'Hedefin ust klasoru bulunamadi.')
    names = set(ROOT_FILES) | {'research/' + n for n in RESEARCH_FILES}
    names |= {'research/configs/' + n for n in CONFIGS}
    for folder, suffixes in (('src/MMSA', {'.py', '.json'}),
                             ('research/results_package_v1', None),
                             ('research/verification', {'.json'})):
        for p in (source / folder).rglob('*'):
            if p.is_file() and (suffixes is None or p.suffix in suffixes):
                names.add(p.relative_to(source).as_posix())
    files, source_hashes = {}, {}
    for name in sorted(names):
        path = source / name
        require(path.is_file() and path.resolve().is_relative_to(source), f'Eksik/dis kaynak: {name}')
        require(path.stat().st_size < 5 * 1024**2, f'Beklenmeyen buyuk kaynak dosyasi: {name}')
        data = path.read_bytes()
        source_hashes[name] = digest(data)
        if name == 'src/MMSA/config/config_regression.json':
            config = json.loads(data.decode('utf-8-sig'))
            config['datasetCommonParams']['dataset_root_dir'] = './data'
            data = (json.dumps(config, indent=2, ensure_ascii=False)+'\n').encode('utf-8')
        files[name] = data
    registry = json.loads(files['research/configs/selfmm_portable_seed_dependencies.json'])
    for name, expected in registry['source_sha256_lf'].items():
        require(name in files and digest(files[name].replace(b'\r\n', b'\n')) == expected,
                f'Dondurulmus kaynak farkli: {name}')
    verifier = source / 'research/verify_results_package.py'
    spec = importlib.util.spec_from_file_location('clean_package_verifier', verifier)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify(source / 'research/results_package_v1')
    require(digest(files['research/results_package_v1/manifest.json']) == registry['package_manifest_sha256'],
            'Public paket kimligi farkli.')
    return source, target, files, source_hashes


def create_copy(source, target, files, source_hashes):
    target.mkdir(exist_ok=False)
    manifest_path = target / 'source_snapshot.json'
    record = dict(schema_version=1, status='preparing', source_repo=str(source),
                  snapshot_kind='allowlisted local source copy, not a Git clone',
                  source_sha256=source_hashes, copied_sha256={n:digest(b) for n,b in files.items()},
                  normalized_default_config='Only dataset_root_dir -> ./data in copied config_regression.json; portable runners use their explicit config.',
                  data_checkpoints_bert_copied=False, historical_runs_copied=False,
                  packages_copied=False, fresh_dependency_install_verified=False,
                  gpu_replay_verified=False)
    try:
        for name, data in files.items():
            destination = target / 'MMSA' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(data)
            require(digest(destination.read_bytes()) == record['copied_sha256'][name], f'Kopya farkli: {name}')
        require(all(digest((source/n).read_bytes()) == h for n,h in source_hashes.items()),
                'Kopyalama sirasinda kaynak degisti.')
        record.update(status='completed', source_files_unchanged=True, copied_file_count=len(files))
    except BaseException as exc:
        record.update(status='failed_or_interrupted', error=str(exc))
        raise
    finally:
        with manifest_path.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--target-root', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    plan = prepare(args.source_repo, args.target_root)
    print(f'KOPYA_PLANI: {len(plan[2])} kucuk dosya; hedef={plan[1]}')
    if args.check:
        print('KONTROL_BASARILI: kaynak/paket/hedef; indirme, venv, model ve dosya yazma yok.')
        return
    create_copy(*plan)
    print('TEMIZ_KAYNAK_KOPYASI_HAZIR:', plan[1])
    print('SONRAKI_ADIM: yeni venv ve sifirdan paket kurulumu; ortam/GPU henuz dogrulanmadi.')


if __name__ == '__main__':
    main()
