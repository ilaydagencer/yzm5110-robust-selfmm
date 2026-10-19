"""Verify the isolated venv, installed dependencies and relocated MMSA; no model loading."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import re
import site
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_requirements(path):
    result = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        require(re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9.+_-]+', line), 'Beklenmeyen requirements satiri.')
        name, version = line.split('==')
        require(name not in result, 'Tekrarlanan paket.')
        result[name] = version
    return result


def verify(venv, repo):
    venv, repo = venv.resolve(), repo.resolve()
    require(sys.flags.isolated == 1, 'Bu kontrolu Python -I -B ile calistirin.')
    require(sys.version_info[:3] == (3, 10, 5) and struct.calcsize('P') == 8,
            'Gozlenen kurulum icin Python 3.10.5 64-bit bekleniyor.')
    require(Path(sys.prefix).resolve() == venv and sys.prefix != sys.base_prefix,
            'Yanlis Python/venv; yeni ortamin Scripts/python.exe yolunu kullanin.')
    require(Path(sys.executable).resolve().is_relative_to(venv), 'Python yeni venv disinda.')
    require(site.ENABLE_USER_SITE is False, 'Kullanici paketleri etkin.')
    cfg = (venv/'pyvenv.cfg').read_text(encoding='utf-8')
    require(re.search(r'^include-system-site-packages\s*=\s*false\s*$', cfg, re.M | re.I),
            'Sistem paketleri venv icinde gorunuyor.')
    require(repo.parent == venv.parent and repo.name == 'MMSA', 'Kaynak ve venv ayni temiz kokte olmali.')
    snapshot_path = repo.parent/'source_snapshot.json'
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    require(snapshot['status'] == 'completed' and snapshot['source_files_unchanged'] is True,
            'Kaynak kopyasi kaydi tamamlanmamis.')
    require(Path(snapshot['source_repo']).resolve() != repo, 'Eski repo kullaniliyor.')
    for name, expected in snapshot['copied_sha256'].items():
        path = (repo/name).resolve()
        require(path.is_relative_to(repo) and path.is_file() and sha(path) == expected,
                f'Kopyalanan kaynak degismis: {name}')
    expected = pinned_requirements(repo/'research/requirements-research.txt')
    expected.update(torch='2.3.1+cu121', MMSA='2.2.1', pip='24.0', setuptools='69.5.1', wheel='0.43.0')
    distributions = {}
    for name, version in expected.items():
        dist = metadata.distribution(name)
        require(dist.version == version, f'Paket surumu farkli: {name}={dist.version}, beklenen={version}')
        origin = Path(dist.locate_file('')).resolve()
        require(origin.is_relative_to(venv) or (name == 'MMSA' and origin.is_relative_to(repo)),
                f'Paket yeni ortam disindan geliyor: {name}: {origin}')
        distributions[name] = dict(version=dist.version, location=str(origin))
    # Preserve the verified Windows import ordering.
    import sentencepiece
    import torch
    import numpy
    import transformers
    import MMSA
    require(Path(MMSA.__file__).resolve() == repo/'src/MMSA/__init__.py', 'MMSA eski/yanlis kaynak klasorunden geliyor.')
    for module in (sentencepiece, torch, numpy, transformers):
        require(Path(module.__file__).resolve().is_relative_to(venv), f'Eski paket yolu: {module.__name__}')
    for entry in sys.path:
        if 'site-packages' in entry.lower():
            require(Path(entry).resolve().is_relative_to(venv), 'Dis site-packages sys.path icinde.')
    check = subprocess.run([sys.executable, '-I', '-B', '-m', 'pip', '--isolated',
                            '--disable-pip-version-check', 'check'], capture_output=True, text=True)
    require(check.returncode == 0, 'pip check basarisiz: '+check.stdout+check.stderr)
    require(torch.cuda.is_available(), 'CUDA gorunmuyor; GPU tekrarindan once incelenmeli.')
    require(torch.version.cuda == '12.1', 'Beklenen CUDA runtime 12.1.')
    return dict(status='environment_checks_passed', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                python=sys.version, executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix,
                platform=platform.platform(), isolated=True, system_site_packages=False,
                user_site_enabled=site.ENABLE_USER_SITE, mmsa_file=MMSA.__file__, packages=distributions,
                pip_check=check.stdout.strip(), cuda_available=True, cuda_runtime=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                snapshot_manifest_sha256=sha(snapshot_path), source_files_verified=len(snapshot['copied_sha256']),
                fresh_install_history='Consult separate pip install logs; checks verify current isolation, versions and paths.',
                same_machine_and_base_python=True, gpu_model_replay_performed=False,
                training_performed=False, model_or_dataset_loaded=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-dir', type=Path, required=True)
    parser.add_argument('--venv-dir', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.output_dir:
        output = args.output_dir.resolve()
        root = args.repo_dir.resolve().parent
        require(output.is_relative_to(root/'checks') and not output.exists(),
                'Yeni cikti temiz kokun checks klasoru altinda olmali; mevcut hedef kullanilmaz.')
    result = verify(args.venv_dir, args.repo_dir)
    print('ORTAM_KONTROLU_BASARILI:', result['executable'])
    print('MMSA:', result['mmsa_file'])
    print('PIP_CHECK:', result['pip_check'])
    print('CUDA:', result['cuda_runtime'], 'GPU:', result['gpu'])
    print('KAPSAM: izole venv, paket surumleri/yollari ve kaynak kopyasi; gercek model/GPU sonucu henuz yok.')
    if args.check:
        print('DOSYA_YAZILMADI')
        return
    output.mkdir(parents=True, exist_ok=False)
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('ORTAM_KAYDI:', output/'manifest.json')


if __name__ == '__main__':
    main()
