"""Run every check a reader can do without a GPU, dataset or checkpoint.

Standard library only. Usage from the repository root (the MMSA folder):

    py -3.10 -B research\\run_cpu_checks.py

Each check runs in its own subprocess and is reported as PASS, FAIL or SKIP.
SKIP means an optional dependency is missing and says how to add it:
  * bootstrap byte check  -> numpy 1.26.4 (any venv:  pip install numpy==1.26.4)
  * figure redraw         -> research/requirements-figures.txt
  * diagnostics self-test -> the research environment of research/README.md
                             (torch 2.3.1, requirements-research.txt, this MMSA copy installed)
Pass another interpreter for the optional checks with --python. Nothing in the
repository is written; the figure check draws into a temporary folder.
Exit code is 1 if any check fails, 0 otherwise.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(python, args, expect, timeout=900):
    try:
        proc = subprocess.run([python, *args], cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, 'zaman asimi'
    output = (proc.stdout + proc.stderr).strip()
    ok = proc.returncode == 0 and expect in output
    lines = [line for line in output.splitlines() if line.strip()]
    return ok, (lines[-1] if lines else '') if ok else '\n'.join(lines[-6:])


def probe(python, code):
    proc = subprocess.run([python, '-c', code], cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                          errors='replace')
    return proc.returncode == 0, (proc.stdout or proc.stderr).strip().splitlines()[-1:] or ['']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python', default=sys.executable,
                        help='Interpreter for the optional numpy/matplotlib/torch checks (default: this one)')
    cli = parser.parse_args(argv)
    base, extra = sys.executable, cli.python
    results = []

    def check(name, python, args, expect, needs=None):
        if needs:
            ok, detail = needs
            if not ok:
                results.append((name, 'SKIP', detail))
                print(f'[SKIP] {name}: {detail}', flush=True)
                return
        print(f'[....] {name}', flush=True)
        ok, detail = run(python, args, expect)
        results.append((name, 'PASS' if ok else 'FAIL', detail))
        print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}', flush=True)

    check('Sonuc paketi v1 (ortalama/SS/farklar, kaynak olcumu)', base,
          ['-I', '-S', '-B', 'research/verify_results_package.py', '--check'], 'PAKET_KONTROLU_BASARILI')
    for split in ('validation', 'evaluation_test'):
        check(f'Protokol v2 ozeti: {split} (hipotez kararlari, tau secimi)', base,
              ['-I', '-S', '-B', 'research/summarize_selfmm_diagnostics.py', '--verify', f'research/diagnostics_v2/{split}'],
              'OZET_DOGRULANDI')
    check('Rapor tablolari (tables.md byte olarak)', base,
          ['-I', '-S', '-B', 'research/make_selfmm_report_tables.py', '--verify'], 'TABLOLAR_DOGRULANDI')

    ok, out = probe(extra, 'import numpy; print(numpy.__version__)')
    numpy_ok = (ok and out[0] == '1.26.4', 'numpy 1.26.4 gerekli (pip install numpy==1.26.4); bulunan: '
                + (out[0] if ok else 'yok'))
    check('Bootstrap araliklari (bootstrap_test.csv byte olarak)', extra,
          ['-B', 'research/bootstrap_selfmm_diagnostics.py', '--verify', 'research/diagnostics_v2/bootstrap'],
          'BOOTSTRAP_DOGRULANDI', numpy_ok)

    ok, out = probe(extra, 'import matplotlib; print(matplotlib.__version__)')
    figure_ok = (ok, 'matplotlib yok (research/requirements-figures.txt)')
    temporary = Path(tempfile.mkdtemp(prefix='selfmm_figures_'))
    try:
        check('Figurler (gecici klasore yeniden cizim)', extra,
              ['-B', 'research/make_selfmm_figures.py', '--output-dir', str(temporary / 'figures')],
              'FIGURLER_YAZILDI', figure_ok)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    code = ('import sentencepiece, torch, MMSA, pathlib; '
            f'print(pathlib.Path(MMSA.__file__).resolve().is_relative_to(pathlib.Path(r"{ROOT}").resolve()))')
    ok, out = probe(extra, code)
    torch_ok = (ok and out[0] == 'True',
                'arastirma ortami yok ya da MMSA bu kopyadan kurulmamis (research/README.md, GPU ortami bolumu)')
    check('Tani betigi oz-testi (sahte modelle CPU hatti)', extra,
          ['-B', 'research/diagnose_selfmm_weighting.py', '--self-test'], 'TANI_OZTESTLERI_BASARILI', torch_ok)

    passed = sum(r[1] == 'PASS' for r in results)
    failed = [r for r in results if r[1] == 'FAIL']
    skipped = sum(r[1] == 'SKIP' for r in results)
    print(f'\nOZET: {passed} gecti, {len(failed)} kaldi, {skipped} atlandi (toplam {len(results)})')
    if failed:
        print('KONTROLLER_BASARISIZ: ' + '; '.join(r[0] for r in failed))
        return 1
    print('KONTROLLER_TAMAM' + (' (atlananlar icin yukaridaki notlara bakin)' if skipped else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
