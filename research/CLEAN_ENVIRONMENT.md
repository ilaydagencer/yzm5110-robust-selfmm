# Temiz Python ortamı ve ayrı kaynak kopyası

Amaç: Önceki sonuçların eski venv paketlerine ve çalışma manifestlerine bağlı
olmadan üretilebildiğini sınamak. Bu işlem aynı Windows bilgisayarı, Python
3.10.5 temel kurulumu, NVIDIA sürücüsü ve mevcut veri/checkpoint/BERT varlıklarını
kullanır. İşletim sistemi kurulumu, yeni bilgisayar, GitHub'dan temiz clone veya
eğitimden baştan tekrar değildir. Kaynak, mevcut yerel repodan hash'leri kayıtlı
bir kopyadır; commit edilmeyen değişiklikleri de kapsayabilir. Son teslimde
Fork commit/tag'i ve public checkout ayrıca doğrulanmalıdır.

**Durum: Hazırlık tamamlandı; yeni bağımlılık kurulumu ve temiz ortam GPU sonucu bekleniyor.**

## Neler hazırlanıyor?

- `prepare_selfmm_clean_copy.py`: yalnız küçük kaynak, paketleme ve araştırma
  dosyalarını yeni klasöre kopyalar; kaynak/çıktı hash'lerini kaydeder. Veri,
  checkpoint, BERT, .git, eski venv ve eski runs kopyalanmaz. Kopyadaki varsayılan
  config'in yalnız `dataset_root_dir` alanı `./data` olur; kişisel veri yolu
  taşınmaz. Portable betikler zaten açık verilen kendi config'lerini kullanır.
- `check_selfmm_clean_environment.py`: yeni Python/venv'i, paket sürüm ve
  yollarını, kopya kaynak hash'lerini, MMSA'nın kopyadan import edildiğini,
  `pip check` ve CUDA'yı denetler. Model/veri yüklemez. `--check` çıktı yazmaz;
  kayıt modu yalnız belirtilen yeni `checks` alt klasörüne yazar.
- Mevcut değerlendirme betikleri ve bağımlılık JSON'ları değiştirilmez.

Kaynaklar: [Python venv](https://docs.python.org/3.10/library/venv.html),
[pip check](https://pip.pypa.io/en/stable/cli/pip_check/),
[PyTorch 2.3.1 kurulumu](https://pytorch.org/get-started/previous-versions/#v231).
Venv'i etkinleştirmek gerekmez; her komutta doğru python.exe yolu kullanılır.

## 1. Teslim dosyalarını mevcut repoya yerleştir

```powershell
Set-Location 'C:\Users\ilayda\YZM5110\MMSA'
$deliveryDir = 'C:\Users\ilayda\Documents\Codex\2026-10-17\referenced-chatgpt-conversation-this-is-an\outputs\research_clean_environment_v1'
& "$deliveryDir\Install-CleanEnvironmentDocs.ps1" -CheckOnly
```

Kontrol başarılıysa:

```powershell
& "$deliveryDir\Install-CleanEnvironmentDocs.ps1"
```

Dört mevcut belge güncellenir; iki doğrulama JSON'u, iki yardımcı Python betiği
ve bu yönerge eklenir: toplam dokuz dosya. Bütün hedefler yazmadan önce denetlenir.
Beklenmeyen içerik üzerine yazılmaz. Aynı paket tekrar çalışırsa yazmadan biter.

## 2. Kaynak kopyasının kontrolü ve oluşturulması

```powershell
$cleanRoot = 'C:\Users\ilayda\YZM5110_clean_replay_v1'
& '..\.venv\Scripts\python.exe' -I -S -B `
    '.\research\prepare_selfmm_clean_copy.py' `
    --target-root $cleanRoot --check
```

Başarılıysa:

```powershell
& '..\.venv\Scripts\python.exe' -I -S -B `
    '.\research\prepare_selfmm_clean_copy.py' `
    --target-root $cleanRoot
if ($LASTEXITCODE -ne 0) { throw 'Kaynak kopyasi basarisiz; sonraki adima gecmeyin.' }
```

`$cleanRoot\MMSA` ve `$cleanRoot\source_snapshot.json` oluşur. Hedef zaten varsa
betik durur; silme veya üzerine yazma yapmaz. Bu adım henüz venv oluşturmaz.

## 3. Yeni venv

```powershell
$basePython = (& '..\.venv\Scripts\python.exe' -I -B -c 'import sys; print(sys._base_executable)').Trim()
if ($LASTEXITCODE -ne 0) { throw 'Temel Python yolu okunamadi.' }
if (Test-Path -LiteralPath "$cleanRoot\.venv") { throw 'Yeni venv hedefi zaten var.' }
& $basePython -I -B -m venv "$cleanRoot\.venv"
if ($LASTEXITCODE -ne 0) { throw 'Venv olusturulamadi.' }
$cleanPython = Join-Path $cleanRoot '.venv\Scripts\python.exe'
$cleanRepo = Join-Path $cleanRoot 'MMSA'
$logs = Join-Path $cleanRoot 'install_logs'
New-Item -ItemType Directory -Path $logs -ErrorAction Stop | Out-Null
```

Eski venv kopyalanmaz. `--system-site-packages` veya `--clear` kullanılmaz.
Bu değişkenler aynı PowerShell oturumunda geçerlidir; oturumu kapatırsan üç
yolu yeniden tanımla, kaynak kopyalama/venv oluşturma adımlarını tekrar çalıştırma.

## 4. Paketleri yeni ortama kur

İnternet ve disk alanı gerekir; CUDA PyTorch indirmesi büyük olabilir. Komutlar
yalnız `$cleanPython` ortamına kurar. Her komut hata verirse sonraki adıma geçme;
hatayı paylaş. Pin'leri gevşetme veya eski site-packages klasörünü kopyalama.

```powershell
& $cleanPython -I -B -m pip --isolated --disable-pip-version-check install `
    --index-url 'https://pypi.org/simple' `
    --log "$logs\01-build-tools.log" `
    'pip==24.0' 'setuptools==69.5.1' 'wheel==0.43.0'
if ($LASTEXITCODE -ne 0) { throw 'Kurulum araclari basarisiz.' }

& $cleanPython -I -B -m pip --isolated --disable-pip-version-check install `
    --index-url 'https://download.pytorch.org/whl/cu121' `
    --log "$logs\02-torch.log" 'torch==2.3.1'
if ($LASTEXITCODE -ne 0) { throw 'PyTorch kurulumu basarisiz.' }

& $cleanPython -I -B -m pip --isolated --disable-pip-version-check install `
    --index-url 'https://pypi.org/simple' `
    --log "$logs\03-requirements.log" `
    -r "$cleanRepo\research\requirements-research.txt"
if ($LASTEXITCODE -ne 0) { throw 'Arastirma paketleri kurulumu basarisiz.' }

& $cleanPython -I -B -m pip --isolated --disable-pip-version-check install `
    --no-deps --no-build-isolation --log "$logs\04-mmsa.log" -e $cleanRepo
if ($LASTEXITCODE -ne 0) { throw 'Yerel MMSA kurulumu basarisiz.' }

& $cleanPython -I -B -m pip --isolated --disable-pip-version-check check
if ($LASTEXITCODE -ne 0) { throw 'Paket bagimliliklari uyumsuz.' }
```

Paket listesi çalışan ortamın gözlenen sürümleridir; bu aşamada internetten
kurulabilirliği henüz sınanmamıştır. Hash kilitli bağımlılık lockfile'ı değildir.
Pip indirme önbelleğini kullanabilir; bu, eski kurulu paketlerin kopyalanması
değildir. Ağ indirmesi ve model tekrarı farklı aşamalardır; çıkarım çevrimdışıdır.
Kurulum logları yereldir; Git'e topluca eklenmez.

## 5. Ortam doğrulaması ve kayıt

```powershell
& $cleanPython -I -B "$cleanRepo\research\check_selfmm_clean_environment.py" `
    --repo-dir $cleanRepo --venv-dir "$cleanRoot\.venv" --check
if ($LASTEXITCODE -ne 0) { throw 'Ortam kontrolu basarisiz.' }
```

`ORTAM_KONTROLU_BASARILI`, yeni MMSA yolu, `No broken requirements found` ve
CUDA 12.1 beklenir. Bu çıktı gerçek model sonucunu henüz doğrulamaz.

Başarılıysa kaydet ve çıktıyı paylaş:

```powershell
& $cleanPython -I -B "$cleanRepo\research\check_selfmm_clean_environment.py" `
    --repo-dir $cleanRepo --venv-dir "$cleanRoot\.venv" `
    --output-dir "$cleanRoot\checks\environment_v1"
if ($LASTEXITCODE -ne 0) { throw 'Ortam kaydi basarisiz.' }

Get-Content "$cleanRoot\checks\environment_v1\manifest.json" -Raw |
    ConvertFrom-Json |
    Select-Object status, executable, prefix, mmsa_file, isolated,
        system_site_packages, pip_check, cuda_runtime, gpu,
        source_files_verified, gpu_model_replay_performed |
    Format-List
```

## 6. Ortam geçtiğinde ilk gerçek GPU doğrulaması

İlk kontrol için daha önce sabitlenmiş seed 1111, K0–K7 ve A/B/C/D kullanılır.
Yeni test ayarı seçilmez. Veri, BERT ve checkpoint'ler salt okunacak mevcut
varlıklardır; eski run manifestlerine bağımlılık yoktur.

```powershell
if (-not (Test-Path -LiteralPath "$cleanRoot\runs")) {
    New-Item -ItemType Directory -Path "$cleanRoot\runs" -ErrorAction Stop | Out-Null
}
$bertDir = Join-Path $env:USERPROFILE '.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594'
$cleanArgs = @(
    '--repo-dir', $cleanRepo,
    '--data-file', 'C:\Users\ilayda\YZM5110\data\MOSI\Processed\unaligned_50.pkl',
    '--baseline-checkpoint', 'C:\Users\ilayda\YZM5110\runs\selfmm_seed1111\saved_models\self_mm-mosi.pth',
    '--dropout-checkpoint', 'C:\Users\ilayda\YZM5110\runs\selfmm_dropout_a20_v20_seed1111\saved_models\self_mm-mosi.pth',
    '--bert-dir', $bertDir,
    '--output-dir', "$cleanRoot\runs\selfmm_portable_k0_k7_test_seed1111_v1"
)
& $cleanPython -I -B "$cleanRepo\research\evaluate_selfmm_portable_k0_k7.py" @cleanArgs --check
if ($LASTEXITCODE -ne 0) { throw 'Model on kontrolu basarisiz.' }
```

Kontrol başarılıysa aynı komut `--check` olmadan çalıştırılır:

```powershell
& $cleanPython -I -B "$cleanRepo\research\evaluate_selfmm_portable_k0_k7.py" @cleanArgs
if ($LASTEXITCODE -ne 0) { throw 'Temiz ortam GPU tekrari basarisiz.' }
```

Başarı ölçütü: completed, 128/128 metrik, 16/16 girdi, 2/2 train istatistiği ve
koruma bayrakları başarılı. Ortam manifestiyle GPU manifesti birlikte kanıttır.
Sadece pip kurulumu, CUDA görünürlüğü veya --check yeterli değildir. Bu ilk
başarı diğer dört seed'in yeni venv'de çalıştığı veya başka bir bilgisayarda
sonuçların doğrulandığı anlamına gelmez; kapsam raporda açık yazılacaktır.

## Fork kaydı

Başlık: `feat: document five-seed replay and prepare clean-environment validation`

```text
Record verified seed-1115 replay and the completed five-seed portable series:
640 metric, 80 input and 10 train-statistics checks pass. Recheck 40 output
hashes and preserve the fixed model, checkpoints and evaluation protocol.

Add a guarded source-copy helper and isolated-environment checker.
Document fresh-venv installation, dependency logs, preflight and a subsequent
seed-1111 GPU replay. Update portable guides and add verification records.

Fresh dependency installation and clean-environment GPU replay remain pending.
```

Yalnız teslimdeki dokuz küçük dosya stage edilir. Yeni temiz kök, venv,
kurulum logları, veri/checkpoint ve bütün runs klasörü stage edilmez.
