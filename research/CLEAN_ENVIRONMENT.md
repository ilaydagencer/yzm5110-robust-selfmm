# Temiz Python ortamı ve ayrı kaynak kopyası

Amaç: Önceki sonuçların eski venv paketlerine ve çalışma manifestlerine bağlı
olmadan üretilebildiğini sınamak. Bu işlem aynı Windows bilgisayarı, Python
3.10.5 temel kurulumu, NVIDIA sürücüsü ve mevcut veri/checkpoint/BERT varlıklarını
kullanır. İşletim sistemi kurulumu, yeni bilgisayar, GitHub'dan temiz clone veya
eğitimden baştan tekrar değildir. Kaynak, mevcut yerel repodan hash'leri kayıtlı
bir kopyadır; commit edilmeyen değişiklikleri de kapsayabilir. Son teslimde
Fork commit/tag'i ve public checkout ayrıca doğrulanmalıdır.

**Durum: Yeni venv kurulumu ve seed 1111 K0–K7 A/B/C/D GPU doğrulaması tamamlandı.**

## Tamamlanan doğrulama — 19 Ekim 2026

Yeni venv'e kurulan 51 paketin konumu/sürümü, izole Python çalışması,
MMSA'nın ayrı kaynak kopyasından import edilmesi ve `pip check` başarıyla
doğrulandı. Python 3.10.5, torch 2.3.1+cu121, CUDA 12.1, numpy 1.26.4,
transformers 4.30.2 ve RTX 3050 Ti Laptop GPU kullanıldı.

| Kontrol | Sonuç |
|---|---|
| Kaynak kopyası | 125/125 dosya özeti tekrar eşleşti. |
| Kurulum kanıtı | Dört pip logu ve ortam manifestinin kayıtlı özetleri eşleşti. |
| Public paket metrikleri | 128/128 geçti; en büyük fark 5.55e-16 (tolerans 1e-5). |
| Bozulmuş girdiler | 16/16 tam eşleşti; iki checkpoint aynı girdilerle değerlendirildi. |
| Train istatistikleri | 2/2 geçti; ortalama/std farkı sıfır; 49878 ses ve 54741 görüntü zaman adımı. |
| Çıktı/girdi bütünlüğü | 8 çıktı ve 106 kayıtlı girdi özeti denetim anında eşleşti. |
| Önceki portable seed 1111 çıktıları | Sekiz çıktı dosyasının tamamı bayt düzeyinde aynı. |
| Tahmin ve ağırlık karşılaştırması | 10976 satır; en büyük tahmin/etiket/ağırlık farkı 0. |
| Kimlik ve koruma kontrolleri | Tüm ağırlıklar=1 hatası 0; hook, metin, uzunluk, padding ve veri korundu. |

32 yöntem/koşul satırı için MAE, Non0 accuracy, weighted F1 ve Corr, kaydedilen
tahminlerden bağımsız olarak yeniden hesaplandı. Float32 ortalama kullanan
MAE ile fark en fazla 7.83e-8 (tolerans 1e-6), diğer metriklerde 5.56e-16
altında kaldı. Bu hesaplama farkı, eski/yeni portable çıktı farkı değildir;
eski/yeni portable çıktı farkı **sıfırdır**.

Küçük public paketle karşılaştırma toplu metrik düzeyindedir. Tekil tahminlerin
birebir karşılaştırmasında referans, yerel önceki **portable seed 1111**
çalışmasıdır; tüm tarihsel eğitim tahminleriyle karşılaştırma iddia edilmez.
[Kişisel mutlak yol içermeyen doğrulama kaydı](verification/clean_environment_seed1111_k0_k7_v1.json),
ortam/kopya/GPU manifestlerinin ve çıktıların SHA256 özetlerini birlikte saklar.

| K0 yöntemi | Önceki MAE | Yeni MAE | Önceki / yeni Non0 accuracy (%) |
|---|---:|---:|---:|
| A: baseline | 0.7448 | 0.7448 | 82.77 / 82.77 |
| B: dropout | 0.7421 | 0.7421 | 83.54 / 83.54 |
| C: baseline + ağırlık | 0.8270 | 0.8270 | 82.62 / 82.62 |
| D: dropout + ağırlık | 0.8413 | 0.8413 | 83.54 / 83.54 |

K1–K7 de önceki portable sonuçlarla tam eşleşti. Seed 1111'de B'nin MAE'si
K0–K7 boyunca A'dan azdır; C'nin A'dan, D'nin B'den fazladır. Bu, eski bulgunun
tekrarıdır. Beş-seed test ortalamasında dropout'un üstün olmadığı sonucu
değişmez. Tekrarı yeni seed gibi sayarak ortalama/SS'ye eklemiyoruz.
Kaydedilen 399.68 saniye yükleme, train istatistiği, çıkarım ve kontrolleri
içeren toplam tekrar süresidir; önceki çıkarım benchmark'ıyla hız kıyası yapılmaz.

Ortam manifestindeki `gpu_model_replay_performed=False`, ortam kontrolünün
model çalıştırmadığını anlatır. O tarihsel kayıt korunur; sonradan tamamlanan
GPU çalışmasının ayrı `status=completed` manifestiyle birlikte değerlendirilir.
Kaynak kopyası manifestindeki önceki hazırlık bayrakları da değiştirilmez.

Bu kanıt aynı bilgisayar, aynı temel Python ve sürücü üzerinde **yeni venv +
ayrı yerel kaynak kopyası + mevcut dış veri/checkpoint/BERT** kapsamındadır.
Yeni venv'de yalnız seed 1111 sınandı. Seed 1112–1115 önceki ortamda doğrulandı.
Public Git checkout, başka bilgisayar ve uçtan uca yeniden eğitim doğrulanmadı.
Test seti daha önce görülmüştür; sabit protokol tekrarında yeni ayar seçilmedi.

## Mevcut çalışmanın ardından

1. Aşağıdaki altı dosyalık belge güncellemesini **asıl MMSA reposuna** uygula.
2. Fork'ta bu dosyaları gözden geçir, commit ve push yap; commit kimliğini sakla.
3. Sonraki adımda public depodaki bu commit'ten checkout ve dağıtılan küçük
   paketin kontrolünü ele al. Yerel kaynak kopyası ile public teslim aynı kanıt değildir.
4. Ardından raporu hocanın PDF isterleriyle yeniden eşleştir; bu doğrulamayı
   yeniden üretilebilirlik bölümüne aktar ve Word/PDF son kontrolünü tamamla.

Bu belge güncellemesi Word raporunu değiştirmez. Temiz kopya kanıtını korumak
için `YZM5110_clean_replay_v1/MMSA` içindeki kaynak/belgeleri güncelleme.

## Yeniden kurulum tarifinin kullanımı

Aşağıdaki adımlar tamamlanan çalışmanın tekrar tarifidir; mevcut temiz köke
yeniden uygulanmaz. Yeni bir tekrar gerektiğinde kullanılmamış hedef adı
seçilmeli ve tüm komutlarda aynı yeni kök kullanılmalıdır.

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

## 1. Kaynak repoyu seç

Araçlar artık repodadır; eski teslim paketini yeniden kurma. Aşağıdaki örnek
tamamlanan yerel çalışmanın yollarını kullanır. Bir sonraki baştan kurulumda
henüz kullanılmamış kök seç.

```powershell
Set-Location 'C:\Users\ilayda\YZM5110\MMSA'
```

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

Bu paket sürümleriyle yeni venv kurulumu bu bilgisayarda doğrulandı. Liste
hash kilitli bir bağımlılık lockfile'ı değildir; gelecekteki indirme erişimi veya
başka bilgisayarda kurulum bu kayıtla garanti edilmez.
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

## Güncel belge güncellemesinin Fork kaydı

Başlık: `docs: verify seed-1111 GPU replay in a fresh environment`

```text
Record successful seed-1111 K0-K7 A/B/C/D GPU replay in a fresh isolated
venv and relocated local source copy on the same Windows computer.

Verify 128 metric, 16 corrupted-input and two train-statistics checks.
Confirm all eight output files, including 10,976 prediction/weight rows,
are byte-identical to the prior portable seed-1111 run. Independently
recompute metrics and recheck 125 copied source files plus 106 run inputs.

Link installation and GPU evidence in a small verification record and
update five guides. State limits: existing assets reused; no retraining,
public Git checkout, other-machine or other-seed clean-environment replay.
```

Beş belge: `README.md`, `PORTABLE_MODEL.md`, `PORTABLE_K0_K7.md`,
`PORTABLE_SEED_TEST.md`, `CLEAN_ENVIRONMENT.md`. Bir yeni kayıt:
`verification/clean_environment_seed1111_k0_k7_v1.json`. Toplam altı dosya.
Yeni temiz kök, venv, kurulum logları, veri/checkpoint ve bütün runs klasörü
stage edilmez. Kişisel veri yolunu içeren orijinal config bu commit'e dahil değildir.
