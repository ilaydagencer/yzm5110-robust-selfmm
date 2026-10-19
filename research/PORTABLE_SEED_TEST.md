# Seed 1112: taşınabilir K0–K7 A/B/C/D test tekrarı

Bu adım, daha önce eğitilmiş seed 1112 baseline ve dropout checkpoint'lerinin
sonuçlarını eski çalışma manifestlerine bağlanmadan yeniden üretir. Eğitim
yapılmaz. Kullanıcı gerçek GPU komutunu çalıştırır; Git işlemlerini Fork'tan yapar.

## Ne değişiyor?

Yeni giriş noktası `evaluate_selfmm_portable_seed_test.py`, `--seed 1112` alır.
Bu teslimde yalnız 1112 desteklenir; 1113–1115 daha sonra aynı akışa eklenecektir.
Seed 1111'in mevcut betikleri ve bağımlılık kayıtları korunur.

| Ayar | Bu çalışmadaki değer ve anlamı |
|---|---|
| Eğitim seed'i | **1112**; kullanılacak iki eğitim checkpoint'ini ve referans metrik satırlarını seçer. |
| Bozulma seed'i | **1111**; önceki deneylerdeki maskeler ve gürültü aynen korunur. |
| Bölüm | Test; 686 örnek. Non0 sınıflandırma metriklerinde 656 örnek. |
| Batch | 4; son batch 2 örnektir. |
| Ağırlıklandırma | Önceki yöntem, τ=1; yeni ayar seçimi yapılmaz. |
| Metin | Bütün koşullarda aynı kalır. |

A: baseline checkpoint'i, standart fusion. B: dropout checkpoint'i, standart
fusion. C: baseline checkpoint'i, tek modlu tahminlerin uyuşmasına dayalı
ağırlıklandırma. D: dropout checkpoint'i, aynı ağırlıklandırma. C/D iki forward
kullanır. B/D eğitiminde uygulanan %20 ses/%20 görüntü dropout'u değerlendirmede
etkin değildir; modeller `eval()` ve `inference_mode()` ile çalışır.

K0 temiz; K1 ses eksik; K2 görüntü eksik; K3 ikisi eksik. K4/K5/K6, her
modalitenin geçerli zaman adımlarını bağımsız %30/%50/%70 olasılıkla sıfırlar.
Maskeler iç içedir. K7, `0.5 × train std` ölçeğinde Gaussian gürültü ekler.
Zaman adımları ve padding kuralları [K0–K7 yönergesindeki](PORTABLE_K0_K7.md)
protokolle aynıdır. Tam eksiklikte de padding korunur; gerçek uzunluklar değiştirilmez.

## Hangi dosya ne işe yarıyor?

- `evaluate_selfmm_portable_seed_test.py`: eğitim seed'ini seçer, varlık kimliklerini
  denetler, train istatistiğini hesaplar ve K0–K7 çıkarımını yapar.
- `configs/selfmm_portable_seed_dependencies.json`: seed 1112'nin iki checkpoint
  SHA256 özeti, kaynak sürümleri, ortak config/protokol ve sonuç paketi kimlikleri.
  SHA256, dosyanın baytlarından hesaplanan kimliktir; yanlış checkpoint seçimini yakalar.
- `PORTABLE_SEED_TEST.md`: bu yönerge.
- `README.md`: yeni giriş noktasına bağlantı ve doğrulama kapsamı.

Yeni kayıt hazırlanırken checkpoint'ler, eğitim config'leri ve önceki test
kayıtları karşılaştırılmıştır. Bu hazırlık işlemi eski kayıtları okudu;
**teslim edilen betik çalışırken eski run manifestlerini veya kişisel config'leri
okumaz.** Eğitim/test manifestlerinin SHA256 değerleri yalnız kaynak izini saklar.

Gerekli mevcut dosyalar: portable K0 yardımcısı, `evaluate_selfmm_abcd_valid.py`,
`selfmm_corruptions.py`, `verify_results_package.py`, `results_package_v1`,
`configs/selfmm_mosi_portable_k0.json` ve `configs/selfmm_portable_k0_k7_dependencies.json`.
Ortak protokoldeki `training_seed: 1111`, **girdi referansının** kökenidir;
seed 1112 checkpoint seçimi yeni bağımlılık kaydından yapılır. Public paketteki
eski provenance checkpoint çifti de seed 1111'e aittir; seed 1112 diye kullanılmaz.

Model ayarları iki seed 1112 eğitim kaydıyla eşleşir. Veri, checkpoint ve yerel
BERT yolları açık komut parametreleridir. `sentencepiece`, torch/MMSA'dan önce
içe aktarılır. Betik çevrimdışıdır, BERT indirmez. Mevcut çalışan venv'i kullanın.
Farklı bir repo kopyasında çalıştırırken MMSA kurulumu da o kopyanın `src` yolunu
kullanmalıdır; yanlış yerel kurulum kontrol sırasında reddedilir.

## 1. Dosyaları yerleştir

```powershell
Set-Location 'C:\Users\ilayda\YZM5110\MMSA'
$deliveryDir = 'C:\Users\ilayda\Documents\Codex\2026-10-17\referenced-chatgpt-conversation-this-is-an\outputs\research_portable_seed1112_v1'
& "$deliveryDir\Install-PortableSeed1112.ps1" -CheckOnly
```

`YERLESTIRME_KONTROLU_BASARILI` gelirse:

```powershell
& "$deliveryDir\Install-PortableSeed1112.ps1"
```

Üç yeni dosya eklenir, README güncellenir. Tüm hedefler yazmadan önce kontrol
edilir; beklenmeyen içerik varsa üzerine yazılmaz. Aynı paket ikinci kez
çalıştırıldığında `PAKET_ZATEN_YERINDE` ile dosya değiştirmeden biter. Git işlemi yoktur.

## 2. Eğitim/model yüklemesi yapmadan --check

Aşağıdaki blok yeni PowerShell oturumunda da kullanılabilir:

```powershell
Set-Location 'C:\Users\ilayda\YZM5110\MMSA'
$bertDir = Join-Path $env:USERPROFILE '.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594'
$portable1112Args = @(
    '--seed', '1112',
    '--data-file', '..\data\MOSI\Processed\unaligned_50.pkl',
    '--baseline-checkpoint', '..\runs\selfmm_baseline_seed1112\saved_models\self_mm-mosi.pth',
    '--dropout-checkpoint', '..\runs\selfmm_dropout_a20_v20_seed1112\saved_models\self_mm-mosi.pth',
    '--bert-dir', $bertDir,
    '--output-dir', '..\runs\selfmm_portable_k0_k7_test_seed1112_v1'
)
& '..\.venv\Scripts\python.exe' -B `
    '.\research\evaluate_selfmm_portable_seed_test.py' `
    @portable1112Args --check
```

`SEED_SECILDI: egitim=1112; bozulma=1111; referans metrik seed=1112`,
`CUDA: True` ve sonunda `KONTROL_BASARILI: seed1112/K0-K7` beklenir.

Bu kontrol kaynak/varlık özetlerini ve sürümleri doğrular, yapay CPU verileriyle
maskeleri, seed tekrarlanabilirliğini, ağırlıklandırmayı, padding'i ve hook
temizliğini sınar. Gerçek dataset/model/checkpoint nesnesi kurulmaz, dosya yazılmaz.
Büyük veri/checkpoint dosyaları SHA256 hesabı için **bayt olarak okunur**;
hash aşaması bu nedenle biraz sürebilir. Gerçek train std ve 128 metrik
karşılaştırması yalnız sonraki gerçek çalışmada doğrulanır.

## 3. Gerçek GPU değerlendirmesi

Kontrol başarılıysa aynı PowerShell oturumunda:

```powershell
& '..\.venv\Scripts\python.exe' -B `
    '.\research\evaluate_selfmm_portable_seed_test.py' `
    @portable1112Args
```

Önce train bölümünün geçerli ses/görüntü zaman adımlarından özellik ortalaması
ve std (`ddof=0`) yeniden hesaplanır. Sesin `-inf` değerleri MMSA'daki gibi
sıfırlanır; padding istatistiklere katılmaz. Std sıfırsa o özelliğe gürültü
eklenmez. Pickle bütün split'leri içerir; istatistik fonksiyonuna yalnız train
dizileri verilir. Test/validation etiketleri veya özellikleri bu hesaba katılmaz.

Sonra baseline A/C ve dropout B/D çalışır; GPU'da aynı anda bir checkpoint
tutulur. Checkpoint yüklemesi `weights_only=True`, model yüklemesi `strict=True`
kullanır. Ağırlıklar eğitilmez. Metin, uzunluklar ve padding korunur. Ek
all-one ağırlık forward'ı, fusion yamasının identity kontrolüdür.

Yeni çıktı klasörü zaten varsa betik durur. Başarılı koşuyu tekrar çalıştırmayın;
hatalı koşuda önce manifest/CSV'leri inceleyin. Toleransları veya beklenen
hash değerlerini değiştirerek kontrolü atlamayın. Eski çıktı klasörleri korunur.
İsteğe bağlı `--repo-dir`, `--config`, `--dependencies`, `--protocol`,
`--package-dir` parametreleri vardır. Varsayılan küçük bağımlılıklar repo altında
aranır; göreli CLI yolları PowerShell'in bulunduğu klasöre göredir.

## 4. Sonuçları paylaş

```powershell
$portable1112Dir = '..\runs\selfmm_portable_k0_k7_test_seed1112_v1'
Get-Content "$portable1112Dir\manifest.json" -Raw |
    ConvertFrom-Json |
    Select-Object status, training_seed, corruption_seed, reference_training_seed,
        input_reference_training_seed, split, conditions,
        historical_run_manifests_required, train_statistics_recomputed_and_matched,
        reference_test_labels_matched, reference_metrics_matched,
        reference_corrupted_inputs_matched, same_corruptions_as_seed1111,
        shared_inputs_verified, hooks_restored, text_unchanged,
        lengths_unchanged, padding_unchanged, protected_files_unchanged,
        data_file_unchanged, metric_rows, prediction_rows,
        reference_check_rows, reference_input_check_rows, train_stats_check_rows |
    Format-List

Import-Csv "$portable1112Dir\reference_checks.csv" |
    Group-Object passed | Select-Object Name, Count

Import-Csv "$portable1112Dir\train_stats_checks.csv" | Format-List

Import-Csv "$portable1112Dir\reference_input_checks.csv" |
    Format-Table checkpoint, condition, shared_inputs, passed -AutoSize

Import-Csv "$portable1112Dir\metrics.csv" |
    Sort-Object condition, method |
    Format-Table method, condition, n,
        @{Name='Non0_acc_2';Expression={[math]::Round([double]$_.Non0_acc_2,4)}},
        @{Name='Non0_F1_score';Expression={[math]::Round([double]$_.Non0_F1_score,4)}},
        @{Name='MAE';Expression={[math]::Round([double]$_.MAE,4)}},
        @{Name='Corr';Expression={[math]::Round([double]$_.Corr,4)}} -AutoSize
```

| Beklenen çıktı | Satır/kapsam |
|---|---|
| `metrics.csv` | 32: 4 yöntem × 8 koşul. |
| `reference_checks.csv` | 128: her sonuç için MAE, Non0 accuracy, weighted F1, Corr. |
| `predictions_and_weights.csv` | 10976: 2 checkpoint × 8 koşul × 686; aynı satırda plain/weighted tahmin. |
| `controls.csv` | 16 identity/girdi/hook kontrolü. |
| `reference_input_checks.csv` | 16: iki checkpoint × sekiz koşul; seed 1111 girdileriyle tam SHA256 eşleşmesi. |
| `corruption_summary.csv` | 16 maske/gürültü özeti. |
| `train_stats_checks.csv` | 2: ses/görüntü train istatistiği karşılaştırması. |
| `train_feature_stats.json` | Yeni hesaplanan train istatistikleri. |
| `manifest.json` | Seed'ler, ayarlar, sürümler, dosya özetleri ve tamamlanma durumu. |

Başarı ölçütü: `status=completed`; metrik 128/128, girdi 16/16 ve train
istatistiği 2/2 başarılı. Metrik toleransı `1e-5`, train istatistiği toleransı
`1e-12`; girdi/etiket SHA256 eşleşmesi tamdır. `historical_run_manifests_required`
False, koruma bayrakları True olmalıdır. Referans metrikler **seed 1112**'ye,
girdi referansı **seed 1111**'e aittir.

## Hazırlık doğrulaması ve sınırlar

13 CPU testi geçti: ortak bozulma/girdi sırası uyumu, train-only istatistikler,
seed seçimi, yanlış/karışmış checkpoint reddi, seed'e göre metrik seçimi,
yapay uçtan uca çıktı ve başarısız karşılaştırmalarda tanı kaydı dahil.
Gerçek dosyalarla yazmasız `--check` başarılıdır. Eski run manifestleri
bulunmayan ayrı kaynak kopyasında, model/dataset yükleyicileri engellenerek
`--check` de sınanır; bu kontrolün sonucu teslim doğrulama kaydında tutulur.

Gerçek seed 1112 GPU tekrarı bu giriş noktasında henüz çalıştırılmadı; kullanıcı
sonraki adımda çalıştıracak. CPU yapay testleri gerçek model doğruluğunun yerine
geçmez. Temiz ML kurulumu ve başka bilgisayarda gerçek GPU çalışması doğrulanmadı.
Eski tekil tahminler public pakette yoktur; kontrol toplu metrikleri ve girdi
özetlerini karşılaştırır. 128 kontrol, 128 bağımsız deney değildir.

Bu bir önceki sabit deneyin tekrarıdır. Test daha önce görülmüştür; yeni τ,
dropout veya seed seçimi yapılmaz. Beş-seed bulgularına yeni bağımsız gözlem
eklenmez. Manifestteki süre yükleme, train istatistiği, model çalışması ve
kontrolleri kapsar; çıkarım benchmark'ı olarak kullanılmaz.

## Fork commit'i

Bu aşamada yalnız yeni betik, yeni seed bağımlılık JSON'u, bu yönerge ve README
stage edilir. Veri, checkpoint, BERT, venv, runs ve kişisel orijinal config
değişikliği eklenmez. Gerçek GPU sonucu henüz alınmadığı için commit'te
başarıyla yeniden üretildiği yazılmaz.

Başlık: `feat: add portable seed-1112 K0-K7 checkpoint replay`

```text
Add a seed-aware portable A/B/C/D test entry point and register the
verified seed-1112 checkpoint identities. Preserve seed-1111 scripts.

Keep corruption seed 1111, batch 4 and tau 1. Recompute noise statistics
from TRAIN frames and compare 128 metrics with seed-1112 references,
plus 16 shared-input fingerprints and two train-statistics checks.

Document offline preflight, explicit asset paths and protected outputs.
Pass 13 CPU tests and read-only preflight. Actual seed-1112 GPU replay
through this new entry point remains pending.
```
