# Seed 1111: K0–K7 taşınabilir A/B/C/D değerlendirmesi

Bu betik, mevcut seed 1111 checkpoint'leriyle test bölümündeki önceki deneyleri
yeniden çalıştırır. Eğitim yapılmaz. K0'da doğrulanan model ayarları, batch 4,
τ=1 ve yöntemler korunur. Yeni çıktılar eski klasörlere yazılmaz.

## Ne yapıyoruz?

K0 tekrarı temiz girdilerle modelin çalıştığını gösterdi. Şimdi ses/görüntü
eksikliğini ve gürültüyü, eski çalışma manifestlerini taşımadan uyguluyoruz.

| Yöntem | Kullanılan checkpoint ve çıkarım |
|---|---|
| A | Baseline, standart fusion. |
| B | Dropout ile eğitilmiş checkpoint, standart fusion. |
| C | Baseline, tek modlu tahminlerin uyuşmasına dayalı ağırlıklandırma. |
| D | Dropout checkpoint'i, aynı ağırlıklandırma. |

Dropout yalnız eğitimde kullanılmıştı. Burada modeller `eval()` ve
`inference_mode()` ile çalışır. Metin değiştirilmez.

| Koşul | Ses ve görüntü girdileri |
|---|---|
| K0 | Temiz, bütün modaliteler mevcut. |
| K1 | Sesin geçerli zaman adımları sıfırlanır. |
| K2 | Görüntünün geçerli zaman adımları sıfırlanır. |
| K3 | Ses ve görüntünün geçerli zaman adımları sıfırlanır. |
| K4 | Her modalitede zaman adımları bağımsız %30 olasılıkla sıfırlanır. |
| K5 | Aynı işlem %50 olasılıkla uygulanır. |
| K6 | Aynı işlem %70 olasılıkla uygulanır. |
| K7 | Her özelliğe `0.5 × eğitim özelliğinin standart sapması` ölçeğinde Gaussian gürültü eklenir. |

Zaman adımı, ses/görüntü dizisinin bir satırıdır; seçildiğinde bütün özellikleri
sıfırlanır. Örneklerin %30'unu tamamen silmek anlamına gelmez. Gerçekleşen maske
oranları hedef olasılığa yakın olur; tam eşitlik beklenmez. K4 maskesi K5'in,
K5 maskesi K6'nın alt kümesidir. Orijinal uzunluklar ve dizilerin sonundaki
padding satırları korunur. Bozulma seed'i 1111; örnek indeksi, modalite ve split
ile birlikte kullanılır. Eğitim seed'i de 1111'dir; amaçları farklıdır.

## Gürültü istatistiği ve bağımlılıklar

Yeni betik yalnız **train** bölümünün ses/görüntü dizilerinden özellik başına
ortalama ve standart sapmayı yeniden hesaplar. Bütün geçerli eğitim zaman
adımları havuzlanır, padding dışarıda tutulur; popülasyon standart sapması
(`ddof=0`) kullanılır. Ses verisindeki `-inf`, MMSA'nın yaptığı gibi sıfıra
çevrilir. Standart sapması sıfır olan özelliğe gürültü eklenmez.

Validation/test değerleri veya etiketleri istatistik hesabına katılmaz.
Pickle bütün split'leri içerdiğinden yüklemede hepsi açılır; istatistik
fonksiyonuna yalnız train bloğu verilir. Daha sonra yalnız test değerlendirilir.

Yeni JSON, önceki sabit protokolden küçük sayısal referanslar taşır: eğitim
istatistikleri, sekiz test girdi özeti ve test etiket özeti. Bunlar hazırlık
sırasında dışarı aktarılmıştır. Çalıştırmada eski manifest/istatistik dosyaları
açılmaz. **Gürültüde kullanılan std yeni hesaplamadan gelir; JSON'daki değerler
yalnız doğrulama içindir.**

Yeni dosyalar:

- `evaluate_selfmm_portable_k0_k7.py`: değerlendirme giriş noktası.
- `configs/selfmm_portable_k0_k7_dependencies.json`: sabit koşullar ve referanslar.
- `PORTABLE_K0_K7.md`: bu yönerge.

README'ye bağlantı eklenir. K0 betiği, sonuçlar, checkpoint'ler ve Self-MM
kaynakları değiştirilmez. Yeni betik mevcut `evaluate_selfmm_portable_k0.py`,
`evaluate_selfmm_abcd_valid.py`, `selfmm_corruptions.py`, portable K0 config ve
bağımlılık JSON'ları, `verify_results_package.py` ve `results_package_v1` kullanır.

Veri, checkpoint ve BERT yolları açık parametrelerdir. Önceki K0 yönergesindeki
ortam gereklilikleri geçerlidir. Temel yardımcı, `sentencepiece` modülünü
torch/MMSA'dan önce içe aktarır. Çalışma çevrimdışıdır; otomatik indirme yapılmaz.
Çalışan venv'i yeniden kurmayın.

## Önce --check

```powershell
Set-Location 'C:\Users\ilayda\YZM5110\MMSA'
$bertDir = Join-Path $env:USERPROFILE '.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594'
$k07Args = @(
    '--data-file', '..\data\MOSI\Processed\unaligned_50.pkl',
    '--baseline-checkpoint', '..\runs\selfmm_seed1111\saved_models\self_mm-mosi.pth',
    '--dropout-checkpoint', '..\runs\selfmm_dropout_a20_v20_seed1111\saved_models\self_mm-mosi.pth',
    '--bert-dir', $bertDir,
    '--output-dir', '..\runs\selfmm_portable_k0_k7_test_seed1111_v1'
)
& '..\.venv\Scripts\python.exe' -B `
    '.\research\evaluate_selfmm_portable_k0_k7.py' `
    @k07Args --check
```

Başta `TEMEL_K0_BAGIMLILIK_KONTROLU` altında K0 yazıları görülebilir; bu,
önceki ortak yardımcının hazırlık kontrolüdür. Son kapsam satırı K0–K7 olmalı;
`KONTROL_BASARILI` ve `CUDA: True` beklenir.

`--check`, SHA256 özetlerini ve yapay CPU verileriyle maskeleri, test/valid RNG
ayrımını, padding'i, standart sapmayı ve ağırlıklandırmayı kontrol eder.
Gerçek pickle/checkpoint yüklenmez, Self-MM/BERT kurulmaz, dosya yazılmaz.
SHA256 için büyük dosyalar bayt olarak okunur. Gerçek train istatistiği ve
gerçek metrik eşleşmesi bu aşamada hesaplanmaz.

## Gerçek değerlendirme

Kontrol başarılıysa aynı PowerShell oturumunda:

```powershell
& '..\.venv\Scripts\python.exe' -B `
    '.\research\evaluate_selfmm_portable_k0_k7.py' `
    @k07Args
```

Önce train istatistikleri, sonra baseline A/C ve dropout B/D değerlendirilir.
GPU'da aynı anda bir checkpoint tutulur. Her koşulda 686 örnek vardır:
171 adet dört örneklik batch ve son iki örneklik batch. C/D iki forward kullanır;
ek bir tüm ağırlıklar=1 forward'ı identity kontrolüdür.

Çıktı klasörü mevcutsa işlem başlamaz. Başarılı koşuyu yeniden çalıştırmak
gerekmez. Hata varsa önce kaydı inceleyin; neden giderildikten sonra yeni çıktı
adı seçin. Hash/tolerans değiştirerek korumaları atlamayın.

İsteğe bağlı `--repo-dir`, `--config`, `--dependencies`, `--protocol` ve
`--package-dir` vardır. Repo varsayılanı betiğin üst klasörüdür; eski portable K0
config/bağımlılıkları bu repo altında aranır. Yeni protokol JSON'u betiğin yanındaki
`configs` klasöründedir. Komutta verilen göreli yollar PowerShell klasörüne göredir.

## Beklenen çıktı

| Dosya | İçerik/satır sayısı |
|---|---|
| `metrics.csv` | 4 yöntem × 8 koşul = 32 satır. |
| `reference_checks.csv` | 32 sonuç × 4 metrik = 128 kontrol. |
| `predictions_and_weights.csv` | 2 checkpoint × 8 koşul × 686 örnek = 10976 satır; her satırda plain/weighted tahmin. |
| `controls.csv` | 16 identity/girdi/hook kontrolü. |
| `reference_input_checks.csv` | 16 girdi özeti karşılaştırması. |
| `corruption_summary.csv` | 2 modalite × 8 koşul = 16 maske/gürültü özeti. |
| `train_stats_checks.csv` | Ses ve görüntü için 2 istatistik kontrolü. |
| `train_feature_stats.json` | Bu çalışmada yeniden hesaplanan train istatistikleri. |
| `manifest.json` | Durum, sürümler, dosya özetleri ve kontroller. |

MAE/Corr bütün 686 örnekte; Non0 accuracy ve support-weighted binary F1,
etiketi sıfır olmayan 656 örnekte hesaplanır. Pozitif sınıf `prediction > 0`.
Metrik toleransı `1e-5`, train istatistiği toleransı `1e-12`; girdi/etiket
SHA256 eşleşmesi tam olmalıdır. Metrik eşleşmesi, eski tekil tahminlerin her
birinin aynı olduğu iddiası değildir; paket eski tekil tahminleri içermez.
128 kontrol, 128 bağımsız istatistiksel tekrar anlamına gelmez.

## Paylaşılacak çıktılar

```powershell
$portableK07Dir = '..\runs\selfmm_portable_k0_k7_test_seed1111_v1'
Get-Content "$portableK07Dir\manifest.json" -Raw |
    ConvertFrom-Json |
    Select-Object status, training_seed, corruption_seed, split, conditions,
        historical_run_manifests_required,
        train_statistics_recomputed_and_matched,
        reference_test_labels_matched, reference_metrics_matched,
        reference_corrupted_inputs_matched, shared_inputs_verified,
        hooks_restored, text_unchanged, lengths_unchanged, padding_unchanged,
        protected_files_unchanged, data_file_unchanged,
        metric_rows, prediction_rows, reference_check_rows,
        reference_input_check_rows, train_stats_check_rows |
    Format-List

$metricChecks = @(Import-Csv "$portableK07Dir\reference_checks.csv")
$failedMetricChecks = @($metricChecks | Where-Object { $_.passed -ne 'True' })
[pscustomobject]@{
    Total = $metricChecks.Count
    Passed = $metricChecks.Count - $failedMetricChecks.Count
    Failed = $failedMetricChecks.Count
} | Format-List

Import-Csv "$portableK07Dir\train_stats_checks.csv" | Format-List
Import-Csv "$portableK07Dir\reference_input_checks.csv" |
    Format-Table checkpoint, condition, shared_inputs, passed -AutoSize
Import-Csv "$portableK07Dir\metrics.csv" |
    Sort-Object condition, method |
    Format-Table method, condition, n, Non0_acc_2, Non0_F1_score, MAE, Corr -AutoSize
```

Başarı: `status=completed`, 128/128 metrik, 16/16 girdi ve 2/2 train istatistik
kontrolü başarılı; koruma bayrakları True. `historical_run_manifests_required`
False kalmalıdır.

## Doğrulama kapsamı

10 CPU kontrol testi geçti: maskeler, train-only hesaplama/padding, yanlış
uzunluk ve sonlu olmayan değerlerin reddi, eski batch özetiyle uyum, --check
ayrımı, yapay uçtan uca çıktı sayıları, mevcut çıktının korunması ve yanlış
metrik/girdi referanslarında hata kaydı.

Ayrı bir gerçek-veri CPU denetiminde iki train istatistiği ve sekiz bozulmuş
test girdisi referansla eşleşti; kaynaklar değişmedi. Model/checkpoint yüklemesi
engellendi. Bu denetim --check'ten ayrıdır: gerçek veri açıldı, GPU tahmini
yapılmadı. Eski çalışma manifestleri olmadan ayrı kaynak kopyasında, model/veri
yükleyicileri engellenerek --check de sınandı.

Yeni giriş noktasının gerçek GPU tahminlerini kullanıcı çalıştıracak. Temiz ML
kurulumu, başka bilgisayarda gerçek GPU tekrarı ve diğer seed'lerin bu giriş
noktasından değerlendirilmesi henüz doğrulanmış değildir.

Test daha önce görülmüştür. Mevcut sabit protokol tekrar edilir; yeni
hiperparametre veya iyi görünen seed seçilmez. Toplam süre istatistik hesabı,
yükleme ve kontrolleri içerir; ayrı çıkarım benchmark'ı yerine kullanılamaz.
Önceki beş-seed bulguları korunur.

## Fork

Yalnız üç yeni dosyayı ve README değişikliğini stage edin. Veri, checkpoint,
BERT, `.venv`, `runs` ve kişisel orijinal config değişikliğini eklemeyin.

Başlık: `feat: add portable seed-1111 K0-K7 checkpoint replay`

```text
Extend the verified K0 setup to frozen K0-K7 A/B/C/D test evaluation.
Recompute noise statistics from valid TRAIN frames and apply the existing
deterministic corruption protocol without historical run manifests.

Add 128 aggregate metric checks, 16 corrupted-input fingerprint checks,
and train-statistics/label controls. Preserve existing scripts and outputs.
Document offline preflight and explicit asset paths.

Pass 10 CPU contract tests and a real-data CPU compatibility audit.
Real GPU replay through the new entry point remains to be verified.
```
