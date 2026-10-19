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

### Tamamlanan GPU tekrarı — 19 Ekim 2026

Kullanıcı mevcut RTX 3050 Ti Laptop GPU ortamında seed 1111 K0–K7 tekrarını
tamamladı. Eski çalışma manifestleri gerekmedi; eğitim yapılmadı. Python
ortamında torch 2.3.1+cu121, numpy 1.26.4 ve transformers 4.30.2 kullanıldı.
Batch 4, τ=1 ve bozulma seed'i 1111 korundu.

| Denetim | Gözlenen sonuç |
|---|---|
| Yöntem/koşul metrik satırı | 32; 4 yöntem × 8 koşul. |
| Referans metrik karşılaştırması | 128/128 geçti; en büyük mutlak fark 5.55e-16. |
| Bozulmuş girdi SHA256 karşılaştırması | 16/16 tam eşleşti; iki checkpoint aynı girdileri aldı. |
| Train istatistiği | 2/2 geçti; ortalama ve std farkları sıfır. |
| Geçerli train zaman adımı | Ses 49878; görüntü 54741. |
| Kaydedilen tahmin satırı | 10976; her satırda plain ve weighted tahmin. |
| Tüm ağırlıklar=1 identity kontrolü | En büyük tahmin farkı sıfır. |
| Dosya özetleri | 8 çıktı ve 106 kayıtlı girdi denetim sırasında eşleşti. |

Kaydedilen tahminlerden MAE, Non0 accuracy, support-weighted F1 ve Corr
ayrıca yeniden hesaplandı. Örnek indeksleri, etiketler, sonlu sayılar ve ağırlık
toplamları kontrol edildi. Bağımsız hesaplamanın float32 MAE ile en büyük
farkı 7.83e-8 idi (tolerans 1e-6); diğer metriklerde fark 5.56e-16 altında kaldı.
Bu karşılaştırma, yukarıdaki public paketle 128 referans karşılaştırmasından
ayrı bir denetimdir. Eski tekil tahmin dosyalarıyla birebir karşılaştırma yapılmadı.

Kayıt ve kontrol CSV'lerinde hook'ların geri alınması, kaynak dizilerin, metnin,
uzunlukların ve padding'in korunması başarılıdır. Kaynak veri, checkpoint'ler
ve korunan dosyalar değişmedi. Küçük, kişisel yol içermeyen
[doğrulama kaydı](verification/portable_k0_k7_seed1111_v1.json) bu sonuçları,
çıktı dosyalarının özetlerini ve kapsam sınırlarını saklar; büyük varlıkları içermez.

| Koşul | Ses: gerçekleşen maske oranı | Görüntü: gerçekleşen maske oranı | Hedef |
|---|---:|---:|---:|
| K4 | %29.94 | %29.98 | %30 |
| K5 | %50.09 | %49.79 | %50 |
| K6 | %70.03 | %70.09 | %70 |

K7'de gürültü RMS / train std oranı seste 0.49889, görüntüde 0.49996 oldu;
0.5 ölçeğiyle uyumludur. Hedefe yakın bu değerler tek başına doğrulama değildir:
16 girdi SHA256 eşleşmesi, önceki sabit koşullarla tam girdi uyumunu da gösterir.
Train std yalnız eğitim verisinden yeniden hesaplandı.

Bu sonuç **mevcut ortamda gerçek GPU tekrarını** doğrular. Temiz ML kurulumu,
taşınmış kaynak kopyasında gerçek GPU çalışması, başka bilgisayarda çalıştırma
henüz doğrulanmış değildir. Diğer seed'lerin güncel durumu
[ortak seed yönergesindedir](PORTABLE_SEED_TEST.md).
Toplam çalışma süresi 496.72 saniyedir (yaklaşık 8 dakika 17 saniye); veri/model
yüklemesini, train istatistiklerini ve koruma kontrollerini de içerir.

Rapora aktarılabilecek bulgu: Sabit seed 1111 checkpoint'leri, aynı veri ve
bozulma protokolüyle, eski çalışma manifestleri kullanılmadan değerlendirilmiştir.
K0–K7 boyunca A/B/C/D yöntemlerinin 128 toplu metriği public sonuç paketiyle
1e-5 tolerans içinde eşleşmiştir. Bu sonuç çalıştırılabilirlik kanıtını güçlendirir;
beş-seed kalite karşılaştırmasına yeni bir bağımsız gözlem eklemez.

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
Verify real GPU replay in the current environment: all 128 metric,
16 corrupted-input and two train-statistics checks pass.
This seed-1111 replay does not establish clean installation or other-seed replay.
```

### Tamamlanan çalışmayı ayrı belge commit'iyle kaydetme

Giriş noktası ve protokol JSON'u önceki commit'te yer aldıysa bu adımda yalnız
`README.md`, `PORTABLE_K0_K7.md`, `PORTABLE_MODEL.md` ve
`verification/portable_k0_k7_seed1111_v1.json` stage edilir.

Başlık: `docs: record verified seed-1111 K0-K7 GPU replay`

```text
Record successful K0-K7 A/B/C/D replay with existing seed-1111 checkpoints
on the current RTX 3050 Ti environment, without historical run manifests.
Verify 128 aggregate metric checks, 16 corrupted-input fingerprints and
two train-statistics checks. Maximum reference metric error is 5.55e-16.

Independently recompute metrics from 10,976 saved prediction rows and
verify eight output hashes and 106 recorded input hashes. Add a small
verification record and update the README and both portable guides.

Keep clean-install, relocated real-GPU and other-seed replay limitations
explicit. No training, model changes or new independent results.
```

## Sıradaki adım

Mevcut ortamda taşınabilir K0–K7 GPU tekrarları
[seed 1112](verification/portable_k0_k7_seed1112_v1.json) ve
[seed 1113](verification/portable_k0_k7_seed1113_v1.json) için de tamamlandı.
[Ortak seed giriş noktasıyla](PORTABLE_SEED_TEST.md) sırada seed 1114 var.
Bozulma seed'i 1111 ve protokol sabit kalır; yeniden eğitim veya testten ayar
seçimi yapılmaz. Ardından seed 1115, temiz ortam/taşınmış kaynak GPU kontrolü
ve raporun son içerik, sayfa, kaynakça ve PDF teslim denetimi ele alınacaktır.
