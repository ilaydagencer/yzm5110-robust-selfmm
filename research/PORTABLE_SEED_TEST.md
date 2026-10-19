# Seed 1115: taşınabilir K0–K7 A/B/C/D test tekrarı

Bu adım, daha önce eğitilmiş seed 1115 baseline ve dropout checkpoint'lerinin
sonuçlarını eski çalışma manifestlerine bağlanmadan yeniden üretir. Eğitim
yapılmaz. Kullanıcı gerçek GPU komutunu çalıştırır; Git işlemlerini Fork'tan yapar.

## Ne değişiyor?

Mevcut giriş noktası `evaluate_selfmm_portable_seed_test.py`, artık `--seed 1115` de alır.
Bu teslimde 1112, 1113, 1114 ve 1115 desteklenir. Seed 1111 ayrı giriş noktasıyla çalışır.
Seed 1111'in mevcut betikleri ve bağımlılık kayıtları korunur.

| Ayar | Bu çalışmadaki değer ve anlamı |
|---|---|
| Eğitim seed'i | **1115**; kullanılacak iki eğitim checkpoint'ini ve referans metrik satırlarını seçer. |
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
- `configs/selfmm_portable_seed_dependencies.json`: seed 1112–1115'in checkpoint
  SHA256 özeti, kaynak sürümleri, ortak config/protokol ve sonuç paketi kimlikleri.
  SHA256, dosyanın baytlarından hesaplanan kimliktir; yanlış checkpoint seçimini yakalar.
- `PORTABLE_SEED_TEST.md`: bu yönerge.
- `README.md`: desteklenen seed'ler, tamamlanan seed 1111–1114 ve bekleyen seed 1115 durumu.
- `verification/portable_k0_k7_seed1114_v1.json`: tamamlanan çalışmanın yeni doğrulama kaydı; önceki kayıtlar korunur.
- `PORTABLE_MODEL.md` ve `PORTABLE_K0_K7.md`: sonraki adım bağlantıları güncellendi.

Kodda yalnız kabul edilen seed listesi genişler; model çalıştırma, metrik,
ağırlıklandırma ve bozulma fonksiyonları önceki sürümle aynıdır. Yeni bağımlılık
kaydına 1115 eklenir; 1112–1114 kayıtları değişmez.

Yeni kayıt hazırlanırken checkpoint'ler, eğitim config'leri ve önceki test
kayıtları karşılaştırılmıştır. Bu hazırlık işlemi eski kayıtları okudu;
**teslim edilen betik çalışırken eski run manifestlerini veya kişisel config'leri
okumaz.** Eğitim/test manifestlerinin SHA256 değerleri yalnız kaynak izini saklar.

Gerekli mevcut dosyalar: portable K0 yardımcısı, `evaluate_selfmm_abcd_valid.py`,
`selfmm_corruptions.py`, `verify_results_package.py`, `results_package_v1`,
`configs/selfmm_mosi_portable_k0.json` ve `configs/selfmm_portable_k0_k7_dependencies.json`.
Ortak protokoldeki `training_seed: 1111`, **girdi referansının** kökenidir;
seed 1115 checkpoint seçimi yeni bağımlılık kaydından yapılır. Public paketteki
eski provenance checkpoint çifti de seed 1111'e aittir; seed 1115 diye kullanılmaz.

Model ayarları iki seed 1115 eğitim kaydıyla eşleşir. Veri, checkpoint ve yerel
BERT yolları açık komut parametreleridir. `sentencepiece`, torch/MMSA'dan önce
içe aktarılır. Betik çevrimdışıdır, BERT indirmez. Mevcut çalışan venv'i kullanın.
Farklı bir repo kopyasında çalıştırırken MMSA kurulumu da o kopyanın `src` yolunu
kullanmalıdır; yanlış yerel kurulum kontrol sırasında reddedilir.

## 1. Dosyaları yerleştir

```powershell
Set-Location 'C:\Users\ilayda\YZM5110\MMSA'
$deliveryDir = 'C:\Users\ilayda\Documents\Codex\2026-10-17\referenced-chatgpt-conversation-this-is-an\outputs\research_portable_seed1115_v1'
& "$deliveryDir\Install-PortableSeed1115.ps1" -CheckOnly
```

`YERLESTIRME_KONTROLU_BASARILI` gelirse:

```powershell
& "$deliveryDir\Install-PortableSeed1115.ps1"
```

Altı mevcut dosya güncellenir, bir doğrulama kaydı eklenir. Tüm hedefler yazmadan önce kontrol
edilir; beklenmeyen içerik varsa üzerine yazılmaz. Aynı paket ikinci kez
çalıştırıldığında `PAKET_ZATEN_YERINDE` ile dosya değiştirmeden biter. Git işlemi yoktur.

## 2. Eğitim/model yüklemesi yapmadan --check

Aşağıdaki blok yeni PowerShell oturumunda da kullanılabilir:

```powershell
Set-Location 'C:\Users\ilayda\YZM5110\MMSA'
$bertDir = Join-Path $env:USERPROFILE '.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594'
$portable1115Args = @(
    '--seed', '1115',
    '--data-file', '..\data\MOSI\Processed\unaligned_50.pkl',
    '--baseline-checkpoint', '..\runs\selfmm_baseline_seed1115\saved_models\self_mm-mosi.pth',
    '--dropout-checkpoint', '..\runs\selfmm_dropout_a20_v20_seed1115\saved_models\self_mm-mosi.pth',
    '--bert-dir', $bertDir,
    '--output-dir', '..\runs\selfmm_portable_k0_k7_test_seed1115_v1'
)
& '..\.venv\Scripts\python.exe' -B `
    '.\research\evaluate_selfmm_portable_seed_test.py' `
    @portable1115Args --check
```

`SEED_SECILDI: egitim=1115; bozulma=1111; referans metrik seed=1115`,
`CUDA: True` ve sonunda `KONTROL_BASARILI: seed1115/K0-K7` beklenir.

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
    @portable1115Args
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
$portable1115Dir = '..\runs\selfmm_portable_k0_k7_test_seed1115_v1'
Get-Content "$portable1115Dir\manifest.json" -Raw |
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

Import-Csv "$portable1115Dir\reference_checks.csv" |
    Group-Object passed | Select-Object Name, Count

Import-Csv "$portable1115Dir\train_stats_checks.csv" | Format-List

Import-Csv "$portable1115Dir\reference_input_checks.csv" |
    Format-Table checkpoint, condition, shared_inputs, passed -AutoSize

Import-Csv "$portable1115Dir\metrics.csv" |
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
False, koruma bayrakları True olmalıdır. Referans metrikler **seed 1115**'e,
girdi referansı **seed 1111**'e aittir.

## Tamamlanan seed 1112 tekrarı — 19 Ekim 2026

Mevcut RTX 3050 Ti Laptop GPU ortamında seed 1112 K0–K7 değerlendirmesi
tamamlandı. 128/128 metrik, 16/16 bozulmuş girdi ve 2/2 train istatistiği
kontrolü geçti. Referans metriklerde en büyük fark 8.88e-16, train ortalama/std
farkları sıfırdı. Uniform ağırlık identity kontrolünün en büyük hatası sıfırdır.

10976 tahmin satırından metrikler ayrıca yeniden hesaplandı. Bağımsız hesaplama
ile float32 MAE arasındaki en büyük fark 1.38e-7 idi (tolerans 1e-6); diğer
metriklerde fark 8.89e-16 altında kaldı. Etiketler, indeksler, sonlu değerler ve
ağırlık toplamları kontrol edildi. 8 çıktı ve 106 girdi dosyasının özeti eşleşti.
Seed 1111 ile beş protokol/kontrol dosyası bayt düzeyinde de aynıdır.
[Doğrulama kaydı](verification/portable_k0_k7_seed1112_v1.json).

| Seed 1112, K0 | Non0 accuracy (%) | MAE | Corr |
|---|---:|---:|---:|
| A | 83.54 | 0.7063 | 0.7877 |
| B | 81.86 | 0.8008 | 0.7457 |
| C | 83.69 | 0.8453 | 0.7595 |
| D | 81.86 | 0.9945 | 0.6730 |

Bu seed'de B'nin K0 MAE'si A'dan yaklaşık 0.0945 daha yüksek; C−A yaklaşık
+0.1390, D−B yaklaşık +0.1937'dir. Ağırlıklandırma bazı sınıf kararlarını
koruyabilse veya iyileştirebilse de duygu şiddetini tahmin etme hatasını artırır.
K0–K7'nin tümünde B'nin MAE'si A'dan, C'nin A'dan ve D'nin B'den yüksektir.
Bu önceki bulguların tekrar üretimidir; beş-seed analizine altıncı gözlem eklenmez.

Kaydedilen 195.19 saniye, yükleme/istatistik/kontroller dahil tekrar süresidir;
çıkarım benchmark'ı veya seed'ler arası hız karşılaştırması olarak yorumlanmaz.
Bu doğrulama mevcut ortam içindir; temiz kurulum/taşınmış kaynakla gerçek GPU
çalışması ve eski tekil tahminlerle birebir karşılaştırma iddiası yoktur.

Dosya özeti denetimi bu seed 1113 güncellemesinden **önce** yapıldı. Güncelleme,
ortak betiği ve seed kayıt dosyasını bilinçli olarak değiştirir. Eski manifestteki
bu iki kaynak özeti tarihsel sürüme aittir; yeni kaynak sürümünün hash'i aynı
olmak zorunda değildir. Doğrulama kaydı denetlenen eski iki özeti de saklar.
Veri ve checkpoint dosyaları değiştirilmez.

## Tamamlanan seed 1113 tekrarı — 19 Ekim 2026

Mevcut RTX 3050 Ti Laptop GPU ortamında 128/128 metrik, 16/16 bozulmuş
girdi ve 2/2 train istatistiği kontrolü geçti. En büyük referans metrik farkı
1.22125e-15 (Corr), MAE ve accuracy farkı sıfırdır. Train ortalama/std farkları
ve uniform ağırlık identity hatası sıfırdır. Ses için 49878, görüntü için
54741 geçerli train zaman adımı aynı istatistikleri verdi.

10976 tahmin satırı üzerinden metrikler bağımsız olarak yeniden hesaplandı.
Float32 MAE hesabıyla en büyük fark 9.75e-8 altında (tolerans 1e-6), diğer
metriklerde 1.23e-15 altındadır. Etiketler, örnek indeksleri, sonlu değerler ve
ağırlık toplamları doğrulandı. 8 çıktı ve 106 kayıtlı girdi dosyasının SHA256
özeti eşleşti. Seed 1111 ile beş protokol/kontrol dosyası bayt düzeyinde aynıdır.
[Doğrulama kaydı](verification/portable_k0_k7_seed1113_v1.json).

| Seed 1113, K0 | Non0 accuracy (%) | MAE | Corr |
|---|---:|---:|---:|
| A | 85.37 | 0.7289 | 0.7860 |
| B | 83.69 | 0.7491 | 0.7636 |
| C | 84.91 | 0.9086 | 0.7266 |
| D | 83.84 | 0.9955 | 0.6964 |

K0'da B−A MAE farkı yaklaşık +0.0202, C−A +0.1797, D−B +0.2464'tür.
MAE'nin artması duygu şiddeti tahmininin kötüleştiğini gösterir. D'nin accuracy'si
B'den yaklaşık 0.15 yüzde puan yüksek olsa da bu, artan MAE'yi ortadan kaldırmaz:
sınıfın işaretini doğru bulmak ve duygu şiddetini doğru tahmin etmek farklıdır.
K0–K7'nin tümünde B'nin MAE'si A'dan, C'nin A'dan ve D'nin B'den yüksektir.
Bu, seed 1113'ün önceki test bulgularını tekrar üretir; yeni bir bağımsız seed
veya test üzerinden yeni yöntem/ayar seçimi değildir.

207.14 saniye, yükleme/istatistik/çıkarım/kontroller dahil toplam tekrar süresidir;
çıkarım benchmark'ı olarak kullanılmaz. Doğrulama mevcut ortam içindir.
Temiz kurulum, taşınmış kaynakta gerçek GPU çalışması ve eski tekil tahminlerle
birebir karşılaştırma bu kaydın kapsamında değildir.

Dosya özetleri seed 1114 desteği eklenmeden önce denetlendi. Doğrulama JSON'u
denetlenen betik ve bağımlılık kaydının eski özetlerini saklar. Bu güncelleme
sonrasında bu iki kaynak dosyanın özeti doğal olarak değişir; tarihsel manifest
önceki sürümün kaydıdır. Veri ve checkpoint dosyaları korunur.

## Tamamlanan seed 1114 tekrarı — 19 Ekim 2026

Mevcut RTX 3050 Ti Laptop GPU ortamında 128/128 metrik, 16/16 bozulmuş
girdi ve 2/2 train istatistiği kontrolü geçti. En büyük referans metrik farkı
9.99201e-16 (Corr), MAE ve accuracy farkları sıfırdır. Train ortalama/std
farkları ve uniform ağırlık identity hatası sıfırdır. Ses için 49878, görüntü
için 54741 geçerli train zaman adımı önceki istatistiklerle eşleşti.

10976 tahmin satırından metrikler bağımsız olarak yeniden hesaplandı. Float32
MAE hesabıyla en büyük fark 1.16e-7 altında (tolerans 1e-6), diğer metriklerde
1e-15 altındadır. Etiketler, indeksler, sonlu değerler ve ağırlık toplamları
doğrulandı. 8 çıktı ve 106 kayıtlı girdinin SHA256 özeti eşleşti. Seed 1111 ile
beş protokol/kontrol dosyası bayt düzeyinde de aynıdır.
[Doğrulama kaydı](verification/portable_k0_k7_seed1114_v1.json).

| Seed 1114, K0 | Non0 accuracy (%) | MAE | Corr |
|---|---:|---:|---:|
| A | 83.38 | 0.7251 | 0.7858 |
| B | 82.32 | 0.7485 | 0.7726 |
| C | 83.38 | 0.8348 | 0.7774 |
| D | 82.32 | 0.8961 | 0.7514 |

K0'da B−A MAE farkı yaklaşık +0.0234, C−A +0.1097,
D−B +0.1476'dır. K0'da C/A ve D/B accuracy değerleri aynı olmasına
rağmen ağırlıklandırma MAE'yi artırır. Doğru sınıf oranının korunması, duygu
şiddetinin de aynı doğrulukla tahmin edildiğini göstermez. K0–K7'nin tamamında
B'nin MAE'si A'dan, C'nin A'dan ve D'nin B'den yüksektir. Bu seed'deki örüntü,
önceki test bulgularının tekrar üretimidir; beş-seed analizine yeni bağımsız
gözlem eklenmez. Test sonuçlarından yeni yöntem veya ayar seçilmez.

199.68 saniye yükleme, train istatistiği, çıkarım ve kontroller dahil toplam
tekrar süresidir; çıkarım benchmark'ı olarak kullanılmaz. Mevcut ortamda
doğrulama yapılmıştır; temiz kurulum, taşınmış kaynakla gerçek GPU çalışması
ve eski tekil tahminlerle birebir karşılaştırma bu kaydın kapsamında değildir.

Dosya özetleri seed 1115 desteği eklenmeden önce denetlendi. Doğrulama kaydı,
denetlenen betik ve bağımlılık JSON'unun eski özetlerini saklar. Bu güncelleme
sonrasında bu iki kaynak dosyanın özeti değişir; tarihsel manifest önceki
sürümün kaydıdır. Veri ve checkpoint dosyaları korunur.

## Hazırlık doğrulaması ve sınırlar

Seed 1115 hazırlığında 17 CPU testi geçti: seed seçimi, yanlış/karışmış checkpoint
reddi, seed'e göre referans metrik seçimi, train-only istatistikler, bozulma/girdi
sırası uyumu, yapay uçtan uca çıktı, hata tanı kayıtları ve seed 1112/1113/1114
akışlarının korunması dahil. Gerçek dosyalarla yazmasız `--check` başarılıdır.
Eski run manifestleri bulunmayan ayrı kaynak kopyasında model/dataset yükleyicileri
engellenerek de `--check` geçti; kopya dosyalar değişmedi, çıktı klasörü oluşmadı.

Seed 1115'in gerçek GPU tekrarı bu giriş noktasında henüz çalıştırılmadı;
kullanıcı sonraki adımda çalıştıracak. CPU yapay testleri gerçek model
doğrulamasının yerine geçmez. Temiz ML kurulumu ve başka bilgisayarda gerçek
GPU çalışması doğrulanmadı. Eski tekil tahminler public pakette yoktur; kontrol
toplu metrikleri ve girdi özetlerini karşılaştırır. 128 kontrol, 128 bağımsız
deney değildir. Test daha önce görülmüştür; yeni τ, dropout veya seed seçilmez.

Seed 1115 tamamlandığında beş eğitim seed'inin mevcut ortamda taşınabilir
test tekrarları tamamlanmış olacak. Sonra temiz ortam/taşınmış kaynakla gerçek
GPU doğrulaması ve raporun içerik, kaynakça, sayfa ve PDF teslim kontrolleri
ele alınacaktır. Bu aşamalar şimdiden tamamlanmış kabul edilmez.

## Fork commit'i

Bu güncellemede yedi dosya stage edilir: ortak seed betiği, seed bağımlılık
JSON'u, README, PORTABLE_SEED_TEST, PORTABLE_MODEL, PORTABLE_K0_K7 ve yeni
seed 1114 doğrulama JSON'u. Veri, checkpoint, BERT, venv, runs ve kişisel
orijinal config değişikliği eklenmez.

Başlık: `feat: verify seed-1114 replay and enable portable seed 1115`

```text
Record completed seed-1114 K0-K7 GPU replay: all 128 metric comparisons,
16 input fingerprints and two train-statistics checks pass. Recompute
metrics from 10,976 prediction rows and verify eight output and 106 input hashes.

Register audited seed-1115 checkpoints and extend CLI seed selection.
Preserve seed-1112/1113/1114 registry data and model, metric and corruption
functions. Update guides and add the seed-1114 verification record.

Pass 17 CPU tests and read-only preflight, including a relocated source
copy without historical run manifests. Seed-1115 GPU replay remains pending.
```
