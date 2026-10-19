# Gerçek model akışını taşıma: seed 1111, K0

Bu giriş noktası, kayıtlı iki checkpoint ile A/B/C/D test tahminlerini yeniden
üretir. İlk kapsam **seed 1111, temiz K0, 686 test örneği, batch 4 ve τ=1** ile
sınırlıdır. Eğitim, hiperparametre seçimi ve yeni hız ölçümü yapmaz.

Küçük sonuç paketiyle farkı şudur: `verify_results_package.py` yalnız kayıtlı
metrikleri yeniden özetler. `evaluate_selfmm_portable_k0.py` ise gerçek veriyi
ve model ağırlıklarını yükler; tahminlerden metrikleri tekrar hesaplar.

## Config, checkpoint ve manifest ne demek?

| Dosya türü | Görevi | Bu giriş noktasında kullanım |
|---|---|---|
| Config | Modelin nasıl kurulacağını belirler. Örneğin BERT, ses/görüntü boyutları ve batch büyüklüğü. | `configs/selfmm_mosi_portable_k0.json`; mevcut deneylerin etkin Self-MM/MOSI ayarları korunur. |
| Checkpoint | Eğitilmiş modelin öğrendiği ağırlıkları içerir. | Baseline ve dropout için iki ayrı `.pth` dosyası, açık komut parametreleriyle verilir. |
| Bağımlılık manifesti | Hangi veri, checkpoint ve kod sürümünün beklendiğini kaydeder. | `configs/selfmm_portable_k0_dependencies.json`; kişisel dosya yolu içermez. |
| Çalışma manifesti | Yeni çalışmada ne yapıldığını, hangi kontrollerin geçtiğini kaydeder. | Yeni çıktı klasöründeki `manifest.json`; işlem sırasında oluşturulur. |

SHA256, dosya içeriğinin kimlik kontrolü için kullanılan bir özettir. Beklenen
özetten farklı veri/checkpoint sessizce kullanılmaz. Bu kontrol dosya değişimini
saptar; kriptografik imzalı bir kaynak doğrulaması değildir.

Eski GPU değerlendirme betikleri önceki `runs` klasörlerindeki yerel config ve
manifest zincirine bağlıdır. Yeni betik bu eski manifestleri okumaz. Yalnız
depoda taşınabilen `results_package_v1` paketini ve yeni bağımlılık manifestini
okur; büyük dosyaların yerini komuttan alır. Eski betikler değiştirilmedi.

## Gereken dosyalar ve ortam

1. Depo kodu, `research/evaluate_selfmm_abcd_valid.py` ağırlıklandırma yardımcısı,
   `verify_results_package.py`, `results_package_v1` ve bu adımın dosyaları.
2. CMU-MOSI `unaligned_50.pkl`: deneylerde kullanılan aynı veri dosyası.
3. Seed 1111 baseline ve dropout checkpoint'leri: eğitim tekrar yapılmaz.
4. Yerel `bert-base-uncased` klasörü: `config.json`, `vocab.txt` ve
   `model.safetensors` veya `pytorch_model.bin`. Varsa tokenizer dosyaları da
   klasörle birlikte korunmalıdır. MMSA, checkpoint'i yüklemeden önce BERT
   nesnesini kurduğu için bu dosyalara hâlâ ihtiyaç duyar.
5. README'deki Python/ML ortamı. Betik torch `2.3.1+cu121`, NumPy `1.26.4`,
   transformers `4.30.2` sürümlerini kontrol eder. Gerçek çalıştırma CUDA ister;
   yalnız `--check` CUDA olmadan da yapılabilir. Gözlenen Python 3.10.5'tir.

**Çalışan proje venv'ini yeniden kurmayın.** Yeni bilgisayar için README kurulum
adımları kullanılır. Veri, checkpoint ve BERT dosyaları Git'e eklenmez. İzinli
yerel kopyalar yeni bilgisayarda başka klasörlerde bulunabilir; komut yolları
buna göre değiştirilir. Checkpoint'ler kamuya açık pakette yer almadığından,
yalnız Git klonu gerçek model tekrarını çalıştırmaya yetmez.

Betik çevrimdışı çalışır; eksik BERT dosyasını otomatik indirmez. Klasörü başka
bilgisayara taşırken cache içindeki sembolik bağlantıların hedeflerinin de
erişilebilir olduğundan veya dosyaların gerçek içeriklerinin kopyalandığından
emin olun. Model dosyaları sağlanmamışsa önce bu bağımlılık tamamlanmalıdır.

## Dosya yolları nasıl belirlenir?

| Parametre | Açıklama |
|---|---|
| `--repo-dir` | İsteğe bağlı. Varsayılan, betiğin bulunduğu `research` klasörünün üstüdür. İçe aktarılan MMSA bu repoya ait olmalıdır. |
| `--config` | İsteğe bağlı. Varsayılan betiğin yanındaki `configs/selfmm_mosi_portable_k0.json`. |
| `--dependencies` | İsteğe bağlı. Varsayılan betiğin yanındaki bağımlılık manifesti. |
| `--package-dir` | İsteğe bağlı. Varsayılan `<repo>/research/results_package_v1`. |
| `--data-file` | Zorunlu. Gerçek MOSI pickle yolu. |
| `--baseline-checkpoint` | Zorunlu. Seed 1111 baseline `.pth` yolu. |
| `--dropout-checkpoint` | Zorunlu. Seed 1111 dropout `.pth` yolu. |
| `--bert-dir` | Zorunlu. Yerel BERT dosyalarının bulunduğu klasör. |
| `--output-dir` | Zorunlu. Üst klasörü mevcut, kendisi henüz bulunmayan yeni çıktı klasörü. |
| `--check` | Hazırlık kontrolü; gerçek tahmin üretmez. |

Komutta yazılan göreli yollar PowerShell'in **mevcut klasörüne** göredir.
Varsayılan config yolu ise **betiğin konumuna** göre çözülür. `--data-file`,
config içindeki `dataset_root_dir` ve `featurePath` birleşimini yalnız bellekte
değiştirir. `--bert-dir` aynı şekilde `pretrained` yolunu yalnız bellekte
değiştirir. Config veya orijinal MMSA kaynaklarına yazılmaz.

Config'in konum dışındaki etkin ayarları manifestteki özetle eşleşmelidir.
Öğrenme oranları gibi miras alınan eğitim alanları da config'de korunmuştur;
bu betik optimizer oluşturmadığı için eğitim başlatmaz. Dropout checkpoint'i
`eval()` modunda çalışır; eğitimdeki giriş dropout'u burada uygulanmaz.

Kaynak kod özetleri CRLF → LF dönüşümünden sonra hesaplanır. Böylece Git'in
satır sonu dönüşümü izin verilen tek normalleştirmedir; başka kod değişikliği
durmaya neden olur. Hata alındığında özeti/toleransı değiştirerek kontrolü
atlamayın; farkı inceleyin.

## Mevcut bilgisayarda adım adım çalıştırma

Önce aşağıdaki parametreleri aynı PowerShell oturumunda tanımlayın. BERT yolu
bu bilgisayarda bulunan snapshot'a aittir; başka ortamda yerel BERT yolunu yazın.

```powershell
Set-Location 'C:\Users\ilayda\YZM5110\MMSA'
$bertDir = Join-Path $env:USERPROFILE '.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594'
$k0Args = @(
    '--data-file', '..\data\MOSI\Processed\unaligned_50.pkl',
    '--baseline-checkpoint', '..\runs\selfmm_seed1111\saved_models\self_mm-mosi.pth',
    '--dropout-checkpoint', '..\runs\selfmm_dropout_a20_v20_seed1111\saved_models\self_mm-mosi.pth',
    '--bert-dir', $bertDir,
    '--output-dir', '..\runs\selfmm_portable_k0_test_seed1111_v1'
)
& '..\.venv\Scripts\python.exe' -B '.\research\evaluate_selfmm_portable_k0.py' @k0Args --check
```

`--check`, büyük dosyaları SHA256 için bayt olarak okur; pickle'ı açmaz,
checkpoint'i modele yüklemez ve Self-MM/BERT nesnesi kurmaz. Küçük yapay CPU
girdileriyle ağırlıklandırma ve hook temizliği sınanır. İndirme veya çıktı
yazma yapılmaz. Bu yüzden disk hızına bağlı olarak kontrol biraz sürebilir.

Kontrol `KONTROL_BASARILI` ile biterse ve `CUDA: True` görünürse gerçek tekrar:

```powershell
& '..\.venv\Scripts\python.exe' -B '.\research\evaluate_selfmm_portable_k0.py' @k0Args
```

Bir checkpoint GPU'dayken diğeri yüklenmez. A/C baseline'dan, B/D dropout
checkpoint'inden hesaplanır. C/D, önce tek modlu tahminlerle ağırlıkları bulur,
sonra ağırlıklı fusion için ikinci forward yapar. Ek bir tüm ağırlıklar=1
forward'ı yalnız doğruluk kontrolüdür. Bu nedenle çalışma süresi önceki
çıkarım benchmark'ı ile karşılaştırılacak bir gecikme ölçümü değildir.

Tüm test bölümü çalışır: 171 adet dört örneklik batch ve son iki örneklik batch.
MMSA pickle dosyasını okurken tüm split sözlüğünü açar; yalnız `mode=test`
örnekleri değerlendirilir. Yeniden eğitim veya testten yeni yöntem seçimi yoktur.

## Çıktı ve başarı ölçütü

```powershell
$portableDir = '..\runs\selfmm_portable_k0_test_seed1111_v1'
Get-Content "$portableDir\manifest.json" -Raw |
    ConvertFrom-Json |
    Select-Object status, training_seed, split, condition,
        historical_run_manifests_required, reference_metrics_matched,
        shared_inputs_verified, hooks_restored, inputs_unchanged,
        protected_files_unchanged, data_file_unchanged,
        metric_rows, prediction_rows, reference_check_rows |
    Format-List

Import-Csv "$portableDir\reference_checks.csv" |
    Format-Table method, metric, absolute_error, tolerance, passed -AutoSize
```

Beklenen dosyalar:

| Dosya | İçerik |
|---|---|
| `metrics.csv` | Dört yöntem için dört metrik; 4 satır. |
| `reference_checks.csv` | Her yöntemin MAE, Non0 accuracy, weighted F1 ve Corr karşılaştırması; 16 satır. |
| `predictions_and_weights.csv` | İki checkpoint × 686 örnek; plain/weighted tahmin ve ağırlıklar, toplam 1372 satır. |
| `controls.csv` | İki checkpoint için identity, hook ve girdi kontrolleri; 2 satır. |
| `manifest.json` | Config, sürümler, dosya özetleri, durum ve kontroller. |

Başarı: `status=completed`, `reference_metrics_matched=True`, 16 karşılaştırmanın
tamamında `passed=True`, kaynak/veri/girdi kontrollerinin tümü True.
`historical_run_manifests_required=False` görünmelidir.

MAE ve Corr 686 örneğe; Non0 accuracy ve support-weighted binary F1, etiketi
sıfır olmayan 656 örneğe dayanır. Pozitif sınıf sınırı `prediction > 0`.
MAE float32 tensör ortalamasından, diğer metrikler yuvarlanmadan hesaplanır.
Paketin seed 1111 K0 kayıtlarıyla mutlak `1e-5` tolerans kullanılır.

**Metrik eşleşmesi, her eski örnek tahmininin birebir eşleştiği anlamına gelmez.**
Kamuya açık küçük pakette eski tekil tahminler bulunmadığı için bu betik
16 toplam metrik kontrolü yapar. Aynı hash'li veri/checkpoint ve ortak girdiler
ayrıca doğrulanır. Eşleşmeyen metrik olduğunda CSV ve `failed_or_interrupted`
manifesti inceleme için saklanır; hata başarı olarak sunulmaz. Var olan çıktı
klasörü yeniden kullanılamaz; hata nedeni giderildikten sonra yeni bir ad seçilir.

## Doğrulanmış kapsam ve sonraki sıra

Hazırlık sırasında mevcut venv ile `--check` geçti. Kod ve küçük paket, eski
`runs` manifestleri olmadan boşluk içeren farklı bir klasöre taşındı; `--check`
yeniden geçti. Bu kontrolde model, checkpoint ve pickle yükleyicileri hata
verecek şekilde engellendi; 98 kopya dosyasının değişmediği görüldü. 10 CPU
sözleşme testi de geçti; bunlar gerçek checkpoint başarısı yerine geçmez.

19 Ekim 2026 tarihinde kullanıcı mevcut Python 3.10.5 ortamında RTX 3050 Ti
GPU ile gerçek checkpoint tekrarını tamamladı. `status=completed`; dört
yöntem için 16 referans karşılaştırmasının tamamı geçti. En büyük mutlak
metrik farkı 5.551115123125783e-16, izin verilen tolerans 1e-5 idi.

Sonradan yapılan bağımsız dosya denetiminde 4 çıktı dosyasının ve kayıtlı
103 girdinin SHA256 özetleri eşleşti. 1372 tahmin satırında örnek sırası,
etiket eşleşmesi, sonlu değerler ve ağırlık toplamları doğrulandı. Bu
tahminlerden MAE, Non0 accuracy, weighted F1 ve Corr yeniden hesaplandı;
float32 MAE ile bağımsız hesaplama farkına 1e-6 tolerans uygulandı.
Diğer metriklerde fark 1e-12 altında kaldı. İki checkpoint'in girdi özetleri
aynı, tüm ağırlıklar=1 identity kontrolünün maksimum hatası sıfırdı.
[Küçük doğrulama kaydı](verification/portable_k0_seed1111_v1.json).

Bu sonuç mevcut ortamda K0 tekrarını doğrular. Taşınmış kaynak kopyasında
gerçek GPU çalışması, temiz bir ML kurulumu veya uçtan uca yeniden eğitim
henüz doğrulanmış değildir. Kaydedilen 77.28 saniye toplam tekrar süresi;
yükleme, kontroller ve dosya özeti hesaplamasını içerir, önceki çıkarım
benchmark'ının yerine kullanılamaz.

19 Ekim 2026 tarihinde ayrı [K0–K7 giriş noktası](PORTABLE_K0_K7.md)
mevcut GPU ortamında tamamlandı. Train-only gürültü istatistikleri yeniden
hesaplandı; 128 metrik, 16 girdi özeti ve 2 train istatistiği kontrolü geçti.
[K0–K7 doğrulama kaydı](verification/portable_k0_k7_seed1111_v1.json).
Bu K0 betiğinin kapsamı değişmedi. Mevcut ortamda K0–K7 tekrarları
[seed 1112](verification/portable_k0_k7_seed1112_v1.json),
[seed 1113](verification/portable_k0_k7_seed1113_v1.json) ve
[seed 1114](verification/portable_k0_k7_seed1114_v1.json) için de tamamlandı.
Seed 1115 de tamamlandı; [beş-seed kaydı](verification/portable_five_seeds_v1.json).
Sıradaki adım [temiz ortam doğrulamasıdır](CLEAN_ENVIRONMENT.md).

Test daha önce eğitim sonlarında ve eski değerlendirmelerde görülmüştür.
Bu bir dondurulmuş protokol tekrarıdır; yeni test-tabanlı seçim veya el
değmemiş holdout iddiası yoktur. K0 tekrarını doğrulamak mevcut araştırma
bulgularını değiştirmez; çalıştırılabilirlik kanıtını güçlendirir.

## Fork kaydı

Bu adımda yalnız `evaluate_selfmm_portable_k0.py`, `PORTABLE_MODEL.md`,
iki yeni config/bağımlılık JSON'u ve güncellenen `README.md` stage edilir.
Kişisel veri yolu içeren orijinal `src/MMSA/config/config_regression.json`,
veri, checkpoint, BERT, venv ve `runs` çıktıları stage edilmez.

Başlık: `feat: add explicit-asset K0 checkpoint replay`

Description:

```text
Add a seed-1111 K0 A/B/C/D inference entry point with explicit data,
checkpoint and local BERT paths. Preserve the effective batch-4 config
and verify portable dependency hashes without historical run manifests.
Add offline preflight checks, output protection and 16 aggregate metric
comparisons against the public results package.
Document setup, scope and limitations. Validate read-only preflight in
a relocated source copy and pass 10 CPU contract tests.
Verify real-checkpoint K0 GPU replay in the existing environment:
all 16 reference metric checks pass, with maximum error 5.55e-16.
Recompute metrics from saved predictions and verify input/output hashes.
Clean installation and relocated real-GPU execution remain unverified.
```

### K0 başarı kaydını ayrı commit olarak ekleme

Önceki kod değişikliği zaten commit edildiyse, bu belge güncellemesinde
yalnız README, PORTABLE_MODEL ve `verification/portable_k0_seed1111_v1.json`
stage edilir. Yeni eğitim/değerlendirme çalıştırılması gerekmez.

Başlık: `docs: record verified seed-1111 K0 GPU replay`

```text
Record successful K0 A/B/C/D replay with the existing seed-1111
checkpoints on the current RTX 3050 Ti environment. All 16 aggregate
metric checks pass; maximum absolute error is 5.55e-16.

Verify four output hashes and 103 input hashes, and independently
recompute metrics from 1,372 saved prediction rows. Update the
reproduction scope and retain clean-install and relocation limitations.
```
