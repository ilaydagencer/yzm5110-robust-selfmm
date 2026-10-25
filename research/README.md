# YZM5110 Self-MM araştırma çalışması

İlayda Gençer — CMU-MOSI üzerinde Self-MM’in uyarlanmış yeniden üretimi ve
tek modlu tahmin uyumunun eksik/gürültülü modalitelerde füzyon ağırlığı olarak
incelenmesi. Kod tabanı [MMSA](https://github.com/thuiar/MMSA), temel çalışma
[Self-MM](https://arxiv.org/abs/2102.04830)’dir.

Beş eğitim seed’i için validation ve test değerlendirmeleri tamamlandı.
İncelenen τ=1 ağırlıklandırması her seed ve K0–K7 koşulunda MAE’yi artırdı.
Dropout’un küçük ortalama validation kazanımı testte korunmadı. Seed 1111 K0
kaynak ölçümünde C/D yaklaşık 2,04 kat süre kullandı; eşleştirilmiş tepe
allocated bellek artışı 6,5 KiB oldu. Bu sonuç belirli kural ve protokole aittir.

## Önce hangi işlemi yapacağım?

| Amaç | Gerekenler | Başlangıç |
|---|---|---|
| Raporun özet tablolarını kontrol etmek | Python 3.10+, bu küçük paket | Aşağıdaki hızlı kontrol |
| Özet CSV’leri yeniden hesaplamak | Aynı dosyalar; boş bir çıktı yolu | Özetleri yeniden üretme |
| Gerçek modelle değerlendirme veya eğitim | CUDA ortamı, MOSI özellikleri, checkpoint/config ve referans kayıtları | GPU ortamı ve deney zinciri |

**Küçük paket kontrolü bağımsız doğrulandı; beş seed'in gerçek model tekrarı da mevcut ortamda tamamlandı.**
Bu yol GPU, PyTorch, internet, veri kümesi veya checkpoint istemez.
Kaydedilmiş seed metriklerinden ortalama/SS/farkları, kaydedilmiş batch
sürelerinden kaynak özetlerini yeniden hesaplar. Tahminleri yeniden üretmez.
GPU deney zincirinin başka bilgisayarda eğitimden baştan sona tekrarı henüz doğrulanmadı.

19 Ekim 2026: aynı bilgisayarda yeni, izole venv ve ayrı kaynak kopyasıyla
**seed 1111 K0–K7 gerçek GPU tekrarı tamamlandı**. 128/128 metrik, 16/16 girdi
ve 2/2 train istatistiği kontrolü geçti. Önceki taşınabilir seed 1111 çalışmasının
sekiz çıktı dosyasıyla bayt düzeyinde eşleşme doğrulandı; 10976 satırdaki
tahmin/etiket/ağırlık farkı sıfırdır.
[Kurulum, sonuç ve kapsam](CLEAN_ENVIRONMENT.md),
[küçük doğrulama kaydı](verification/clean_environment_seed1111_k0_k7_v1.json).
Diğer dört seed'in yeni venv'de tekrarı, başka bilgisayar, public Git checkout
ve eğitimden baştan tekrar bu doğrulamanın kapsamına girmez.

## Hızlı kontrol

Komutları deponun ana klasöründe, yani `research` klasörünün bir üstünde çalıştır.
Python 3.10 kurulu bir bilgisayarda:

```powershell
py -3.10 -I -S -B '.\research\verify_results_package.py' --check
```

Mevcut proje venv’iyle aynı işlem:

```powershell
& '..\.venv\Scripts\python.exe' -I -S -B `
    '.\research\verify_results_package.py' --check
```

`-I -S` kullanıcı/site paketlerini yüklemeden çalıştırır; `-B` bytecode dosyası
yazılmasını engeller. Denetim yalnız Python standart kütüphanesini kullanır.
Sonuçta `PAKET_KONTROLU_BASARILI` görülmelidir. `--check` dosya yazmaz.

Kontrol, 19 paket dosyasının SHA-256/boyut/satır sayısını; iki bölmenin
160’ar seed/yöntem/koşul satırını; ortalama, örneklem SS ve eşleştirilmiş
farklarını; kaynak ölçümünün 4.128 batch ve 24 tur kaydını denetler.
Örnek tahminlerinden MAE/F1 hesaplamaz; eski checkpoint veya veri hash’lerini
yerelde yeniden doğrulamaz. Kaydedilmiş doğruluk bayrakları tarihsel kanıttır.

## Özetleri yeniden üretme

Kontrol geçtikten sonra aşağıdaki komut, aynı hesapları yeni klasöre 11 CSV
olarak yazar. Orijinal paket değişmez. Var olan hedef üzerine yazılmaz.

```powershell
if (-not (Test-Path -LiteralPath '..\runs')) {
    New-Item -ItemType Directory -Path '..\runs' | Out-Null
}

& '..\.venv\Scripts\python.exe' -I -S -B `
    '.\research\verify_results_package.py' `
    --output-dir '..\runs\selfmm_public_summary_recheck_v1'
```

Venv’i olmayan okuyucu aynı komutta başlatıcı olarak `py -3.10` kullanabilir.
`OZETLER_YENIDEN_HESAPLANDI` mesajını bekle. Hedef varsa önceki sonucu silmek
yerine yeni bir klasör adı seç. Yeniden hesaplanan sayılar karşılaştırma
toleransı içinde eşleşir; CSV satır sırası veya sayı yazımı nedeniyle dosya
byte’larının orijinal CSV ile aynı olması gerekmez.

## Paket içeriği ve sonuçları okuma

```text
research/
  .gitattributes
  README.md
  verify_results_package.py
  requirements-research.txt
  environment-observed.txt
  configs/selfmm_mosi_batch4.example.json
  results_package_v1/
    manifest.json
    protocol.json
    provenance.json
    DATA_DICTIONARY.md
    validation/            # Beş seed validation, 5 CSV
    evaluation_test/       # Beş seed test, 5 CSV
    inference/             # Seed 1111 K0 kaynak ölçümü, 6 CSV
```

`evaluation_test` adı, depo kökündeki `test*` ignore kuralının sonuçları
gizlememesi için kullanılır. Ayrıntılı kolon açıklamaları
[veri sözlüğündedir](results_package_v1/DATA_DICTIONARY.md).
`.gitattributes`, hash denetimli paket ve kontrol betiğinde Git’in otomatik
LF/CRLF dönüşümünü kapatır. Böylece başka bilgisayardaki checkout dosya
özetlerini sırf satır sonu dönüşümü nedeniyle değiştirmez; bu dosya da commit’e eklenir.

Örnek görüntüleme:

```powershell
Import-Csv '.\research\results_package_v1\evaluation_test\summary_metrics.csv' |
    Where-Object { $_.condition -eq 'K0' } |
    Format-Table method, MAE_mean, MAE_sd, Non0_acc_2_mean -AutoSize
```

| K0 yöntemi | Validation MAE ± SS | Test MAE ± SS |
|---|---:|---:|
| A | 0,7168 ± 0,0202 | 0,7240 ± 0,0146 |
| B | 0,7049 ± 0,0117 | 0,7574 ± 0,0244 |
| C | 0,8665 ± 0,0513 | 0,8424 ± 0,0412 |
| D | 0,9293 ± 0,0667 | 0,9163 ± 0,0747 |

Kaynak: [validation](results_package_v1/validation/summary_metrics.csv) ve
[test](results_package_v1/evaluation_test/summary_metrics.csv) özetleri.
MAE düşük olduğunda iyidir. Accuracy/F1 CSV’de 0–1 ölçeğindedir.
`Non0_acc_pp_delta` zaten yüzde puandır. SS, beş eğitim seed’i üzerinde
ddof=1 ile hesaplanır. Eşleştirilmiş fark önce aynı seed içinde alınır.
SS güven aralığı değildir; sekiz ilişkili koşul sekiz bağımsız tekrar sayılmaz.

Kaynak ölçümünde A/B/C/D süreleri sırasıyla 4,9833 / 5,0731 / 10,1926 /
10,3496 ms/örnektir. Bunlar batch 4 üzerinden amortize maliyettir;
batch=1 gecikmesi değildir. C/A=2,0454 ve D/B=2,0402, altı tur içindeki
oranların ortalamasıdır. Yükleme, aktarım, ısınma ve kontrol geçişleri süreye
dahil değildir. Bellek PyTorch allocated/reserved ölçüsüdür; toplam sistem
GPU belleği değildir. Kaynak SS’si zamanlama tekrarlarına aittir.
[Kaynak özeti](results_package_v1/inference/summary.csv),
[eşleştirilmiş maliyet](results_package_v1/inference/paired_summary.csv).

## Yöntemler ve koşullar

| Yöntem | Eğitim | Çıkarım |
|---|---|---|
| A | Standart Self-MM | Özgün füzyon, tek forward |
| B | Yalnız eğitimde örnek başına bağımsız %20 ses/%20 görüntü giriş dropout’u | Özgün füzyon, tek forward |
| C | A’nın aynı checkpoint’i | Uyum ağırlığı ile iki forward |
| D | B’nin aynı checkpoint’i | Uyum ağırlığı ile iki forward |

Metin ve uzunluklar korunur. Dropout değerlendirmede kapalıdır. Üç yardımcı
tahmin için d_m = abs(y_m − mean(y_other)); w = 3 × softmax(−d/τ), τ=1.
Ağırlıklar ilk füzyon katmanından önce temsil bloklarına uygulanır.
Toplamları 3’tür; olasılık veya kalibre güven değeri değildir. Eksik girdinin
ağırlığı otomatik sıfır olmaz. Yeni öğrenilen parametre eklenmez.

| Koşul | Ses/görüntü işlemi |
|---|---|
| K0 | Temiz girdiler |
| K1 | Tam ses girdisi sıfır |
| K2 | Tam görüntü girdisi sıfır |
| K3 | İkisi de sıfır |
| K4/K5/K6 | Geçerli zaman adımlarında bağımsız %30/%50/%70 maskeleme; oranlar arasında iç içe maskeler |
| K7 | Train geçerli adımlarından bulunan özellik std’sinin 0,5 katı ölçekli Gauss gürültüsü |

K4–K7’de padding korunur. Ses/görüntü RNG’leri bağımsızdır; aynı bölme/koşul
girdileri bütün modellerde ortaktır. Eğitim seed’leri 1111–1115, bozulma
seed’i 1111’dir. Validation/test RNG alanları ayrıdır. Train/valid/test
1284/229/686; Non0 valid/test 216/656 örnektir.

## Tanı deneyleri: ağırlıklandırma neden işe yaramadı? (protokol v2)

Yukarıdaki sonuçlar τ=1 kuralının zararlı olduğunu gösterir, nedenini göstermez.
[Protokol v2](selfmm_diagnostics_protocol_v2.md), aynı 10 checkpoint üzerinde
yalnız çıkarımla şu soruları ayırır: τ taraması (ölçekleme miktarı), metin
ağırlığı sabit varyant (baskın metin bloğu), örnekler arası karıştırılmış ağırlık
(sinyal içeriği), bilinen eksik modaliteye sıfır veren oracle (iyileşme alanı)
ve `[UNK]` ile metin bozulması K8–K10 (baskın modalite bozulduğunda). Hipotezler
ve karar ölçütleri sonuçlardan önce yazılıp commit edilmiştir; τ yalnız
validation'da seçilir, test bir kez çalışır.
[Çalıştırma yönergesi](DIAGNOSTICS_V2.md).

Validation sonucu (beş seed): altı karar da önceden yazılan beklentiyle aynı çıktı.
Zarar, baskın metin bloğunun ölçeklenmesinden gelir (metin ağırlığı sabit
tutulunca kaybolur); uyum sinyali rastgele karıştırılmış ağırlıklardan daha iyi
değildir; ses/görüntü bozulmasında oracle bile iyileşme sağlamaz; metin
bozulduğunda kural yine yardımcı olmaz. Seçilen τ\* = 16 (ağırlıksıza en yakın).
Test, validation paketi commit edildikten sonra bir kez çalıştı; altı kararın altısı
testte de tekrarlandı.
[Validation paketi](diagnostics_v2/validation), [test paketi](diagnostics_v2/evaluation_test),
[ayrıntılı tablolar](DIAGNOSTICS_V2.md#validation-sonucu-25-ekim-2026-beş-seed-baseline-checkpoint).

## Belirsizlik, rapor tabloları ve figürler

Rapordaki bütün tablo ve figürler depodaki dosyalardan betikle üretilir; GPU,
veri kümesi veya checkpoint gerekmez. Komutlar depo kökünde çalışır.

| Çıktı | Betik | Girdi |
|---|---|---|
| [Sıkıştırılmış test tahminleri](diagnostics_v2/predictions) (2,97 MB, kayıpsız float32) | `export_selfmm_predictions.py` | `runs/` içindeki beş test koşusu (bir kez çalıştırıldı) |
| [Bootstrap aralıkları](diagnostics_v2/bootstrap/bootstrap_test.csv) | `bootstrap_selfmm_diagnostics.py` | Sıkıştırılmış test tahminleri |
| [Rapor tabloları](report_tables/tables.md) | `make_selfmm_report_tables.py` | Sonuç paketleri, bootstrap, [eğitim kayıtları](report_tables/training_resources.csv), [yayımlanan skorlar](configs/published_reference.json) |
| [Figürler](figures) | `make_selfmm_figures.py` | Sonuç paketleri, bootstrap, sıkıştırılmış tahminler |

Bootstrap, 686 test örneğini 2000 kez yeniden örnekler; aynı örnek kümesi bütün
seed, yöntem ve koşullara uygulanır (eşleştirilmiş). İstatistik beş seed ortalamalı
farktır; aralık %95 yüzdeliktir. Seed SS eğitim değişkenliğini, bootstrap aralığı
test örneklemesini gösterir; ikisi birlikte raporlanır.

Yeniden doğrulama (dosya yazmaz):

```powershell
& '..\.venv\Scripts\python.exe' -B '.\research\bootstrap_selfmm_diagnostics.py' --verify '.\research\diagnostics_v2\bootstrap'
& '..\.venv\Scripts\python.exe' -I -S -B '.\research\make_selfmm_report_tables.py' --verify
```

`BOOTSTRAP_DOGRULANDI` ve `TABLOLAR_DOGRULANDI` beklenir. Bootstrap doğrulaması
byte düzeyindedir ve numpy 1.26.4'e bağlıdır.

Figürler, doğrulanmış GPU ortamını değiştirmemek için ayrı bir venv'de çizilir:

```powershell
py -3.10 -m venv ..\.venv-figures
& '..\.venv-figures\Scripts\python.exe' -m pip install -r '.\research\requirements-figures.txt'
& '..\.venv-figures\Scripts\python.exe' -B '.\research\make_selfmm_figures.py' --replace
```

PNG baytları yazı tipi derlemesine göre değişebilir; girdilerin özetleri
`figures/figures_manifest.json` içindedir.

## GPU ortamını yeni bilgisayarda hazırlama

Bu bölüm gerçek eğitim/değerlendirmeyi planlayan okuyucu içindir. Yukarıdaki
paket kontrolü için bu kurulum gerekmez. Çalışan proje venv’ini değiştirmeden,
ayrı boş bir proje ana klasöründe uygulanmalıdır.

```powershell
# Boş bir proje ana klasöründeyken; mevcut MMSA klasörü kullanılıyorsa klonlama adımını geç.
git clone --branch yzm5110-selfmm https://github.com/ilaydagencer/yzm5110-robust-selfmm.git MMSA
py -3.10 -m venv .venv
Set-Location '.\MMSA'

& '..\.venv\Scripts\python.exe' -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
& '..\.venv\Scripts\python.exe' -m pip install -r '.\research\requirements-research.txt'
& '..\.venv\Scripts\python.exe' -m pip install --no-deps -e .
& '..\.venv\Scripts\python.exe' -m pip check
& '..\.venv\Scripts\python.exe' -B -c "import sentencepiece; import torch; import MMSA; print(torch.__version__); print(torch.cuda.is_available()); print(MMSA.__file__)"
```

PyTorch 2.3.1 CUDA 12.1 wheel komutu
[resmî eski sürüm yönergesine](https://pytorch.org/get-started/previous-versions/#v231)
dayanır. CUDA görünürlüğü uygun NVIDIA sürücüsüne bağlıdır. torchvision ve
torchaudio bu hazır özellik deneyinde kullanılmadığından eklenmemiştir.
`MMSA.__file__` bu klonun `src/MMSA` klasörünü göstermelidir.
Windows uyumluluğu için **sentencepiece, torch/MMSA’dan önce import edilir**.

`requirements-research.txt`, çalışan Windows Python 3.10.5 ortamının gözlenen
paket sürümlerini sabitler; torch ve yerel MMSA yukarıdaki ayrı adımlarla kurulur.
Tam ortam bilgisi [environment-observed.txt](environment-observed.txt) içindedir.
Bu liste hash kilitli bir lockfile değildir. Aynı bilgisayarda yeni venv’e
kurulum, paket izolasyonu ve seed 1111 GPU tekrarı doğrulandı. Kullanılan
kurulum komutları, sürümler ve sınırlar [temiz ortam kaydındadır](CLEAN_ENVIRONMENT.md).
Başka bilgisayarda kurulum henüz doğrulanmadı.

## Veri ve konfigürasyon

Klasör düzeni:

```text
proje_ana_klasoru/
  .venv/
  MMSA/                     # Bu depo
  data/MOSI/Processed/unaligned_50.pkl
  local_configs/            # Makineye özel; Git'e eklenmez
  runs/                     # Deneyler; Git'e topluca eklenmez
```

Hazır MOSI özelliklerine [MMSA veri erişim bölümündeki](https://github.com/thuiar/MMSA#2-datasets)
resmî bağlantılar üzerinden erişilir. Veri sağlayıcının erişim/kullanım
koşulları geçerlidir. Bu repo ham videoları veya `.pkl` dosyasını dağıtmaz.
Bu çalışma BERT girdili `unaligned_50.pkl` kullanır; benzer isimli başka
özellik dosyasıyla skorların aynı olması beklenmez. İndirdikten sonra:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath '..\data\MOSI\Processed\unaligned_50.pkl'
```

Beklenen SHA-256:
`78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524`.
İlk model kurulumu `bert-base-uncased` dosyalarını indirebilir. Kaynak benchmark’ı
offline çalışır ve önceden hazırlanmış BERT önbelleğini ister.

[Config örneği](configs/selfmm_mosi_batch4.example.json), kullanılan config’in
yalnız veri kökü `../data` yapılmış kopyasıdır; repo kökünden çalıştırma varsayılır.
Batch=4, BERT fine-tuning=True, early_stop=8, update_epochs=4 korunur.
Bu örnek eski deney kayıtlarının üzerine kopyalanmaz. Mevcut eğitim betikleri
`--config` seçeneği almaz; yerel config’i kaynak config’den üretir ve önceki
config ile karşılaştırır. Örnek JSON, deney ayarlarını kişisel yol olmadan
incelemek içindir; tam taşınabilir GPU başlatıcısı olduğu iddia edilmez.

## Gerçek deney zinciri ve mevcut betikler

Bu betikler önceki adımların çıktılarına bağlı bir deney arşivi oluşturur.
Checkpoint’ler, `local_configs`, ham tahmin CSV’leri ve orijinal manifestler
küçük pakete dahil değildir. Manifestlerde makineye özel mutlak yollar ve
SHA-256 korumaları vardır. `--repo-dir` tek başına taşınmış eski kayıtları
uyumlu hale getirmez; bu korumaları kapatmak yeniden üretim yöntemi değildir.

| Sıra | Betik | Ön koşul / ürettiği kanıt |
|---|---|---|
| 1 | run_selfmm_smoke.py | Veri ve kaynak config → seed 1111 baseline/config |
| 2 | run_selfmm_dropout.py | Baseline config/checkpoint → seed 1111 dropout |
| 3 | evaluate_selfmm_checkpoints.py | İki checkpoint → eski A/B K0–K3 test referansı |
| 4 | evaluate_selfmm_abcd_valid.py | İki checkpoint → A/B/C/D K0–K3 validation |
| 5 | check_selfmm_corruptions.py | Veri → train std ve valid bozulma protokolü |
| 6 | evaluate_selfmm_abcd_corrupt_valid.py | 4 ve 5 → seed 1111 K0–K7 validation |
| 7 | run_selfmm_seed.py ve evaluate_selfmm_abcd_seed_valid.py | Seed 1111 referansları → 1112–1115 eğitimleri/validation |
| 8 | analyze_selfmm_five_seeds.py | Beş tam validation kaydı → validation özetleri |
| 9 | evaluate_selfmm_abcd_test.py | Seed 1111 valid/protokol/eski test kayıtları → K0–K7 test |
| 10 | evaluate_selfmm_abcd_seed_test.py | Seed 1111 test, seçilen seed eğitim/valid kayıtları → 1112–1115 test |
| 11 | analyze_selfmm_test_five_seeds.py | Beş tam test kaydı → test özetleri |
| 12 | benchmark_selfmm_inference.py | Tamamlanmış seed 1111 test kayıtları → K0 kaynak ölçümü |

**Eski baseline betiğinin özel davranışı:** `run_selfmm_smoke.py --check`
yerel config dosyasını yazar; diğer yeni `--check` akışlarıyla aynı değildir.
Bu eski betikte checkpoint/çıktı üzerine yazmayı engelleyen yeni betiklerdeki
koruma da yoktur. Mevcut seed 1111 arşivinde tekrar çalıştırılmaz. Sıfırdan
tekrar için ayrı boş proje konumu ve aşağıdaki iki hedefin yokluğu gerekir:
`../local_configs/selfmm_mosi_batch4.json`, `../runs/selfmm_seed1111`.

Seed 1112–1115 eğitim betiği mevcut hedefte `--check` dahil durur.
Bu, tamamlanan çalışmayı bozmayı önleyen beklenen davranıştır. Bütün
deneyleri tekrar başlatan toplu bir komut bu README’de verilmemiştir.

Yerel tam arşiv mevcutsa, aşağıdaki kontroller yeni boş çıktı adlarıyla yapılabilir:

```powershell
& '..\.venv\Scripts\python.exe' -B '.\research\analyze_selfmm_five_seeds.py' `
    --output-dir '..\runs\selfmm_valid_recheck_v1' --check

& '..\.venv\Scripts\python.exe' -B '.\research\analyze_selfmm_test_five_seeds.py' `
    --output-dir '..\runs\selfmm_test_recheck_v1' --check

& '..\.venv\Scripts\python.exe' -B '.\research\benchmark_selfmm_inference.py' `
    --seed 1111 --output-dir '..\runs\selfmm_inference_repeat_v1' --check
```

İlk iki komuttan `--check` kaldırılırsa yeni özet dosyaları yazılır; model
çalıştırılmaz. Son komuttan kaldırılırsa gerçek GPU çıkarım ölçümü başlar.
Küçük paket bu yerel arşivin yerine geçmez. Tam arşivi olmayan okuyucu hızlı
paket kontrolünü kullanabilir. İlk eğitimleri ve arşiv zincirini yeni makinede
baştan oluşturmanın ayrıca doğrulanması teslim öncesi açık iştir.

`eval_robust.py`, `evaluate_consistency.py`, `evaluate_mask_only.py`,
`probe_consistency_fusion.py`, `inspect_*` ve `run_lmf_smoke.py` erken
keşif dosyalarıdır. Bu paketteki nihai A/B/C/D sonuç zincirinin giriş noktaları
olarak kullanılmaz; burada LMF/MulT yeniden eğitilmiş gibi gösterilmez.

## Yorum sınırları ve teslim durumu

MMSA eğitim sonunda otomatik test yapmıştır; daha önceki test incelemeleri de
vardır. Test, hiç görülmemiş bağımsız holdout diye sunulamaz. Son sabit protokol
karşılaştırmalarına göre yeni τ, dropout oranı veya iyi görünen seed seçilmedi.
Metin korunur; tek veri kümesi ve tek bozulma çekilişi vardır. Düşük bozulma
duyarlılığı tek başına güçlü çok modlu dayanıklılık kanıtı değildir.

Kaynak kodunda gradyan birikiminin son eksik grubu işlenmez; validation seçim
Loss’u eşit örnek ağırlıklı MAE ile aynı olmayabilir. Bunlar raporda açıklanan
miras alınmış sınırlılıklardır; sonuçlardan sonra eski kaynaklar değiştirilmedi.

Paket kontrolü gözlenen Python 3.10 ortamında, yalnız teslim dosyaları bulunan
ayrı bir klasöre taşınarak `-I -S -B` ile başarıyla çalıştırılmıştır. 11 özet
CSV yeniden yazılmış; kaynak dosyaların değişmediği kontrol edilmiştir. Sekiz
kontrol testi, bozuk özet/tekrar kayıt/son batch ve üzerine yazma durumlarını da sınar.
Bu, yeni GPU ortamı kurma veya gerçek bir uzak Git klonu testi değildir.
Doğrulama sonucu ve teslim commit’i ayrı
değerlendirilmelidir. `provenance.json` paketleme öncesi HEAD’i kaydeder;
son teslim commit/tag’i öğrenci tarafından Fork üzerinden sabitlenecektir.
Word raporunun son içerik/sayfa denetimi ve nihai
`Gencer_YZM5110_Research.pdf` teslimi ayrıca tamamlanacaktır.

Kod taslakları, kontroller, tablo üretimi ve açıklamalarda ChatGPT/Codex
desteği kullanıldı. Öğrenci GPU komutlarını çalıştırdı ve Git işlemlerini
kendisi yürütür. Öğrencinin yöntemleri ve kaynakları anlayıp açıklaması gerekir.

## Fork üzerinden eklenecek dosyalar

Bu teslim adımında `.gitattributes`, README, kontrol betiği, ortam listeleri, config örneği ve
`results_package_v1` seçilerek stage edilir. Veri, `.venv`, checkpoint,
kişisel `local_configs`, bütün `runs` klasörü ve eski mutlak yollu manifestler
eklenmez. Kaynak `src/MMSA/config/config_regression.json` içindeki kişisel
veri yolu değişikliği bu commit’e dahil edilmez. Word son aşamada güncellenecektir.

Önerilen başlık: `docs: add Self-MM research guide and verifiable result package`

Önerilen açıklama: Document methods, setup, data requirements and the frozen
experiment chain. Add portable quality/resource summaries and a standard-library
verifier that recalculates means, sample SDs and paired differences. Record
provenance and distinguish package checks from full GPU reproduction.

## Açık dosya yollarıyla gerçek model tekrarı: K0

Seed 1111 için temiz K0 A/B/C/D çıkarımına yeni giriş noktası eklendi:
[`evaluate_selfmm_portable_k0.py`](evaluate_selfmm_portable_k0.py).
[Adım adım yönerge](PORTABLE_MODEL.md), gerekli config ve bağımlılık manifestini,
veri/checkpoint/yerel BERT yollarını, çevrimdışı `--check` ve gerçek çalıştırmayı açıklar.
Bu giriş noktası eski `runs` manifestlerini gerektirmez; public sonuç paketindeki
16 K0 metriğiyle yeni tahminlerden hesaplanan metrikleri karşılaştırır.
Eski arşiv betiklerinin yukarıdaki bağımlılıkları değişmemiştir.

Mevcut ortamda ve ayrı bir kaynak kopyasında yazmasız `--check` doğrulandı;
10 CPU sözleşme testi geçti. 19 Ekim 2026 tarihinde kullanıcı mevcut ortamında
RTX 3050 Ti GPU ile seed 1111 K0 tekrarını tamamladı. Dört yöntemin toplam
16 referans metrik kontrolü geçti; en büyük mutlak fark 5.55e-16 idi.
Kaydedilen 1372 tahmin satırından metrikler ayrıca yeniden hesaplandı;
4 çıktı ve 103 girdi dosyasının SHA256 özetleri doğrulandı.
[Doğrulama kaydı](verification/portable_k0_seed1111_v1.json).
Bu giriş noktası K0 ile sınırlıdır; seed 1111 K1–K7 tekrarı aşağıdaki ayrı
giriş noktasında doğrulandı. Seed 1112–1115'in mevcut ortamdaki taşınabilir
tekrarları da aşağıdaki ortak girişte tamamlandı. Daha sonra yeni venv ve ayrı
kaynak kopyasıyla seed 1111 K0–K7 tekrarı doğrulandı; bu gelişme K0 yardımcısının
kapsamını genişletmez. Eğitimden baştan tekrar henüz doğrulanmadı.

## Açık dosya yollarıyla K0–K7 tekrarı

[`evaluate_selfmm_portable_k0_k7.py`](evaluate_selfmm_portable_k0_k7.py), seed 1111
checkpoint'lerini bütün K0–K7 test koşullarında çalıştırmak için hazırlandı.
[Adım adım yönerge](PORTABLE_K0_K7.md), train-only gürültü istatistiklerinin
yeniden hesabını, 128 metrik ve 16 girdi özeti kontrolünü açıklar.
K0 yardımcısı ve eski deneyler değiştirilmez; eski çalışma manifestleri okunmaz.

Hazırlıkta 10 CPU kontrol testi geçti. Ayrı gerçek-veri CPU denetiminde iki train
istatistiği ve sekiz koşulun girdi özetleri referansla eşleşti. Model/veri yüklemesi
engellenmiş ayrı kaynak kopyasında yazmasız --check doğrulandı.

19 Ekim 2026 tarihinde kullanıcı mevcut RTX 3050 Ti ortamında gerçek K0–K7
tekrarını tamamladı: 32 metrik satırı, 128/128 referans metrik kontrolü,
16/16 bozulmuş girdi özeti ve 2/2 train istatistiği kontrolü doğrulandı.
En büyük referans metrik farkı 5.55e-16, train istatistiği farkı sıfırdı.
10976 tahmin satırından metrikler ayrıca yeniden hesaplandı; 8 çıktı ve
106 kayıtlı girdi dosyasının SHA256 özetleri denetim sırasında eşleşti.
[Doğrulama kaydı](verification/portable_k0_k7_seed1111_v1.json).

Bu tekrar önceki sonuçların mevcut ortamda eski çalışma manifestlerine
bağlanmadan üretilebildiğini gösterir. Yeni eğitim veya bağımsız deney değildir.
Seed 1112–1115'in mevcut ortamda K0–K7 tekrarları aşağıdaki giriş noktasında
tamamlandı. Sonraki ayrı kaynak kopyası ve yeni venv denetimi seed 1111 için
gerçek GPU sonucuyla tamamlandı; [kanıt ve kapsam](CLEAN_ENVIRONMENT.md).


## Eğitim seed'ini seçerek taşınabilir test tekrarı: 1112–1115

[`evaluate_selfmm_portable_seed_test.py`](evaluate_selfmm_portable_seed_test.py),
`--seed 1112`, `--seed 1113`, `--seed 1114` veya `--seed 1115` ile kayıtlı baseline/dropout checkpoint'lerini
K0–K7 test koşullarında değerlendirir. [Adım adım yönerge](PORTABLE_SEED_TEST.md).
Seed 1111 kendi giriş noktalarını kullanır; diğer dört seed bu ortak betiktedir.

`configs/selfmm_portable_seed_dependencies.json`, eğitim/test kayıtlarıyla
karşılaştırılmış checkpoint kimliklerini ve ortak bağımlılıkları saklar.
Gerçek çalışmada eski run manifestleri ve kişisel config dosyaları okunmaz.
Bozulma seed'i 1111, batch 4 ve τ=1 sabittir. Train-only std yeniden hesaplanır.
128 metrik seçilen eğitim seed'inin public paket satırlarıyla; 16 bozulmuş
girdi özeti ortak seed 1111 referansıyla karşılaştırılır.

19 Ekim 2026 tarihinde kullanıcı mevcut RTX 3050 Ti ortamında seed 1112
K0–K7 tekrarını tamamladı: 128/128 metrik, 16/16 girdi ve 2/2 train istatistiği
kontrolü geçti. En büyük referans metrik farkı 8.88e-16, train istatistiği farkı
sıfırdı. 10976 tahmin satırından metrikler ayrıca yeniden hesaplandı; 8 çıktı ve
106 kayıtlı girdinin SHA256 özetleri güncelleme öncesi denetimde eşleşti.
[Seed 1112 doğrulama kaydı](verification/portable_k0_k7_seed1112_v1.json).

19 Ekim 2026 tarihinde seed 1113 K0–K7 tekrarı da mevcut ortamda tamamlandı.
128/128 metrik, 16/16 girdi ve 2/2 train istatistiği kontrolü geçti. En büyük
referans metrik farkı 1.23e-15 altında; MAE ve accuracy farkları sıfırdır.
10976 tahmin satırından metrikler ayrıca yeniden hesaplandı. 8 çıktı ve
106 kayıtlı girdinin özeti bu güncelleme öncesi denetimde eşleşti.
[Seed 1113 doğrulama kaydı](verification/portable_k0_k7_seed1113_v1.json).

19 Ekim 2026 tarihinde seed 1114 K0–K7 tekrarı da mevcut ortamda tamamlandı.
128/128 metrik, 16/16 girdi ve 2/2 train istatistiği kontrolü geçti. En büyük
referans metrik farkı 1e-15 altında; MAE ve accuracy farkları sıfırdır.
10976 tahmin satırından metrikler ayrıca yeniden hesaplandı. 8 çıktı ve
106 kayıtlı girdinin özeti bu güncelleme öncesi denetimde eşleşti.
[Seed 1114 doğrulama kaydı](verification/portable_k0_k7_seed1114_v1.json).

## Tamamlanan seed 1115 ve beş-seed taşınabilir tekrarları — 19 Ekim 2026

Seed 1115'te 128/128 metrik, 16/16 bozulmuş girdi ve 2/2 train istatistiği
kontrolü geçti. En büyük referans farkı 1.11023e-15 (Corr), MAE ve accuracy
farkları sıfırdır. Train istatistiklerinde ve uniform ağırlık identity
kontrolünde fark sıfırdır. 10976 tahmin satırından bağımsız metrik hesabı
doğrulandı; float32 MAE ile en büyük fark 1.05e-7 altında (tolerans 1e-6).
8 çıktı ve 106 girdinin dosya özeti denetim anında eşleşti.
[Seed 1115 kaydı](verification/portable_k0_k7_seed1115_v1.json).

| Seed 1115, K0 | Non0 accuracy (%) | MAE | Corr |
|---|---:|---:|---:|
| A | 84.45 | 0.7149 | 0.7843 |
| B | 83.08 | 0.7463 | 0.7692 |
| C | 84.45 | 0.7965 | 0.7681 |
| D | 82.93 | 0.8539 | 0.7512 |

K0 MAE farkları yaklaşık B−A +0.0314, C−A +0.0816 ve D−B +0.1076'dır.
K0–K7'nin tümünde B'nin MAE'si A'dan, C'nin A'dan ve D'nin B'den yüksektir.
C, K0 accuracy'sini korurken duygu şiddeti hatasını artırır. Bu önceki test
bulgusunun yeniden üretimidir; beş-seed istatistiğine yeni gözlem eklenmez.
200.42 saniye toplam tekrar süresidir; çıkarım benchmark'ı değildir.

Seed 1111–1115'in tamamı mevcut ortamda tekrarlandı. Toplam 640 metrik,
80 girdi ve 10 train istatistiği kontrolü başarılıdır. Beş çalışmanın manifest
kimlikleri ve toplam 40 çıktı dosyası yeniden denetlendi. En büyük referans
metrik farkı 1.23e-15 altındadır. Bu toplamlar bağımsız deney sayısı değildir.
[Beş-seed doğrulama özeti](verification/portable_five_seeds_v1.json).

Bu adım değerlendirme betiklerini, kayıtlı checkpoint kimliklerini veya sabit
protokolü değiştirmez. Sonraki temiz ortam aşaması seed 1111 için tamamlandı:
[sonuçlar ve yeniden kurulum yönergesi](CLEAN_ENVIRONMENT.md).
Eski doğrulama JSON'ları kendi tarihsel kapsamlarıyla korunur; yeni kanıt ayrı kayıttadır.


Temiz ortam hazırlığı için iki stdlib tabanlı araç eklendi:
`prepare_selfmm_clean_copy.py`, seçili küçük kaynakları ayrı klasöre kopyalar;
`check_selfmm_clean_environment.py`, yeni venv'in izolasyonunu, sürümlerini,
paket/kaynak yollarını, `pip check` sonucunu ve CUDA görünürlüğünü denetler.
Kullanıcı yeni ortamın paket kurulumunu ve seed 1111 gerçek GPU tekrarını
tamamladı. 51 paketin ortam içindeki yolları, 125 kopya kaynak ve dört kurulum
logunun dosya özetleri denetlendi. Bu araçların --check sonucu ile gerçek GPU
sonucu ayrı kanıtlardır; ikisi yeni doğrulama kaydında birlikte izlenebilir.
