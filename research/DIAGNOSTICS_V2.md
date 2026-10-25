# Ağırlıklandırma tanı deneyleri (protokol v2): çalıştırma yönergesi

Ne ve neden: [protokol v2](selfmm_diagnostics_protocol_v2.md).
Betik: [`diagnose_selfmm_weighting.py`](diagnose_selfmm_weighting.py).

Bu adım yeni eğitim yapmaz. Mevcut 10 checkpoint'i (5 seed × baseline/dropout)
önce **validation**'da, K0–K10 koşulları ve protokoldeki ağırlık varyantlarıyla
çalıştırır. Test yalnız validation özeti commit edildikten sonra, bir kez çalışır.

## Sıra

1. Protokol ve betik commit edilir (GPU koşusundan **önce**).
2. Validation koşuları (bu belge, bölüm 2–3).
3. Beş seed'in validation özeti, hipotez kararları ve τ\* commit edilir (ayrı adım).
4. Test koşuları `--selection` ile bir kez yapılır (ayrı adım).

## 1. GPU gerektirmeyen öz-denetim

Depo kökünde (`MMSA` klasörü):

```powershell
& '..\.venv\Scripts\python.exe' -B '.\research\diagnose_selfmm_weighting.py' --self-test
```

Beklenen son satırlar `TANI_OZTESTLERI_BASARILI` ve "Ozdenetim dosya yazmadi".
Bu test sahte (dummy) bir modelle bütün işlem hattını CPU'da çalıştırır; gerçek
veri, checkpoint veya BERT açmaz.

## 2. Beş seed için --check

`--check`, veri ve checkpoint'leri bayt olarak okuyup SHA256 özetlerini, kaynak
kodunu, config'i ve sonuç paketini doğrular. Model veya veri kümesi kurulmaz,
dosya yazılmaz.

PowerShell'i depo kökünde (`MMSA` klasörü) aç; komutlardaki `..\` yolları
bu klasöre göredir.

```powershell
$bertDir = Join-Path $env:USERPROFILE '.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594'
function Invoke-Diagnostics([int]$Seed, [string]$Split, [string[]]$Extra) {
    $baseline = if ($Seed -eq 1111) { '..\runs\selfmm_seed1111' } else { "..\runs\selfmm_baseline_seed$Seed" }
    & '..\.venv\Scripts\python.exe' -B '.\research\diagnose_selfmm_weighting.py' `
        --seed $Seed --split $Split `
        --data-file '..\data\MOSI\Processed\unaligned_50.pkl' `
        --baseline-checkpoint "$baseline\saved_models\self_mm-mosi.pth" `
        --dropout-checkpoint "..\runs\selfmm_dropout_a20_v20_seed$Seed\saved_models\self_mm-mosi.pth" `
        --bert-dir $bertDir `
        --output-dir "..\runs\selfmm_diagnostics_v2_${Split}_seed$Seed" @Extra
}
foreach ($seed in 1111, 1112, 1113, 1114, 1115) {
    Invoke-Diagnostics $seed 'valid' @('--check')
    if ($LASTEXITCODE -ne 0) { break }
}
```

Her seed için `KONTROL_BASARILI: seed…/valid` beklenir.

## 3. Validation koşuları (GPU)

Önce yalnız seed 1111'i çalıştırıp süreyi ve referans kontrolünü gör:

```powershell
Invoke-Diagnostics 1111 'valid' @()
```

Her koşul için `baseline/K… : plain MAE=…; tau1 MAE=…; … sn` satırı yazılır.
Sonunda `TANI_V2_TAMAMLANDI: seed1111/valid; 128/128 referans kontrolu gecti`
beklenir. Bu satır, yeni betiğin K0–K7'de v1 validation A/B/C/D sonuçlarını
1e-5 toleransla yeniden ürettiğini gösterir; yeni sayılara ancak bundan sonra
güvenilir.

Sonra kalan dört seed:

```powershell
foreach ($seed in 1112, 1113, 1114, 1115) {
    Invoke-Diagnostics $seed 'valid' @()
    if ($LASTEXITCODE -ne 0) { break }
}
```

Bir koşu hata verirse çıktı klasöründeki `manifest.json` içindeki `status` ve
`error` alanlarına bak. Toleransı veya protokolü değiştirme; hatayı paylaş.
Var olan çıktı klasörünün üzerine yazılmaz; tekrar denemek için klasörü sil
veya yeni ad ver ve bunu not et.

## Çıktılar (`..\runs\selfmm_diagnostics_v2_valid_seed<seed>\`)

| Dosya | İçerik |
|---|---|
| `metrics.csv` | 294 satır: 2 checkpoint × 11 koşul × varyant (+ tek modlu başlıklar) |
| `reference_checks.csv` | 128 satır: K0–K7 plain/τ=1 ile paket A/B/C/D karşılaştırması |
| `predictions.csv` | 5038 satır: örnek başına bütün varyant tahminleri, tek modlu tahminler, d ve ağırlıklar |
| `weight_summary.csv` | Varyant başına ağırlık ortalaması/SS/min/maks |
| `input_summary.csv` | Koşul başına maskelenme oranları ve girdi özeti |
| `controls.csv`, `manifest.json`, `train_*` | Kontroller, sürümler, girdi özetleri |

`runs` klasörü Git'e eklenmez. Beş seed'in küçük özeti ayrı adımda depoya girer.

## 4. Beş seed validation özeti ve τ seçimi (GPU yok)

[`summarize_selfmm_diagnostics.py`](summarize_selfmm_diagnostics.py) yalnız Python
standart kütüphanesini kullanır. Beş koşunun tamamlandığını, aynı protokol/betik
sürümüyle çalıştığını, aynı bozulmuş girdileri aldığını ve çıktı özetlerinin
değişmediğini denetler; sonra protokolün 5. ve 6. bölümündeki kuralları uygular.

```powershell
& '..\.venv\Scripts\python.exe' -I -S -B '.\research\summarize_selfmm_diagnostics.py' `
    --runs-dir '..\runs' --split valid --output-dir '.\research\diagnostics_v2\validation'
```

Paket ([diagnostics_v2/validation](diagnostics_v2/validation)): seed başına
metrikler, beş seed ortalama/SS, varyant − plain eşleştirilmiş farklar, hipotez
kararları (`hypotheses.csv`), ağırlık ve tek modlu başlık yayılımı özetleri ve
test için `selection.json`. Örnek düzeyindeki tahminler `runs` içinde kalır.

Paketi GPU, veri veya checkpoint olmadan yeniden doğrulamak için:

```powershell
& '..\.venv\Scripts\python.exe' -I -S -B '.\research\summarize_selfmm_diagnostics.py' `
    --verify '.\research\diagnostics_v2\validation'
```

`OZET_DOGRULANDI` beklenir. Bu komut bütün özet tablolarını, hipotez kararlarını
ve τ\* değerini seed başına metriklerden yeniden hesaplar ve dosya özetlerini denetler.

### Validation sonucu (25 Ekim 2026, beş seed, baseline checkpoint)

| Kod | Karar | Ana sayı |
|---|---|---|
| H1 | Desteklendi | K0–K7 ortalama MAE τ=0,25'te 1,230 → τ=16'da 0,722; K0'da τ16 − plain = +0,003 |
| H2 | Desteklendi | K0'da metin sabit varyant − plain = +0,0005; τ=1 − plain = +0,150; 5/5 seed |
| H3 | Desteklenmedi | K0'da τ=1 − karıştırılmış = +0,040; τ=1 hiçbir seed'de karıştırılmıştan iyi değil |
| H4 | Desteklendi | Oracle − plain: K1 +0,0001, K2 −0,0003, K3 +0,0002 |
| H5a | Desteklendi | plain MAE K0 0,717 → K10 1,464 (+0,748); Non0 accuracy %84,9 → %45,6 |
| H5b | Desteklenmedi | τ=1 − plain: K9 +0,036 (0/5 seed), K10 +0,001 (3/5 seed) |

Altı kararın altısı da önceden yazılan beklentiyle aynıdır. Seçilen τ\* = 16, yani
ızgaranın en büyük değeri: validation'da en iyi ağırlıklandırma, ağırlıksız
füzyona en yakın olanıdır. Kararlar betimseldir; p-değeri yoktur.

## 5. Test koşuları (bir kez, GPU)

Yalnız validation paketi ve `selection.json` commit edilip push edildikten sonra.
Bölüm 2'deki fonksiyon tanımlıyken:

```powershell
foreach ($seed in 1111, 1112, 1113, 1114, 1115) {
    Invoke-Diagnostics $seed 'test' @('--selection', '.\research\diagnostics_v2\validation\selection.json')
    if ($LASTEXITCODE -ne 0) { break }
}
```

Test 686 örnek içerdiği için seed başına yaklaşık 15–20 dakika sürer. Her koşu
K0–K7'de v1 test A/B/C/D sonuçlarıyla 128 metrik ve sekiz girdi özeti kontrolü yapar.
Sonra özet:

```powershell
& '..\.venv\Scripts\python.exe' -I -S -B '.\research\summarize_selfmm_diagnostics.py' `
    --runs-dir '..\runs' --split test `
    --selection '.\research\diagnostics_v2\validation\selection.json' `
    --output-dir '.\research\diagnostics_v2\evaluation_test'
```

Testteki hipotez satırları yeni karar değildir; validation kararlarının testte
tekrar edip etmediğini betimler.

### Test sonucu (25 Ekim 2026, beş seed, baseline checkpoint, τ\* = 16)

Beş test koşusu validation paketi commit edildikten sonra, commit edilmiş
`selection.json` ile birer kez çalıştı. Her koşuda 128/128 metrik ve 16/16 girdi
özeti kontrolü geçti. [Test paketi](diagnostics_v2/evaluation_test).

| Kod | Validation | Test | Test ana sayısı |
|---|---|---|---|
| H1 | Desteklendi | Tekrarlandı | K0–K7 ortalama MAE τ=0,25'te 1,205 → τ=16'da 0,724 |
| H2 | Desteklendi | Tekrarlandı | K0'da metin sabit − plain = −0,0004; τ=1 − plain = +0,118 |
| H3 | Desteklenmedi | Tekrarlandı | K0'da τ=1 − karıştırılmış = +0,040; 0/5 seed |
| H4 | Desteklendi | Tekrarlandı | Oracle − plain: K1 −0,0007, K2 −0,0002, K3 −0,0009 |
| H5a | Desteklendi | Tekrarlandı | plain MAE K0 0,724 → K10 1,399; Non0 accuracy %83,9 → %55,1 |
| H5b | Desteklenmedi | Tekrarlandı | τ=1 − plain: K9 +0,029 (0/5), K10 −0,0004 (2/5) |

- τ\* = 16 testte plain füzyonla aynıdır (K0 farkı +0,0002); seçilen "en iyi"
  ağırlıklandırma pratikte ağırlıklandırmama demektir.
- Tutarsız ayrıntı: K10'da metne sıfır veren oracle, validation'da plain'den
  iyidir (1,430 ve 1,464), testte kötüdür (1,422 ve 1,399). Bu karşılaştırma için
  önceden karar ölçütü yoktu; yön bölmeler arasında tutarlı değildir.
- τ=1'de ortalama metin ağırlığı K0'dan K10'a beş seed'in beşinde artar
  (testte 0,71–0,81 → 0,91–1,18). Metin bozuldukça kural metne daha çok ağırlık verir.
