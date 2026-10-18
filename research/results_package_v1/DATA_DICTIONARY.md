# Küçük sonuç paketinin veri sözlüğü

Bu paket kaydedilmiş deneylerin özet hesabını incelemek içindir. CSV sayıları
orijinal analizlerden değiştirilmeden kopyalanmıştır; sentetik sonuç içermez.
Veri kümesi, örnek transkripti, tahmin/etiket listesi ve checkpoint içermez.

## Kalite tabloları

`validation/` 229 örneği (Non0=216), `evaluation_test/` 686 örneği (Non0=656) kapsar.
Her iki klasörde aynı dosyalar vardır:

| Dosya | Satır | Anlam |
|---|---:|---|
| per_seed_metrics.csv | 160 | Beş seed × dört yöntem × sekiz koşul; özet hesabının girdisi |
| summary_metrics.csv | 32 | Yöntem/koşul bazında beş seed ortalaması ve örneklem SS |
| paired_differences_by_seed.csv | 160 | Her seed içinde B−A, C−A, D−B, D−A farkları |
| paired_summary.csv | 32 | Beş eşleştirilmiş farkın ortalaması, SS ve iyileşen/eşit/kötüleşen seed sayısı |
| degradation_summary.csv | 32 | Aynı seed ve yöntemde koşul eksi K0; sonra beş seed özeti |

MAE düşük, Non0 accuracy/F1 ve Corr yüksek olduğunda iyidir. Accuracy/F1 CSV’de
0–1 ölçeğindedir; accuracy yüzde için 100 ile çarpılır. `Non0_acc_pp_delta`
zaten yüzde puandır, tekrar 100 ile çarpılmaz. `Non0_acc_2_delta` ise oran
farkıdır. MAE gerçek etiket ölçeğindedir. Corr Pearson korelasyonudur.
`_sd` örneklem standart sapmasıdır, ddof=1; güven aralığı veya standart hata
değildir. Pozitif MAE farkı kötüleşmedir. Eşitlik toleransı 1e-12’dir.

## Kaynak ölçümü

`inference/` yalnız seed 1111, K0, batch 4 ve altı zamanlama tekrarını içerir.

| Dosya | Satır | Anlam |
|---|---:|---|
| batch_timings.csv | 4128 | 4 yöntem × 6 tur × 172 batch; son batch iki örnek |
| rounds.csv | 24 | Tur süreleri, işleme hızı ve ölçülen bellek tepe değerleri |
| summary.csv | 4 | Altı tur üzerinde yöntem özeti |
| paired_rounds.csv | 12 | Aynı tur içinde C/A ve D/B kaynak karşılaştırmaları |
| paired_summary.csv | 2 | Eşleştirilmiş kaynak karşılaştırmalarının ortalama ve SS’si |
| correctness_checks.csv | 4 | Önceki K0 tahminlerine karşı kaydedilmiş doğrulama sonucu |

`ms_per_sample` toplam zamanlanan batch süresi / 686’dır; batch=1 gecikmesi
değildir. `samples_per_second` yalnız zamanlanan işlem kapsamına dayanır.
Yükleme, aktarım, ısınma ve doğruluk kontrolleri süreye dahil değildir.
`full_batch_ms_p50/p95` yalnız dört örnekli batch’lerden lineer enterpolasyonla
hesaplanır. C/D gerçek iki forward ve ağırlık/hook işlemlerini içerir.
`peak_allocated_mib_max` altı turdaki en yüksek PyTorch canlı tahsis tepesidir;
`peak_reserved_mib_max` allocator önbelleğini de kapsar. Toplam sistem GPU
belleği değildir. MiB=2^20 bayt; KiB=2^10 bayt. Bellek girdileri ve ara
tensörleri kapsar. Aynı anda tek checkpoint GPU’dadır.

## Kanıt ve sınırlar

`manifest.json` dağıtılan 19 dosyanın boyut/satır/SHA-256 listesidir.
`provenance.json` kaynak CSV ve eski manifestlerin özetlerini, repo göreli kod
yollarını ve paketleme anındaki Git HEAD’i içerir. Bu HEAD son teslim commit’i
değildir. Kişisel mutlak dosya yolları içeren eski manifestler kopyalanmamıştır.
Hash kontrolü dosyaların eşleştiğini gösterir; kriptografik imza veya deneyin
bağımsız doğruluğunun kanıtı değildir. Paket ve manifest birlikte değiştirilirse
hash denetimi tek başına bunu saptayamaz; Git geçmişi ayrıca kullanılmalıdır.

`verify_results_package.py` kalite istatistiklerini seed metriklerinden,
zaman istatistiklerini batch kayıtlarından yeniden hesaplar. Bellek tepe
ölçümlerini yeniden ölçmez; kayıtlı tur değerlerinin özetini denetler. Örnek
tahminlerinden MAE/F1 hesaplamaz, veri kümesini/modeli yüklemez, eğitim yapmaz.
Bu denetim, bütün GPU deneylerinin temiz ortamdan yeniden üretimi yerine geçmez.
