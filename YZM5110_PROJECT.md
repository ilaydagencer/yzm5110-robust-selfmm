# YZM5110 — Self-MM ile eksik ve gürültülü modalitelerde duygu analizi

İlayda Gençer · YZM5110 araştırma projesi · [THUIAR/MMSA](https://github.com/thuiar/MMSA) fork'u

Bu dosya kısa özettir. Kurulum, veri erişimi, komutlar ve bütün tablolar
[research/README.md](research/README.md) içindedir.

## Araştırma sorusu

Self-MM'in öz-denetimli tek modlu tahminleri (metin, ses, görüntü) arasındaki
uyum, ses veya görüntü eksik ya da bozuk olduğunda füzyon katkılarını belirlemek
için kullanılabilir mi? Bu kural temiz girdide başarıyı korurken bozuk girdide
hatayı azaltıyor mu? Eğitimde ses/görüntü dropout'u bu davranışı değiştiriyor mu?

## Ana makale

Yu, W., Xu, H., Yuan, Z. ve Wu, J. (2021). *Learning Modality-Specific
Representations with Self-Supervised Multi-Task Learning for Multimodal
Sentiment Analysis.* AAAI. [arXiv:2102.04830](https://arxiv.org/abs/2102.04830)

## Yöntemler

| Yöntem | Eğitim | Çıkarım |
|---|---|---|
| A | Standart Self-MM | Özgün füzyon |
| B | Yalnız eğitimde örnek başına %20 ses / %20 görüntü giriş dropout'u | Özgün füzyon |
| C | A ile aynı checkpoint | Tek modlu tahmin uyumuna göre ağırlıklı füzyon (τ=1), iki forward |
| D | B ile aynı checkpoint | C ile aynı ağırlıklandırma |

Ağırlık kuralı: d_m = \|y_m − ortalama(y_diğer)\|, w = 3 × softmax(−d/τ).
Ağırlıklar ilk füzyon katmanından önce modalite temsillerine uygulanır; yeni
öğrenilen parametre eklenmez.

## Veri ve koşullar

- CMU-MOSI, MMSA hazır özellikleri: `MOSI/Processed/unaligned_50.pkl`
  (SHA-256 `78e0f8b5…2e02524`). Train/valid/test 1284/229/686 örnek.
  Veri bu depoda dağıtılmaz; erişim [MMSA veri bölümünden](https://github.com/thuiar/MMSA#2-datasets).
- Koşullar: K0 temiz; K1 ses, K2 görüntü, K3 ikisi birden sıfır; K4/K5/K6
  geçerli zaman adımlarında %30/%50/%70 maskeleme; K7 train std'sinin 0,5 katı
  Gauss gürültüsü. Metin bütün koşullarda korunur.
- Eğitim seed'leri 1111–1115, bozulma seed'i 1111.

## Ölçülen sonuçlar (beş seed, test, ortalama ± örneklem SS)

| K0 yöntemi | MAE ↓ |
|---|---:|
| A | 0,7240 ± 0,0146 |
| B | 0,7574 ± 0,0244 |
| C | 0,8424 ± 0,0412 |
| D | 0,9163 ± 0,0747 |

- τ=1 ağırlıklandırması her seed'de ve K0–K7'nin her koşulunda MAE'yi artırdı.
- Dropout'un validation'daki küçük MAE kazancı (B−A = −0,0119 ± 0,0281) testte korunmadı.
- A'nın test MAE'si ses ve görüntü birlikte sıfırlandığında da neredeyse aynı kaldı
  (K0 0,7240, K3 0,7235).
- İki forward geçişi nedeniyle örnek başına çıkarım süresi oranı C/A 2,05 ve
  D/B 2,04 oldu; tepe allocated bellek farkı yaklaşık 6,5 KiB (seed 1111, K0, batch 4).

Kaynak dosyalar: [test özeti](research/results_package_v1/evaluation_test/summary_metrics.csv),
[validation özeti](research/results_package_v1/validation/summary_metrics.csv),
[çıkarım maliyeti](research/results_package_v1/inference/paired_summary.csv).
Makalede yayımlanan skorlar ile bu ölçümler raporda ayrı tablolarda verilir.

## Ortam

Windows 11 · NVIDIA RTX 3050 Ti Laptop GPU (4 GB) · Python 3.10.5 ·
PyTorch 2.3.1 + CUDA 12.1 · MMSA 2.2.1. Paket sürümleri:
[requirements-research.txt](research/requirements-research.txt),
[environment-observed.txt](research/environment-observed.txt).

## Durum (25 Ekim 2026)

- Tamamlandı: beş seed A/B eğitimi; A/B/C/D K0–K7 validation ve test
  değerlendirmesi; çıkarım maliyeti ölçümü; beş seed'in checkpoint'ten gerçek
  GPU tekrarı; seed 1111'in yeni venv'de tekrarı; ağırlıklandırmanın neden
  başarısız olduğunu ayıran, önceden kaydedilmiş tanı deneyleri (protokol v2,
  validation ve tek test koşusu).
- Açık: bootstrap aralıkları, figür/tablo betikleri, son rapor PDF'i, sunum ve
  teslim tag'i.

## Tanı deneylerinin sonucu (protokol v2)

Önceden yazılan altı kararın altısı validation'da beklentiyle aynı çıktı ve
testte tekrarlandı. Zarar, baskın metin temsilinin ölçeklenmesinden gelir;
uyum sinyali rastgele karıştırılmış ağırlıklardan iyi değildir. Ses ve görüntü
başlıklarının tahminleri örnekler arasında çok az değişir (K0 test SS'si ses
0,001–0,17, görüntü 0,004–0,44; metin 0,64–1,00), bu yüzden kural metni zayıf
tahminlerle karşılaştırır.
Ses/görüntü bozulmasında oracle bile iyileşme sağlamaz; metin bozulduğunda kural
metne daha çok ağırlık verir. Ayrıntı: [DIAGNOSTICS_V2.md](research/DIAGNOSTICS_V2.md).

Kod ve metin hazırlığında yapay zekâ desteği kullanılmıştır; araçlar ve
katkıları raporun ekinde açıklanır.
