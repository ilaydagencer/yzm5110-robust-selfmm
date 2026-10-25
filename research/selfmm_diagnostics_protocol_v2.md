# Self-MM ağırlıklandırma tanı protokolü sürüm 2

Bu protokol 25 Ekim 2026'da, tanı deneylerinin hiçbir sonucu görülmeden önce
yazılmıştır. Belge ve [`diagnose_selfmm_weighting.py`](diagnose_selfmm_weighting.py)
GPU koşusundan önce commit edilir; Git tarihi sıralamanın kanıtıdır. Sonuçlar
görüldükten sonra bu belge değiştirilmez. Gerekirse yeni bir sürüm açılır ve fark
raporda sapma olarak yazılır.

## 1. Neden bu deneyler?

Sürüm 1'in sabit sonuçları (beş seed, validation ve test):

- τ=1 uyum ağırlıklandırması (C/D) K0–K7'nin her koşulunda ve her seed'de
  MAE'yi artırdı. Temiz validation'da C−A = +0,1497 ± 0,0414.
- A'nın test MAE'si ses ve görüntü birlikte sıfırlandığında da neredeyse
  değişmedi (K0 0,7240; K3 0,7235).

Bu iki gözlem, olumsuz sonucun nedenini açıklamaz. Bu protokol, olası nedenleri
birbirinden ayırmak için yeni eğitim yapmadan, mevcut 10 checkpoint üzerinde
yalnız çıkarım yapan kontrollü deneyler tanımlar.

**Önceden görülenler:** v1 validation ve test sonuçları görülmüştür. H4 ve H5a
beklentileri kısmen bu gözlemlere dayanır; bağımsız bir keşif gibi sunulmaz.
K8–K10 metin koşulları ve bütün yeni ağırlık varyantları daha önce hiçbir
bölmede çalıştırılmamıştır.

## 2. Değişmeyenler

- Checkpoint'ler: eğitim seed'leri 1111–1115, baseline ve dropout (toplam 10).
  Kimlikler mevcut kayıtlardan okunur:
  `configs/selfmm_portable_k0_dependencies.json` (1111) ve
  `configs/selfmm_portable_seed_dependencies.json` (1112–1115).
- Batch 4, `shuffle=False`, FP32, `eval()` ve `inference_mode()`.
- K0–K7 ses/görüntü bozulmaları v1 ile aynı modül ve bozulma seed'i 1111 ile
  üretilir. Gürültü ölçeği yalnız train verisinden yeniden hesaplanır.
- Ağırlıklar, v1'deki gibi ilk füzyon katmanının girdisindeki metin (768),
  ses (16) ve görüntü (32) temsil bloklarına uygulanır. Tek modlu başlıklar
  değişmez.

## 3. Yeni koşullar: metin bozulması

| Koşul | Metin | Ses ve görüntü |
|---|---|---|
| K8 | Geçerli token'ların her biri bağımsız %30 olasılıkla `[UNK]` | K0 (temiz) |
| K9 | Aynı işlem %60 | K0 |
| K10 | Bütün geçerli token'lar `[UNK]` | K0 |

- `[CLS]` (ilk) ve `[SEP]` (son geçerli) token'lar korunur; attention mask,
  segment id ve padding değişmez. Bu, MMSA'nın TFR-Net için kullandığı eksik
  metin kuralıyla (`[UNK]`=100, CLS/SEP korunur) aynıdır.
- Rastgelelik `selfmm-diagnostics-v2|1111|bölme|örnek|text|unk` anahtarından
  türetilir; model, batch ve çalıştırma sırası katılmaz. K8 maskesi K9'un,
  K9 maskesi K10'un alt kümesidir.
- Gerçekleşen maskelenme oranları ayrı raporlanır.

## 4. Ağırlık varyantları

Her checkpoint ve her koşul (K0–K10) için aynı girdilerle:

| Varyant | Ağırlık (metin, ses, görüntü) | Ayırdığı soru |
|---|---|---|
| `plain` | Ağırlık yok (A veya B) | Referans |
| `uniform` | (1, 1, 1) | Kimlik kontrolü; `plain` ile aynı çıkmalı |
| `agree_tau{τ}` | 3·softmax(−d/τ), τ ∈ {0,25; 0,5; 1; 2; 4; 8; 16} | Zarar ölçeklemenin miktarıyla mı artıyor? τ=1, C/D'dir |
| `agree_av_tau1` | Metin 1; ses ve görüntü 2·softmax(−[d_A, d_V]/1) | Zarar metin bloğunun ölçeklenmesinden mi geliyor? |
| `shuffled_tau1` | τ=1 ağırlık vektörleri örnekler arasında sabit bir permütasyonla yer değiştirir | Uyum sinyali örneğe özgü bilgi taşıyor mu? |
| `oracle` | Bilinen eksik modalite 0, diğerleri 1. Yalnız K1 (ses), K2 (görüntü), K3 (ikisi), K10 (metin) | Sinyal mükemmel olsaydı iyileşme alanı var mıydı? |

d_m = |ŷ_m − ortalama(ŷ_diğer)|, ŷ_m tek modlu başlıkların ilk geçiş tahminidir.
Permütasyon `selfmm-diagnostics-v2|1111|bölme|koşul|shuffle` anahtarından üretilir;
iki checkpoint ve beş seed aynı permütasyonu kullanır. Tek modlu başlıkların
(`head_T`, `head_A`, `head_V`) etiketle ölçütleri de raporlanır.

## 5. Hipotezler ve karar ölçütleri

Birincil ölçüt MAE'dir. Ölçütler validation'da (229 örnek), **baseline
checkpoint** ile ve beş seed üzerinden uygulanır. "Ortalama", beş seed'in
ortalamasıdır. Eşleştirilmiş fark önce aynı seed içinde alınır. Dropout
checkpoint'i (B/D) aynı tablolarla ikincil olarak raporlanır; ayrı karar
ölçütü yoktur.

| Kod | Hipotez | Destek ölçütü | Önceden beklenti |
|---|---|---|---|
| H1 | Zarar, ölçeklemenin miktarıyla artar | K0–K7 ortalama MAE, τ 0,25'ten 16'ya her adımda azalır (1e-4'e kadar eşitlik kabul) **ve** K0'da \|agree_tau16 − plain\| ≤ 0,01 | Desteklenir |
| H2 | Zararın ana kaynağı metin bloğunun ölçeklenmesidir | K0'da ortalama (agree_av_tau1 − plain) < 0,25 × ortalama (agree_tau1 − plain) **ve** agree_av_tau1 < agree_tau1, 5/5 seed | Desteklenir |
| H3 | Uyum sinyali örneğe özgü yararlı bilgi taşır | K0'da agree_tau1 < shuffled_tau1 en az 4/5 seed'de **ve** ortalama fark ≤ −0,01 | Desteklenmez |
| H4 | Ses/görüntü bozulmasında iyileşme alanı küçüktür | K1, K2 ve K3'ün her birinde \|ortalama (oracle − plain)\| ≤ 0,01 | Desteklenir |
| H5a | Model metne baskın biçimde dayanır | plain için ortalama MAE(K10) − MAE(K0) ≥ 0,10 | Desteklenir |
| H5b | Kural, baskın modalite bozulduğunda yararlıdır | K9 veya K10'dan en az birinde agree_tau1 < plain en az 4/5 seed'de **ve** ortalama fark ≤ −0,01 | Desteklenmez |

Beklentinin tutmaması bir hata değil, bulgudur; raporda "doğrulanmadı" diye
yazılır. Ölçütler betimseldir: p-değeri hesaplanmaz, çoklu karşılaştırma
düzeltmesi yapılmaz ve beş seed bağımsız örneklem büyüklüğü gibi yorumlanmaz.
Örnek düzeyinde bootstrap aralıkları ayrı bir analizde eklenir.

## 6. τ seçimi ve test kuralı

- **τ\*** = baseline checkpoint'te, K0–K10 koşullarının eşit ağırlıklı
  ortalaması alınarak beş seed ortalaması en düşük validation MAE'yi veren τ.
  1e-6 içindeki eşitlikte büyük τ seçilir. Seçim yalnız validation'dan yapılır ve
  validation özetiyle birlikte commit edilir.
- Test, her seed için **bir kez** ve `--selection` ile verilen bu dosyayla çalışır.
  Test çalıştırması bütün varyantları hesaplar; raporda τ\* başlık sonucu, diğer
  τ'lar tam eğri olarak verilir. Testte iyi görünen varyant ya da τ seçilmez.
- Test sonrası bir hata bulunursa düzeltme sapma olarak yazılır; ilk ve
  düzeltilmiş sonuçlar birlikte saklanır.

## 7. Çalışma kontrolleri

Bir koşu ancak bunların hepsi geçerse tamamlanmış sayılır:

- **Referans tekrarı:** K0–K7'de `plain` ve `agree_tau1`, sonuç paketindeki
  A/B/C/D metrikleriyle (validation veya test, seçilen seed) 1e-5 toleransla
  eşleşir: 32 satır × 4 ölçüt = 128 kontrol. Yeni sayılara ancak bu geçerse güvenilir.
- Test bölmesinde K0–K7 bozulmuş girdi özetleri v1 referanslarıyla aynıdır.
- Train istatistikleri v1 referanslarıyla eşleşir.
- `uniform` her batch'te `plain` ile aynıdır (1e-6); geçici hook'lar kaldırılır;
  tek modlu başlıklar ağırlıklandırmadan etkilenmez; model girdileri değişmez;
  iki checkpoint aynı bozulmuş girdileri alır.
- Veri, checkpoint, config, kaynak kodu ve sonuç paketi dosyalarının SHA256
  özetleri koşu boyunca değişmez.

## 8. Çıktılar

Her seed ve bölme için yeni, boş bir klasöre:

- `manifest.json`: sürümler, girdi özetleri, protokol ve betik özeti, durum.
- `metrics.csv`: checkpoint × koşul × varyant; MAE, Non0 accuracy, Non0 F1, Corr.
- `reference_checks.csv`: 128 referans kontrolü.
- `predictions.csv`: örnek başına etiket, bütün varyant tahminleri, tek modlu
  tahminler, d ve τ=1 ağırlıkları.
- `weight_summary.csv`: koşul ve varyant başına ağırlık ortalaması/SS/min/maks.
- `input_summary.csv`: koşul başına girdi özeti, maskelenme oranları.
- `controls.csv`, `train_feature_stats.json`, `train_stats_checks.csv`.

Beş seed'in özeti, hipotez kararları ve τ\* ayrı bir özet betiğiyle üretilir.

## 9. Sınırlar

Tek veri kümesi, tek bozulma seed'i, özellik düzeyinde sentetik bozulma ve
küçük validation kümesi (229 örnek) vardır. Bütün varyantlar eğitim sonrası
(post-hoc) uygulanır; ağırlıklandırmayla yeniden eğitim bu protokolün kapsamı
dışındadır. `oracle`, gerçek kullanımda bilinmeyen bir bilgiyi kullanır ve
yalnız üst sınır tanısıdır.
