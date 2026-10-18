# Self-MM bozulma protokolü sürüm 1

Bu protokol, yeni bozulmalar altında model sonuçları ölçülmeden önce tanımlanmıştır.
Önceki K0–K3 sonuçları görülmüştür; bütün araştırma için ön kayıt iddiası taşımaz.

## Koşullar

| Koşul | İşlem |
|---|---|
| K0 | Özellikler değişmez |
| K1 | Sesin geçerli zaman adımları sıfır |
| K2 | Görüntünün geçerli zaman adımları sıfır |
| K3 | Ses ve görüntünün geçerli zaman adımları sıfır |
| K4 | Her modalitenin geçerli zaman adımlarında bağımsız %30 Bernoulli maskeleme |
| K5 | Aynı işlem %50 olasılıkla |
| K6 | Aynı işlem %70 olasılıkla |
| K7 | Ses ve görüntüye özellik başına sıfır ortalamalı Gauss gürültüsü; sigma = 0,5 × eğitim std |

K4–K6, her örnekte tam %30/%50/%70 sayıda adım silmez; oranlar olasılıktır.
Gerçekleşen oranlar ayrı raporlanır. Seçilen zaman adımının tüm özellikleri sıfırlanır.
Maskeleme oranları aynı uniform çekilişlerini paylaşır; K4 maskesi K5’in, K5 K6’nın alt kümesidir.
Ses ve görüntü ayrı rastgele üreteçler kullanır. K7, maske koşullarıyla birleştirilmez.

## Veri ve korumalar

- İlk aşama 229 validation örneğidir; test ölçümü yoktur.
- Metin, etiketler, özgün dizi uzunlukları ve padding korunur.
- Geçerli adımlar ilk `length` adımdır; MMSA’nın paketlenmiş LSTM davranışıyla uyumludur.
- K1–K3 padding’i korusa da padding LSTM’ye girmediğinden önceki tam sıfırlamayla geçerli girdiler aynıdır.
- Eğitim istatistikleri yalnızca 1284 train örneğinin padding olmayan adımlarından hesaplanır.
- Her özellik için bütün geçerli train adımları havuzlanır; population std (`ddof=0`) kullanılır.
- MMSA ile aynı şekilde float32 kullanılır ve sesin negatif sonsuz değerleri sıfırlanır. Diğer geçerli adımlarda sonlu olmayan değer bulunursa çalışma durur.
- Standart sapması sıfır özelliklere gürültü eklenmez. Standartlaştırma veya clipping yapılmaz.
- Pickle bütün bölmeleri içerir; yükleme sırasında dosya açılır ancak yalnız train/valid dizileri kullanılır.

## Tekrarlanabilirlik ve sonraki değerlendirme

Bozulma seed’i 1111’dir ve model eğitim seed’inden ayrı bir kavramdır. Rastgelelik;
protokol sürümü, seed, split, örnek indeksi, modalite ve bozulma ailesinden türetilir.
Model adı, batch sırası veya çalıştırma sırası rastgeleliğe katılmaz. Böylece A/B/C/D aynı bozuk girdiyi alır.
Tek bozulma seed’i şimdilik sabittir; bu kontrol rastgele bozulmaların belirsizliğini ölçmez.

Kontrol betiği model yüklemez, eğitim yapmaz ve başarı metriği hesaplamaz. Sonraki
değerlendirmede A/B/C/D ve sabit tau=1 önce validation üzerinde incelenecektir.
Yeni ayar seçimi bu kontrolün parçası değildir. MAE, Non0 Accuracy, F1, Corr ve temiz koşula göre değişim birlikte raporlanacaktır.

## Kontrol çıktıları

- corruption_summary.csv: oranlar, geçerli frame sayıları, değişim RMS ve normalize gürültü RMS.
- corruption_per_sample.csv: örnek bazındaki bozulma miktarı.
- train_feature_stats.json: gürültü ölçeği için eğitim istatistikleri.
- manifest.json: korumalar, veri ve kod özetleri, sürüm bilgisi.

K7’de `noise_rms_div_train_std` sıfır varyanslı özellikler dışlanarak hesaplanır;
çok sayıda çekilişte 0,5’e yakın olması beklenir, tam 0,5 olması zorunlu değildir.
