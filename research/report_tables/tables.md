# Rapor tabloları

Bu dosya `research/make_selfmm_report_tables.py` ile üretilir; elle düzenlenmez.
Ondalık ayırıcı virgüldür. "±" beş eğitim seed’inin örneklem SS’sidir (ddof=1).
Köşeli parantez, 686 test örneği üzerinde eşleştirilmiş bootstrap %95 aralığıdır (2000 tekrar).

## Tablo 1. Yayımlanan ve ölçülen temiz test sonuçları (MOSI, K0)

| Kaynak | MAE ↓ | Non0 Acc % ↑ | Non0 F1 % ↑ | Corr ↑ |
|---|---:|---:|---:|---:|
| Self-MM makalesi, Tablo 1 (yazarların bildirdiği) | 0,713 | 85,98 | 85,95 | 0,798 |
| MMSA sonuç tablosu (result-stat.md) | 0,7080 | 85,46 | 85,43 | 0,7963 |
| Bu çalışma, A (ölçülen; 5 seed) | 0,7240 ± 0,0146 | 83,90 ± 1,01 | 83,88 ± 0,92 | 0,7834 ± 0,0057 |
| Fark: bu çalışma − makale | +0,0110 | -2,08 | -2,07 | -0,0146 |

Makale satırı yazarların beş koşu ortalamasıdır; SS bildirilmediği için yazılmamıştır. MMSA tablosundaki MAE ve Corr 100 ölçeklidir ve 100’e bölünmüştür.

## Tablo 2. Koşullara göre A/B/C/D, test, MAE ↓

| Koşul | A | B | C | D |
|---|---:|---:|---:|---:|
| K0 | 0,7240 ± 0,0146 | 0,7574 ± 0,0244 | 0,8424 ± 0,0412 | 0,9163 ± 0,0747 |
| K1 | 0,7239 ± 0,0146 | 0,7571 ± 0,0246 | 0,8428 ± 0,0410 | 0,9173 ± 0,0758 |
| K2 | 0,7237 ± 0,0150 | 0,7571 ± 0,0244 | 0,8398 ± 0,0375 | 0,9118 ± 0,0709 |
| K3 | 0,7235 ± 0,0150 | 0,7568 ± 0,0246 | 0,8403 ± 0,0374 | 0,9115 ± 0,0703 |
| K4 | 0,7239 ± 0,0143 | 0,7573 ± 0,0245 | 0,8426 ± 0,0410 | 0,9155 ± 0,0743 |
| K5 | 0,7234 ± 0,0141 | 0,7570 ± 0,0247 | 0,8417 ± 0,0410 | 0,9148 ± 0,0742 |
| K6 | 0,7236 ± 0,0142 | 0,7571 ± 0,0248 | 0,8418 ± 0,0404 | 0,9146 ± 0,0738 |
| K7 | 0,7238 ± 0,0145 | 0,7574 ± 0,0244 | 0,8426 ± 0,0415 | 0,9157 ± 0,0740 |
| K8 | 0,9703 ± 0,0223 | 1,0117 ± 0,0445 | 1,0436 ± 0,0259 | 1,1047 ± 0,0594 |
| K9 | 1,1903 ± 0,0167 | 1,2212 ± 0,0582 | 1,2193 ± 0,0154 | 1,2579 ± 0,0559 |
| K10 | 1,3994 ± 0,0199 | 1,4291 ± 0,0773 | 1,3990 ± 0,0208 | 1,4294 ± 0,0736 |

## Tablo 3. Koşullara göre A/B/C/D, test, Non0 accuracy % ↑

| Koşul | A | B | C | D |
|---|---:|---:|---:|---:|
| K0 | 83,90 ± 1,01 | 82,90 ± 0,79 | 83,81 ± 0,90 | 82,90 ± 0,82 |
| K1 | 83,90 ± 1,01 | 82,90 ± 0,79 | 83,81 ± 0,90 | 82,90 ± 0,82 |
| K2 | 83,84 ± 1,14 | 82,74 ± 0,94 | 83,75 ± 1,05 | 82,84 ± 0,97 |
| K3 | 83,84 ± 1,14 | 82,74 ± 0,94 | 83,78 ± 1,05 | 82,80 ± 0,93 |
| K4 | 83,84 ± 1,07 | 82,80 ± 0,84 | 83,81 ± 0,86 | 82,80 ± 0,84 |
| K5 | 83,96 ± 1,00 | 82,77 ± 0,89 | 83,84 ± 0,99 | 82,80 ± 0,93 |
| K6 | 83,96 ± 1,02 | 82,77 ± 0,89 | 83,93 ± 0,97 | 82,80 ± 0,93 |
| K7 | 83,87 ± 0,94 | 82,90 ± 0,79 | 83,84 ± 0,92 | 82,84 ± 0,83 |
| K8 | 75,37 ± 1,16 | 73,35 ± 2,53 | 75,18 ± 1,22 | 73,48 ± 2,46 |
| K9 | 66,68 ± 1,11 | 63,93 ± 4,51 | 66,62 ± 1,08 | 64,12 ± 4,66 |
| K10 | 55,06 ± 3,82 | 54,57 ± 7,08 | 55,00 ± 3,94 | 54,48 ± 7,00 |

K0–K7 değerleri v1 sonuç paketindekiyle aynıdır (en büyük MAE farkı 0,0e+00); K8–K10 yalnız protokol v2’de ölçülmüştür. A: baseline, B: eğitimde ses/görüntü dropout’u, C/D: A/B checkpoint’ine τ=1 uyum ağırlığı.

## Tablo 4. Eşleştirilmiş farklar ve belirsizlik, test

| Fark | Koşul | MAE farkı [%95 BA] | Seed SS | MAE’si düşen seed | Non0 Acc farkı yp [%95 BA] |
|---|---:|---:|---:|---:|---:|
| B−A | K0 | +0,0334 [+0,0166; +0,0499] | 0,0364 | 1/5 | -1,01 [-2,09; +0,12] |
| B−A | K3 | +0,0333 [+0,0165; +0,0496] | 0,0367 | 1/5 | -1,10 [-2,10; -0,06] |
| B−A | K7 | +0,0336 [+0,0167; +0,0502] | 0,0364 | 1/5 | -0,98 [-2,03; +0,12] |
| B−A | K9 | +0,0309 [+0,0189; +0,0426] | 0,0719 | 2/5 | -2,74 [-4,30; -1,13] |
| B−A | K10 | +0,0297 [+0,0226; +0,0366] | 0,0926 | 4/5 | -0,49 [-2,11; +1,12] |
| C−A | K0 | +0,1184 [+0,0909; +0,1475] | 0,0416 | 0/5 | -0,09 [-0,25; +0,06] |
| C−A | K3 | +0,1167 [+0,0896; +0,1466] | 0,0398 | 0/5 | -0,06 [-0,19; +0,06] |
| C−A | K7 | +0,1188 [+0,0911; +0,1482] | 0,0419 | 0/5 | -0,03 [-0,19; +0,12] |
| C−A | K9 | +0,0291 [+0,0180; +0,0404] | 0,0062 | 0/5 | -0,06 [-0,48; +0,34] |
| C−A | K10 | -0,0004 [-0,0010; +0,0002] | 0,0019 | 2/5 | -0,06 [-0,21; +0,09] |
| D−B | K0 | +0,1589 [+0,1283; +0,1930] | 0,0616 | 0/5 | +0,00 [-0,09; +0,09] |
| D−B | K3 | +0,1547 [+0,1240; +0,1885] | 0,0559 | 0/5 | +0,06 [+0,00; +0,15] |
| D−B | K7 | +0,1583 [+0,1280; +0,1923] | 0,0609 | 0/5 | -0,06 [-0,18; +0,00] |
| D−B | K9 | +0,0367 [+0,0243; +0,0491] | 0,0167 | 0/5 | +0,18 [-0,03; +0,40] |
| D−B | K10 | +0,0004 [-0,0007; +0,0014] | 0,0043 | 1/5 | -0,09 [-0,33; +0,15] |

Negatif MAE farkı iyileşmedir. Seed SS eğitim değişkenliğini, bootstrap aralığı test örneklemesini yansıtır.

## Tablo 5. Önceden kaydedilmiş hipotezler (protokol v2)

| Kod | Hipotez | Önceden beklenti | Validation kararı | Test (tekrar) |
|---|---:|---:|---:|---:|
| H1 | Zarar ölçekleme miktarıyla artar (τ taraması) | Desteklenir | Desteklendi | Desteklendi |
| H2 | Zararın ana kaynağı metin bloğunun ölçeklenmesidir | Desteklenir | Desteklendi | Desteklendi |
| H3 | Uyum sinyali örneğe özgü yararlı bilgi taşır | Desteklenmez | Desteklenmedi | Desteklenmedi |
| H4 | Ses/görüntü bozulmasında iyileşme alanı küçüktür (oracle) | Desteklenir | Desteklendi | Desteklendi |
| H5a | Model metne baskın biçimde dayanır | Desteklenir | Desteklendi | Desteklendi |
| H5b | Kural, metin bozulduğunda yararlıdır | Desteklenmez | Desteklenmedi | Desteklenmedi |

Kararlar validation’da verilmiştir; test satırı aynı kuralın testte tekrar edip etmediğini gösterir.

## Tablo 6. Tanı varyantları, baseline checkpoint, test

| Karşılaştırma | Koşul | MAE farkı [%95 BA] | MAE’si düşen seed |
|---|---:|---:|---:|
| τ=1 (C) − plain | K0 | +0,1184 [+0,0909; +0,1475] | 0/5 |
| τ=1 (C) − plain | K3 | +0,1167 [+0,0896; +0,1466] | 0/5 |
| τ=1 (C) − plain | K10 | -0,0004 [-0,0010; +0,0002] | 2/5 |
| τ=16 − plain | K0 | +0,0002 [-0,0019; +0,0023] | 2/5 |
| τ=16 − plain | K3 | -0,0003 [-0,0024; +0,0020] | 3/5 |
| τ=16 − plain | K10 | -0,0000 [-0,0001; +0,0000] | 2/5 |
| Metin sabit, A/V ağırlıklı − plain | K0 | -0,0004 [-0,0011; +0,0003] | 4/5 |
| Metin sabit, A/V ağırlıklı − plain | K3 | -0,0002 [-0,0003; -0,0001] | 3/5 |
| Metin sabit, A/V ağırlıklı − plain | K10 | -0,0004 [-0,0008; +0,0001] | 3/5 |
| τ=1 − karıştırılmış τ=1 | K0 | +0,0404 [+0,0225; +0,0596] | 0/5 |
| τ=1 − karıştırılmış τ=1 | K3 | +0,0451 [+0,0279; +0,0623] | 0/5 |
| τ=1 − karıştırılmış τ=1 | K10 | +0,0003 [-0,0003; +0,0009] | 1/5 |
| Oracle − plain | K1 | -0,0007 [-0,0016; +0,0002] | 3/5 |
| Oracle − plain | K2 | -0,0002 [-0,0006; +0,0001] | 4/5 |
| Oracle − plain | K3 | -0,0009 [-0,0021; +0,0003] | 4/5 |
| Oracle − plain | K10 | +0,0224 [+0,0098; +0,0353] | 2/5 |
| plain K8 − plain K0 | K8 | +0,2463 [+0,1974; +0,3018] | 0/5 |
| plain K9 − plain K0 | K9 | +0,4662 [+0,4051; +0,5293] | 0/5 |
| plain K10 − plain K0 | K10 | +0,6754 [+0,6100; +0,7490] | 0/5 |

## Tablo 7. τ taraması, baseline, K0–K7 ortalama MAE

| Bölme | plain | τ=0,25 | τ=0,5 | τ=1 | τ=2 | τ=4 | τ=8 | τ=16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0,7179 | 1,2300 | 1,0347 | 0,8677 | 0,7797 | 0,7413 | 0,7272 | 0,7216 |
| Test | 0,7237 | 1,2052 | 1,0055 | 0,8417 | 0,7600 | 0,7320 | 0,7252 | 0,7238 |

Validation’da K0–K10 ölçütüyle seçilen τ* = 16 (ızgaranın en büyük değeri).

## Tablo 8. Çıkarım maliyeti (seed 1111, K0, test, batch 4, RTX 3050 Ti, 6 tur)

| Yöntem | ms/örnek | örnek/sn | Tepe allocated MiB | Forward sayısı |
|---|---:|---:|---:|---:|
| A | 4,983 ± 0,026 | 200,7 | 436,54 | 1 |
| B | 5,073 ± 0,048 | 197,1 | 437,98 | 1 |
| C | 10,193 ± 0,057 | 98,1 | 436,54 | 2 |
| D | 10,350 ± 0,050 | 96,6 | 437,99 | 2 |

C/A = 2,045 ± 0,016; D/B = 2,040 ± 0,013; tepe allocated bellek farkı 6,5 KiB.
Batch 4 üzerinden amortize süredir; batch=1 gecikmesi değildir. Yükleme ve ısınma dahil değildir.

## Tablo 9. Eğitim süresi ve bellek (MMSA_run toplamı)

| Seed | Model | Toplam sn | Tepe allocated MiB |
|---|---:|---:|---:|
| 1111 | dropout | 402,4 | 2106,2 |
| 1112 | baseline | 461,3 | 2106,2 |
| 1112 | dropout | 568,7 | 2106,2 |
| 1113 | baseline | 492,9 | 2106,2 |
| 1113 | dropout | 618,7 | 2106,2 |
| 1114 | baseline | 445,5 | 2106,2 |
| 1114 | dropout | 482,3 | 2106,2 |
| 1115 | baseline | 418,0 | 2106,2 |
| 1115 | dropout | 535,4 | 2106,2 |

Seed 1112–1115: baseline 454,5 ± 31,3 sn, dropout 551,3 ± 57,3 sn (n=4). Seed 1111 baseline için süre kaydı yoktur. Süre yükleme, eğitim, validation, son test ve temizliği içerir; dropout modelleri farklı epoch’ta durduğu için fark yalnız dropout maliyeti değildir.

## Tablo 10. Mekanizma: tek modlu başlıkların örnekler arası yayılımı (baseline, K0, test)

| Başlık | Tahmin SS aralığı (5 seed) |
|---|---:|
| Metin | 0,639 – 1,001 |
| Ses | 0,001 – 0,171 |
| Görüntü | 0,004 – 0,436 |

## Tablo 11. τ=1 kuralında ortalama metin ağırlığı (baseline, test)

| Koşul | Ortalama ± seed SS | Seed aralığı |
|---|---:|---:|
| K0 | 0,783 ± 0,041 | 0,712 – 0,814 |
| K3 | 0,774 ± 0,047 | 0,692 – 0,815 |
| K8 | 0,859 ± 0,012 | 0,847 – 0,873 |
| K9 | 0,930 ± 0,054 | 0,879 – 1,018 |
| K10 | 0,994 ± 0,107 | 0,912 – 1,177 |

Ağırlıkların toplamı 3’tür; eşit ağırlık 1’dir. Metin bozuldukça metin ağırlığı artar.

