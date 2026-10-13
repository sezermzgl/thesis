# Tasarımdan sonuçlara: tez çalışması nasıl ilerledi (30 Eylül – 13 Ekim 2026)

*Bu doküman, hocaya göstereceğin İngilizce süreç raporunun Türkçesi. Toplantıda anlatacaklarını buradan çalışabilirsin. Bölümler ve tablolar İngilizce sayfayla birebir aynı sırada. Her bölümün sonunda, o bölümü toplantıda nasıl anlatacağına dair kısa bir **🗣 Anlatım notu** var.*

---

## 0. Başlangıç noktası

**Hedef.** Tiroid nodülleri için bir triage katmanı. Bu katman sadece hata oranının garanti edilebildiği yerlerde otomatik karar veriyor, diğer bütün nodülleri doktora yönlendiriyor:
- **otomatik iyi huylu:** "iyi huylu" denen nodüllerin en fazla %5'i kanser olabilir (R1, α₁ = %5),
- **otomatik kötü huylu:** "kötü huylu" denen nodüllerin en fazla %20'si iyi huylu olabilir (R2, α₂ = %20),
- her iki garanti de calibration verisi üzerinden %95 olasılıkla geçerli (δ = 0,05). Yöntem: Learn then Test (LTT).

**Sınıflandırıcı.** Ra et al. (2025) pipeline'ını mamografiden tiroid ultrasonuna uyarladık:
- nodülün PyRadiomics özellikleri,
- mRMR ile seçim,
- her özellik için bir İngilizce cümle ("The 'Elongation' feature is measured at −1.140."),
- 4-bit QLoRA ve sınıflandırma katmanıyla eğitilmiş bir dil modeli (Qwen2.5-1.5B).

**Belirsizlik sinyali (30 Eylül'deki orijinal tasarım).** Modele aynı soruyu prompt'u hafifçe değiştirerek birkaç kez soruyoruz ve kaç kez "kötü huylu" dediğini sayıyoruz. Beklenti: yanlış cevaplar kararsız olacak, oylar bölünecek ve bu nodüller doktora gidecek.

**Veri.** TN3K: 3.493 ultrason görüntüsü (2.421 hasta), nodül maskeleri, biyopsiye dayalı etiketler, %35 kötü huylu.

> 🗣 **Anlatım notu:** "Amacım hata oranını garanti edebilen bir triage katmanıydı. Sınıflandırıcı olarak Ra et al.'ın radyomik özellikleri metne çevirip LLM'e verme yöntemini tiroide uyarladım. Belirsizlik için başta pertürbasyon oylarını planlamıştım."

---

## 1. Resmi split'te ilk triage çalıştırmaları (2 Ekim)

**Ne yaptık?** Triage katmanını test edilmiş bir Python paketi olarak yazdık (`conformal_triage/`, o gün 21 test). Tasarım notundaki örnek hesap birebir tuttu. Resmi TN3K split'inde 8 v1 özelliğiyle çalıştırdık. Model test AUC'si 0,672.

**Run 1, karışık pertürbasyonlar** (özellik sırasını karıştırma, özellik atma, farklı cümle kalıpları). Pertürbasyonlar modeli yoklamak yerine bozdu:
- farklı kalıplar hiçbir zaman "kötü huylu" cevabı vermedi,
- sadece sırayı karıştırmak cevapların %29'unu değiştirdi,
- hiçbir nodül 10 oyda 6'dan fazla "kötü huylu" almadı.

Oylar belirsizliği değil, eğitim formatından ne kadar uzaklaşıldığını ölçtü.

**Run 2, formatı koruyan pertürbasyonlar** (standartlaştırılmış değerlere Gauss gürültüsü, tek özellik atma). Model formatta kaldı, cevapların sadece %5–12'si değişti, ama:

| run 2 sinyali (resmi split) | ilk prompt'a göre değişen cevap oranı (cal) | doğru/yanlış tahminleri ayırma AUC'si (cal / test) |
|---|---|---|
| softmax güveni | – | 0,629 / 0,577 |
| gürültü oyları (SD 0,2) | 0,106 | 0,579 / 0,533 |
| özellik atma oyları | 0,117 | 0,652 / 0,571 |

| calibration'da LTT (resmi split) | en temiz otomatik iyi huylu grubu | en temiz otomatik kötü huylu grubu |
|---|---|---|
| gürültü oyları | v = 0: 165 nodül, %19,4 kanser (hedef ≤ %5) | v = 10: 116 nodül, %45,7 iyi huylu (hedef ≤ %20) |
| atma oyları | v = 0: 175 nodül, %20,0 kanser | v = 9: 58 nodül, %31,0 iyi huylu |
| softmax | sertifika yok | sertifika yok |

- Oy uyumu, doğru ve yanlış tahminleri softmax olasılığından daha iyi ayıramadı.
- Model **"emin ama yanlış"tı:** 0/10 oy alan nodüllerin %19'u kanserdi, hataların %56–69'unda on sorgunun hepsi aynı yanlış cevabı verdi.
- **LTT hiçbir şeyi sertifikalamadı**, ne oylarla ne softmax'la.
- Split conformal prediction calibration'da kapsamını tuttu, ama test'teki otomatik iyi huylu kararlarının %23'ü kanserdi.

**Bir veri sorunu ortaya çıktı.** Bir sınıflandırıcı calibration nodüllerini resmi test nodüllerinden ayırt edebiliyordu (AUC 0,63, p = 0,002). Conformal garantiler calibration ve test verisinin exchangeable olmasını gerektiriyor. Bu split'te değildiler.

**Yönü belirleyen sonuçlar:**
1. Sınır triage yönteminde değil, temel modelde. Hatalar kararlı olduğu için hiçbir tutarlılık sinyali onları yakalayamaz.
2. Softmax olasılığı en az oylar kadar iyi bir skor. Pertürbasyon yaklaşımı bırakıldı.
3. Herhangi bir garantiyi test etmeden önce split düzeltilmeli.

> 🗣 **Anlatım notu:** "İlk denemede LTT hiçbir şey vermedi. Ama asıl öğrendiğim şuydu: model yanlış bildiği yerlerde de kararlı, yani oylar işe yaramıyor. Ayrıca calibration ve test aynı dağılımdan değildi, bu da garantiyi baştan geçersiz kılıyordu."

---

## 2. Hata analizi ve daha gevşek bir garanti (7 Ekim)

Yeni eğitim yapmadan run 2'nin çıktılarını yeniden inceledik:
- LTT'yi engelleyen hatalar zaten pertürbasyonsuz prompt'ta vardı. Pertürbasyonlar onları ne yarattı ne gösterdi.
- **Farklı bir garanti denedik: kaçırma oranı.** Bütün kanserlerin otomatik iyi huylu grubuna düşen payı ≤ %5. Aynı binom LTT testiyle test ettik.

| kaçırma oranı kuralı, softmax (resmi split) | eşik | test'te otomatik iyi huylu | test'te kaçırılan kanser |
|---|---|---|---|
| kanserlerin ≤ %5'i | P ≤ 0,22 | %7,2 | %5,1 |
| kanserlerin ≤ %10'u | P ≤ 0,28 | %24,4 | %15,3 (kayma yüzünden ihlal edildi) |

Bu analiz, çalışmanın geri kalanında süren bir ayrımı getirdi. İkisi de **aynı hatayla** ilgili: kanser olduğu hâlde "iyi huylu" denen nodül (false negative, FN). Fark, FN'nin **neye bölündüğünde**:
- **Kaçırma oranı** = FN / bütün kanserler. Program düzeyinde bir soru: "kanserlerin yüzde kaçını kaçırıyorum?" İstatistikte false negative rate, yani 1 − duyarlılık.
- **R1** = FN / "iyi huylu" denen bütün nodüller. Hasta düzeyinde bir soru: "iyi huylu dediğim bir hasta aslında yüzde kaç ihtimalle kanser?" İstatistikte false omission rate, yani 1 − NPV.

Örnek: 1.000 nodül, 350 kanser, 200'üne "iyi huylu" deniyor, bunların 20'si kanser. Kaçırma oranı 20/350 = %5,7, R1 = 20/200 = %10. Kanser oranı yüksekken R1'i %5'te tutmak çok daha zor. False positive (iyi huyluya "kötü huylu" demek) ise R2'nin konusu.

> 🗣 **Anlatım notu:** "Burada iki farklı garanti türü olduğunu fark ettim. Kaçırma oranı program düzeyinde: 'kanserlerin %95'ini yakalarım'. R1 hasta düzeyinde: 'iyi huylu dediğimin %95'i gerçekten iyi huylu'. İkincisi çok daha zor."

---

## 3. Model, referans, split ve veri denetimi (8 Ekim)

**Daha büyük model (resmi split).** Qwen2.5-7B, 1.5B ile aynı test AUC'yi verdi. Yeniden eğitilen 1.5B ilk modelle tutarlıydı (Spearman 0,97).

**Klasik referans.** Aynı özelliklerle XGBoost, LLM'i yakaladı ya da geçti.

| resmi split, test AUC | 8 özellik | 16 özellik |
|---|---|---|
| Qwen2.5-1.5B (yeniden eğitilmiş) | 0,667 | – |
| Qwen2.5-7B | 0,669 | 0,666 |
| XGBoost | 0,696 | 0,709 |

**Pooled, exchangeable split.** 3.493 görüntünün hepsini birleştirip etiket ve kaynağa göre tabakalı olarak yeniden böldük: train 2.095 / validation 436 / calibration 524 / test 438.

| two-sample test (bir sınıflandırıcı iki seti ayırt edebiliyor mu?) | AUC |
|---|---|
| validation – calibration (kontrol) | 0,51 |
| calibration – resmi test | **0,63** (p = 0,002) |
| calibration – test, pooled split | **0,43** |

Bu split'te daha fazla özellik test'te de kazandırmaya başladı (XGBoost, 67 özellik: 0,792). Önceki "daha fazla özellik işe yaramıyor" bulgusu kaymadan kaynaklanıyormuş.

**Görüntü kalitesi denetimi.** Orijinal özellik çıkarımında dört sorun:
- doku özellikleri nodül içinde sadece yaklaşık 9 gri seviyeyle hesaplanıyordu (bin width 25),
- şekil özellikleri piksel cinsindendi ama görüntü boyutları farklıydı (resmi test görüntüleri daha büyük, "kaymanın" büyük kısmı buradan),
- 298 maske birden fazla parçadan oluşuyordu,
- 356 maske görüntü kenarına değiyordu.

**Kayda geçen sınırlama.** Hasta kimlikleri veri setinde yok, bu yüzden pooled split görüntü düzeyinde. Aynı hastanın görüntüleri farklı split'lere düşebilir. Bu, bütün modellerin mutlak AUC'sini aynı ölçüde şişirir ve sınırlama olarak raporlanıyor.

> 🗣 **Anlatım notu:** "Bu gün iki temel düzeltme yaptım. Birincisi split: veriyi birleştirip yeniden böldüm, exchangeability testi 0,63'ten 0,43'e indi. İkincisi özellikler: denetimde dört hata buldum. XGBoost'u da bu noktadan sonra referans model olarak kullandım."

---

## 4. Yeni özellikler ve pooled split'teki ilk LLM run'ları (8–9 Ekim)

**v2 çıkarımı.** 64 gri seviye, görüntüye göre normalize edilmiş boyut, yedi özellik ailesi (102 özellik), maskenin sadece en büyük parçası. Tek parçalı maskelerde v1'i birebir tekrar üretiyor.

**Yol üstünde bulunan iki düzeltme.**
- mRMR neredeyse aynı özellikleri seçiyordu (2D'de piksel yüzeyi ile mesh yüzeyi aynı sayı). mRMR'den önce bir kopya filtresi eklendi (|r| > 0,95, sadece train).
- Tahminler bfloat16 hassasiyetinde hesaplanıyordu ve olasılıklar yuvarlanıyordu (524 nodülde sadece 195 farklı değer). Tahmin tam hassasiyete taşındı.

**Sonuçlar (A–F run'ları, pooled split, 16 özellik).**

| run | değişiklik | val | cal | test |
|---|---|---|---|---|
| B | v1 özellikler | 0,709 | 0,705 | 0,699 |
| D | v2 özellikler (referans) | 0,750 | 0,752 | **0,754** |
| E | z-score ifadesi | 0,729 | 0,756 | 0,759 |
| F | persentil değerleri | 0,709 | 0,713 | 0,734 |
| C | Qwen2.5-7B | 0,708 | 0,705 | 0,713 |
| – | XGBoost, aynı 16 özellik | 0,799 | 0,823 | **0,805** |

- v1 → v2: test'te +0,055, eşleştirilmiş aralık [+0,013, +0,089].
- Değerlerin z-score olduğunu söylemek bir şey değiştirmedi. Persentil kötüleştirdi, yani model sayıların büyüklüğünü kullanıyor.
- Qwen2.5-7B kararsız eğitildi: validation duyarlılığı epoch'lar boyunca 0,00, 0,01, 0,93, 0,04, 0,62.
- XGBoost, LLM'in yaklaşık 0,05 önünde kalıyor.

> 🗣 **Anlatım notu:** "Yeni özellikler LLM'i 0,70'ten 0,75'e çıkardı ve bu anlamlı bir kazanç. Format denemeleri modelin sayıların büyüklüğünü kullandığını gösterdi. Ama XGBoost hâlâ 0,05 önde."

---

## 5. Ra et al.'ın ayarları, açıklamalı prompt'lar, split CP (9 Ekim)

| run | değişiklik | val | cal | test | dakika |
|---|---|---|---|---|---|
| D | referans (Qwen2.5-1.5B, decimal) | 0,750 | 0,752 | 0,754 | 26 |
| G | Llama-2-7B, Ra et al.'ın ayarları | 0,705 | 0,709 | 0,731 | 182 |
| H | rich prompt (sade dille anlam eklenmiş) | 0,732 | 0,711 | 0,744 | 83 |

- **Llama-2-7B, Ra et al.'ın ayarlarıyla** (LR 1e-4, LoRA bütün katmanlarda): Qwen2.5-1.5B'nin altında ve yedi kat daha yavaş.
- **"Rich" prompt'lar** (z-score + persentil + "ortalamadan düşük, yani daha az yuvarlak bir sınır" gibi bir anlam): katkı yok.
- **Split conformal prediction** her run'a eklendi. Mondrian CP (kanserlerin %95'ini kapsama) ortalamada vaat ettiğini veriyor, ama otomatik iyi huylu grubunda hâlâ %10–17 kanser kalıyor. Marginal kapsamayı çoğunluk sınıfı olan iyi huylular taşıyor.
- **LTT'de bir keşif bulgusu:** planlanan fixed sequence en sıkı eşikten başlıyor, orada sadece birkaç nodül var ve ilk başarısızlıkta duruyor. Adayları validation p-değerlerine göre sıralamak (garantiyi bozmadan) bir run'da otomatik kötü huyluyu sertifikalattı. Keşif olarak işaretlendi, doğrulanması gerekiyor.

> 🗣 **Anlatım notu:** "Ra et al.'ın kendi ayarlarını da denedim, kazandırmadı. Sade dille açıklama eklemek de işe yaramadı. Burada LTT'nin sıra tasarımında bir zayıflık fark ettim, bunu 13 Ekim'de çözdüm."

---

## 6. TI-RADS'tan esinlenen özellikler (9–10 Ekim)

**Fikir.** PyRadiomics nodülü tek başına inceliyor. ACR TI-RADS'ın bazı işaretleri ise tam da onun dışarıda bıraktığı şeylere bağlı: nodülün yönelimi (taller-than-wide) ve çevre dokuyla kontrastı.

**Ne yaptık?** Görüntü ve maskeden altı özellik yazdık. Ayarları etiketlere bakmadan önce sabitledik.

| özellik | TI-RADS işareti | tek başına train AUC'si | yön |
|---|---|---|---|
| taller-than-wide oranı | şekil | 0,658 | beklendiği gibi |
| solidity | kenar | 0,367 | beklendiği gibi (düşük = düzensiz = kötü huylu) |
| sıvı (anechoic) oranı | yapı | 0,409 | beklendiği gibi (kistik = iyi huylu) |
| ekojenite oranı | ekojenite | 0,552 | ters (sıvıyı ölçüyor) |
| kenar keskinliği | kenar | 0,563 | belirsiz |
| parlak nokta yoğunluğu | ekojenik odaklar | 0,487 | sinyal yok |

**Sonuçlar.**

| model | özellikler | val | cal | test |
|---|---|---|---|---|
| XGBoost | radyomik 16 | 0,799 | 0,823 | 0,805 |
| XGBoost | radyomik 16 + 6 TI-RADS | 0,828 | 0,848 | **0,831** |
| XGBoost | sadece 6 TI-RADS | 0,786 | 0,787 | 0,775 |
| LLM (run D) | radyomik 16 | 0,750 | 0,752 | 0,754 |
| LLM (run I) | radyomik 16 + 6 TI-RADS | 0,795 | 0,812 | **0,804** |
| LLM (run J) | + 7. özellik (katı kısım ekojenitesi) | 0,786 | 0,802 | 0,781 |
| LLM (run K) | + 6 TI-RADS, rich format | 0,803 | 0,785 | 0,801 |

- Herhangi bir değişikliğin getirdiği en büyük kazanç: LLM test'te +0,050, eşleştirilmiş aralık [+0,015, +0,084].
- 7. özellik ters yönü düzeltti ama bir şey eklemedi. Rich format decimal ile eşit çıktı, ama dört kat daha uzun eğitimle (132 dakikaya karşı 36).

> 🗣 **Anlatım notu:** "En büyük kazancı buradan aldım. TI-RADS'ın işaret ettiği ama PyRadiomics'in ölçmediği bilgileri, yönelim ve kenar şekli gibi, kendim hesapladım. Üç özellik TI-RADS'ın beklediği yönde çalıştı. Ekojenite oranı ters çıktı, çünkü aslında sıvıyı ölçüyordu."

---

## 7. Büyük modele adil bir şans (11 Ekim)

7B'deki kararsızlık LR'nin fazla yüksek olduğuna işaret ediyordu. Bu yüzden Qwen2.5-7B'yi LR 5e-5 ve TI-RADS özellikleriyle yeniden eğittik (run L).

| model | özellikler | test AUC |
|---|---|---|
| Qwen2.5-1.5B (run I) | radyomik + TI-RADS | **0,804** |
| Qwen2.5-7B, LR 2e-4 (run C) | radyomik | 0,713 |
| Llama-2-7B, LR 1e-4 (run G) | radyomik | 0,731 |
| Qwen2.5-7B, LR 5e-5 (run L) | radyomik + TI-RADS | 0,693 |

Eğitim daha sakindi, ama sonuç 1.5B'nin çok altında kaldı (aralık [−0,148, −0,076]) ve TI-RADS özellikleri 7B'ye hiçbir şey kazandırmadı. Üç büyük model denemesi aynı sonuca varıyor: bu kurulumda daha büyük model kazandırmıyor.

> 🗣 **Anlatım notu:** "Büyük model sorusunu üç kez denedim, öğrenme oranını düşürmek dahil. Hepsi 1.5B'nin altında kaldı."

---

## 8. Sonuç gerçek mi? Seed'ler, birleştirme, conformal kontroller (12 Ekim)

**Seed tekrarları.** En iyi kurulumu ve TI-RADS'sız kurulumu seed 1 ve 2 ile yeniden eğittik.

| test AUC | seed 42 | seed 1 | seed 2 | ortalama ± SD |
|---|---|---|---|---|
| TI-RADS özellikleriyle (I, M, N) | 0,804 | 0,768 | 0,783 | **0,785 ± 0,018** |
| TI-RADS özellikleri olmadan (D, O, P) | 0,754 | 0,712 | 0,713 | 0,726 ± 0,024 |
| eşleştirilmiş kazanç | +0,050 | +0,056 | +0,070 | **+0,059** |
| kazancın %95 aralığı | [+0,015, +0,084] | [+0,013, +0,098] | [+0,026, +0,112] | |

Seed 42 iki tarafta da şanslı üst uçtu. Eğitim seed'i tek başına AUC'yi 0,03–0,04 oynatıyor, bu da birçok tek run karşılaştırması kadar büyük bir etki.

**XGBoost ile birleştirme.** LLM, XGBoost'un bilmediği bir şey biliyor mu?

| model (aynı 22 özellik) | val | cal | test |
|---|---|---|---|
| XGBoost | 0,827 | 0,848 | **0,835** |
| LLM, 3 seed ortalaması | 0,792 | 0,803 | 0,793 |
| iki olasılığın ortalaması | 0,820 | 0,838 | 0,824 |
| stacking (validation'da eğitilmiş) | 0,827 | 0,848 | 0,834 |

Stacker'ın ağırlıkları: XGBoost 0,99, LLM −0,04. İki modelin skorları 0,87 korelasyonlu. **LLM, XGBoost'un ötesinde hiçbir bilgi eklemiyor.**

**Conformal kontroller.**

| LTT'nin sertifikalayabileceği en küçük grup (δ = 0,05) | 0 hata | 1 | 2 | 3 |
|---|---|---|---|---|
| otomatik iyi huylu, α = %5 | 59 | 93 | 124 | 153 |
| otomatik kötü huylu, α = %20 | 14 | 22 | 30 | 37 |

| calibration kuyruğu (524 nodül) | en düşük %10 | %15 | %20 | planlanan LTT |
|---|---|---|---|---|
| XGBoost | 52'de 2 kanser | 78'de 2 | 104'te 3 | sertifika yok |
| LLM, 3 seed ortalaması | 52'de 4 | 78'de 7 | 104'te 10 | sertifika yok |

| Mondrian split CP (kanserlerin %95'i kapsanır) | bizim bölünmede kapsam | 4.000 yeniden bölmede | otomatik iyi huylu | aralarındaki kanser |
|---|---|---|---|---|
| XGBoost | 0,941 | 0,951 ± 0,024 | %25,7 | %6,4 |
| LLM, 3 seed ortalaması | 0,934 | 0,950 ± 0,024 | %16,7 | %9,8 |

- En güçlü model olan XGBoost bile %5'te hiçbir şeyi sertifikalamıyor. En temiz 78 nodülündeki 2 kanser için 124 nodül gerekirdi.
- Mondrian'ın kanser kapsamı bizim bölünmede 16 run'ın 12'sinde 0,95'in altındaydı, ama yeniden bölmelerde ortalaması 0,95. **Garanti tutuyor, bizim bölünme şanssızdı.**

> 🗣 **Anlatım notu:** "Bu gün sonuçların sağlam olup olmadığını test ettim. TI-RADS kazancı üç seed'de de tutuyor, ama en iyi gerçek değer 0,804 değil 0,785. LLM'i XGBoost'la birleştirince LLM'in hiçbir şey katmadığı ortaya çıktı. Conformal tarafta XGBoost bile R1'i alamıyor. Mondrian'ın garantisi ise yeniden bölmelerle doğrulandı."

---

## 9. Geçebilen bir tasarım ve son bir özellik fikri (13 Ekim)

**Önceden kaydedilmiş LTT tasarımı.** Çalıştırmadan önce yazılıp commit'lendi:
- aday eşikler validation skorlarının yüzdeliklerinde (her tarafta nodüllerin %5'i ile %30'u arası),
- sıra validation p-değerlerine göre,
- test calibration'da fixed sequence ile.

2.000 yeniden bölmede değerlendirildi:

| sertifikalanan bölünme oranı | otomatik iyi huylu %5 | otomatik kötü huylu %20, eski tasarım | otomatik kötü huylu %20, yeni tasarım | karar verilen / aralarında iyi huylu |
|---|---|---|---|---|
| XGBoost | %0 | %0 | %38 | %10,7 / %14,9 |
| LLM, 3 seed ortalaması | %0 | %1 | **%99** | %7,9 / %7,4 |
| LLM run I | %0 | %13 | %22 | %11,5 / %18,4 |

Otomatik kötü huylu ulaşılabilir hâle geliyor. Otomatik iyi huylu ise hiçbir model ve hiçbir tasarımda sertifikalanmıyor.

**Peritümöral radyomik.** Nodülün etrafındaki bir halkada PyRadiomics özellikleri (93 özellik). Kural önceden belliydi: XGBoost'u calibration'da iyileştirirlerse LLM'e geçilecek.

| XGBoost özellikleri | val | cal | test |
|---|---|---|---|
| nodül + TI-RADS | 0,828 | 0,848 | 0,831 |
| + 8 halka özelliği | 0,832 | 0,851 | 0,834 |
| sadece halka | 0,699 | 0,671 | 0,693 |

Calibration +0,002 [−0,009, +0,014]: kazanç yok, bu yüzden LLM eğitilmedi. Halka tek başına sinyal taşıyor, ama nodül ve TI-RADS özelliklerinin zaten içerdiği bilgiyi tekrarlıyor.

> 🗣 **Anlatım notu:** "LTT'nin sıra sorununu, tasarımı çalıştırmadan önce sabitleyerek çözdüm. Otomatik kötü huylu garantisi artık neredeyse her bölünmede alınıyor. Otomatik iyi huyluda ise sınır modelin kendisi. Son olarak nodül çevresini denedim, önceden koyduğum kurala göre kazanç olmadığı için LLM'e geçmedim."

---

## 10. Sonuçlar bir bakışta

**Sınıflandırma (test AUC, aksi belirtilmedikçe pooled split).**

| adım | LLM (Qwen2.5-1.5B) | XGBoost |
|---|---|---|
| resmi split, v1, 8 özellik | 0,672 | 0,696 |
| pooled split, v1, 16 özellik | 0,699 | 0,723 |
| pooled split, v2, 16 özellik | 0,754 | 0,805 |
| + 6 TI-RADS özelliği | **0,785 ± 0,018** (3 seed) | **0,835** |
| + halka özellikleri | çalıştırılmadı (XGBoost'ta kazanç yok) | 0,834 |
| daha büyük modeller (7B) | 0,693–0,731 | – |
| LLM ve XGBoost birleşik | 0,834 (= XGBoost) | |

**Conformal garantiler (en iyi kurulum, pooled split).**

| garanti | ulaşılabilir mi? | ne sağlıyor |
|---|---|---|
| R1: otomatik iyi huylularda ≤ %5 kanser (LTT) | **hayır**, hiçbir model ya da tasarımla | – |
| R2: otomatik kötü huylularda ≤ %20 iyi huylu (LTT, yeni tasarım) | **evet**, yeniden bölmelerin %99'unda (LLM) | nodüllerin ~%8'ine karar, bunların %6–7'si iyi huylu |
| kanserlerin ≤ %5'i kaçırılır (kaçırma oranı LTT'si) | **evet** | test nodüllerinin %12'si (LLM) ya da %24'ü (XGBoost) otomatik iyi huylu, kanserlerin %4,6'sı kaçırılıyor |
| kanserlerin %95'i kapsanır (Mondrian split CP) | **evet**, ortalamada 0,95 | %17 (LLM) ya da %26 (XGBoost) otomatik iyi huylu, aralarında ~%10 / %6 kanser |

> 🗣 **Anlatım notu:** "Bu iki tablo bütün tezin özeti. Birinci tablo: her kazanç daha iyi ya da yeni bilgiden geldi, XGBoost'la arada hep 0,05 fark kaldı. İkinci tablo: hangi garanti mümkün, hangisi değil ve mümkün olan ne sağlıyor."

---

## 11. Çalışmanın geldiği nokta

| araştırma sorusu | cevap |
|---|---|
| RQ1: LLM radyomik prompt'lardan nodül sınıflandırabilir mi, klasik modelle nasıl karşılaştırılır? | Evet, orta düzeyde: en iyi özelliklerle test AUC 0,785 ± 0,018. Aynı özelliklerle XGBoost'un yaklaşık 0,05 altında ve XGBoost'a bilgi eklemiyor. |
| RQ2: Pipeline'daki hangi seçimler önemli? | Özelliklerin içeriği önemli: v2 ile +0,05, TI-RADS ile +0,06. Prompt dili, prompt formatı, model boyutu ve peritümöral özellikler önemli değil. |
| RQ3: LTT otomatik kararları sertifikalayabilir mi, pertürbasyon oyları yardımcı olur mu? | Oylar yardımcı olmuyor. %5'te otomatik iyi huylu, XGBoost dahil denenen hiçbir modelle ulaşılamıyor. %20'de otomatik kötü huylu, önceden kaydedilmiş bir tasarımla ulaşılabilir. Kaçırılan kanserler üzerine garantiler tutuyor, ama otomatik iyi huylu kararlarının ~%10'u kanser olarak kalıyor. |

**Ana mesaj.** *Hangi garanti, ne zaman ulaşılabilir ve ne işe yarar.* LTT katmanı dürüst davrandı: kanıtın desteklemediği garantileri vermedi.
- 524 calibration nodülü ve %35 kanser oranıyla, her "iyi huylu" kararı için %5 garantisi yaklaşık AUC 0,9 olan bir model gerektiriyor.
- Kaçırılan kanserler üzerine garanti şu an ulaşılabilir, ama geride TI-RADS TR4 benzeri risk taşıyan bir grup bırakıyor. Bu düzeyde biyopsi hâlâ öneriliyor.
- Geçerli bir garanti, güvenli bir karar demek değil.

> 🗣 **Anlatım notu:** Bu bölümü yavaş anlat, tezin ana mesajı burada. "Yöntem başarısız olmadı. Hangi garantinin ne zaman mümkün olduğunu sayılarla gösterdi."

---

## 12. Yöntemsel dersler

- **Bir garantiye güvenmeden önce exchangeability'yi kontrol et.** Resmi split onu bozuyordu, birleştirip yeniden bölmek düzeltti.
- **Özellikleri denetle.** v2 yeniden çıkarımı, herhangi bir modelleme değişikliği kadar değerliydi.
- **Seed'lerle tekrarla.** Tek run'lar 0,03–0,05 oynadı. Ana karşılaştırma eşleştirilmiş seed'lerle raporlanıyor.
- **Düzgün değerlendirilmiş bir tablo modeliyle karşılaştır ve ek bilgiyi test et.** Birleştirme, "LLM bir şey katıyor mu?" sorusunu bir AUC farkından daha doğrudan cevaplıyor.
- **Karar kurallarını çalıştırmadan önce sabitle.** LTT tasarımı da peritümöral deney de önceden yazılmış kurallarla karara bağlandı.
- **Skoru oynatan şey yeni bilgiydi**, yeni ifade ya da büyüklük değil.

---

## 13. Sınırlamalar

- Tek veri seti.
- Hasta kimliği olmadan görüntü düzeyinde split.
- Uzman maskeleriyle tek 2D görüntüler.
- Sezgisel (heuristic) TI-RADS özellikleri.
- Ayarlanmamış LLM hiperparametreleri.
- 524 nodüllük tek bir calibration seti.
- Bazı tasarım seçimleri calibration ya da test sonuçlarından etkilendi (geçtikleri yerde belirtildi).
- Bootstrap aralıkları eğitim gürültüsünü ve hasta kümelenmesini içermiyor.
- Sevk popülasyonuna ait bir kanser oranı (%35), taramaya taşınmaz.

---

## 14. Bir sonraki aşama için seçenekler (tartışmak için)

1. Önceden kaydedilmiş LTT tasarımını yeni veride doğrulamak: yeni rastgele split'lerde yeniden eğitim ya da ikinci bir veri setine (ThyroidXL) uygulama.
2. Görüntünün kendisini kullanmak (bir CNN ya da vision-language model). Otomatik iyi huylu garantisi için yeterince iyi bir modele giden tek gerçekçi yol, ama kapsam değişikliği demek.
3. Yazım: sınıflandırıcıyla ilgili deneysel sorular cevaplandı.

> 🗣 **Anlatım notu, kapanış:** "Hocam, sizin görüşünüzü almak istediğim üç konu var: bu çerçeve tez için uygun mu, LTT tasarımını yeni veride nasıl doğrulamalıyım, ve görüntü tabanlı bir model tezin kapsamına girer mi?"
