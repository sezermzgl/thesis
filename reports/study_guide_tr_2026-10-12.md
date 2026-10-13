# Toplantı çalışma materyali (güncelleme: 2026-10-13)

Bu doküman, hocana tezin şu anki durumunu anlatabilmen için hazırlandı. Önce kavramları sıfırdan açıklıyor, sonra her deneyi "ne sorduk, ne yaptık, ne bulduk, neden önemli" sırasıyla anlatıyor. Sonda 10 dakikalık bir anlatım akışı ve muhtemel sorular ile cevapları var.

İngilizce rapor: `thesis_repo/reports/consolidated_report_2026-10-12.md` ve rapor sayfası (aynı içerik). Tablolar ve sayılar oradakilerle aynı.

---

## Bugün için çalışma planı

Önerilen sıra ve süre (toplam yaklaşık 3–4 saat):

1. **Bölüm 0 ve 1 (30 dk):** kavramlar. Özellikle 1.6 (AUC), 1.8 (bootstrap), 1.9 (seed) ve 1.11 (bilgi tavanı). Bunları kendi cümlelerinle anlatabilmelisin.
2. **Bölüm 2 (60 dk, en önemlisi):** conformal kavramları. 2.3'teki "iki farklı hata oranı" örneğini kâğıda kendin yeniden hesapla. 2.4'teki binom testini bir örnekle dene: 93 nodülde 1 kanser varsa p = P(Binom(93; 0,05) ≤ 1) ≈ 0,05.
3. **Bölüm 3 ve 4 (60 dk):** deneylerin hikâyesi ve conformal sonuçlar. Her tablonun "tek cümlelik mesajını" yaz.
4. **Bölüm 7 (20 dk):** 10 dakikalık anlatımı sesli prova et.
5. **Bölüm 8 (30 dk):** soruları cevaplarına bakmadan önce kendin cevaplamaya çalış.

**Kendini kontrol soruları** (cevapları metinde var):
- AUC 0,785 ne demek, bir cümleyle?
- Bootstrap aralığı neden seed farkını kapsamıyor?
- Mondrian'ın garantisi ile LTT'nin garantisi arasındaki fark ne? (ortalamada mı, olasılıkla mı)
- %5 hedefi için 2 kanserle en az kaç nodül gerekir? (124)
- Planlanan LTT XGBoost'ta bile neden sertifika vermedi? (iki sebep)
- Yeni LTT tasarımı neyi değiştirdi, neyi değiştiremedi?
- Neden daha büyük model ya da daha iyi prompt bizi otomatik iyi huylu garantisine götürmez?
- Halka özellikleri neden işe yaramadı, TI-RADS özellikleri neden yaradı?

---

## Bölüm 0: Tek cümlelik özet

> TN3K'da radyomik özellikleri metne çevirip LLM'e sınıflandırma yaptırdık. TI-RADS'tan esinlenen 6 yeni özellik LLM'i belirgin şekilde iyileştirdi, ama LLM aynı bilgiyi gören XGBoost'un altında kaldı ve ona hiçbir bilgi eklemedi. Conformal katmanda otomatik "iyi huylu" garantisi (grubun en fazla %5'i kanser) hiçbir modelle sağlanamadı. Otomatik "kötü huylu" garantisi (en fazla %20 iyi huylu) ise önceden sabitlenmiş yeni bir test tasarımıyla neredeyse her bölünmede sağlandı. Daha gevşek garantiler (kaçırılan kanser oranı ≤ %5) da sağlandı, ama bu garantilerin otomatik "iyi huylu" grubunda hâlâ yaklaşık %10 kanser var.

Tezin ana mesajı: **"Hangi garanti ne zaman mümkün ve ne işe yarar."** Yöntem başarısız olmadı. Hangi garantinin bu veriyle ve bu modellerle mümkün olduğunu ölçtük.

---

## Bölüm 1: Kavramlar (sıfırdan)

### 1.1 Radyomik özellikler
Görüntüdeki nodülün maskesinden hesaplanan sayılar: şekil (yuvarlaklık, uzunluk), parlaklık istatistikleri (ortalama, aralık) ve doku (gri tonların nasıl dağıldığı; GLCM, GLSZM gibi aileler). Biz PyRadiomics ile 102 özellik çıkardık.

### 1.2 z-score
Her özelliği train setinin ortalaması ve standart sapmasıyla ölçekliyoruz: z = (değer − ortalama) / std. 0 "ortalama bir nodül", +2 "ortalamanın 2 std üstü" demek. Prompt'taki sayılar bu z-score'lar.

### 1.3 mRMR (minimum redundancy, maximum relevance)
Özellik seçme yöntemi. Etiketle ilişkisi yüksek, ama birbirine benzemeyen özellikleri seçer. Biz 16 özellik seçiyoruz. Ondan önce |r| > 0,95 olan neredeyse kopya özellikleri atıyoruz, çünkü mRMR tek başına "PixelSurface" ve "MeshSurface" gibi 2D'de aynı olan iki özelliği birlikte seçebiliyordu.

### 1.4 LLM sınıflandırıcı
Her nodül için bir metin yazıyoruz: *"The 'Elongation' feature is measured at -1.140."* (22 özellik için 22 cümle). Bu metni Qwen2.5-1.5B'ye veriyoruz.

**Önemli:** model cevabı yazıyla üretmiyor. Modelin son katmanına yeni bir doğrusal katman (classification head) ekleniyor. Bu katman son token'ın iç temsilini iki sayıya (benign, malignant) çeviriyor, bunlar softmax'tan geçince P(malignant) olasılığı çıkıyor. Ra et al. de aynısını yapıyor.

### 1.5 LoRA ve QLoRA
1,5 milyar parametreyi baştan eğitmek pahalı. **LoRA**, modelin ağırlıklarını dondurup her katmana küçük, eğitilebilir "ek matrisler" ekliyor. Rank 16, bu matrislerin boyutu. Q/V, dikkat (attention) katmanının query ve value kısımlarına eklendiği anlamına geliyor. **QLoRA**, dondurulmuş modeli 4-bit'e sıkıştırarak bellek tasarrufu sağlıyor.

### 1.6 AUC
Rastgele seçilmiş bir kanser ile bir iyi huylu nodülü model doğru sıraya koyuyor mu? AUC = bunun olasılığı. 0,5 yazı-tura, 1,0 mükemmel. Eşik seçmeden modelin **sıralama** kalitesini ölçer.

### 1.7 Split'ler ve rolleri
| split | nodül | ne için |
|---|---|---|
| train | 2.095 | modeli eğitmek |
| validation | 436 | en iyi epoch'u seçmek |
| calibration | 524 | conformal eşiklerini seçmek (sadece burada!) |
| test | 438 | sadece raporlamak |

Her split'te kanser oranı %34,6–34,7.

### 1.8 Paired bootstrap
İki modeli **aynı nodüller** üzerinde karşılaştırıyoruz. Test setinden tekrar çekerek (with replacement) 2.000 sahte test seti oluşturuyoruz, her birinde iki modelin AUC farkını hesaplıyoruz. Bu farkların %2,5 ve %97,5 değerleri, %95 güven aralığını veriyor. **Aralık sıfırı içermiyorsa fark anlamlı.**

Sınırı: modeli sabit kabul ediyor. Modeli başka bir seed ile eğitince ortaya çıkan oynamayı kapsamıyor. Bu yüzden seed tekrarları da yaptık.

### 1.9 Seed
Eğitimdeki rastgeleliğin başlangıç noktası: LoRA ağırlıklarının başlangıcı ve verinin karıştırılma sırası. Aynı ayarlarla farklı seed, farklı model demek. Bizde seed tek başına test AUC'yi yaklaşık 0,03–0,04 oynatıyor.

### 1.10 XGBoost
Karar ağaçlarını art arda ekleyen klasik tablo modeli. Tablo verisinde genelde en güçlü yöntemlerden biri. Biz ayarlamadan (300 ağaç, derinlik 3) "baseline" olarak kullanıyoruz.

### 1.11 Bilgi tavanı
Bir model, kendisine verilen bilginin ötesine geçemez. LLM ve XGBoost aynı 22 sayıyı görüyor. Bu sayılarla ulaşılabilecek en iyi ayırma gücü kabaca XGBoost'un seviyesi (0,835). Daha büyük bir LLM ya da daha iyi bir prompt, LLM'i en fazla bu tavana yaklaştırabilir, tavanı yükseltemez. Tavanı yükseltmenin tek yolu yeni bilgi eklemek: TI-RADS özellikleri bunu yaptı (+0,06), halka özellikleri yapamadı.

---

## Bölüm 2: Conformal kavramları (en önemli kısım)

### 2.1 Exchangeability (değiştirilebilirlik)
Conformal yöntemlerin tek varsayımı: calibration ve test nodülleri aynı "torbadan" rastgele çekilmiş olmalı. Böylece calibration'da gördüğümüz hata oranı, test için güvenilir bir tahmin olur.

**Bizim hikâyemiz:** resmi TN3K split'inde calibration ile test farklı kaynaklardan geliyordu. Bir sınıflandırıcı bu iki seti %63 AUC ile ayırt edebiliyordu, yani exchangeable değillerdi. Bu yüzden bütün veriyi birleştirip yeniden böldük (pooled split). Aynı test bu kez 0,43 verdi, yani artık ayırt edilemiyorlar.

**Dikkat:** split görüntü düzeyinde. Aynı hastanın farklı görüntüleri farklı split'lere düşmüş olabilir, çünkü hasta kimlikleri veri setinde yok. Bu, bütün modellerin AUC'sini biraz şişirir. Conformal garantiyi bozmaz, ama garanti "bu havuzdan yeni bir görüntü" için geçerli olur, "yeni bir hasta" için değil.

### 2.2 Üçlü triage kuralı
Modelin olasılığına göre:
- P ≤ t ise **otomatik iyi huylu**,
- P ≥ u ise **otomatik kötü huylu**,
- arada kalırsa **doktora yönlendir**.

Soru: t ve u'yu nasıl seçelim ki hata oranları garanti altında olsun?

### 2.3 İki farklı "hata oranı", tezin kalbi
Aynı kural için iki farklı soru sorulabilir:

1. **Kaçırma oranı (miss rate):** bütün kanserlerin yüzde kaçı otomatik "iyi huylu" grubuna düştü? Payda bütün kanserler.
2. **Grup içi kanser oranı (R1):** otomatik "iyi huylu" denen nodüllerin yüzde kaçı aslında kanser? Payda o gruba düşen nodüller.

Örnek: 1.000 nodül, 350 kanser. Kural 200 nodüle "iyi huylu" diyor ve bunların 20'si kanser.
- Kaçırma oranı = 20/350 = **%5,7**
- Grup içi kanser oranı = 20/200 = **%10**

Hastaya "iyi huylu" denince onu ilgilendiren soru ikincisi. Ama ikincisini sağlamak çok daha zor.

### 2.4 LTT (Learn-then-Test)
Planımızdaki yöntem buydu.
1. Önceden aday eşikler belirliyoruz: t = 0,05, 0,10, … 0,45.
2. Her aday için şu hipotezi test ediyoruz: **H0: "bu eşikte otomatik iyi huylu grubundaki kanser oranı %5'ten fazla."**
3. Calibration'da bu eşiğin altında n nodül var, k tanesi kanser. Eğer gerçek oran %5 olsaydı, k veya daha az kanser görme olasılığı p = P(Binom(n, 0,05) ≤ k). Bu p-değeri.
4. p ≤ δ = 0,05 ise H0'ı reddediyoruz. Yani %95 güvenle "bu eşikte oran %5'in altında" diyoruz.
5. **Fixed sequence:** adayları en güvenliden (en küçük t) başlayarak sırayla test ediyoruz ve ilk başarısızlıkta duruyoruz. Sıra önceden sabitlendiği için çoklu test düzeltmesi gerekmiyor.

**Garanti tipi:** "%95 olasılıkla (calibration çekilişi üzerinden), seçilen eşiğin gerçek riski α'nın altında." Buna yüksek olasılıklı garanti (PAC tipi) denir. İki risk (R1 ve R2) birlikte %90 olasılıkla tutar (1 − 2δ).

**Binom testi neden geçerli?** Eşiğin altına düşen n nodül, "eşiğin altındaki nodüller" popülasyonundan rastgele bir örnek. Dolayısıyla içlerindeki kanser sayısı Binom(n, gerçek oran) dağılır.

### 2.5 Split conformal prediction (CP) ve Mondrian
Farklı bir mantık: her nodül için bir **etiket kümesi** üretiyoruz: {iyi huylu}, {kötü huylu} veya {ikisi}. Tek etiketli küme otomatik karar, iki etiketli küme "doktora gönder" demek.

- **Marginal:** bütün nodüllerin %90'ında (α = 0,10) doğru etiket kümede. Ama bu ortalamayı çoğunluk sınıfı (iyi huylular) taşıyabilir: α = 0,10'da kanserlerin sadece %70–86'sı kapsanıyordu.
- **Mondrian:** her sınıf için ayrı eşik. Biz kanserler için α = 0,05 kullandık, yani kanserlerin en az %95'inin kümesinde "kötü huylu" etiketi bulunuyor. Bu, kanserlerin en fazla %5'inin tek başına "iyi huylu" etiketi alması demek. Yani **kaçırma oranı garantisi**.

**Garanti tipi:** "ortalama olarak" (calibration ve test çekilişleri üzerinden beklenen değer). LTT'deki "%95 olasılıkla" ile karıştırılmamalı. Tek bir test setinde gerçekleşen değer bunun altında da üstünde de çıkabilir.

### 2.6 Kaçırma oranı LTT'si (miss-rate rule)
LTT ile aynı test, ama payda bütün calibration kanserleri (182). "Kaçırma oranı ≤ %5" sorusunu %95 güvenle test ediyor. Planlanan LTT başarısız olduktan sonra eklediğimiz için "keşif amaçlı" dedik, ama yöntem olarak geçerli.

### 2.7 Özet tablo
| yöntem | neyi sınırlıyor | garanti tipi | sonuç |
|---|---|---|---|
| LTT (R1) | otomatik "iyi huylu" grubundaki kanser oranı ≤ %5 | %95 olasılıkla | **hiçbir modelde sertifika yok** |
| LTT (R2) | otomatik "kötü huylu" grubundaki iyi huylu oranı ≤ %20 | %95 olasılıkla | planlanan sırayla yok |
| kaçırma oranı LTT'si | kanserlerin ≤ %5'i otomatik "iyi huylu" | %95 olasılıkla | **sağlandı** |
| Mondrian CP | kanserlerin ≥ %95'i kümede "kötü huylu" ile | ortalamada | **sağlandı** |

---

## Bölüm 3: Deneylerin hikâyesi (kronolojik)

### 3.1 İlk çalıştırma (2 Ekim): resmi split, v1 özellikler
- LLM test AUC'si 0,672. LTT hiçbir şeyi sertifikalamadı.
- Pertürbasyonlar (girdiyi hafifçe değiştirip modele tekrar sorup oy toplamak), softmax olasılığından daha iyi bir belirsizlik sinyali vermedi.
- Model, hatalarında "emin ama yanlış": hatalarının çoğunda bütün sorgularda aynı cevabı veriyor.
- Calibration ile test exchangeable değildi.

**Sonuç:** sorun conformal yöntemde değil, modelin kendisinde ve split'te.

### 3.2 Altyapıyı düzeltme
1. **Pooled split:** exchangeability'yi sağladık.
2. **Veri kalitesi kontrolü:** v1'de doku özellikleri nodül içinde sadece yaklaşık 9 gri seviyeyle hesaplanıyordu, şekil özellikleri piksel cinsindendi (görüntü boyutları farklı), 298 maske birden fazla parçadan oluşuyordu.
3. **v2 özellikleri:** 64 gri seviye, boyuta göre normalize edilmiş şekil özellikleri, 7 özellik ailesi (102 özellik).
4. **Teknik düzeltme:** tahmin aşamasında bf16 yuvarlaması olasılıkları bozuyordu. fp32'ye geçtik.

**Sonuç:** v2 özellikleri LLM'i +0,05 iyileştirdi (0,699 → 0,754).

### 3.3 Format denemeleri: "LLM sayıları mı kullanıyor, kelimeleri mi?"
| format | örnek | test AUC |
|---|---|---|
| decimal | "… is measured at -1.140." | 0,754 |
| z-score | "… has a z-score of -1.14 relative to the training nodules." | 0,759 (fark yok) |
| percentile | "… is at percentile 13 …" | 0,734 (uç değerler sıkışıyor, daha kötü) |
| rich | z-score + persentil + "ortalamadan düşük, yani daha az yuvarlak" gibi açıklama | 0,744 (tekrar run'ında 0,698) |

**Yorum:** model sayıların büyüklüğünü kullanıyor. Etrafına kelime eklemek bir şey kazandırmıyor.

### 3.4 Model boyutu: "Daha büyük LLM daha iyi olur mu?"
| model | test AUC |
|---|---|
| Qwen2.5-1.5B | 0,754 |
| Qwen2.5-7B (LR 2e-4) | 0,713 |
| Llama-2-7B (Ra et al.'ın ayarları) | 0,731 |
| Qwen2.5-7B, TI-RADS ile (LR 5e-5) | 0,693 |

7B modellerin eğitimi kararsızdı: duyarlılık epoch'tan epoch'a 0 ile 0,9 arasında zıpladı, gradient norm 200'e çıktı. LR'yi düşürmek zıplamayı azalttı ama sonucu değiştirmedi. **Bu kurulumda (QLoRA, yaklaşık 2.000 örnek, 5 epoch) büyük model kazandırmıyor.**

### 3.5 TI-RADS'tan esinlenen özellikler (en önemli olumlu bulgu)
**Fikir:** radyologlar TI-RADS'ta şeye bakıyor: nodülün dik mi yatık mı olduğu (taller-than-wide), çevresine göre ne kadar karanlık olduğu (ekojenite), kenarının düzgünlüğü, sıvı içeriği, parlak noktalar. PyRadiomics bunların bir kısmını hiç ölçmüyor, çünkü nodülü tek başına inceliyor, çevresine ya da yönüne bakmıyor. Biz bu işaretleri görüntüden hesaplayan 6 özellik yazdık.

**Sonuç (3 eşleştirilmiş seed):**
| | test AUC |
|---|---|
| TI-RADS özellikleriyle (3 seed ortalaması ± SD) | 0,785 ± 0,018 |
| TI-RADS özellikleri olmadan | 0,726 ± 0,024 |
| aynı seed'lerle fark | +0,059 (en küçüğü +0,050) |

Her seed'de ve her split'te kazanç pozitif ve aralık sıfırı dışlıyor. XGBoost'ta da +0,026 kazandırdı.

**Hangi özellik beklendiği gibi davrandı?**
- **Taller-than-wide, solidity, sıvı oranı:** TI-RADS'ın söylediği yönde.
- **Ekojenite oranı:** ters yönde çıktı, çünkü aslında sıvıyı ölçüyor (sıvı oranıyla korelasyonu −0,78). Sadece katı kısımda hesaplayınca yön düzeldi, ama ek bilgi gelmedi.
- **Parlak noktalar (mikrokalsifikasyon yaklaşımı):** sinyal yok.

### 3.6 Seed tekrarları: "Sonuçlarımız şans mı?"
En iyi kurulum tek seed'le test 0,804 vermişti. Üç seed: 0,804 / 0,768 / 0,783. Yani 0,804 şanslı üst uçtu. **Raporlanacak sayı 0,785 ± 0,018.** Seed tek başına 0,03–0,04 fark yaratıyor, bu yüzden tek run'lar arasındaki küçük farkları yorumlamıyoruz.

### 3.7 LLM ile XGBoost'u birleştirme (hibrit)
**Soru:** LLM, XGBoost'un bilmediği bir şey biliyor mu?

**Yöntem:** aynı 22 özellikle XGBoost eğittik ve iki modelin olasılıklarını birleştirdik:
- ortalama,
- sıra ortalaması,
- stacking: validation'da iki olasılığı girdi alan bir lojistik regresyon eğitmek.

**Sonuç:**
| model | test AUC |
|---|---|
| XGBoost | 0,835 |
| LLM (3 seed ortalaması) | 0,793 |
| ortalama | 0,824 (XGBoost'tan biraz kötü) |
| stacking | 0,834 (XGBoost'la aynı) |

- Stacking, XGBoost'a 0,99, LLM'e −0,04 ağırlık verdi. Yani LLM'i tamamen göz ardı etti.
- İki modelin olasılıkları 0,87 korelasyonlu.

**Anlamı:** LLM, XGBoost'un sıralamasının daha zayıf bir kopyasını öğreniyor.

### 3.8 Nodül çevresi (peritümöral) özellikler: "Daha fazla görüntü bilgisi işe yarar mı?"
**Fikir:** TI-RADS özellikleri nodülün çevresiyle ilişkisini ölçünce kazanç gelmişti. Peki nodülün etrafındaki dokunun kendisi de bilgi taşıyor mu?

**Yöntem:** nodülün etrafında bir halka tanımladık (genişliği nodülün eşdeğer yarıçapının yarısı, en az 5 piksel) ve bu halkada PyRadiomics özellikleri çıkardık (93 özellik).

**Karar kuralı, önceden sabitlendi:** halka özellikleri XGBoost'a eklendiğinde calibration'da anlamlı kazanç olursa LLM'e geçecekti.

| XGBoost | val | cal | test |
|---|---|---|---|
| nodül + TI-RADS | 0,828 | 0,848 | 0,831 |
| + 8 halka özelliği | 0,832 | 0,851 | 0,834 |
| sadece halka | 0,699 | 0,671 | 0,693 |

Calibration farkı +0,002, aralık [−0,009, +0,014], yani **kazanç yok, LLM'e geçilmedi.** Halkada sinyal var (tek başına 0,69), ama bu bilgi nodülün kendi özelliklerinde ve TI-RADS özelliklerinde zaten mevcut. Seçilen halka özellikleri çoğunlukla parlaklık istatistikleri, yani ekojenite oranının zaten karşılaştırdığı şey.

**Ders:** TI-RADS özellikleri, standart radyomiğin hiç ölçmediği yönelim ve kenar şekli gibi bilgileri getirdiği için işe yaradı. Halka ise eldeki bilgiyi tekrarladı. Skoru yükselten şey özellik sayısı değil, yeni bilgi.

---

## Bölüm 4: Conformal sonuçlar

### 4.1 LTT neden hiçbir şeyi sertifikalamadı?
Önce sayıların ne gerektirdiğine bakalım. %5 hedefi için:

| gruptaki kanser sayısı | gereken minimum grup büyüklüğü |
|---|---|
| 0 | 59 |
| 1 | 93 |
| 2 | 124 |
| 3 | 153 |

Modellerin calibration'da sunduğu en temiz kuyruklar:

| | en düşük %10 | %15 | %20 |
|---|---|---|---|
| XGBoost | 52'de 2 kanser | 78'de 2 | 104'te 3 |
| LLM | 52'de 4 | 78'de 7 | 104'te 10 |

- XGBoost'un kuyruğu yaklaşıyor (78'de 2 kanser) ama 2 kanserle sertifika için 124 nodül gerekiyor. LLM'in kuyruğu 2–3 kat daha kirli.
- **İkinci sebep, sıra tasarımı:** fixed sequence en sıkı eşikten başlıyor ve orada çok az nodül var. Örneğin XGBoost'ta u = 0,95'te sadece 7 nodül var, 3'ü yanlış. Test orada başarısız oluyor ve sıra duruyor. Oysa u = 0,85 tek başına geçerdi (54 nodülde 4 hata, p = 0,010). Yani sorun kısmen tasarımda: en sıkı adaylar hiçbir zaman geçemeyecek kadar küçük.

**Hocaya söylenecek:** "LTT dürüst davrandı. Garanti istediğimiz seviyede değil, çünkü ne LLM ne XGBoost o kadar temiz bir kuyruk üretiyor. Eski bir simülasyona göre 500 nodüllük calibration ile %5 hedefi için AUC'nin 0,9'un üstünde olması gerekiyor."

### 4.2 Validation sıralı LTT (keşif)
Adayları en sıkıdan başlayarak değil, validation'daki p-değerlerine göre sıralarsak (validation, calibration'dan bağımsız olduğu için garanti bozulmaz), LLM için otomatik "kötü huylu" eşiği sertifikalanıyor:
- Run I: u = 0,85, test'te 43 nodül, %11,6 iyi huylu.
- 3 seed ortalaması: u = 0,90, test'te 28 nodül, 0 iyi huylu.

Ama bu yöntemi sonuçları gördükten sonra seçtik. **Yeni bir split'te önceden sabitlenip doğrulanması gerekiyor.**

### 4.2b Önceden sabitlenmiş tasarım (13 Ekim)
Tasarımı çalıştırmadan önce yazıp GitHub'a commit'ledik. Böylece "sonuca bakıp tasarımı ayarladık" denemez.
- **Aday eşikler:** sabit olasılıklar (0,95, 0,90…) yerine validation skorlarının yüzdelikleri: en düşük/en yüksek %5, %10, … %30. Böylece her adayda yeterince nodül var.
- **Sıra:** adaylar validation p-değerlerine göre sıralanıyor.
- **Test:** calibration'da fixed sequence, aynı α ve δ.

Modeller sabit tutulup calibration ve test 2.000 kez yeniden bölündü.

| model | otomatik iyi huylu (%5) | otomatik kötü huylu (%20), eski tasarım | otomatik kötü huylu (%20), yeni tasarım |
|---|---|---|---|
| XGBoost | hiç | %0 | %38 |
| LLM (3 seed ortalaması) | hiç | %1 | **%99** |

- **Otomatik iyi huylu tarafı hiçbir tasarımda geçmiyor.** Orada sınır modelin kendisi, prosedür değil.
- **Otomatik kötü huylu tarafı artık neredeyse her bölünmede sertifikalanıyor (LLM).** Nodüllerin yaklaşık %8'i otomatik "kötü huylu" deniyor ve bunların %6–7'si aslında iyi huylu (sınır %20). Bizim bölünmede eşik u = 0,833: test nodüllerinin %9,8'i, bunların %14'ü iyi huylu.
- **XGBoost'ta neden daha az?** Validation sırası onun en üst %5'lik grubunu başa koyuyor, ama o grup calibration'da kirli (7 nodülde 3 iyi huylu). Sıra orada takılıyor.
- **Dürüst sınırlama:** tasarımı, run I'deki keşif sonucunu gördükten sonra yazdık. Yeniden bölmeler de sadece hangi nodülün calibration'a hangisinin test'e düştüğünü değiştiriyor. Kesin doğrulama yeni veri ister: yeniden eğitim ya da ikinci veri seti.

**Hocaya söylenecek:** "Prosedürü düzeltince otomatik kötü huylu garantisi ulaşılabilir hale geldi. Otomatik iyi huylu garantisi için ise daha iyi bir model gerekiyor. Bu, tezin 'hangi garanti ne zaman mümkün' sorusuna net bir cevap."

### 4.3 Kaçırma oranı kuralı
| model | test'te otomatik oran | test kaçırma oranı | otomatik "iyi huylu" içindeki kanser |
|---|---|---|---|
| XGBoost | %23,7 | %4,6 | 104'te 7 (%6,7) |
| LLM | %12,1 | %4,6 | 53'te 7 (%13,2) |

### 4.4 Mondrian ve önemli bir teori kontrolü
Tek test setimizde kanser kapsamı 16 run'ın 12'sinde 0,95'in altında çıktı (0,90–0,94). "Garanti bozuluyor mu?" sorusu doğal.

**Kontrol:** modeli sabit tutup calibration + test'i (962 nodül) 4.000 kez rastgele yeniden böldük.
- Bütün modellerde ortalama kanser kapsamı **0,950–0,951**, standart sapma 0,024.
- Bölünmelerin yaklaşık %45'inde 0,95'in altında.

**Yani garanti tutuyor.** Bizim tek bölünmemiz yaklaşık 1 standart sapma şanssızdı. Bütün run'lar aynı bölünmeyi paylaştığı için hepsi birlikte düşük çıktı.

Aynı yeniden bölmede otomatik "iyi huylu" grubu:

| model | otomatik "iyi huylu" oranı | grup içindeki kanser oranı |
|---|---|---|
| XGBoost | %25,7 | %6,4 (%2,3–10,5) |
| LLM | %16,7 | %9,8 (%4,7–14,6) |

### 4.5 "Geçerli garanti ≠ güvenli karar"
Sağlanabilen garantiler (Mondrian, kaçırma oranı) nodüllerin %17–26'sını otomatik "iyi huylu" yapıyor. Bu grupta kanser oranı başlangıçtaki %35'ten %6–10'a iniyor. Ama ACR TI-RADS'ta %5–20 risk TR4 kategorisine karşılık geliyor ve o kategoride belirli bir boyutun üstündeki nodüllere hâlâ biyopsi öneriliyor. Yani bu kurallar istatistiksel olarak geçerli, ama biyopsiyi atlamayı haklı çıkarmaz.

---

## Bölüm 5: Teori kontrolünde neleri doğruladık

1. **LTT kodu doğru:** binom p-değeri, fixed sequence ve iki risk için 1 − 2δ. Unit testler geçiyor (40/40).
2. **Split CP kodu doğru:** quantile indeksi ⌈(n+1)(1−α)⌉. Yeniden bölme deneyi teoriyle birebir uyumlu (0,951).
3. **Kaçırma oranı kuralı geçerli bir LTT.** Notebook'ta "calibration ve test exchangeable değil" diyen eski bir açıklama kalmıştı. Bu sadece eski resmi split için doğruydu, düzelttim.
4. **Sızıntı (leakage) yok:** ölçekleme, filtre ve mRMR sadece train'de, epoch seçimi ve stacking sadece validation'da yapılıyor. Calibration sadece conformal için, test sadece raporlama için kullanılıyor.

**Dürüstçe söylenmesi gerekenler:**
- Bootstrap aralıkları modeli sabit kabul ediyor ve hasta kümelenmesini hesaba katmıyor (hasta kimliği yok), yani biraz dar.
- Bazı tasarım seçimleri (7. özelliği çıkarmak, decimal'de kalmak, validation sıralı LTT) calibration/test sonuçlarına bakılarak yapıldı. Ana sonuçlar bu seçimlere bağlı değil.
- Çok sayıda karşılaştırma yaptık ve çoklu test düzeltmesi uygulamadık. Sınırdaki tekil aralıkları abartmamak gerekiyor. Ana bulgular seed'ler ve split'ler boyunca tutarlı.
- %35 kanser oranı bir biyopsi popülasyonu. Grup içi kanser oranı gibi riskler prevalansa bağlı, başka bir popülasyona doğrudan taşınmaz.

---

## Bölüm 6: Ra et al. (2025) ile ilişki

Onların yöntemini tiroide uyarladık: PyRadiomics → mRMR → metin → LoRA'lı LLM sınıflandırıcı. Onlar "LLM klasik modelleri geçiyor" diyor, biz tersini bulduk. Olası sebepler:
1. **Baseline'larının AUC'si accuracy'lerine neredeyse eşit** (en fazla 0,004 fark, XGBoost AUC 0,605). Bu, AUC'nin olasılıktan değil 0/1 tahminden hesaplandığını düşündürüyor. Öyleyse baseline'lar haksız şekilde zayıf görünüyor.
2. **Kendi ablation'larında**, LLM dondurulup sadece son katman eğitildiğinde AUC 0,710, LoRA ile 0,700. Yani LLM'i eğitmek kazandırmamış.
3. **Özellik isimlerini gizleyince** AUC sadece 0,005 düşüyor ve bu fark test edilmemiş.
4. **Etiketleri BI-RADS skoru, biyopsi değil.** Güven aralığı yok, tek seed, 500+ epoch.

Biz düzgün değerlendirilmiş bir XGBoost ve seed tekrarlarıyla LLM'in 0,04–0,05 geride kaldığını ve XGBoost'a bilgi eklemediğini gösterdik.

---

## Bölüm 7: Toplantıda 10 dakikalık anlatım akışı

1. **(1 dk) Soru:** "İki soru vardı: LLM radyomik metinden nodül sınıflandırabilir mi, ve conformal katman hangi garantileri verebilir?"
2. **(1 dk) Altyapı:** "İlk denemede split exchangeable değildi ve özelliklerde hatalar vardı. Pooled split ve v2 özelliklerle düzelttik. v2, LLM'e +0,05 kazandırdı."
3. **(2 dk) Ana olumlu bulgu:** "TI-RADS'tan esinlenen 6 özellik yazdım. LLM'i 3 eşleştirilmiş seed'de ortalama +0,06 AUC iyileştirdi, her seed'de anlamlı. En iyi LLM test AUC'si 0,785 ± 0,018."
4. **(2 dk) LLM ve XGBoost:** "Aynı 22 özellikle XGBoost 0,835. Daha büyük modeller (7B, Llama), farklı prompt formatları ve Ra et al.'ın ayarları kazandırmadı. İkisini birleştirince stacking LLM'e −0,04 ağırlık verdi. LLM ek bilgi taşımıyor."
5. **(3 dk) Conformal:** "Planladığımız LTT hiçbir modelde, XGBoost'ta bile sertifika vermedi. İki sebebi var: kuyruklar yeterince temiz değil (%5 için 93 nodülde en fazla 1 kanser gerekiyor) ve fixed sequence en sıkı, çok küçük eşikten başlıyor. İkinci sorunu çözen bir tasarımı çalıştırmadan önce sabitledim: adaylar validation yüzdeliklerinde, sıra validation p-değerlerine göre. 2.000 yeniden bölmede otomatik kötü huylu garantisi LLM için %99 oranında sağlandı. Otomatik iyi huylu garantisi ise hiçbir tasarımda sağlanmadı, orada sınır model. Kaçırma oranı ve Mondrian garantileri de sağlanıyor, teoriyle birebir tuttuğunu yeniden bölmelerle doğruladım. Ama otomatik iyi huylu grubunda hâlâ %6–10 kanser var: geçerli garanti, güvenli karar demek değil."
6. **(1 dk) Skoru yükseltme denemeleri:** "Model boyutu, prompt formatı ve nodül çevresi özellikleri kazandırmadı. Tek gerçek kazanç yeni bilgiden, TI-RADS özelliklerinden geldi. Skoru daha fazla yükseltmenin yolu görüntünün kendisini kullanmak."
7. **(1 dk) Sonraki adımlar ve sorular:** aşağıdaki "hocaya sorulacaklar" listesi.

---

## Bölüm 8: Muhtemel sorular ve cevaplar

**S: LTT hiçbir şey vermediyse yöntem başarısız mı?**
C: Hayır. LTT, istenen garantinin bu modelle mümkün olmadığını dürüstçe söyledi. Bu bir bulgu. Hangi koşulda mümkün olacağını da sayıyla gösterebiliyoruz: örneğin 93 nodülde en fazla 1 kanser gibi. Daha gevşek ama anlamlı garantiler (kaçırma oranı) mümkün.

**S: Neden LLM'e bu kadar yatırım yaptın, XGBoost zaten daha iyi?**
C: Tezin sorusu Ra et al.'ın iddiasını test etmekti. Düzgün bir baseline ve seed tekrarlarıyla iddianın bu veride tutmadığını gösterdik, hibrit analizle de LLM'in ek bilgi taşımadığını. Bu, literatüre olumsuz ama sağlam bir katkı.

**S: 0,785 ile 0,804 arasında hangisi gerçek?**
C: Tek run'lar 0,77–0,80 arasında oynuyor. Raporladığım sayı üç seed'in ortalaması: 0,785 ± 0,018.

**S: Mondrian'da kapsam 0,95'in altında çıkmış, garanti bozuk mu?**
C: Garanti ortalama üzerinden. Calibration + test'i 4.000 kez yeniden böldüm, ortalama 0,951 çıktı. Bizim bölünme yaklaşık 1 standart sapma şanssızdı ve bütün run'lar aynı bölünmeyi paylaşıyor.

**S: Hasta düzeyinde sızıntı var mı?**
C: Olabilir. Hasta kimlikleri veri setinde yok, split görüntü düzeyinde. Bu bütün modellerin AUC'sini aynı yönde şişirir. Karşılaştırmaları etkilemez, conformal garantiyi de bozmaz, ama garanti "yeni hasta" değil "bu havuzdan yeni görüntü" için geçerli. Bunu sınırlama olarak yazıyorum.

**S: Daha büyük model denedin mi?**
C: Üç kez: Qwen 7B (iki farklı LR) ve Llama-2-7B. Hepsi 1.5B'nin altında kaldı ve eğitimleri kararsızdı.

**S: TI-RADS özellikleri gerçekten TI-RADS mi?**
C: Hayır, görüntüden otomatik hesaplanan yaklaşımlar. Üçü TI-RADS'ın beklediği yönde çalıştı, ekojenite oranı sıvıyı ölçtüğü için ters yönde çıktı, parlak nokta özelliği sinyal vermedi. Ayarlar etiketlere bakmadan önceden sabitlendi.

**S: LTT tasarımını sonuçları gördükten sonra değiştirdin. Bu hile değil mi?**
C: Haklı bir soru, ve sınırlamayı açıkça yazıyorum. Yeni tasarımı çalıştırmadan önce yazıp GitHub'a commit'ledim, git geçmişinde tarih belli. Tasarım garantiyi bozmuyor, çünkü aday sırası sadece validation verisine bakıyor, calibration'a dokunmuyor. Ama fikir, run I'deki keşif sonucundan sonra geldi ve modeller bu veriyle seçildi. Bu yüzden sonucu "doğrulanmış" değil, "yeni veride doğrulanması gereken güçlü bir aday" olarak sunuyorum.

**S: Neden daha büyük model ya da daha iyi prompt denemiyorsun?**
C: Bilgi tavanı yüzünden. LLM, XGBoost'un gördüğü 22 sayıyı görüyor ve hibrit analiz LLM'in bu sayılardan ek bilgi çıkarmadığını gösterdi. En iyi senaryoda LLM XGBoost'a (0,835) yetişir. Otomatik iyi huylu garantisi için ise AUC yaklaşık 0,9 gerekiyor ve XGBoost bile hiçbir bölünmede bu garantiyi alamadı. Üç büyük model denemesi de bunu destekliyor.

**S: Nodül çevresinin özelliklerini denedin mi?**
C: Evet. Halkada 93 radyomik özellik çıkardım ve önceden bir karar kuralı koydum. XGBoost'ta kazanç yoktu (calibration +0,002, aralık sıfırı içeriyor), bu yüzden LLM'e geçmedim. Halka tek başına 0,69 AUC veriyor ama bilgisi zaten mevcut özelliklerde var.

**S: %35 kanser oranı gerçekçi mi?**
C: Hayır, bu bir biyopsi popülasyonu. Gerçek tarama popülasyonunda oran çok daha düşük, bu da grup içi kanser oranını kolaylaştırır. Sonuçlar doğrudan taşınmaz.

---

## Bölüm 9: Hocaya sorulacaklar

1. **Ana çerçeve:** "Hangi garanti ne zaman mümkün" çerçevesi tez için uygun mu?
2. **LTT tasarımı:** önceden sabitlenmiş yeni tasarım otomatik kötü huylu garantisini sağlıyor. Bunu yeni veride doğrulamak için modeli yeni split'lerde yeniden eğitmek mi, yoksa ikinci bir veri seti mi daha uygun?
2b. **Görüntü tabanlı model:** otomatik iyi huylu garantisine yaklaşmanın tek gerçekçi yolu görüntünün kendisini kullanmak (CNN ya da vision-language model). Bu tezin kapsamına girer mi?
3. **İkinci veri seti:** ThyroidXL'i harici doğrulama için eklemek tezin kapsamına girer mi?
4. **Raporlama:** seed ortalamaları ve bootstrap aralıklarını birlikte raporlamak yeterli mi, çoklu karşılaştırma düzeltmesi istenir mi?
5. **Klinik anlam:** "otomatik iyi huylu grubunda %6–10 kanser" sonucunu TI-RADS risk kategorileriyle bu şekilde yorumlamak doğru mu?
