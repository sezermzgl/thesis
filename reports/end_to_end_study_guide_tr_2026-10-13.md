# Uçtan uca çalışma materyali: tezde yaptığımız her şey (30 Eylül – 13 Ekim 2026)

Bu doküman raporlardan bağımsız. Amacı, ilk günden bugüne yaptığımız her şeyi tek bir yerden, kavramlarıyla birlikte öğrenmen. Her aşamada şu sırayı izliyor:
- **Neden?** O adımı neden attık.
- **Ne yaptık?** Somut olarak ne denedik.
- **Ne bulduk?** Sayılarla sonuç.
- **Ders:** Sonraki adımı nasıl belirledi.

Sonda sözlük, ezberlemen gereken sayılar ve kendini sınama soruları var.

---

## Kısım A: Temel kavramlar

### A1. Problem
Tiroid nodüllerinin çoğu iyi huylu. Hangisine biyopsi (FNA, ince iğne aspirasyonu) yapılacağına ultrason görüntüsüne bakılarak karar veriliyor. Radyologlar bunun için ACR TI-RADS adlı bir puanlama sistemi kullanıyor. Bu sistem nodülün yapısına, parlaklığına, şekline, kenarına ve içindeki parlak noktalara puan veriyor.

Bizim fikrimiz: bir makine öğrenmesi modeli bazı nodüllere otomatik karar versin, ama sadece hata oranını **garanti edebildiği** yerlerde. Geri kalan nodüller doktora gitsin.

### A2. Veri: TN3K
- 3.493 ultrason görüntüsü, 2.421 hastadan.
- Her görüntüde nodülün maskesi (sınırları çizilmiş hâli) var.
- Etiketler biyopsiden geliyor: iyi huylu ya da kötü huylu. Kanser oranı %35. Bu, normal bir tarama popülasyonundan çok yüksek, çünkü veri setindekiler zaten şüpheli bulunup biyopsi yapılmış hastalar.
- Hasta kimlikleri veri setinde yok. Bu yüzden aynı hastanın farklı görüntüleri farklı split'lere düşmüş olabilir.

### A3. Radyomik özellikler ve PyRadiomics
Maskenin içindeki piksellerden hesaplanan sayılar:
- **Şekil:** uzunluk, yuvarlaklık (sphericity), uzama (elongation).
- **Birinci derece (first-order):** parlaklığın ortalaması, aralığı, yüzdelikleri.
- **Doku (texture):** gri tonların komşuluk ilişkileri. GLCM, GLSZM, GLRLM, GLDM ve NGTDM birer doku ailesi. Örneğin GLSZM "aynı gri tondaki bölgelerin boyutlarını" sayar.

PyRadiomics bu özellikleri hesaplayan standart kütüphane.

### A4. Ön işleme
- **Medyan doldurma:** eksik değerler train setinin medyanıyla doldurulur.
- **z-score:** z = (değer − train ortalaması) / train std. Örneğin +2, "train ortalamasının 2 standart sapma üstü" demek.
- **Kopya filtresi:** iki özellik |r| > 0,95 ile korelasyonluysa, etiketle daha az ilişkili olanı atılır.
- **mRMR:** etiketle ilişkisi en yüksek, ama birbirine en az benzeyen özellikleri sıralar. İlk 16'sını kullanıyoruz.
- Hepsi **sadece train** verisiyle öğreniliyor, diğer split'lere aynen uygulanıyor. Böylece sızıntı (leakage) olmuyor.

### A5. LLM sınıflandırıcı
Her nodül için bir metin oluşturuyoruz: *"Classify this thyroid nodule as benign or malignant… The 'Elongation' feature is measured at -1.140. …"*

Bu metni Qwen2.5-1.5B'ye veriyoruz. Model cevabı **yazıyla üretmiyor**. Son katmana yeni bir doğrusal katman (classification head) ekleniyor ve bu katman iki sayı veriyor. Softmax'tan sonra bunlar P(kötü huylu) olasılığına dönüşüyor.

### A6. LoRA ve QLoRA
- **LoRA:** modelin milyarlarca ağırlığını dondurup her katmana küçük, eğitilebilir "ek matrisler" ekliyor. Rank 16, bu matrislerin boyutu. Q/V, dikkat (attention) mekanizmasının query ve value kısımları.
- **QLoRA:** donmuş modeli 4-bit'e sıkıştırıyor, böylece tek bir GPU'ya sığıyor.
- **Eğitim ayarları:** 5 epoch, öğrenme oranı (LR) 2e-4, efektif batch 16. Sınıf ağırlıklı kayıp fonksiyonu kullanılıyor, kanserleri kaçırmak daha pahalı. En iyi epoch validation AUC'ye göre seçiliyor.

### A7. XGBoost
Karar ağaçlarını art arda ekleyen klasik tablo modeli: 300 ağaç, derinlik 3, ayarsız. Bizim referans modelimiz.

### A8. AUC
Rastgele bir kanser ve rastgele bir iyi huylu nodül seçtiğinde modelin kansere daha yüksek olasılık verme ihtimali. 0,5 yazı-tura, 1,0 mükemmel. Eşikten bağımsız, modelin **sıralama** kalitesini ölçer.

### A9. Split'ler
| split | nodül | görevi |
|---|---|---|
| train | 2.095 | eğitim ve bütün ön işleme |
| validation | 436 | en iyi epoch'u seçmek |
| calibration | 524 | sadece conformal eşikleri |
| test | 438 | sadece raporlama |

### A10. Paired bootstrap ve seed
- **Paired bootstrap:** iki modelin aynı test nodülleri üzerindeki AUC farkını, nodülleri 2.000 kez yeniden çekerek hesaplıyoruz. %95 aralık sıfırı içermiyorsa fark anlamlı.
- **Bootstrap'ın sınırı:** modeli sabit kabul ediyor. Yeniden eğitimin getirdiği rastgeleliği kapsamıyor.
- **Seed:** eğitimdeki rastgeleliğin başlangıç noktası. Bizde seed tek başına AUC'yi 0,03–0,04 oynatıyor. Bu yüzden ana karşılaştırmayı 3 seed ile yaptık.

---

## Kısım B: Conformal kavramları

### B1. Exchangeability (değiştirilebilirlik)
Conformal yöntemlerin tek varsayımı: calibration ve test nodülleri aynı "torbadan" rastgele çekilmiş olmalı. Böylece calibration'da ölçtüğümüz hata oranı gelecekteki nodüller için de geçerli olur.

Test etmek için "two-sample classifier" kullanılır: bir model iki seti birbirinden ayırt etmeye çalışır. AUC 0,5 civarındaysa ayırt edilemiyor demektir, exchangeability makul. 0,5'ten belirgin şekilde uzaksa (bizde 0,63) setler farklı dağılımlardan geliyor demektir.

### B2. Üçlü kural
- P ≤ t ise **otomatik iyi huylu**,
- P ≥ u ise **otomatik kötü huylu**,
- arada kalırsa **doktora gönder**.

### B3. İki farklı hata oranı (en önemli kavram)
Örnek: 1.000 nodül, 350 kanser. Kural 200 nodüle "iyi huylu" diyor ve bunların 20'si kanser.
- **Kaçırma oranı:** 20 / 350 = %5,7. Payda bütün kanserler.
- **Grup içi kanser oranı (R1):** 20 / 200 = %10. Payda "iyi huylu" denen nodüller.

Hastaya "iyi huylu" denince onu ilgilendiren soru ikincisi. Kanser oranı yüksek olduğunda (%35) ikincisini sağlamak çok daha zor.

### B4. LTT (Learn then Test)
1. Aday eşikler önceden belirlenir.
2. Her aday için şu hipotez test edilir: **H0: "bu eşikte gerçek risk α'dan büyük."**
3. Eşiğin altında n nodül ve aralarında k kanser varsa, p-değeri = P(Binom(n, α) ≤ k).
4. p ≤ δ = 0,05 ise eşik sertifikalanır.
5. **Fixed sequence:** adaylar önceden belirlenmiş sırayla test edilir ve ilk başarısızlıkta durulur. Sıra calibration verisine bakmadan belirlendiği için çoklu test düzeltmesi gerekmez.

**Garanti tipi:** "%95 olasılıkla, seçilen eşiğin gerçek riski α'nın altında." İki risk birlikte %90 olasılıkla tutar (1 − 2δ, union bound).

**Binom testi neden geçerli?** Eşiğin altına düşen n nodül, "eşiğin altındaki nodüller" popülasyonundan rastgele bir örnek. İçlerindeki kanser sayısı Binom(n, gerçek risk) dağılır.

**%5 hedefi için gereken minimum grup büyüklükleri:**

| gruptaki kanser sayısı | 0 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| gereken nodül sayısı | 59 | 93 | 124 | 153 | 208 |

%20 hedefi için bu sayılar çok daha küçük: 14, 22, 30, 37, 50.

### B5. Split conformal prediction (CP)
Her nodül için bir etiket kümesi üretilir: {iyi huylu}, {kötü huylu} veya {ikisi}. Tek etiketli küme otomatik karar, iki etiketli küme "doktora gönder" demek.
- **Marginal:** bütün nodüllerin en az %90'ında doğru etiket kümede.
- **Mondrian:** her sınıf ayrı kalibre edilir. Biz kanserler için %95 kullandık, yani kanserlerin en fazla %5'i tek başına "iyi huylu" etiketi alır. Bu bir kaçırma oranı garantisi.
- **Garanti tipi:** "ortalamada". Tek bir test setinde gerçekleşen değer biraz altında da üstünde de çıkabilir.

### B6. Kaçırma oranı LTT'si
LTT ile aynı test, ama payda bütün calibration kanserleri. "Kanserlerin en fazla %5'i otomatik iyi huylu grubuna düşer" garantisini %95 olasılıkla veriyor.

---

## Kısım C: Kronolojik süreç

### C0. Başlangıç (30 Eylül): tasarım
- **Neden?** Ra et al. (2025), radyomik özellikleri metne çevirip LLM'e verince klasik modellerden daha iyi sonuç aldıklarını söylüyordu. Biz bunu tiroide uyarlayıp üstüne garantili bir triage katmanı eklemek istedik.
- **Tasarım:**
  - LTT ile R1 ≤ %5 ve R2 ≤ %20.
  - Belirsizlik sinyali olarak "pertürbasyon oyları": aynı nodülü modele birkaç kez, hafif değiştirilmiş prompt'larla soruyoruz ve kaç kez "kötü huylu" dediğini sayıyoruz.
  - Beklenti: model yanlış bildiği nodüllerde kararsız olacak, oylar bölünecek ve bu nodüller doktora gidecek.

### C1. İlk triage çalıştırmaları (2 Ekim)
- **Ne yaptık?** Conformal katmanı test edilmiş bir Python paketi olarak yazdık (o gün 21 test). Tasarım notundaki örnek hesap birebir tuttu. Resmi TN3K split'i, 8 özellik, Qwen2.5-1.5B kullandık. Model test AUC'si 0,672.
- **Run 1, karışık pertürbasyonlar** (özellik sırasını karıştırma, özellik atma, farklı cümle kalıpları):
  - Farklı kalıplar hiçbir zaman "kötü huylu" cevabı vermedi.
  - Sadece sırayı karıştırmak cevapların %29'unu değiştirdi.
  - Hiçbir nodül 10 oyda 6'dan fazla "kötü huylu" almadı.
  - **Ders:** pertürbasyonlar belirsizliği değil, eğitim formatından ne kadar uzaklaşıldığını ölçtü.
- **Run 2, formatı koruyan pertürbasyonlar** (değerlere küçük gürültü, tek özellik atma):
  - Model formatta kaldı, cevapların sadece %5–12'si değişti.
  - Ama oylar, doğru ve yanlış tahminleri softmax olasılığından daha iyi ayıramadı.
  - **"Emin ama yanlış":** 0/10 oy alan nodüllerin %19'u kanserdi. Hataların %56–69'unda model 10 sorgunun hepsinde aynı yanlış cevabı verdi.
- **LTT:** hiçbir şey sertifikalanmadı.
- **Split CP:** test'teki otomatik iyi huylu grubunda %23 kanser vardı.
- **Veri sorunu:** calibration ile resmi test ayırt edilebiliyordu (AUC 0,63, p = 0,002). Exchangeability bozuktu.
- **Dersler:**
  1. Sorun yöntemde değil, modelde: hatalar kararlı olduğu için hiçbir tutarlılık sinyali onları yakalayamaz.
  2. Softmax olasılığı en az oylar kadar iyi bir skor. Pertürbasyon yaklaşımı bırakıldı.
  3. Garantileri test edebilmek için önce split düzeltilmeli.

### C2. Hata analizi ve daha gevşek bir garanti (7 Ekim)
- **Ne yaptık?** Run 2'nin çıktılarını yeni eğitim yapmadan yeniden inceledik.
- **Bulgu 1:** LTT'yi engelleyen hatalar zaten pertürbasyonsuz prompt'ta vardı. Pertürbasyonlar ne yarattı ne gösterdi.
- **Bulgu 2, kaçırma oranı denemesi:** "kanserlerin ≤ %5'i otomatik iyi huylu" garantisi sertifikalandı, ama nodüllerin sadece yaklaşık %5'i için. %10 hedefinde ise resmi test'te ihlal edildi (%15 kaçırıldı), çünkü split exchangeable değildi.
- **Ders:** iki farklı garanti türü var. Kaçırma oranı program düzeyinde, R1 ise hasta düzeyinde. Bu ayrım tezin geri kalanının omurgası oldu.

### C3. Model, referans, split ve veri denetimi (8 Ekim)
- **Büyük model (resmi split):** Qwen2.5-7B, 1.5B ile aynı sonucu verdi (0,669 ve 0,667). Yeniden eğitilen 1.5B ilk modelle tutarlıydı (Spearman 0,97).
- **XGBoost referansı:** aynı özelliklerle LLM'i yakaladı ya da geçti (8 özellikle 0,696, 16 ile 0,709).
- **Pooled split:** 3.493 görüntüyü birleştirip etiket ve kaynağa göre tabakalı olarak yeniden böldük. Exchangeability testi 0,43'e düştü, yani setler artık ayırt edilemiyor. Bu split'te daha fazla özellik test'te de kazandırmaya başladı (XGBoost, 67 özellik: 0,792). Önceki "fazla özellik işe yaramıyor" bulgusu kaymadan kaynaklanıyormuş.
- **Görüntü denetimi, dört sorun:**
  1. Doku özellikleri nodül içinde sadece yaklaşık 9 gri seviyeyle hesaplanıyordu (bin width 25).
  2. Şekil özellikleri piksel cinsindendi, ama görüntü boyutları farklıydı. Resmi test görüntüleri daha büyük, "kaymanın" önemli bir kısmı buradan geliyordu.
  3. 298 maske birden fazla parçadan oluşuyordu.
  4. 356 maske görüntü kenarına değiyordu.
- **Kayıt altına alınan sınırlama:** split görüntü düzeyinde, hasta düzeyinde değil.

### C4. Yeni özellikler ve ilk LLM run'ları (8–9 Ekim)
- **v2 özellikleri:** 64 gri seviye, görüntüye göre normalize edilmiş boyut, 7 özellik ailesi (102 özellik), maskenin sadece en büyük parçası. Tek parçalı maskelerde v1 ile birebir aynı değerleri veriyor.
- **Yol üstünde bulunan iki hata:**
  - mRMR, neredeyse aynı iki özelliği birlikte seçiyordu (2D'de piksel yüzeyi ve mesh yüzeyi aynı sayı). Kopya filtresi eklendi.
  - Tahminler bf16 hassasiyetinde hesaplanıyordu ve olasılıklar yuvarlanıyordu: 524 nodülde sadece 195 farklı değer. Tam hassasiyete geçildi.
- **Sonuçlar (test AUC):**
  - v1 → v2: 0,699 → 0,754, anlamlı (+0,05).
  - Değerin z-score olduğunu söylemek bir şey değiştirmedi (0,759). Persentil kullanmak kötüleştirdi (0,734). Yani model sayıların büyüklüğünü kullanıyor.
  - Qwen2.5-7B: 0,713. Eğitim kararsızdı, duyarlılık epoch'tan epoch'a 0 ile 0,93 arasında zıpladı.
  - XGBoost aynı 16 özellikle 0,805. LLM yaklaşık 0,05 geride.

### C5. Ra et al.'ın ayarları, açıklamalı prompt'lar, split CP (9 Ekim)
- **Llama-2-7B, Ra et al.'ın ayarlarıyla:** 0,731, 1.5B'nin altında. Run başına 182 dakika.
- **"Rich" prompt'lar** (z-score + persentil + "ortalamadan düşük, yani daha az yuvarlak" gibi bir açıklama): 0,744, katkı yok.
- **Split CP her run'a eklendi:** Mondrian ortalamada çalışıyor, ama otomatik iyi huylu grubunda %10–17 kanser kalıyor.
- **LTT'de keşif bulgusu:** planlanan sıra en sıkı eşikten başlıyor ve orada çok az nodül var, bu yüzden hemen duruyor. Sırayı validation p-değerlerine göre kurunca bir run'da otomatik kötü huylu sertifikalandı. Keşif olarak not edildi.

### C6. TI-RADS'tan esinlenen özellikler (9–10 Ekim)
- **Neden?** PyRadiomics nodülü tek başına inceliyor. TI-RADS'ın bazı işaretleri ise yönelime (dik mi, yatık mı) ve nodülün çevresiyle kontrastına bağlı.
- **Ne yaptık?** Görüntü ve maskeden 6 özellik yazdık. Ayarları etiketlere bakmadan sabitledik.

| özellik | ne ölçüyor | sonuç |
|---|---|---|
| taller-than-wide | dikey/yatay uzunluk | beklendiği gibi |
| ekojenite oranı | nodül/çevre parlaklığı | ters yönde, sıvıyı ölçüyor |
| parlak nokta yoğunluğu | mikrokalsifikasyon yaklaşımı | sinyal yok |
| solidity | alan / dışbükey zarf alanı | beklendiği gibi |
| kenar keskinliği | sınırdaki gradyan | belirsiz |
| sıvı oranı | çok karanlık piksel payı | beklendiği gibi |

- **Sonuç:** XGBoost 0,805 → 0,831. LLM 0,754 → 0,804, şimdiye kadarki en büyük kazanç.
- **Yan denemeler:**
  - 7. özellik (sadece katı kısımda ekojenite): yönü düzeltti ama kazandırmadı.
  - Rich format + TI-RADS: decimal ile eşit, ama 4 kat daha yavaş.

### C7. Büyük modele adil bir şans (11 Ekim)
- **Neden?** 7B'nin kararsızlığı LR'nin fazla yüksek olmasından olabilirdi.
- **Ne yaptık?** Qwen2.5-7B'yi LR 5e-5 ve TI-RADS özellikleriyle yeniden eğittik.
- **Sonuç:** eğitim daha sakindi, ama test 0,693. TI-RADS özellikleri 7B'ye hiçbir şey kazandırmadı. Üç büyük model denemesi aynı yönde.

### C8. "Sonuç gerçek mi?" Seed'ler, birleştirme, conformal kontroller (12 Ekim)
- **Seed tekrarları:**

| | seed 42 | seed 1 | seed 2 | ortalama |
|---|---|---|---|---|
| TI-RADS'lı | 0,804 | 0,768 | 0,783 | **0,785 ± 0,018** |
| TI-RADS'sız | 0,754 | 0,712 | 0,713 | 0,726 ± 0,024 |

  Aynı seed'lerle kazanç ortalama +0,059, en küçüğü +0,050, her çiftte aralık sıfırın üstünde. Seed 42 iki tarafta da şanslı üst uçtu.
- **Hibrit (birleştirme):** aynı 22 özellikle XGBoost 0,835. İki modeli birleştiren stacking, LLM'e −0,04, XGBoost'a 0,99 ağırlık verdi. İki modelin skorları 0,87 korelasyonlu. **LLM, XGBoost'un bilmediği bir şey bilmiyor.**
- **Conformal kontroller:**
  - LTT, XGBoost'ta bile hiçbir şeyi sertifikalamadı. En temiz %15'lik grubunda 78 nodülde 2 kanser var, sertifika için 2 kanserle 124 nodül gerekiyor.
  - Mondrian'da kanser kapsamı 16 run'ın 12'sinde 0,95'in altındaydı. Calibration ve test'i 4.000 kez yeniden bölünce ortalama 0,950–0,951 çıktı. **Garanti tutuyor, bizim bölünme şanssızdı.**
  - Mondrian'ın otomatik iyi huylu grubunda beklenen kanser oranı: LLM'de yaklaşık %10, XGBoost'ta %6.

### C9. Geçebilen bir tasarım ve son bir özellik denemesi (13 Ekim)
- **Önceden sabitlenmiş LTT tasarımı.** Çalıştırmadan önce GitHub'a commit'ledik.
  - Adaylar: validation skorlarının yüzdelikleri (%5–30).
  - Sıra: validation p-değerlerine göre.
  - Test: calibration'da fixed sequence.
  - Sonuç (2.000 yeniden bölme):
    - Otomatik kötü huylu (%20): LLM'de bölünmelerin **%99'unda** sertifikalandı. Nodüllerin yaklaşık %8'i otomatik, bunların %6–7'si iyi huylu. Eski tasarımla bu oran %1'di.
    - Otomatik iyi huylu (%5): **hiçbir modelde, hiçbir zaman** sertifikalanmadı.
- **Peritümöral (nodül çevresi) özellikler.** Nodülün etrafındaki halkada 93 radyomik özellik çıkardık. Kural önceden belliydi: XGBoost'ta calibration kazancı yoksa LLM'e geçmiyoruz. Calibration +0,002, aralık [−0,009, +0,014]: kazanç yok. Halka tek başına 0,69 AUC veriyor ama bilgisi zaten mevcut özelliklerde var.

---

## Kısım D: Sonuç ve büyük resim

### D1. Araştırma sorularının cevapları
- **RQ1, LLM sınıflandırabilir mi?** Evet, orta düzeyde: 0,785 ± 0,018. Ama aynı özelliklerle XGBoost'un yaklaşık 0,05 gerisinde ve ona bilgi eklemiyor.
- **RQ2, neler önemli?** Özelliklerin içeriği: v2 ile +0,05, TI-RADS ile +0,06. Prompt dili, format, model boyutu ve halka özellikleri önemli değil.
- **RQ3, LTT garanti verebilir mi?**
  - Oylar işe yaramıyor.
  - Otomatik iyi huylu %5: hiçbir modelle mümkün değil.
  - Otomatik kötü huylu %20: yeni tasarımla mümkün.
  - Kaçırma oranı garantisi: mümkün, ama otomatik iyi huylu grubunda yaklaşık %10 kanser bırakıyor.

### D2. Ana mesaj
**"Hangi garanti ne zaman mümkün ve ne işe yarar."** LTT dürüst davrandı: kanıtın desteklemediği garantiyi vermedi.
- 524 calibration nodülü ve %35 kanser oranıyla, her "iyi huylu" kararı için %5 garantisi kabaca AUC 0,9 gerektiriyor.
- Kaçırma oranı garantisi şimdi bile mümkün, ama bıraktığı grup TI-RADS TR4 düzeyinde riskli. Bu düzeyde biyopsi hâlâ öneriliyor.
- **Geçerli garanti, güvenli karar demek değil.**

### D3. Neden daha büyük model ya da daha iyi prompt çözüm değil? (Bilgi tavanı)
LLM, XGBoost'un gördüğü 22 sayıyı görüyor. Bu sayılardaki bilginin tavanı yaklaşık 0,835. Büyük model ya da iyi prompt en fazla bu tavana yaklaştırır, tavanı yükseltemez. Tavanı yükselten tek şey yeni bilgi oldu: TI-RADS özellikleri.

### D4. Yöntemsel dersler
1. Garantiye güvenmeden önce exchangeability'yi kontrol et.
2. Özellikleri denetle: v2, herhangi bir model değişikliği kadar kazandırdı.
3. Seed'lerle tekrarla: tek run'lar 0,03–0,05 oynuyor.
4. Düzgün bir tablo modeliyle karşılaştır ve "ek bilgi var mı?" sorusunu birleştirmeyle test et.
5. Karar kurallarını sonuçları görmeden önce sabitle.

### D5. Sınırlamalar
- Tek veri seti.
- Hasta kimliği olmadan, görüntü düzeyinde split.
- Tek 2D görüntü, uzman maskesi.
- Sezgisel (heuristic) TI-RADS özellikleri.
- Ayarlanmamış LLM hiperparametreleri.
- 524 nodüllük tek bir calibration seti.
- Bazı seçimler calibration ya da test sonuçlarına bakılarak yapıldı.
- Bootstrap, seed ve hasta kümelenmesini kapsamıyor.
- %35 kanser oranı taramaya taşınmaz.

### D6. Sonraki adım seçenekleri
1. Yeni LTT tasarımını yeni veride doğrulamak: yeniden eğitim ya da ThyroidXL.
2. Görüntünün kendisini kullanmak (CNN ya da vision-language model). Otomatik iyi huylu garantisine giden tek gerçekçi yol, ama kapsamı değiştirir.
3. Tezi yazmak: sınıflandırıcıyla ilgili deney soruları cevaplandı.

---

## Kısım E: Ezberlenecek sayılar

| ne | sayı |
|---|---|
| veri | 3.493 görüntü, 2.421 hasta, %35 kanser |
| split | 2.095 / 436 / 524 / 438 |
| ilk model (resmi split) | test 0,672 |
| v1 → v2 | 0,699 → 0,754 |
| TI-RADS ile LLM (3 seed) | 0,785 ± 0,018 |
| TI-RADS kazancı (eşleştirilmiş) | +0,059 (en az +0,050) |
| XGBoost, aynı 22 özellik | 0,835 |
| stacking ağırlıkları | XGBoost 0,99, LLM −0,04 |
| exchangeability | resmi 0,63 → pooled 0,43 |
| %5 için gereken | 1 kanser: 93 nodül, 2: 124 |
| Mondrian yeniden bölme | kapsam 0,951 |
| otomatik iyi huylu grubundaki kanser | LLM ~%10, XGBoost ~%6 |
| yeni LTT tasarımı (otomatik kötü huylu) | LLM'de %99 sertifika |
| 7B denemeleri | 0,713 / 0,693, Llama 0,731 |
| halka özellikleri | calibration +0,002, kazanç yok |

---

## Kısım F: Sözlük

- **ACR TI-RADS:** tiroid nodülleri için ultrason puanlama sistemi. TR1–TR5, risk arttıkça yükselir.
- **AUC:** sıralama kalitesi; 0,5 rastgele, 1,0 mükemmel.
- **Bootstrap:** veriden tekrar tekrar örnek çekerek belirsizliği tahmin etme.
- **Calibration seti:** sadece conformal eşiklerini belirlemek için ayrılan veri.
- **Classification head:** LLM'in sonuna eklenen, iki sınıf skoru veren doğrusal katman.
- **Exchangeability:** veri noktalarının sırasının önemli olmaması, kabaca "aynı torbadan çekilmiş olmak".
- **Fixed-sequence testing:** önceden sabitlenmiş sırayla test edip ilk başarısızlıkta durmak.
- **FNA:** ince iğne aspirasyonu biyopsisi.
- **LoRA / QLoRA:** büyük modeli küçük ek matrislerle eğitme; QLoRA'da donmuş model 4-bit.
- **LTT (Learn then Test):** eşik seçimini hipotez testine çeviren, yüksek olasılıklı risk garantisi veren yöntem.
- **Miss rate (kaçırma oranı):** bütün kanserlerin otomatik iyi huylu grubuna düşen payı.
- **Mondrian CP:** her sınıfı ayrı kalibre eden split conformal prediction.
- **mRMR:** etiketle ilişkili ama birbirine benzemeyen özellikleri seçme yöntemi.
- **Peritümöral:** nodülün çevresindeki doku.
- **Seed:** rastgeleliğin başlangıç değeri.
- **Stacking:** iki modelin tahminlerini girdi alan üçüncü bir model.
- **Two-sample test:** iki veri setinin aynı dağılımdan gelip gelmediğini test etme.
- **z-score:** (değer − ortalama) / standart sapma.

---

## Kısım G: Kendini sınama soruları

1. Pertürbasyon oyları neden işe yaramadı? İki ayrı sebep söyle.
2. Resmi split neden conformal için uygun değildi, nasıl düzelttik?
3. v2 özellikleri v1'den hangi dört noktada farklı?
4. Persentil formatının kötü sonuç vermesi neyi gösteriyor?
5. TI-RADS özelliklerinden hangisi ters yönde çıktı, neden?
6. "0,804" yerine neden "0,785 ± 0,018" raporluyoruz?
7. Hibrit analiz LLM hakkında neyi gösterdi? Stacking ağırlıkları ne?
8. %5 hedefiyle, grupta 2 kanser varken kaç nodül gerekir? Bu sayı nereden geliyor?
9. Kaçırma oranı ile R1 arasındaki fark ne? Sayısal bir örnekle anlat.
10. Mondrian garantisi "ortalamada", LTT garantisi "%95 olasılıkla". Fark ne?
11. Mondrian kapsamı 0,95'in altında çıkınca neden "garanti bozuk" demedik?
12. Yeni LTT tasarımı neyi çözdü, neyi çözemedi? Neden?
13. Yeni tasarımı sonradan bulduk. Bunu nasıl dürüstçe sunuyoruz?
14. Halka özellikleri neden işe yaramadı, TI-RADS özellikleri neden yaradı?
15. Daha büyük bir LLM neden bizi otomatik iyi huylu garantisine götürmez?
16. "Geçerli garanti, güvenli karar demek değil" cümlesini bir örnekle açıkla.
