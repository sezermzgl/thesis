# Uçtan uca çalışma materyali: tezde yaptığımız her şey (30 Eylül – 13 Ekim 2026)

*Toplantıda anlatacakların için ayrı bir doküman var: `toplanti_anlatimi_TR.md`, İngilizce süreç raporunun bölüm bölüm Türkçesi. Bu doküman ise öğrenmek için.*

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

**Dikkat:** ikisi de false negative'le (kanser olduğu hâlde "iyi huylu" demek) ilgili, sadece paydaları farklı:
- kaçırma oranı = FN / tüm kanserler = 1 − duyarlılık (false negative rate),
- R1 = FN / "iyi huylu" denenler = 1 − NPV (false omission rate).

False positive (iyi huyluya "kötü huylu" demek) ise R2'nin konusu: R2 = FP / "kötü huylu" denenler = 1 − PPV.

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

#### B4b. LTT'yi sayılarla anlamak

**LTT tablolarını nasıl okumalı?** Otomatik iyi huylu grupta *hata*, iyi huylu dediğimiz ama aslında kanser olan nodül demek. Garanti **gerçek oran** hakkında: bu kuralın gelecekte iyi huylu diyeceği bütün nodüller arasındaki kanser oranı. Bunu hiçbir zaman tam bilemeyiz. Calibration grubu bize sadece **gözlenen oranı**, yani bir örneği veriyor. LTT şunu soruyor: *gerçek oran %5 olsaydı, gördüğümüz kadar az kanser görme ihtimalimiz ne olurdu?* Bu ihtimal δ = %5'in altındaysa, gerçek oranın %5'in altında olduğu sertifikalanıyor.

**Örnek 1: grupta hiç kanser yok.** Gerçek oran %5 ise her nodülün iyi huylu olma ihtimali 0,95.

| grup | gerçek oran %5 olsaydı hepsinin iyi huylu çıkma ihtimali | sonuç |
|---|---|---|
| 30 nodül, 0 kanser | 0,95³⁰ ≈ %21 | sık olur, bir şey kanıtlamaz |
| 59 nodül, 0 kanser | 0,95⁵⁹ ≈ %4,8 | %5'in altında: **sertifika** |

**Örnek 2: grupta kanser var.** Her kanser, gereken kanıtı büyütüyor:

| gruptaki kanser | sertifika için en küçük grup | o durumda gözlenen oran |
|---|---|---|
| 0 | 59 | %0 |
| 1 | 93 | %1,1 |
| 2 | 124 | %1,6 |
| 3 | 153 | %2,0 |

Yani gözlenen oranın %5'in biraz altında olması yetmiyor, %0–2 civarında olması gerekiyor. Aradaki fark, küçük örneğin belirsizliği. Otomatik kötü huylu tarafta hedef %20, bu yüzden 14 temiz nodül yetiyor ve o taraf geçebiliyor. **Çorba benzetmesi:** bir kaşık tuzsuz diye bütün tencere tuzsuz değildir. Ne kadar çok kaşık tadarsan tencere hakkında o kadar emin olursun.

**En temiz kuyruk.** Her model calibration'daki 524 nodülün her birine bir kanser olasılığı veriyor. Küçükten büyüğe sıralayınca ilk %10 (52 nodül), modelin iyi huylu olduğundan en emin olduğu nodüller. Eşik oraya konsaydı bunlar otomatik iyi huylu olacaktı. %10–20'lik dilimlere bakmamızın sebebi: %5'lik dilim (26 nodül) hiç kanser olmasa bile sertifika alamaz (59 gerekiyor), daha geniş dilimler ise daha çok kanser içeriyor. LTT'nin kendisi bu dilimleri kullanmıyor; dilimler sadece neden geçemediğini gösteriyor.

**Bizim veride.** XGBoost'un en temiz 52 nodülünde 2 kanser var: gözlenen oran %3,8, %5'in altında. Ama gerçek oran %5 olsaydı bile 52 nodülde 2 ya da daha az kanser görme ihtimali yaklaşık %51 (binom hesabı: X ~ Binom(52; 0,05) için P(X ≤ 2); LTT'nin δ = 0,05 ile karşılaştırdığı p-değeri bu). Yani bu bir kanıt değil. 2 kanserle 124 nodül gerekirdi. Grubu 104'e büyütünce üçüncü kanser giriyor ve 153 gerekiyor. LLM'in kuyruğu (52'de 4, yaklaşık %8) gözlenen oran olarak bile %5'in üstünde.

> 🗣 **Anlatım notu:** "%5'in altında olduğunu %95 güvenle söyleyebilmek için en temiz grupta kanser oranının %0–2 civarında olması ve grubun 60–150 nodül büyüklüğünde olması gerekiyor. Örneğin 59 nodülün hepsi temizse yeter; 30 temiz nodül ise bir şey kanıtlamaz, çünkü oran %5 olsa bile bu her beş denemeden birinde olur. XGBoost'un en temiz grubu oranı tutturuyor ama çok küçük; LLM'inki oranı da tutturmuyor."

**Binom testi, permütasyon testi değil.** p-değeri doğrudan binom dağılımından hesaplanıyor: gerçek oran %5 ise kanser sayısı X ~ Binom(n; 0,05). Etiketleri karıştırmak (permütasyon) burada yok.

**Ne olsaydı R1 geçerdi?** (1) Daha iyi bir model: en temiz grupta oran gerçekten %0–2 olmalı, kabaca AUC 0,9 gerekir. (2) Çok daha fazla calibration verisi, ama sadece oranı zaten %5'in altında olan model için: XGBoost'un %3,8'iyle sertifika için iyi huylu grubun yaklaşık 880 nodül olması gerekirdi, yani yaklaşık 9.000 calibration nodülü (bizde 524). LLM'in oranı %5'in üstünde, ne kadar veri eklense de geçmez. (3) Daha gevşek bir hedef (kaçırma oranı): geçiyor ama grupta %6–10 kanser kalıyor.

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
| sinyal (run 2) | cal'da değişen cevap | doğru/yanlış ayırma AUC'si (cal / test) |
|---|---|---|
| softmax güveni | – | 0,629 / 0,577 |
| gürültü oyları | %10,6 | 0,579 / 0,533 |
| özellik atma oyları | %11,7 | 0,652 / 0,571 |

| LTT (calibration) | en temiz otomatik iyi huylu grup | en temiz otomatik kötü huylu grup |
|---|---|---|
| gürültü oyları | v = 0: 165 nodül, %19,4 kanser | v = 10: 116 nodül, %45,7 iyi huylu |
| atma oyları | v = 0: 175 nodül, %20,0 kanser | v = 9: 58 nodül, %31,0 iyi huylu |
| softmax | sertifika yok | sertifika yok |

- **LTT:** hiçbir şey sertifikalanmadı.
- **Split CP:** test'teki otomatik iyi huylu grubunda %23 kanser vardı.
- **Model neye dayanıyordu?** Her özelliği tek tek çıkarıp bakınca en büyük etki MajorAxisLength'ten geliyordu: çıkarınca cevapların %28'i değişti. Bu, sonradan "piksel cinsinden boyut" sorununun ne kadar önemli olduğunu gösterdi.
- **Veri sorunu:** calibration ile resmi test ayırt edilebiliyordu (AUC 0,63, p = 0,002). Exchangeability bozuktu.
- **Dersler:**
  1. Sorun yöntemde değil, modelde: hatalar kararlı olduğu için hiçbir tutarlılık sinyali onları yakalayamaz.
  2. Softmax olasılığı en az oylar kadar iyi bir skor. Pertürbasyon yaklaşımı bırakıldı.
  3. Garantileri test edebilmek için önce split düzeltilmeli.

### C2. Hata analizi ve daha gevşek bir garanti (7 Ekim)
- **Ne yaptık?** Run 2'nin çıktılarını yeni eğitim yapmadan yeniden inceledik.
- **Bulgu 1:** LTT'yi engelleyen hatalar zaten pertürbasyonsuz prompt'ta vardı. Pertürbasyonlar ne yarattı ne gösterdi.
- **Bulgu 2, kaçırma oranı denemesi:** "kanserlerin ≤ %5'i otomatik iyi huylu" garantisi sertifikalandı, ama nodüllerin sadece yaklaşık %5'i için. %10 hedefinde ise resmi test'te ihlal edildi (%15 kaçırıldı), çünkü split exchangeable değildi.
| kaçırma oranı kuralı (resmi split) | eşik | test'te otomatik | test'te kaçırılan |
|---|---|---|---|
| ≤ %5 | P ≤ 0,22 | %7,2 | %5,1 |
| ≤ %10 | P ≤ 0,28 | %24,4 | %15,3 (ihlal) |

- **Ders:** iki farklı garanti türü var. Kaçırma oranı program düzeyinde, R1 ise hasta düzeyinde. Bu ayrım tezin geri kalanının omurgası oldu.

### C3. Model, referans, split ve veri denetimi (8 Ekim)
- **Büyük model (resmi split):** Qwen2.5-7B, 1.5B ile aynı sonucu verdi (0,669 ve 0,667). Yeniden eğitilen 1.5B ilk modelle tutarlıydı (Spearman 0,97).
- **XGBoost referansı:** aynı özelliklerle LLM'i yakaladı ya da geçti.

| resmi split, test AUC | 8 özellik | 16 özellik |
|---|---|---|
| Qwen2.5-1.5B | 0,667 | – |
| Qwen2.5-7B | 0,669 | 0,666 |
| XGBoost | 0,696 | 0,709 |

- **Teknik düzeltme:** gradient accumulation kullanırken kayıp fonksiyonu yanlış ölçekleniyordu. Düzeltildi. Yeniden eğitilen model eskisiyle tutarlı çıktı (Spearman 0,97), yani eski sonuçlar bundan etkilenmemişti.
- **Pooled split:** 3.493 görüntüyü birleştirip etiket ve kaynağa göre tabakalı olarak yeniden böldük.

| two-sample test | AUC |
|---|---|
| validation – calibration (kontrol) | 0,51 |
| calibration – resmi test | **0,63** (p = 0,002) |
| calibration – test, pooled | **0,43** | Bu split'te daha fazla özellik test'te de kazandırmaya başladı (XGBoost, 67 özellik: 0,792). Önceki "fazla özellik işe yaramıyor" bulgusu kaymadan kaynaklanıyormuş.
- **Görüntü denetimi, dört sorun:**
  1. Doku özellikleri nodül içinde sadece yaklaşık 9 gri seviyeyle hesaplanıyordu (bin width 25).
  2. Şekil özellikleri piksel cinsindendi, ama görüntü boyutları farklıydı. Resmi test görüntüleri daha büyük, "kaymanın" önemli bir kısmı buradan geliyordu.
  3. 298 maske birden fazla parçadan oluşuyordu.
  4. 356 maske görüntü kenarına değiyordu.
- **Denetimin yan bulguları:**
  - Görüntü genişliği tek başına kötü huyluluğu zayıf da olsa tahmin ediyor (AUC 0,57). Bu, çekimle ilgili bir karıştırıcı (confounder) olabileceğini düşündürüyor.
  - Çok parçalı ve kenara değen maskelerde kötü huylu oranı daha düşük (%22, genel %36).
- **Kayıt altına alınan sınırlama:** split görüntü düzeyinde, hasta düzeyinde değil. TN3K'da hastadan 1.072 fazla görüntü var. En kötü senaryoda test görüntülerinin yaklaşık %39'unun train'de aynı hastadan bir görüntüsü olabilir. Gerçek oran bilinmiyor.

### C4. Yeni özellikler ve ilk LLM run'ları (8–9 Ekim)
- **v2 özellikleri:** 64 gri seviye, görüntüye göre normalize edilmiş boyut, PyRadiomics'in yedi özellik ailesi (102 özellik: şekil 9, first-order 18, GLCM 24, GLRLM 16, GLSZM 16, GLDM 14, NGTDM 5. Şekil geometriyi, first-order parlaklık dağılımını, beş doku ailesi de gri tonların diziliş biçimini ölçüyor: komşu piksel çiftleri, düz çizgideki tekrarlar, bitişik bölgeler, komşu benzerliği, komşu ortalamasından fark), maskenin sadece en büyük parçası. Tek parçalı maskelerde v1 ile birebir aynı değerleri veriyor.
- **Yol üstünde bulunan iki hata:**
  - mRMR, neredeyse aynı iki özelliği birlikte seçiyordu (2D'de piksel yüzeyi ve mesh yüzeyi aynı sayı). Kopya filtresi eklendi.
  - Tahminler bf16 hassasiyetinde hesaplanıyordu ve olasılıklar yuvarlanıyordu: 524 nodülde sadece 195 farklı değer. Tam hassasiyete geçildi.
- **Sonuçlar:**

| run | değişiklik | val | cal | test |
|---|---|---|---|---|
| B | v1 özellikler | 0,709 | 0,705 | 0,699 |
| D | v2 özellikler (referans) | 0,750 | 0,752 | **0,754** |
| E | z-score ifadesi | 0,729 | 0,756 | 0,759 |
| F | persentil | 0,709 | 0,713 | 0,734 |
| C | Qwen2.5-7B | 0,708 | 0,705 | 0,713 |
| – | XGBoost, aynı 16 özellik | 0,799 | 0,823 | **0,805** |

  - v1 → v2: +0,055, aralık [+0,013, +0,089], yani anlamlı.
  - z-score demek bir şey değiştirmedi, persentil kötüleştirdi. Model sayıların büyüklüğünü kullanıyor: persentil, z = 2 ile z = 5'i neredeyse aynı yapıyor (98 ve 100).
  - 7B kararsız eğitildi: duyarlılık 0,00 → 0,01 → 0,93 → 0,04 → 0,62.
  - XGBoost yaklaşık 0,05 önde.

### C5. Ra et al.'ın ayarları, açıklamalı prompt'lar, split CP (9 Ekim)
| run | değişiklik | val | cal | test | dakika |
|---|---|---|---|---|---|
| D | referans | 0,750 | 0,752 | 0,754 | 26 |
| G | Llama-2-7B, Ra et al. ayarları | 0,705 | 0,709 | 0,731 | 182 |
| H | rich prompt | 0,732 | 0,711 | 0,744 | 83 |

- **Llama-2-7B, Ra et al.'ın ayarlarıyla:** 1.5B'nin altında, 7 kat daha yavaş.
- **"Rich" prompt'lar** (z-score + persentil + "ortalamadan düşük, yani daha az yuvarlak" gibi bir açıklama): katkı yok. Aynı ayarlarla tekrarlanınca 0,698 çıktı, yani tek run'lar ciddi oynuyor.
- **Split CP her run'a eklendi:** Mondrian ortalamada çalışıyor, ama otomatik iyi huylu grubunda %10–17 kanser kalıyor.
- **LTT'de keşif bulgusu:** planlanan sıra en sıkı eşikten başlıyor ve orada çok az nodül var, bu yüzden hemen duruyor. Sırayı validation p-değerlerine göre kurunca bir run'da otomatik kötü huylu sertifikalandı. Keşif olarak not edildi.

### C6. TI-RADS'tan esinlenen özellikler (9–10 Ekim)
- **Neden?** PyRadiomics nodülü tek başına inceliyor. TI-RADS'ın bazı işaretleri ise yönelime (dik mi, yatık mı) ve nodülün çevresiyle kontrastına bağlı.

**Arka plan: ACR TI-RADS nedir?** (Tessler et al., 2017). Radyologlar bir tiroid nodülünü ultrasonda beş kategoride puanlıyor ve puanları topluyor. Toplam puan bir risk seviyesi veriyor: TR1 iyi huylu … TR5 çok şüpheli. Bu seviye, nodülün boyutuyla birlikte biyopsi (FNA) önerilip önerilmeyeceğini belirliyor.

| TI-RADS kategorisi | radyolog neye bakıyor | seçenekler ve puanlar | bizim özelliğimiz |
|---|---|---|---|
| yapı (composition) | kistik mi, katı mı | kistik ya da süngerimsi 0 · karışık kistik-katı 1 · katı 2 | sıvı oranı |
| ekojenite | çevredeki tiroid dokusundan koyu mu, parlak mı | anekoik 0 · hiper ya da izoekoik 1 · hipoekoik 2 · çok hipoekoik 3 | ekojenite oranı |
| şekil | enine kesitte boyu eninden uzun mu | eni boyundan geniş 0 · boyu eninden uzun 3 | taller-than-wide oranı |
| kenar (margin) | düzgün, loblu ya da düzensiz | düzgün 0 · belirsiz 0 · loblu ya da düzensiz 2 · tiroid dışına taşma 3 | solidity, kenar keskinliği |
| ekojenik odaklar | parlak noktalar, örneğin mikrokalsifikasyon | yok ya da büyük kuyruklu yıldız 0 · makrokalsifikasyon 1 · çevresel (rim) 2 · noktasal 3 (türler toplanır) | parlak nokta yoğunluğu |

Toplam puan → TR1 (0), TR2 (2), TR3 (3), TR4 (4–6), TR5 (≥ 7). Biyopsi (FNA) önerisi: TR3'te 2,5 cm'den, TR4'te 1,5 cm'den, TR5'te 1 cm'den itibaren. Bizim özelliklerimiz bu işaretlerin otomatik, sürekli (continuous) yaklaşımları. Radyoloğun puanını birebir üretmiyorlar.

**Kenar keskinliği hakkında not.** ACR TI-RADS'ta belirsiz (ill-defined) kenar 0 puan alıyor. Kenar puanı loblu ya da düzensiz dış hattan geliyor, onu da Solidity ölçüyor. Yani kenar keskinliği, ACR'nin puanladığı bir işaretin birebir karşılığı değil, kenar kategorisinden esinlenmiş ek bir ölçüm.

**Feature engineering: TI-RADS, PyRadiomics'e ne kattı?** PyRadiomics zaten şekli, parlaklığı ve dokuyu ölçüyor. Ama tasarımı gereği (a) dönmeden bağımsız (rotation-invariant), yani nodülün nasıl durduğuna bakmıyor; (b) sadece maskenin içindeki piksellere bakıyor, yani nodülü çevresindeki dokuyla hiç karşılaştırmıyor. TI-RADS'ın iki işareti tam olarak bunlara dayanıyor.

| tür | TI-RADS neye bakıyor | bizim özelliğimiz | PyRadiomics neden kaçırıyor |
|---|---|---|---|
| yeni bakış | yön | taller-than-wide oranı = maskenin dikey / yatay uzunluğu | MajorAxisLength, Elongation nodülün kendi eksenlerine göre: yatan ve dik duran elips aynı değeri alır |
| yeni bakış | çevreyle kontrast | ekojenite oranı = içerinin ortalaması / çevredeki halkanın ortalaması | first-order sadece maskenin içindeki mutlak parlaklığı görür, bu da cihazın gain ayarına bağlı |
| yeni bakış | sınır (içerisi ve dışarısı) | kenar keskinliği = sınır boyunca parlaklık değişimi / çevrenin dağınıklığı | sınırın iki tarafındaki pikseller gerekir; PyRadiomics sadece içeriyi görür |
| kısmi örtüşme | kenar düzensizliği | solidity = alan / dışbükey zarf alanı | Sphericity, PerimeterSurfaceRatio ile akraba; solidity loblanmayı doğrudan ölçer |
| kısmi örtüşme | sıvı içeriği | sıvı oranı = halkadan çok koyu piksellerin payı | first-order yüzdelikleriyle akraba, ama çevreye göreli |
| kısmi örtüşme | noktasal parlak odaklar | parlak nokta yoğunluğu = 1.000 pikselde küçük, ayrı parlak nokta sayısı | first-order max / 90. yüzdelik parlak pikseli görür, küçük ayrı noktalar halinde olup olmadığını görmez |

Yani mühendislik katkısı: PyRadiomics'te yapısal olarak olmayan iki bakış (yön ve çevre), artı PyRadiomics'in birçok özelliğe dağıttığı bilgiyi klinik tanımlı tek bir sayıda toplayan üç özellik. Bunun gerçekten yeni bilgi olduğunun kanıtı: aynı seed'lerle bu özellikleri eklemek LLM'in AUC'sini ortalama +0,059 artırdı (en az +0,050).

- **Ne yaptık?** Görüntü ve maskeden 6 özellik yazdık. Ayarları etiketlere bakmadan sabitledik.

| özellik | ne ölçüyor | sonuç |
|---|---|---|
| taller-than-wide | dikey/yatay uzunluk | beklendiği gibi |
| ekojenite oranı | nodül/çevre parlaklığı | ters yönde, sıvıyı ölçüyor |
| parlak nokta yoğunluğu | mikrokalsifikasyon yaklaşımı | sinyal yok |
| solidity | alan / dışbükey zarf alanı | beklendiği gibi |
| kenar keskinliği | sınırdaki gradyan | belirsiz |
| sıvı oranı | çok karanlık piksel payı | beklendiği gibi |

- **Sonuç:**

| model | özellikler | val | cal | test |
|---|---|---|---|---|
| XGBoost | radyomik 16 | 0,799 | 0,823 | 0,805 |
| XGBoost | + 6 TI-RADS | 0,828 | 0,848 | **0,831** |
| XGBoost | sadece 6 TI-RADS | 0,786 | 0,787 | 0,775 |
| LLM (D) | radyomik 16 | 0,750 | 0,752 | 0,754 |
| LLM (I) | + 6 TI-RADS | 0,795 | 0,812 | **0,804** |
| LLM (J) | + 7. özellik | 0,786 | 0,802 | 0,781 |
| LLM (K) | + 6 TI-RADS, rich | 0,803 | 0,785 | 0,801 |

- LLM'de +0,050, aralık [+0,015, +0,084]. Şimdiye kadarki en büyük kazanç.
- **Yan denemeler:**
  - 7. özellik (sadece katı kısımda ekojenite): yönü düzeltti ama kazandırmadı.
  - Rich format + TI-RADS: decimal ile eşit, ama 4 kat daha yavaş (132 dk'ya karşı 36).
- **Not:** XGBoost'ta kalan 38 radyomik özelliği eklemek hiçbir şey kazandırmadı (0,796), 6 TI-RADS özelliği ise kazandırdı. Yani kazanç özellik sayısından değil, yeni bilgiden geliyor.

### C7. Büyük modele ikinci bir şans (11 Ekim)
- **Neden?** 7B'nin kararsızlığı LR'nin fazla yüksek olmasından olabilirdi.
- **Ne yaptık?** Qwen2.5-7B'yi LR 5e-5 ve TI-RADS özellikleriyle yeniden eğittik.
| model | özellikler | test AUC |
|---|---|---|
| Qwen2.5-1.5B (I) | radyomik + TI-RADS | **0,804** |
| Qwen2.5-7B, LR 2e-4 (C) | radyomik | 0,713 |
| Llama-2-7B, LR 1e-4 (G) | radyomik | 0,731 |
| Qwen2.5-7B, LR 5e-5 (L) | radyomik + TI-RADS | 0,693 |

- **Sonuç:** eğitim daha sakindi (duyarlılık 0,85 → 0,30 → 0,31 → 0,46 → 0,58), ama test 0,693. L ile I arasındaki fark anlamlı: [−0,148, −0,076]. TI-RADS özellikleri 7B'ye hiçbir şey kazandırmadı. Üç büyük model denemesi aynı yönde.
- **Çekince.** Bu run'lar tek seed'le yapıldı ve 7B için öğrenme oranı (LR) araması yapılmadı. Run C, 1.5B için seçilen LR'yi kullandı. Run L dört kat düşük LR kullandı ve validation AUC'si 5. epoch'ta hâlâ yükseliyordu (0,691 → 0,736 → 0,742 → 0,751 → 0,753), yani model tam oturmadan durmuş olabilir. Llama-2'de (run G) TI-RADS özellikleri yoktu. Bu yüzden savunulabilir iddia sınırlı: *denediğimiz ayarlarda büyük model kazandırmadı*. Büyük modelin hiçbir zaman kazandıramayacağı gösterilmedi.
- **Neden ana model hâlâ Qwen2.5-1.5B?**
  1. Tek birebir karşılaştırmada (aynı 16 özellik, aynı LR, epoch ve efektif batch: run C ile run D) 7B daha düşük: 0,713'e karşı 0,754, eşleştirilmiş aralık [−0,069, −0,004].
  2. Üç büyük model run'ı, üç farklı ayarda, hepsi karşılık gelen 1.5B run'ının altında kaldı; hiçbiri 0,804'e yaklaşmadı.
  3. 1.5B sonucu kararlı ve üç seed'de tekrarlandı (0,785 ± 0,018); 7B eğitimleri kararsızdı (gradient norm 200'e kadar).
  4. Bilgi tavanı: aynı 22 sayıyla XGBoost 0,835'e ulaşıyor ve LLM onun üstüne hiçbir şey katmıyor (stacking ağırlığı −0,04). Daha büyük bir LLM en iyi ihtimalle bu tavana yaklaşır, tavanı yükseltemez.
  5. Maliyet: eğitim süresi üç ile yedi kat.

### C8. "Sonuç gerçek mi?" Seed'ler, birleştirme, conformal kontroller (12 Ekim)
- **Seed tekrarları:**

| | seed 42 | seed 1 | seed 2 | ortalama |
|---|---|---|---|---|
| TI-RADS'lı | 0,804 | 0,768 | 0,783 | **0,785 ± 0,018** |
| TI-RADS'sız | 0,754 | 0,712 | 0,713 | 0,726 ± 0,024 |

| | seed 42 | seed 1 | seed 2 |
|---|---|---|---|
| eşleştirilmiş kazanç | +0,050 | +0,056 | +0,070 |
| %95 aralık | [+0,015, +0,084] | [+0,013, +0,098] | [+0,026, +0,112] |

  Ortalama kazanç +0,059. Her çiftte aralık sıfırın üstünde. Seed 42 iki tarafta da şanslı üst uçtu.
- **Hibrit (birleştirme):**

| model (aynı 22 özellik) | val | cal | test |
|---|---|---|---|
| XGBoost | 0,827 | 0,848 | **0,835** |
| LLM, 3 seed ortalaması | 0,792 | 0,803 | 0,793 |
| olasılık ortalaması | 0,820 | 0,838 | 0,824 |
| stacking | 0,827 | 0,848 | 0,834 |

  Stacking ağırlıkları: XGBoost 0,99, LLM −0,04. Skorlar 0,87 korelasyonlu, LLM'in iki seed'i arasında ise 0,91. **LLM, XGBoost'un bilmediği bir şey bilmiyor.** Ayrıca seed'lerin ortalaması (0,793), tek run'ların ortalamasından (0,785) biraz daha iyi.
- **Conformal kontroller:**

| calibration kuyruğu | en düşük %10 | %15 | %20 | planlanan LTT |
|---|---|---|---|---|
| XGBoost | 52'de 2 | 78'de 2 | 104'te 3 | sertifika yok |
| LLM, 3 seed | 52'de 4 | 78'de 7 | 104'te 10 | sertifika yok |

| Mondrian | bizim bölünme | 4.000 yeniden bölme | otomatik iyi huylu | içindeki kanser |
|---|---|---|---|---|
| XGBoost | 0,941 | 0,951 ± 0,024 | %25,7 | %6,4 |
| LLM, 3 seed | 0,934 | 0,950 ± 0,024 | %16,7 | %9,8 |

  - XGBoost bile hiçbir şeyi sertifikalamadı: 78 nodülde 2 kanser var, 2 kanserle sertifika için 124 nodül gerekiyor.
  - Otomatik kötü huylu tarafında başka bir sorun var: XGBoost'un u = 0,85 eşiği tek başına geçerdi (54 nodülde 4 iyi huylu, p = 0,010). Ama sıra u = 0,95'ten başlıyor, orada 7 nodülde 3 iyi huylu var. Test orada başarısız olup duruyor. **Sorunun bir kısmı tasarımda.**
  - Mondrian'da kanser kapsamı 16 run'ın 12'sinde 0,95'in altındaydı. Yeniden bölmede ortalama 0,951 ve bölünmelerin yaklaşık %45'i 0,95'in altında. **Garanti tutuyor, bizim bölünme yaklaşık 1 standart sapma şanssızdı.**

### C9. Geçebilen bir tasarım ve son bir özellik denemesi (13 Ekim)
- **Önceden sabitlenmiş LTT tasarımı.** Çalıştırmadan önce GitHub'a commit'ledik.
  - Adaylar: validation skorlarının yüzdelikleri (%5–30).
  - Sıra: validation p-değerlerine göre.
  - Test: calibration'da fixed sequence.
  - Sonuç (2.000 yeniden bölme):

| sertifika oranı | oto. iyi huylu %5 | oto. kötü huylu, eski | oto. kötü huylu, yeni | karar verilen / içinde iyi huylu |
|---|---|---|---|---|
| XGBoost | %0 | %0 | %38 | %10,7 / %14,9 |
| LLM, 3 seed | %0 | %1 | **%99** | %7,9 / %7,4 |
| LLM run I | %0 | %13 | %22 | %11,5 / %18,4 |

  - Otomatik kötü huylu artık ulaşılabilir. Otomatik iyi huylu hiçbir zaman sertifikalanmadı, orada sınır model.
  - XGBoost'ta neden daha az? Validation sırası en üst %5'lik grubu başa koyuyor, ama o grup calibration'da kirli.
  - Sınırlama: tasarım run I'deki keşif bulgusundan sonra yazıldı. Kesin doğrulama yeni veri ister.

#### C9b. Eski ve yeni tasarım, adım adım

**Eski tasarım gerçek sayılarla (LLM, 3 seed, calibration).** Eşikler önceden sabit olasılık değerleriydi; sıra en sıkıdan başlıyordu ve ilk başarısızlıkta duruyordu.

| sıra | eşik (oto. kötü huylu) | nodül | aslında iyi huylu | p-değeri | sonuç |
|---|---|---|---|---|---|
| 1 | ≥ 0,95 | 8 | 1 | 0,50 | kaldı, test durdu |
| 2 | ≥ 0,90 | 30 | 2 | 0,044 | geçerdi, hiç denenmedi |
| 3 | ≥ 0,85 | 43 | 4 | 0,051 | denenmedi |

İyi huylu tarafta ilk eşik (≤ 0,05) sadece 1 nodül içeriyordu; ama orada sıra sonucu değiştirmiyor: bütün eşiklerde kanser oranı %9–13, p-değerleri 0,95'in üstünde.

**Planlanan tasarım neden başarısız oldu?** LTT aday eşikleri sabit bir sırayla test ediyor ve ilk başarısızlıkta duruyor; garantiyi koruyan şey bu kural. Planlanan sıra en güvenli otomatik kötü huylu eşikten başlıyordu: u = 0,95. Olasılığı 0,95'in üstünde olan nodül neredeyse yok. 14'ten az nodülle hiçbir şey sertifikalanamaz, bu yüzden test ilk adımda duruyor ve geçeceği u ≈ 0,80'e hiç ulaşamıyordu. Benzetme: ilk basamağı eksik bir merdiven.

**İki değişiklik.**

1. Adaylar olasılık değeriyle değil, nodül payıyla tanımlandı (validation'da en yüksek %5, %10, …, %30). Böylece hiçbir aday boş kalmıyor (calibration'ın %5'i ≈ 26 nodül).
2. Sıra validation setinde belirlendi, en umut verici aday önce; sonra calibration'da bu sırayla test edildi.

**Sayısal örnek (LLM, 3 seed ortalaması, otomatik kötü huylu taraf).** Validation'da her adaya bir p-değeri veriliyor: *gerçek hata oranı %20 olsaydı, bu kadar az iyi huylu görme ihtimalim ne?* Küçük p = daha umut verici. Validation'da hiçbir aday geçmiyor; validation sadece sırayı belirliyor. Calibration bu sırayla test ediyor ve ilk başarısızlıkta duruyor.

| aday (eşik) | validation: nodül / iyi huylu / p | sıra | calibration: nodül / iyi huylu / p | sonuç |
|---|---|---|---|---|
| en yüksek %5 (0,891) | 22 / 5 / 0,73 | 1 | 34 / 2 / 0,023 | geçti |
| en yüksek %10 (0,833) | 44 / 12 / 0,91 | 2 | 50 / 4 / 0,019 | geçti |
| en yüksek %20 (0,710) | 88 / 28 / 0,997 | 3 | 105 / 24 / 0,81 | kaldı: dur |
| en yüksek %15, %25, %30 | p 0,999–1,00 | 4–6 | – | sıra gelmedi |

Geçenlerin en cömerti kullanılıyor: eşik 0,833, yani en yüksek %10.

**Garanti neden bozulmuyor?** Sabit sıralı testin tek şartı, sıranın calibration verisi kullanılmadan belirlenmesi. Validation ayrı ve bağımsız bir küme. Tasarımın bütün detayları (adaylar, sıra, α, δ) çalıştırılmadan önce Git'e commit edildi, yani sonuca bakıp ayar oynanmadı.

**"Otomatik kötü huylu" ne demek?** Triage her nodülü üç kutudan birine koyuyor: *otomatik iyi huylu* (olasılık çok düşük; biyopsiye gerek yok), *belirsiz* (arada; doktor karar veriyor, nodüllerin çoğu burada) ve *otomatik kötü huylu* (olasılık çok yüksek; sistem doktora sormadan "kötü huylu" diyor ve yüksek riskli olarak işaretliyor). Otomatik kötü huylu garantisi: bu kararların en fazla %20'si aslında iyi huylu. Bizim bölünmede sertifikalanan eşik 0,833: 438 test nodülünün %9,8'i (yaklaşık 43) otomatik kötü huylu oldu, bunların %14'ü (yaklaşık 6) aslında iyi huyluydu, %20'nin altında. 2.000 yeniden bölmenin ortalamasında nodüllerin yaklaşık %8'i bu kutuya düşüyor, aralarında iyi huylu oranı %7,4.

**Otomatik iyi huylu neden hâlâ %0?** Orada sorun sıra değil, kuyruğun kendisi (yukarıdaki conformal kontroller). **LLM'in AUC'si daha düşükken neden XGBoost'tan daha sık geçiyor (%99'a %38)?** AUC bütün sıralamayı ölçüyor; bu test sadece en üstteki %5–30'a bakıyor ve orada LLM'in 3 seed ortalaması daha temiz. **Akılda tutulması gereken:** tasarım ana sonuçlardan sonra yazıldı; önceden kayıtlı ama aynı veride test edildi. Bu yüzden yeni veride (ör. ThyroidXL) doğrulanana kadar keşifsel (exploratory).

> 🗣 **Anlatım notu:** "Planlanan LTT en sıkı eşikten başlıyordu; orada neredeyse hiç nodül olmadığı için ilk adımda duruyordu. Eşikleri nodül payıyla tanımlayıp sırayı bağımsız validation setine göre belirleyince otomatik kötü huylu taraf 2.000 yeniden bölmenin %99'unda sertifikalandı: sistem hastaların yaklaşık %8–10'una kendi başına 'kötü huylu' diyebiliyor ve bu kararların en fazla %20'si yanlış (pratikte %7–14). Otomatik iyi huylu taraf hâlâ geçmiyor, çünkü orada sorun tasarım değil, en temiz grubun yeterince temiz olmaması."

- **Peritümöral (nodül çevresi) özellikler.** Nodülün etrafındaki halkada (genişlik: eşdeğer yarıçapın yarısı) 93 radyomik özellik çıkardık. Kural önceden belliydi: XGBoost'ta calibration kazancı yoksa LLM'e geçmiyoruz.

| XGBoost | val | cal | test |
|---|---|---|---|
| nodül + TI-RADS | 0,828 | 0,848 | 0,831 |
| + 8 halka özelliği | 0,832 | 0,851 | 0,834 |
| sadece halka | 0,699 | 0,671 | 0,693 |

  Calibration +0,002, aralık [−0,009, +0,014]: kazanç yok, LLM eğitilmedi. Halkada sinyal var, ama seçilen özellikler çoğunlukla parlaklık istatistikleri, yani ekojenite oranının zaten karşılaştırdığı şey.

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

### D7. Sonuçlar bir bakışta

| adım (test AUC) | LLM | XGBoost |
|---|---|---|
| resmi split, v1, 8 özellik | 0,672 | 0,696 |
| pooled, v1, 16 | 0,699 | 0,723 |
| pooled, v2, 16 | 0,754 | 0,805 |
| + 6 TI-RADS | **0,785 ± 0,018** | **0,835** |
| + halka | çalıştırılmadı | 0,834 |
| büyük modeller | 0,693–0,731 | – |
| birleşik | 0,834 (= XGBoost) | |

| garanti | ulaşılabilir mi? | ne sağlıyor |
|---|---|---|
| R1: oto. iyi huyluda ≤ %5 kanser | **hayır** | – |
| R2: oto. kötü huyluda ≤ %20 iyi huylu (yeni tasarım) | **evet** (%99) | nodüllerin ~%8'i, %6–7'si iyi huylu |
| kanserlerin ≤ %5'i kaçırılır | **evet** | LLM %12, XGBoost %24 otomatik iyi huylu |
| Mondrian, kanserlerin %95'i | **evet** (ortalamada) | otomatik iyi huylularda ~%10 (LLM) / %6 (XGBoost) kanser |

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

## Kısım E2: Ra et al. (2025) ile ilişki

Onların yöntemini tiroide uyarladık: PyRadiomics, mRMR, metin, LoRA'lı LLM sınıflandırıcı. Onlar "LLM klasik modelleri geçiyor" diyor, biz tersini bulduk. Makalelerindeki zayıf noktalar:
1. **Baseline AUC'leri accuracy'lerine neredeyse eşit** (7 klasik modelde en fazla 0,004 fark, XGBoost AUC 0,605). Bu, AUC'nin olasılıktan değil 0/1 tahminden hesaplandığını düşündürüyor ve baseline'ları haksız şekilde zayıf gösteriyor.
2. **Kendi ablation'ları:** LLM dondurulup sadece son katman eğitildiğinde AUC 0,710, LoRA ile 0,700. LLM'i eğitmek kazandırmamış.
3. **Özellik isimlerini gizleyince** AUC sadece 0,005 (iç test) ve 0,013 (dış test) düşüyor ve bu fark test edilmemiş. Oysa ana iddiaları "isimler anlam katıyor".
4. **Etiketleri BI-RADS skoru**, yani radyoloğun kararı, biyopsi değil. Güven aralığı yok, tek seed var, 500'den fazla epoch.

Biz düzgün bir XGBoost ve seed tekrarlarıyla LLM'in 0,04–0,05 geride kaldığını ve XGBoost'a bilgi eklemediğini gösterdik. Bizim etiketlerimiz biyopsiye dayandığı için görev daha zor ama daha anlamlı. AUC'lerimiz doğrudan karşılaştırılamaz.

---

## Kısım E3: Düşünüp yapmadıklarımız (ve neden)

| fikir | neden yapmadık |
|---|---|
| daha büyük LLM (14B, 32B) | Bilgi tavanı: aynı 22 sayı. 3 büyük model denemesi kazandırmadı. XGBoost seviyesine çıksa bile R1 yine kapalı. |
| dondurulmuş 7B + lojistik regresyon | Ucuz ve temiz bir deney olurdu, ama beklenen kazanç düşük. İstenirse yapılabilir. |
| prompt teknikleri (chain-of-thought, benzer vaka getirme, verbalizer) | Model cevap üretmiyor, sınıflandırma katmanıyla çalışıyor. Bilgi tavanı aynı. En mantıklısı benzer vaka getirmeydi, öncelik düşük. |
| XGBoost tahminini prompt'a yazmak | LLM sadece kopyalar, LLM'in katkısını ölçemeyiz. |
| conformal selection (Jin & Candès, 2023) | R1'e en yakın teori (ortalamada kontrol), ama %35 prevalansta çok temkinli kalır. Teori bölümünde "neden kullanmadık" diye anılabilir. |
| görüntünün kendisi (CNN / VLM) | En büyük potansiyel, ama tezin kapsamını değiştiriyor. Hocayla karar verilecek. |
| ThyroidXL | Planlı, ayrı bir veri seti olarak. Yeni LTT tasarımını doğrulamak için ideal. |

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
17. Ra et al.'ın makalesindeki baseline'ların neden haksız şekilde zayıf olabileceğini açıkla.
18. Hasta kimliklerinin olmaması hangi sonuçları etkiler, hangilerini etkilemez?
19. Seed'lerin ortalamasını almak neden tek run'lardan daha iyi bir LLM skoru veriyor, ve bu conformal için neden önemli?
20. Halka denemesinde "önceden konulmuş kural" neden önemliydi? LLM eğitimini neden durdurduk?
