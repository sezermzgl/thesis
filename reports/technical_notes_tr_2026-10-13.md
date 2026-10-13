# Ayrıntılı teknik notlar: her adım, her parametre (Türkçe)

*Bu not, tezde yaptığın her şeyin "nasıl" kısmını kodda uygulandığı hâliyle anlatıyor. Kaynaklar: `conformal_triage/` paketi, `experiments/` script'leri ve `experiments/tn3k_v2_pipeline_colab.ipynb`. Sayılar `runs/` kayıtlarından alındı. Kronoloji için `uctan_uca_calisma_materyali.md`, anlatım için `toplanti_anlatimi_TR.md` dosyalarına bak. Bu not, bir tez savunmasında "bunu nasıl yaptın?" sorusunun cevabı.*

---

## 1. Veri: TN3K

| özellik | değer |
|---|---|
| görüntü sayısı | 3.493 (2.421 hasta) |
| resmi bölünme | trainval 2.879 (1.905 iyi huylu, 974 kötü huylu) + test 614 (378 iyi huylu, 236 kötü huylu) |
| kötü huylu oranı | %34,6 (trainval %33,8, resmi test %38,4) |
| etiket kaynağı | sitoloji/patoloji (FNA biyopsisi), radyolog skoru değil |
| maske | piksel düzeyinde nodül maskesi, 127 eşiğinde ikiliye çevriliyor |
| görüntü boyutu | değişken. Medyan: trainval 390×336, resmi test 436×375 piksel |
| hasta kimliği | yayınlanmamış |

- **Resmi bölünme hasta düzeyinde:** bir hastanın bütün görüntüleri ya trainval'da ya da test'te.
- **İlk deneylerde (2–8 Ekim)** trainval görüntü düzeyinde üçe bölünmüştü: train 2.015 / validation 432 / calibration 432. Test resmi test setiydi (614).

---

## 2. Pooled split: nasıl yapıldı?

**Script:** `experiments/make_pooled_splits.py`, seed 42.

1. 3.493 görüntünün hepsi tek bir havuzda birleştirildi.
2. Her görüntüye bir "katman" etiketi verildi: resmi kaynak × sınıf. Örneğin `trainval_1` = trainval'dan gelen kötü huylu.
3. scikit-learn `train_test_split` bu katmanlara göre tabakalı (stratified) olarak üç kez uygulandı:
   - önce %60 train, kalanı %40,
   - kalanın 12,5/40'ı validation,
   - kalanın 15/27,5'i calibration, geri kalanı test.

| split | görüntü | kötü huylu | resmi test'ten gelen |
|---|---|---|---|
| train | 2.095 | %34,6 | %17,6 |
| validation | 436 | %34,6 | %17,4 |
| calibration | 524 | %34,7 | %17,7 |
| test | 438 | %34,7 | %17,6 |

**Neden tabakalı?** Her split'te hem sınıf oranı hem de iki resmi kaynağın karışımı aynı kalsın diye. Böylece calibration ve test aynı dağılımdan gelir.

**Bedeli:**
- Test artık bağımsız bir "harici" örnek değil, TN3K popülasyonunun tamamını temsil ediyor.
- Görüntü düzeyinde bölündüğü için aynı hastanın görüntüleri farklı split'lere düşebilir. TN3K'da hastadan 1.072 fazla görüntü var. En kötü senaryoda test görüntülerinin yaklaşık %40'ının train'de aynı hastadan bir görüntüsü olabilir. Gerçek oran bilinmiyor.

---

## 3. Exchangeability testi (two-sample classifier test)

**Kod:** `conformal_triage/evaluation.py`, `two_sample_test`.

1. İki set (örneğin calibration ve test) üst üste konur. Etiket "hangi setten geldi" olur: 0 ya da 1.
2. Bir lojistik regresyon, özelliklerden bu etiketi tahmin etmeye çalışır.
3. 5 katlı tabakalı çapraz doğrulamayla her görüntü için "test setinden gelme" olasılığı hesaplanır, buradan AUC çıkar.
4. **p-değeri:** etiketler 500 kez rastgele karıştırılır, her seferinde AUC yeniden hesaplanır. p = (1 + gerçek AUC'den büyük ya da eşit karışık AUC sayısı) / (1 + 500).

**Yorum:** AUC ≈ 0,5 ise setler ayırt edilemiyor, exchangeability makul. Sonuçlar:

| karşılaştırma | AUC | p |
|---|---|---|
| validation – calibration (resmi, kontrol) | 0,506 | 0,37 |
| calibration – resmi test | 0,632 | 0,002 |
| validation – calibration (pooled) | 0,476 | – |
| calibration – test (pooled) | 0,434 | 1,0 |

---

## 4. Özellik çıkarımı

### 4.1 v1 (ilk çıkarım, 2–8 Ekim)
- **PyRadiomics, 2D mod, sadece orijinal görüntü** (wavelet ya da LoG filtresi yok, Ra et al. gibi).
- **Ön işleme:** görüntü griye çevrilir ve kendi min–max değerleriyle 0–255'e ölçeklenir. Piksel aralığı (spacing) 1 kabul edilir, gerçek fiziksel boyut bilinmiyor.
- **Doku hesabı:** sabit aralık genişliği, binWidth = 25 gri değer.
- **Aileler:** shape2D, firstorder, GLCM, GLSZM. Toplam 67 kullanılabilir özellik.

### 4.2 Veri denetiminde bulunan dört sorun
1. **Az gri seviye:** binWidth 25 ile nodül içinde medyan yaklaşık 9 gri seviye kalıyor (çeyrekler arası 7–10). Doku matrisleri çok kaba.
2. **Boyut pikselde:** görüntü boyutları farklı, resmi test görüntüleri daha büyük. MajorAxisLength ile görüntü genişliği arasındaki korelasyon 0,42. "Boyut" kısmen görüntü boyutunu ölçüyor.
3. **Çok parçalı maskeler:** 298 maske (trainval'ın %7,8'i, test'in %11,7'si) birden fazla bölgeden oluşuyor. Şekil özellikleri birkaç bölgeyi birden tarif ediyor.
4. **Kenara değen maskeler:** 356 maske (trainval'ın %11,2'si, test'in %5,4'ü). Nodülün bir kısmı kadrajın dışında olabilir.

**Yan bulgular:**
- Çok parçalı ve kenara değen maskelerde kötü huylu oranı daha düşük (%22, genel %36).
- Görüntü genişliği tek başına kötü huyluluğu zayıf da olsa tahmin ediyor (AUC 0,57), yani çekimle ilgili bir karıştırıcı olabilir.

### 4.3 v2 (8 Ekim'den itibaren)

| ayar | v1 | v2 |
|---|---|---|
| doku için gri seviye | binWidth 25 (~9 seviye) | **binCount 64** (her görüntüde sabit 64 seviye) |
| boyut özellikleri | piksel | **görüntüye göre:** uzunluklar / √(W·H), alanlar / (W·H), çevre/alan oranı × √(W·H) |
| aileler | 4 | **7:** + GLRLM, GLDM, NGTDM |
| çok parçalı maske | olduğu gibi | **sadece en büyük parça** (önceden sabitlenmiş kural) |
| kenara değen maske | olduğu gibi | tutuluyor, `touches_border` olarak işaretleniyor |
| en küçük maske | – | 20 piksel |
| özellik sayısı | 67 | **102** |

- W ve H görüntünün genişliği ve yüksekliği. "Relative" özellikler: MajorAxisLength, MinorAxisLength, MaximumDiameter ve Perimeter √(W·H)'ye bölünüyor. MeshSurface ve PixelSurface W·H'ye bölünüyor. PerimeterSurfaceRatio √(W·H) ile çarpılıyor.
- Birimsiz şekil özellikleri (Sphericity, Elongation) değişmiyor.
- **Tutarlılık kontrolü:** tek parçalı maskelerde v1 ile ortak olan özellikler birebir aynı çıktı (Mean, Median, Maximum, Elongation için fark 0, korelasyon 1).
- **Kaynak ayrımı kontrolü:** göreli şekil özellikleriyle resmi kaynaklar hâlâ ayırt edilebiliyor (AUC 0,709, piksel şekil özellikleriyle 0,654). Yani iki kaynak arasındaki fark sadece piksel ölçeğinden kaynaklanmıyor. Pooled split bu farkı her split'e eşit dağıtarak sorunu çözüyor.

---

## 5. Ön işleme ve özellik seçimi

Hepsi **sadece train** verisiyle öğreniliyor, diğer split'lere aynen uygulanıyor:

1. **Temizlik:** sonsuz değerler eksik kabul edilir, train'de sabit olan özellikler atılır.
2. **Eksik değerler:** train medyanıyla doldurulur.
3. **Standartlaştırma:** z = (x − train ortalaması) / train standart sapması.
4. **Kopya filtresi** (`conformal_triage/selection.py`, |r| > 0,95):
   - Özellikler etiketle olan |korelasyonlarına| göre büyükten küçüğe sıralanır.
   - Sırayla gezilir. Bir özellik, o ana kadar tutulan her özellikle |r| ≤ 0,95 ise tutulur, değilse atılır.
   - Sonuç: iki kopyadan etiketle daha ilişkili olanı kalıyor.
   - v2'nin 102 özelliğinden 48'i atıldı, 54 kaldı.
5. **mRMR** (mrmr-selection paketi, `mrmr_classif`):
   - Özellikleri açgözlü (greedy) bir şekilde tek tek seçer. Her adımda şu skoru en yüksek olanı alır: alaka / tekrar.
   - **Alaka:** özelliğin etiketle ilişkisi (F istatistiği).
   - **Tekrar:** o ana kadar seçilen özelliklerle ortalama |korelasyonu|.
   - İlk 16 özellik kullanılıyor, mRMR sırasıyla.
   - **Neden filtre gerekti?** Tekrar *ortalama* korelasyon olduğu için, seçilenlerden sadece biriyle aynı olan bir özellik, diğerleriyle düşük korelasyonu sayesinde yine seçilebiliyordu. Örnek: PixelSurfaceRelative ve MeshSurfaceRelative, 2D'de aynı sayı ama ikisi de seçilmişti.
6. **TI-RADS özellikleri** filtreye ve mRMR'ye girmiyor. Kullanıldıklarında prompt'ta her zaman ilk sırada.

---

## 6. Prompt'taki özellikler tek tek

### 6.1 En iyi kurulumda seçilen 16 radyomik özellik (mRMR sırası)

Değerler train'e göre z-score. Nodülün sınırını belirleyen maske içinde hesaplanıyorlar.

| # | özellik (aile) | ne ölçüyor |
|---|---|---|
| 1 | MajorAxisLengthRelative (şekil) | Nodülle aynı ikinci momentlere sahip elipsin uzun ekseni, görüntü boyutuna göre. Kabaca **nodülün uzunluğu**. |
| 2 | SmallAreaLowGrayLevelEmphasis (GLSZM) | Küçük ve **koyu** bölgelerin (aynı gri tondaki bitişik piksel grupları) ağırlığı. Çok sayıda küçük koyu lekede yüksek. |
| 3 | Elongation (şekil) | √(küçük öz değer / büyük öz değer). 1 = daire, küçüldükçe **uzamış**. |
| 4 | PixelSurfaceRelative (şekil) | Maskenin piksel alanı / görüntü alanı. **Nodülün göreli büyüklüğü.** |
| 5 | Range (birinci derece) | Nodül içindeki en parlak ve en koyu piksel farkı. **Parlaklık aralığı.** |
| 6 | PerimeterSurfaceRatioRelative (şekil) | Çevre / alan, görüntüye göre ölçeklenmiş. **Düzensiz ya da küçük** sınırlarda yüksek. |
| 7 | DependenceVariance (GLDM) | Her pikselin "benzer gri tondaki komşu sayısı"nın varyansı. **Homojenliğin ne kadar değiştiği.** |
| 8 | MeanAbsoluteDeviation (birinci derece) | Parlaklığın ortalamadan ortalama mutlak sapması. **Parlaklık yayılımı.** |
| 9 | Idm (GLCM) | Inverse difference moment. Komşu piksel çiftleri benzer gri tonlardaysa yüksek. **Yerel homojenlik.** |
| 10 | Sphericity (şekil) | Aynı alana sahip dairenin çevresi / gerçek çevre. 1 = daire. **Yuvarlaklık**, düşükse düzensiz. |
| 11 | DependenceNonUniformityNormalized (GLDM) | Bağımlılık büyüklüklerinin görüntü boyunca ne kadar eşit dağıldığı, normalize. |
| 12 | DependenceNonUniformity (GLDM) | Aynısı, normalize edilmemiş, yani nodül büyüklüğünden de etkilenir. |
| 13 | MaximumProbability (GLCM) | En sık görülen komşu gri ton çiftinin oranı. **Tek bir dokunun baskınlığı.** |
| 14 | SizeZoneNonUniformity (GLSZM) | Bölge büyüklüklerinin ne kadar değişken olduğu, normalize edilmemiş. |
| 15 | ZoneEntropy (GLSZM) | Bölge büyüklükleri ve gri tonlarının rastgeleliği. **Doku karmaşıklığı.** |
| 16 | SizeZoneNonUniformityNormalized (GLSZM) | 14'ün normalize hâli. |

**Gruplar hâlinde:**
- **Boyut ve şekil (1, 3, 4, 6, 10):** uzunluk, uzama, alan, çevre/alan oranı, yuvarlaklık.
- **Parlaklık dağılımı (5, 8):** aralık ve sapma.
- **Doku (geri kalanlar):** homojenlik, bölge yapısı, karmaşıklık.

### 6.2 İlk deneylerin 8 özelliği (v1, resmi split)
1. MajorAxisLength (piksel cinsinden uzunluk)
2. HighGrayLevelZoneEmphasis (parlak bölgelerin ağırlığı)
3. Elongation
4. ClusterShade (GLCM matrisinin asimetrisi)
5. Range
6. LargeAreaHighGrayLevelEmphasis (büyük ve parlak bölgeler)
7. MaximumDiameter (piksel cinsinden en uzun çap)
8. Maximum (en parlak piksel)

Tek tek çıkarma deneyinde model en çok MajorAxisLength'e dayanıyordu: çıkarınca cevapların %28'i değişti. Sonra bunun kısmen görüntü boyutu olduğu anlaşıldı.

### 6.3 TI-RADS'tan esinlenen 6 özellik: tam hesap (`conformal_triage/clinical.py`)

Hepsi **ham görüntü** üzerinde hesaplanıyor (görüntü başına ölçekleme yok). Hepsi oran ya da yoğunluk, yani görüntü boyutundan ve cihazın genel parlaklık ayarından bağımsız. Ayarlar etiketlere bakmadan sabitlendi.

**Ortak yapı: çevre halkası**
- Nodülün eşdeğer yarıçapı: r = √(alan / π).
- Halka genişliği = max(4 piksel, round(0,25 × r)).
- Halka = maskenin bu genişlikte genişletilmiş hâli, eksi nodülün kendisi.
- Gri değeri 5'in altındaki pikseller halkadan çıkarılıyor. Bunlar ultrason sektörünün dışındaki siyah alan.

| özellik | formül | TI-RADS karşılığı |
|---|---|---|
| **TallerThanWideRatio** | maskenin dikey uzunluğu / yatay uzunluğu (sınırlayıcı kutu). Görüntü çekildiği gibi kabul ediliyor, derinlik aşağıda. | şekil: "boyu eninden uzun" (> 1) şüpheli |
| **EchogenicityRatio** | nodül içi ortalama parlaklık / halkanın ortalama parlaklığı | ekojenite: < 1 hipoekoik (şüpheli) |
| **PunctateFociDensity** | parlak nokta sayısı / nodül alanı × 1.000 (hesap aşağıda) | ekojenik odaklar (mikrokalsifikasyon) |
| **Solidity** | alan / dışbükey zarf (convex hull) alanı | kenar: < 1 loblu ya da düzensiz |
| **MarginSharpness** | sınırdaki ortalama gradyan büyüklüğü / √(0,5 × (nodül içi varyans + halka std²)) | kenar keskinliği |
| **AnechoicFraction** | nodül piksellerinin halka ortalamasının %25'inden koyu olanlarının payı | yapı: sıvı (kistik) kısım |

**Parlak nokta tespiti, ayrıntılı:**
1. Nodülün kenarından 2 piksel içeri girilir (alan > 50 ise), kenar etkisinden kaçınmak için.
2. White top-hat filtresi uygulanır (disk yarıçapı 3): yaklaşık 7 pikselden dar parlak yapıları öne çıkarır.
3. **Eşik:** top-hat değeri > medyan + 6 × MAD (MAD × 1,4826, en az 1) **ve** piksel, halka ortalamasından parlak. MAD (median absolute deviation) aykırı değerlerden etkilenmeyen bir yayılım ölçüsü. Böylece noktaların kendisi eşiği yükseltmiyor.
4. Birbirine bağlı bölgeler sayılır, sadece 2–40 piksel arasındakiler alınır. Tek piksel speckle gürültüsüdür, büyük bölgeler nokta değildir.

**Kenar keskinliği, ayrıntılı:** görüntü Gauss filtresiyle (σ = 1) yumuşatılır. Sobel ile gradyan büyüklüğü hesaplanır (/8). Sınır piksellerinde ortalaması alınır.

**Yedinci özellik (SolidEchogenicityRatio, run J):** ekojenite oranı sadece "sıvı olmayan" pikseller (halka ortalamasının %25'inden parlak olanlar) üzerinden hesaplanıyor. TI-RADS ekojeniteyi katı kısımda değerlendirdiği için denendi. Yönü düzeltti ama bilgi eklemedi, varsayılan ayarlarda çıkarıldı.

**Train'de tek başına AUC'ler:** TallerThanWide 0,658, Solidity 0,367, AnechoicFraction 0,409, EchogenicityRatio 0,552, MarginSharpness 0,563, PunctateFoci 0,487. 0,5'in altı "düşük değer kötü huylu" demek. Ekojenite oranı sıvı oranıyla −0,78 korelasyonlu: aslında sıvıyı ölçüyor.

### 6.4 Peritümöral halka (`conformal_triage/peritumoral.py`)
- **Genişlik:** max(5 piksel, round(0,5 × r)). Nodül ve sektör dışı hariç.
- **Hesap:** Öklid mesafe dönüşümüyle (distance transform). Disk genişletmesiyle aynı sonucu veriyor ama daha hafif.
- **Özellikler:** PyRadiomics, v2 ile aynı ayarlar (min–max 0–255, binCount 64). Aileler: firstorder, GLCM, GLSZM, GLRLM, GLDM, NGTDM. Şekil hariç, çünkü halkanın şekli maskeden geliyor. 93 özellik, kopya filtresinden sonra 49.
- **Bellek düzeltmesi:** PyRadiomics'e sadece halkanın sınırlayıcı kutusu (+2 piksel) veriliyor. Tam görüntüyle sonuç birebir aynı.

---

## 7. Prompt'un tam metni

```
Classify this thyroid nodule as benign or malignant based on the following standardized radiomic features.
The 'Taller Than Wide Ratio' feature is measured at 1.204. The 'Echogenicity Ratio' feature is measured at -0.512.
… The 'Major Axis Length Relative' feature is measured at 0.335. The 'Elongation' feature is measured at -1.140. …
```

- **Değer:** train'e göre z-score, 3 ondalık basamak.
- **İsimler okunabilir hâle getiriliyor:** `original_shape2D_Elongation` → "Elongation". Aynı isim birden fazla ailede varsa ailesi ekleniyor, örneğin "Gray Level Non Uniformity (GLDM)".
- **Sıra:** önce TI-RADS özellikleri, sonra mRMR sırası. Halka özellikleri kullanılsaydı en sonda olacaktı.
- **Uzunluk:** 16 özellik ≤ 384 token, TI-RADS ile ≤ 512, rich ~780 (TI-RADS ile ~1.536). Eğitimden önce en uzun prompt kontrol ediliyor. Kesilecek bir prompt varsa çalışma durduruluyor.

**Formatlar:**

| format | cümle |
|---|---|
| decimal | "The '…' feature is measured at -1.140." |
| zscore | "The '…' feature has a z-score of -1.14 relative to the training nodules." |
| percentile | "The '…' feature is at percentile 13 of the training nodules." (train içindeki sıra) |
| rich | z-score + persentil + bant + anlam |
| semantic_only | bant + anlam, sayı yok (çalıştırılmadı) |

- **Rich formattaki bantlar:** persentil < 10 "much lower than typical", < 30 "lower than typical", < 70 "close to typical", < 90 "higher than typical", geri kalanı "much higher than typical".
- **Rich formattaki anlam cümleleri:** özelliğin ne ölçtüğünü söylüyor ("meaning a less round, more irregular outline"), ama **hangi yönün kötü huylu olduğunu asla söylemiyor**. Böylece modele cevap sızdırılmıyor.
- Elongation için PyRadiomics tanımı da prompt'a ekleniyor.

---

## 8. Dil modeli

| ayar | değer | açıklama |
|---|---|---|
| model | Qwen2.5-1.5B (ayrıca Qwen2.5-7B, Llama-2-7B) | önceden eğitilmiş, metin üretmek için |
| kafa | `AutoModelForSequenceClassification`, 2 çıkış | son (padding olmayan) token'ın gizli durumundan iki logit. Rastgele başlatılıyor. |
| olasılık | softmax(logit)[1] = P(kötü huylu) | |
| ağırlıklar | 4-bit NF4, double quantization, hesaplama bf16 | QLoRA: ana ağırlıklar donmuş ve sıkıştırılmış |
| LoRA | rank 16, α = 32, dropout 0,05 | etkin öğrenme ölçeği α/r = 2 |
| LoRA yerleri | dikkat katmanlarının q_proj ve v_proj'u. Llama run'ında yedi doğrusal katmanın hepsi (q, k, v, o, gate, up, down). | |
| pad token | eos token | modelde pad token yok |
| eğitilen | LoRA matrisleri + sınıflandırma kafası | |

**LoRA nedir?** Bir ağırlık matrisi W donuk kalıyor. Eğitilen şey küçük bir düzeltme: ΔW = B·A, burada A (r × d) ve B (d × r), r = 16. Böylece milyonlarca yerine birkaç milyon parametre eğitiliyor.

---

## 9. Eğitim

| ayar | değer |
|---|---|
| kayıp | sınıf ağırlıklı cross-entropy. Ağırlık = n / (2 × n_sınıf) → iyi huylu 0,765, kötü huylu 1,445 |
| optimizer | AdamW (Trainer varsayılanı) |
| öğrenme oranı | 2 × 10⁻⁴. Llama 10⁻⁴, run L 5 × 10⁻⁵ |
| zamanlama | cosine, ısınma (warmup) toplam adımların %5'i |
| batch | 8 × 2 gradient accumulation = efektif 16 (7B: 4 × 4) |
| epoch | 5 → epoch başına 131 adım, toplam 655 |
| hassasiyet | L4'te bf16, T4'te fp16. Gradient checkpointing açık. |
| model seçimi | her epoch sonunda validation AUC. En iyi epoch geri yükleniyor (`load_best_model_at_end`). |
| tahmin | eğitimden sonra **tam hassasiyette** (fp32) |
| seed | 42 (tekrarlarda 1 ve 2): LoRA başlangıcı ve batch sırası |

**Yol üstünde düzeltilen iki teknik hata:**
1. **Gradient accumulation kaybı:** eski sürümde kayıp, accumulation sayısıyla yanlış ölçekleniyordu. Düzeltildi (`model_accepts_loss_kwargs = False`). Yeniden eğitilen model eskisiyle tutarlı (Spearman 0,97).
2. **bf16 yuvarlaması:** Trainer tahmin yaparken bf16 otomatik dönüşümünü açık bırakıyordu. Olasılıklar logit uzayında 1/64 adımlarla yuvarlanıyordu (524 nodülde 195 farklı değer). Model sarmalayıcısından çıkarılıp fp32'de tahmin edilerek düzeltildi (524'te 524 farklı değer).

**Neden sınıf ağırlığı?** Kötü huylu sınıf azınlıkta (%35). Ağırlık olmasa model "hep iyi huylu" demeye kayar. AUC eşikten bağımsız olduğu için ağırlık AUC'yi pek etkilemez, ama 0,5 eşiğindeki duyarlılığı etkiler.

---

## 10. XGBoost (referans model)

| ayar | değer |
|---|---|
| ağaç sayısı | 300 |
| derinlik | 3 |
| öğrenme oranı | 0,05 |
| alt örnekleme | satırların %80'i (subsample), sütunların %80'i (colsample_bytree) |
| sınıf dengesi | scale_pos_weight = iyi huylu sayısı / kötü huylu sayısı |
| ayar | **yapılmadı** (sabit parametreler) |
| ön işleme | LLM ile aynı: medyan, z-score, filtre, mRMR, train üzerinde |

---

## 11. Değerlendirme

- **AUC:** P(kötü huylu)'dan, eşiksiz. Ana ölçü.
- **Accuracy ve duyarlılık:** 0,5 eşiğinde. Sınıf ağırlıkları yüzünden yorumu dikkat ister.
- **Paired bootstrap:**
  - Aynı test nodüllerinden 2.000 kez yerine koyarak örnek çekiliyor.
  - Her örnekte iki modelin AUC farkı hesaplanıyor. Tek sınıflı örnekler atlanıyor.
  - %2,5 ve %97,5 yüzdelikleri %95 aralığı veriyor.
- **Bootstrap'ın kapsamadıkları:**
  - Eğitim gürültüsü: modeli sabit tutuyor. Bunu seed tekrarları karşılıyor.
  - Hasta kümelenmesi: hasta kimliği yok.
- **Seed tekrarları:** ana karşılaştırma eşleştirilmiş seed'lerle (42, 1, 2). Raporlananlar: ortalama ± SD, seed başına fark ve seed'lerin olasılık ortalamasının bootstrap aralığı.
- **Doğru/yanlış ayırma AUC'si:** bir güven skorunun "tahmin doğru mu?" olayını ne kadar iyi ayırdığı. İlk aşamada oyları softmax ile karşılaştırmak için kullanıldı.

---

## 12. Pertürbasyon oyları (ilk aşama, 2 Ekim)

**Run 1 (karışık):** 10 sorgu.
- Sorgu 0: eğitimdeki prompt'un aynısı.
- Sorgu 1–9: her birinde özellik sırası karıştırılıyor, 0,5 olasılıkla bir özellik atılıyor ve üç cümle kalıbından biri seçiliyor.

**Run 2 (formatı koruyan):** nodül başına 36 sorgu.
- 1 temel sorgu.
- **Gürültü:** her standart sapma seviyesi (0,05, 0,1, 0,2) için 9 sorgu. Her standartlaştırılmış değere N(0, SD²) gürültü ekleniyor. Seviye, validation'da önceden belirlenmiş bir kuralla seçiliyor: oy oranının AUC'si en yüksek olan. Seçilen 0,2.
- **Atma:** 8 sorgu, her birinde bir özellik çıkarılıyor.

**Oy ve sinyaller:**
- Oy v = P ≥ 0,5 olan sorgu sayısı.
- Gürültü oyları: temel + 9 gürültü, m = 10.
- Atma oyları: temel + 8 atma, m = 9.
- **Tekrarlanabilirlik:** her nodülün gürültüsü kendi kimliğinden türetilen bir seed'le üretiliyor (crc32). Sonuç satır sırasına bağlı değil.

---

## 13. Conformal yöntemler: matematik

### 13.1 Tanımlar
Model "iyi huylu" derse (P ≤ t):
- FN = gerçekte kanser olan sayısı.
- **Kaçırma oranı** = FN / tüm kanserler = 1 − duyarlılık (false negative rate).
- **R1** = FN / "iyi huylu" denen tüm nodüller = 1 − NPV (false omission rate).

Model "kötü huylu" derse (P ≥ u):
- FP = gerçekte iyi huylu olan sayısı.
- **R2** = FP / "kötü huylu" denen tüm nodüller = 1 − PPV (false discovery rate).

### 13.2 LTT (`conformal_triage/ltt.py`)
- **Aday eşikler:** t ∈ {0,05; 0,10; …; 0,45}, u ∈ {0,95; 0,90; …; 0,55}. Her t her u'dan küçük. Oy skoru için t ∈ {0..4}, u ∈ {10..6}.
- **Her aday için:** n = calibration'da P ≤ t olan nodül sayısı, k = aralarındaki kanser sayısı.
  - **H0:** R1(t) > α₁.
  - **p-değeri:** p = P(Binom(n, α₁) ≤ k). n = 0 ise p = 1.
- **Fixed sequence:** adaylar en güvenliden başlayarak sırayla test edilir. İlk p > δ'da durulur. Sertifikalanan en geniş eşik kullanılır. Hiçbiri sertifikalanmazsa o tarafta otomatik karar yok.
- **Neden geçerli?**
  - n verildiğinde, eşiğin altındaki nodüller o bölgeden bağımsız bir örnek. Kanser sayısı Binom(n, R1(t)) dağılır.
  - Sıra calibration verisine bakmadan belirlendiği için ailesel hata oranı (FWER) ≤ δ.
- **Garanti:** calibration çekilişi üzerinden en az 1 − δ olasılıkla, seçilen eşiğin gerçek riski ≤ α. İki taraf birlikte en az 1 − 2δ = 0,90 olasılıkla tutar (union bound).
- **Gereken en küçük grup (δ = 0,05):**

| hata | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| α = %5 | 59 | 93 | 124 | 153 | 181 | 208 |
| α = %20 | 14 | 22 | 30 | 37 | 44 | 50 |

  Örnek: 0 hatayla 59 nodül gerekiyor, çünkü 0,95⁵⁹ ≈ 0,048 ≤ 0,05.

### 13.3 Kaçırma oranı LTT'si
- **Aday eşikler:** t ∈ {0,01; 0,02; …; 0,50}, artan sırada.
- **n** = tüm calibration kanserleri (pooled'da 182). **k** = P ≤ t olan kanser sayısı.
- p = P(Binom(182, 0,05) ≤ k). İlk başarısızlıkta durulur.
- Geçerli bir LTT. Planlanan kural başarısız olduktan sonra eklendiği için "keşif amaçlı" deniyor.

### 13.4 Split conformal prediction (`conformal_triage/split_cp.py`)
- **Uyumsuzluk skoru:** s(x, y) = 1 − P(y | x).
- **Eşik:** calibration'daki doğru etiket skorlarının k'ıncı en küçüğü, k = ⌈(n + 1)(1 − α)⌉. k > n ise eşik sonsuz.
- **Küme:** {y : s(x, y) ≤ q̂}. Tek etiket otomatik karar, iki etiket doktora.
- **Marginal:** tek eşik, α = 0,10 ya da 0,05.
- **Mondrian:** her sınıf kendi nodülleriyle kalibre ediliyor. α_iyi huylu = 0,10, α_kötü huylu = 0,05.
- **Garanti:** calibration ve test çekilişleri üzerinden **ortalamada** P(y ∈ C(x) | Y = y) ≥ 1 − α_y. Tek bir test setinde gerçekleşen değer bunun altında da çıkabilir.

### 13.5 Validation sıralı LTT ve önceden sabitlenmiş tasarım
- **Keşif hâli (9 Ekim):** aynı aday ızgarası, ama sıra validation p-değerlerine göre. Validation, calibration'dan bağımsız olduğu için garanti bozulmuyor.
- **Önceden sabitlenmiş tasarım (13 Ekim, `experiments/ltt_design_resplits.py`, çalıştırmadan önce commit 4f42659):**
  - **Adaylar:** validation skorlarının yüzdelikleri. Otomatik iyi huylu için q ∈ {%5, %10, …, %30} yüzdeliği, otomatik kötü huylu için (1 − q) yüzdeliği.
  - **Sıra:** validation p-değerine göre, eşitlikte küçük q önce.
  - **Test:** calibration'da fixed sequence, aynı α ve δ. En geniş sertifikalı eşik kullanılıyor.
  - **Değerlendirme:** modeller sabit. Calibration + test (962 nodül) 2.000 kez, aynı boyutlar ve sınıf oranlarıyla yeniden bölündü.
  - **Raporlanan:** sertifika oranı, test'te karar verilen pay, test'teki ve 962'nin tamamındaki hata payı.

---

## 14. Birleştirme (hibrit, `experiments/hybrid_xgb_llm.py`)
- XGBoost, run I'in tam olarak aynı 22 özelliği ve aynı ön işlemeyle eğitiliyor. LLM skoru üç seed'in olasılık ortalaması.
- **Üç birleştirme yolu:**
  1. olasılıkların ortalaması,
  2. sıraların ortalaması,
  3. **stacking:** logit(P_XGB) ve logit(P_LLM) üzerinde lojistik regresyon. Olasılıklar [10⁻⁶, 1 − 10⁻⁶] aralığına kırpılıyor. **Sadece validation'da** eğitiliyor.
- Calibration ve test eğitime dokunmadığı için conformal katman birleşik skora da uygulanabilir.
- **Raporlanan:** AUC'ler, iki skorun korelasyonu (Pearson, Spearman), XGBoost'a göre paired bootstrap, Mondrian tablosu ve stacking ağırlıkları.

---

## 15. Conformal kontroller (`experiments/conformal_checks.py`)
1. LTT için gereken en küçük grup tablosu.
2. XGBoost ve 3 seed ortalamalı LLM için: planlanan LTT, validation sıralı LTT, kaçırma oranı kuralı ve en temiz kuyruklar.
3. **Mondrian'ın yeniden bölme kontrolü:**
   - Model sabit. Calibration + test 4.000 kez yeniden bölünüyor: 524 calibration'ın 182'si kanser.
   - Her seferinde yeniden kalibre ediliyor.
   - Kapsamın ortalaması ve standart sapması, otomatik iyi huylu oranı ve aralarındaki kanser oranı raporlanıyor.

---

## 16. Bütün run'lar

| run | model | özellikler | format | diğer | val | cal | test |
|---|---|---|---|---|---|---|---|
| resmi | Qwen2.5-1.5B | v1, 8 | decimal | T4, ilk model | 0,704 | 0,707 | 0,672 |
| resmi | Qwen2.5-1.5B | v1, 8 | decimal | L4, yeniden | 0,697 | 0,714 | 0,667 |
| resmi | Qwen2.5-7B | v1, 8 | decimal | | 0,697 | 0,686 | 0,669 |
| resmi | Qwen2.5-7B | v1, 16 | decimal | | 0,702 | 0,715 | 0,666 |
| A | Qwen2.5-1.5B | v2, 16, filtresiz | decimal | | 0,753 | 0,757 | 0,750 |
| B | Qwen2.5-1.5B | v1, 16 | decimal | | 0,709 | 0,705 | 0,699 |
| C | Qwen2.5-7B | v2, 16 | decimal | batch 4×4 | 0,708 | 0,705 | 0,713 |
| D | Qwen2.5-1.5B | v2, 16 | decimal | referans | 0,750 | 0,752 | 0,754 |
| E | Qwen2.5-1.5B | v2, 16 | zscore | | 0,729 | 0,756 | 0,759 |
| F | Qwen2.5-1.5B | v2, 16 | percentile | | 0,709 | 0,713 | 0,734 |
| G | Llama-2-7B | v2, 16 | decimal | LR 1e-4, LoRA her yerde | 0,705 | 0,709 | 0,731 |
| H | Qwen2.5-1.5B | v2, 16 | rich | | 0,732 | 0,711 | 0,744 |
| I | Qwen2.5-1.5B | v2, 16 + 6 TI-RADS | decimal | **en iyi kurulum** | 0,795 | 0,812 | 0,804 |
| J | Qwen2.5-1.5B | v2, 16 + 7 TI-RADS | decimal | | 0,786 | 0,802 | 0,781 |
| K | Qwen2.5-1.5B | v2, 16 + 6 TI-RADS | rich | | 0,803 | 0,785 | 0,801 |
| L | Qwen2.5-7B | v2, 16 + 6 TI-RADS | decimal | LR 5e-5 | 0,751 | 0,712 | 0,693 |
| M | Qwen2.5-1.5B | I ile aynı | decimal | seed 1 | 0,771 | 0,777 | 0,768 |
| N | Qwen2.5-1.5B | I ile aynı | decimal | seed 2 | 0,787 | 0,794 | 0,783 |
| O | Qwen2.5-1.5B | D ile aynı | decimal | seed 1 | 0,706 | 0,695 | 0,712 |
| P | Qwen2.5-1.5B | D ile aynı | decimal | seed 2 | 0,723 | 0,715 | 0,713 |

- A–P pooled split'te. "Filtre", kopya filtresi (|r| > 0,95).
- **Eğitim süreleri:** 1.5B 26–37 dk, rich 83–132 dk, Qwen 7B 90–123 dk, Llama 182 dk.
- **Donanım:** önce Colab T4 (ücretsiz), sonra Colab Pro L4.

---

## 17. Kod ve kayıtlar

| yer | ne |
|---|---|
| `conformal_triage/` | LTT, split CP, prompt'lar, anlam sözlüğü, kopya filtresi, TI-RADS ve halka özellikleri, değerlendirme. 43 birim testi (`tests/`). |
| `experiments/tn3k_v2_pipeline_colab.ipynb` | Ana notebook. Part A çıkarım (v2, TI-RADS, halka), Part B XGBoost, Part C LLM + conformal. |
| `experiments/make_pooled_splits.py` | pooled split |
| `experiments/hybrid_xgb_llm.py` | birleştirme |
| `experiments/conformal_checks.py` | conformal kontroller |
| `experiments/ltt_design_resplits.py` | önceden sabitlenmiş LTT tasarımı |
| `runs/…/` | her run için config, metrikler, eğitim kaydı, nodül başına olasılıklar ve çalışmış notebook |
| `manuscript/` | tez (LaTeX, sapthesis) |
