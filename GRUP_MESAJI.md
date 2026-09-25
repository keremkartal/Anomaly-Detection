# Gruba atacağım mesaj

`gorevdagilim.md` dosyasıyla birlikte gönder. Uzun geldiyse boş satırdan bölüp
arka arkaya at.

---

Arkadaşlar merhaba,

Hakem yorumları geldi, üçünü de tek tek okudum ve iddiaları kodda kontrol
ettim. Detaylı dosyayı ekliyorum ama önemli kısımları buraya yazayım.

**Önce iyi haber:** Kimse "reddedilsin" dememiş, üçü de düzeltme istiyor.
Hakem 1 makaleyi *"sınırlılıkları konusunda alışılmadık derecede şeffaf"*
bulmuş, Hakem 3 katkıyı *"potansiyel olarak değerli ampirik katkı"* diye
tanımlamış. İngilizce ikisinde temiz, birinde geliştirilmeli. Toplam 25 madde
var.

**Bir uyarı:** Elimizdeki `LSTM_GAT_Uc_Hakem_Revizyon_Yol_Haritasi.docx`
dosyası hakem numaralarını karıştırmış. Orada "birinci hakem" diye geçen
aslında Hakem 2, "ikinci hakem" Hakem 3, "üçüncü hakem" Hakem 1. İçerikleri
tek tek eşleştirdim, madde sayıları da doğruluyor (13-4-8). Yanıt mektubu
hakem hakem yazıldığı için bu önemli, o kodlarla yazarsak her maddeyi yanlış
hakeme cevaplamış oluruz. Ekteki dosyada gerçek numaralar var.

**Kötü haber ve dürüst olmam gereken kısım:** Hakem 3 kodu gerçekten okumuş
gibi, iki gerçek hata bulmuş ve ikisi de bende.

Birincisi, çapraz doğrulamanın havuzlanmış analizinde bütün fold'ların
eşiklerinin ortalamasını alıp hepsine uygulamışım. Her fold kendi eşiğini
kullanmalıydı. Eşikler 0,23 ile 0,94 arasında değişiyor, hepsine 0,79
uygulanmış. Bu bir sızıntı, hakem haklı.

Etkisini ölçtüm: Tablo 13'te fark küçük, ama **Tablo 14'te uzun duraklama
bulgusu +0,0335'ten +0,0074'e düşüyor.** O bulgu makalenin tek istatistiksel
olarak anlamlı pozitif sonucuydu. Büyük ihtimalle anlamlılığını yitirecek,
yani elimizde "uzamsal dal işe yarıyor" diyebileceğimiz kanıt kalmayabilir.
Buna hazırlıklı olalım, özet ve sonuç bölümü yeniden yazılabilir.

İkincisi, grafı seçerken kullandığım "etiket saflığı" gerekçesi test
etiketlerini de hesaba katıyormuş. Hocamın bir önceki turdaki "performansa
göre graf seçimi" eleştirisini çözmek için koymuştum ama daha kötüsünü
yapmışım. Düzeltilecek.

**İyi tarafı:** Bu iki düzeltme yeniden eğitim gerektirmiyor, 130 koşunun
tahminleri önbellekte duruyor. Birkaç günlük iş.

**Kapsam:** Kodu ve veriyi denetledikten sonra çıkan sonuç şu: 25 maddenin
**17'sini ekipten hiçbir şey beklemeden tam kapatabiliyorum**, 5'ini de
savunulabilir biçimde kapatıp sizin katkınızla güçlendirebiliyorum. Yani siz
hiçbir şey vermeseniz bile 22/25 kapanır ve makale gönderilebilir olur.

Başta veri ekibine bağlı sandığım iki madde de bana geçti, çünkü veride
zaten gerekli bilgi varmış (`GNN_DATA.csv` içinde hangi hız limiti ve şerit değerinin OSM'den,
hangisinin varsayılan olduğu etiketli).

Yapacağım işler kabaca: eşik düzeltmesi, graf gerekçesi, tekrarlı bölümleme,
grafsız ve kenarsız kontroller, klasik modelleri CV'ye alma, iki grafta
sistematik duyarlılık, indüktif graf kontrolü, pencere/stride duyarlılığı,
imputasyon duyarlılığı, gürültü ve yaygınlık duyarlılığı, baseline
hiperparametre belgelemesi, gate çökme tanıları, alt tip tam raporu, bütün
tablo ve figürler, yanıt mektubu.

**Sizden beş şey isteyeceğim:**

1. **Markov geçiş olasılıkları** nasıl seçildi veya kalibre edildi? Hakem 1
   yeniden üretilebilirlik için istiyor, bende bu bilgi yok.

2. **Etiket üretim kuralının tamamı.** Hakem 3'ün "etiket üretimindeki hız
   limiti ile modele verilen aynı mı" sorusunu ölçerek cevapladım: **evet,
   aynı.** Kuralı (`v > 1,10 · v_max`) grafın düğüm özniteliğiyle elle
   uyguladım, 13 tip-1 pozitifin 13'ü de sağlıyor. Ayrıca o pozitiflerin 5'i
   imputasyonlu limitli segmentlerde — yani döngüsellik gerçek.
   **Ama kuralda ek bir koşul olmalı:** 128 pencere limiti aşıyor ama tip-1
   etiketi almamış. Süre koşulu mu, süreklilik mi? Tam kuralı yazarsanız
   tartışmayı kesinleştiririm. Diğer alt tipler için de aynı şey lazım.

3. **Alt tip tasarımı** — veriye baktım, taksonomide dokuz tip var ama
   **tip 6 hiç üretilmemiş**. Tip 3, 5, 7, 8 çok seyrek (testte 0–3 pencere).
   Bu tasarım kararı mıydı yoksa üretim parametresi sonucu mu? Hakem 3
   dokuz tipin tamamı için sayı istiyor, gerekçesini bilmemiz lazım.

4. **Bağımsız veri varyantları** — en büyük iş bu. Hakem 2 ve 3 farklı trafik
   akışı, gürültü seviyesi, anomali yaygınlığı ve şiddetiyle üretilmiş
   bağımsız veri kümeleri istiyor. Ben gürültü ve yaygınlığı mevcut veriden
   türetebiliyorum ama o perturbasyon analizi olur, bağımsız üretim değil.
   3–4 varyant yeterli olur diye düşünüyorum, formatı mevcutla aynı olsun.
   Zaman sıkıntımız yok, haftalar sürebilir, ben paralel çalışırım.

5. **İkinci yol ağı veya gerçek veri** — bu grup kararı. Üç hakem de istiyor.
   Kocaeli dışında ikinci bir OSM ağı üretebilir miyiz? Üretebilirsek kabul
   şansımız ciddi artar. Üretemezsek genelleme iddiasını tamamen çıkarıp
   sonuçları üretici koşullarına bağlamamız gerekiyor.

**Hocam size bir sorum var:** Hakem 3, STANet'in yayın durumunu soruyor.
Yayınlandı mı, hangi dergide, hangi statüde? Tablo 1'de ve atıflarda buna
göre düzeltme yapmamız gerekiyor.

**Sıra konusunda:** Sizin işiniz beni bloke etmiyor, ilk birkaç hafta
kendi kısmımla başlayabilirim, siz paralel çalışırsınız. Asıl bağımlılık
şurada: veri varyantları gelmezse (4. madde) o deneyler yapılamaz ve
2.1/3.5 maddelerini sadece yazıyla karşılarız.

Zaman sıkıntımız olmadığı için kapsamı geniş tutuyorum, yapabileceğimiz her
şeyi yapalım istiyorum. Bir eksiğimiz kalmasın.

**Başlıyorum:** Eşik düzeltmesini bekletmeden yapıyorum, çünkü sonucu diğer
her şeyin çerçevesini belirliyor. Dwell bulgusunun ayakta kalıp kalmadığını
çıkar çıkmaz yazarım, kararları o sayıyı görerek konuşuruz.

Ekteki dosyada her maddenin kimde olduğu, çalışma sırası ve bağımlılıklar
tablo hâlinde var.

Kolay gelsin.

---

## Kısa sürüm (WhatsApp için çok uzunsa)

Arkadaşlar merhaba,

Hakem yorumları geldi, üçünü de okudum ve iddiaları kodda kontrol ettim.
Kimse reddedilsin dememiş, hepsi düzeltme istiyor, 25 madde var. Detaylı
dosyayı ekliyorum.

Önce bir uyarı: elimizdeki yol haritası dosyası hakem numaralarını
karıştırmış, orada birinci hakem diye geçen aslında Hakem 2. Yanıt mektubu
hakem hakem yazıldığı için önemli, ekteki dosyada doğrusu var.

Dürüst olmam gereken kısım: Hakem 3 iki gerçek hata bulmuş, ikisi de bende.
Çapraz doğrulamada bütün fold'ların eşiklerinin ortalamasını alıp hepsine
uygulamışım, her fold kendi eşiğini kullanmalıydı. Etkisini ölçtüm, uzun
duraklama bulgusu +0,0335'ten +0,0074'e düşüyor. O bizim tek anlamlı pozitif
sonucumuzdu, muhtemelen anlamlılığını yitirecek. İkincisi graf seçiminde test
etiketlerini kullanmışım. İkisi de yeniden eğitim gerektirmiyor, birkaç günlük
iş.

25 maddenin 17'sini ekipten bir şey beklemeden tam, 5'ini kısmen
kapatabiliyorum. Siz hiçbir şey vermeseniz bile 22/25 kapanıyor.

Sizden beş şey isteyeceğim:

1. Markov geçiş olasılıkları nasıl seçildi
2. Etiket üretim kuralının tamamı. "Hız limiti aynı mı" sorusunu ölçerek
   cevapladım, evet aynı (13/13). Ama kuralda ek bir koşul olmalı, 128 pencere
   limiti aşıyor ama etiket almamış. Tam kural lazım
3. Tip 6 neden hiç üretilmemiş, tip 3/5/7/8 neden bu kadar seyrek
4. Farklı akış, gürültü, anomali yaygınlığı ve şiddetiyle 3–4 bağımsız veri
   varyantı
5. İkinci yol ağı üretebilir miyiz — grup kararı

Hocam, Hakem 3 STANet'in yayın durumunu soruyor, hangi dergide hangi statüde?

Sizin işiniz beni bloke etmiyor, paralel çalışabiliriz. Zaman sıkıntımız
olmadığı için kapsamı geniş tutuyorum. Eşik düzeltmesiyle hemen başlıyorum,
sonucu çıkınca yazarım.

Kolay gelsin.
