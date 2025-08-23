# 🐾 Metin Benzerlik Arama Projesi

## 📄 Summary
Bu proje, Sentence-BERT (SBERT), Otoencoder ve çeşitli string benzerlik algoritmaları kullanarak metinler arasındaki anlam ve karakter tabanlı benzerlikleri bulur.

Metinler sayısal vektörlere dönüştürülür, boyutu küçültülür ve farklı benzerlik metrikleri ile analiz edilerek en yakın eşleşmeler keşfedilir.

## 📝 Description
Proje adımları:  

1. **Metinleri yükleme**  
   `.txt` dosyası veya Python listesi üzerinden metinler alınır.  

2. **SBERT ile embedding**  
   Cümleler çok boyutlu sayısal vektörlere dönüştürülür.  

3. **Otoencoder ile sıkıştırma**  
   Vektörler 768 boyuttan **64 boyuta** indirgenerek daha hızlı benzerlik araması yapılır.  

4. **Benzerlik algoritmaları**  
- 📐 **Cosine Similarity** → Anlam tabanlı benzerlik  
- ✍️ **Levenshtein Distance** → Karakter bazlı düzenleme mesafesi  
- 🔗 **Jaro-Winkler Distance** → Küçük yazım farklılıklarına toleranslı ölçüm  

5. **Nihai skor hesaplama**  
      ```python
      final_sim = 0.7 * cosine + 0.2 * levenshtein + 0.1 * jaro_winkler
      ```
6. **SQLite veritabanı kaydı**
İlk 50 en benzer eşleşme results.db dosyasına kaydedilir.
- Her algoritmanın ham skorları ayrı kolonlarda tutulur.
- Ayrıca katkı oranları tek kolon içinde şu formatta saklanır:
   ```python
   cos:0.637|lev:0.120|jaro:0.075
   ```
- A-B ve B-A tekrarları engellenir.

## Küme Analizi ve Görselleştirme

- Benzerlik skorlarına göre kümeler oluşturulur.
- Her küme, NetworkX kullanılarak grafik olarak çizilir.
- Düğümler spiral düzenlemede konumlandırılır, kenarlar benzerlik skorunu temsil eder.
- 2 düğümlü izole kümeler ayrı olarak analiz edilir.
- Her kümenin görseli `kume_gorselleri` klasörüne kaydedilir.

## Özellikler

- SBERT ile çok dilli embedding
- Otoencoder ile sıkıştırma (768 → 64 boyut)
- Cosine, Levenshtein, Jaro-Winkler benzerlikleri
- SQLite veritabanına kayıt (ham skor + katkılar + final skor)
- Algoritma katkılarının analiz edilebilir olması
- Küme analizi ve spiral görselleştirme

## ⚡Hız Karşılaştırması
Benzerlik algoritmalarının hız sıralaması:
- Cosine Similarity → En hızlı (vektör matematiği ile çalışır).
- Jaro-Winkler → Orta seviye hız.
- Levenshtein → En yavaş (karakter karşılaştırmaları daha yoğun işlem gerektirir).

## 📦 Kullanılan Kütüphaneler

- `numpy`
- `sqlite3`
- `sentence_transformers`
- `tensorflow (Keras)`
- `scikit-learn`
- `rapidfuzz (Levenshtein için)`
- `jellyfish (Jaro-Winkler için)`
- `networkx (küme analizi ve grafikler)`
- `matplotlib (grafik çizimi)`

## 🛠 Kurulum
      pip install numpy sentence-transformers tensorflow scikit-learn rapidfuzz jellyfish networkx matplotlib
      
## 🖥 Kullanım
1. Metin benzerlik analizi
- TextSimilarityAE.py dosyasını çalıştırın.
   - Metinleri işler, embedding üretir ve otoencoder modelini eğitir.
   - Her cümle için en benzer 50 sonuç, tüm algoritma skorları ve katkı oranları ile birlikte results.db veritabanındaki search_results_yazir tablosuna kaydedilir.

2. Küme analizi ve görselleştirme
- cluster_analysis.py dosyasını çalıştırın.
   - Benzerlik eşik değerini threshold değişkeninden ayarlayın.
   - Çalıştırıldığında her küme için görseller oluşturulur ve kume_gorselleri klasörüne kaydedilir.


