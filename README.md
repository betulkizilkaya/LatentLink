# 🐾 Metin Benzerlik Arama Projesi

## 📄 Summary
Bu proje, **Sentence-BERT (SBERT)** ve **Otoencoder** kullanarak metinler arasındaki **anlam tabanlı benzerlikleri** bulur.  
Metinler sayısal vektörlere dönüştürülür, boyutu küçültülür ve en benzer metinler hızlıca keşfedilir. 💡  

## 📝 Description
Proje adımları:  

1. **Metinleri yükleme** → `.txt` dosyası veya Python listesi üzerinden metinler alınır.  
2. **SBERT ile embedding** → Cümleler çok boyutlu sayısal vektörlere dönüştürülür.  
3. **Otoencoder ile sıkıştırma** → Vektörler 64 boyuta indirgenerek daha hızlı benzerlik araması yapılır.  
4. **Cosine similarity hesaplama** → Metin çiftleri arasındaki benzerlik puanları çıkarılır.  
5. **SQLite veritabanı kaydı** → İlk 50 en benzer eşleşme `results.db` dosyasına kaydedilir.  
   - Birebir aynı cümlelere **1.0** benzerlik değeri atanır.  
   - **A-B ve B-A tekrarları** önlenir.  

## 🚀 Özellikler
- 📥 **SBERT ile çok dilli embedding**  
- 🔄 **Otoencoder ile sıkıştırma (768 → 64 boyut)**  
- 📊 **Cosine Similarity ile benzerlik puanlama**  
- 🗄 **SQLite veritabanına kayıt**  

## 📦 Kullanılan Kütüphaneler
- `numpy`  
- `sqlite3`  
- `sentence_transformers`  
- `tensorflow` (Keras)  
- `scikit-learn`  

## 🛠 Kurulum
```bash
pip install numpy sentence-transformers tensorflow scikit-learn
````

## 🖥 Kullanım

1. `TextSimilarityAE.py` dosyasını çalıştırın.  
2. Kod, metinleri işler, embedding üretir ve otoencoder modelini eğitir.  
3. Her cümle için en benzer 50 sonuç `results.db` veritabanındaki `search_results_yazir` tablosuna kaydedilir.  
