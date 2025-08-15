# 🐾 Metin Benzerlik Arama Projesi

## 📄 Summary
Bu proje, **Sentence-BERT (SBERT)** ve **Otoencoder** kullanarak metinler arasındaki anlam tabanlı benzerliği bulur.  
Metinler sayısal vektörlere dönüştürülür, boyutu küçültülür ve en benzer metinler hızlıca bulunur. 💡

## 📝 Description
Proje adımları:

1. **Metinleri yükleme**: Örnek veya kendi metinlerinizi Python listesi veya `.txt` dosyası ile ekleyebilirsiniz.  
2. **SBERT embedding**: Cümleler çok boyutlu sayısal vektörlere dönüştürülür.  
3. **Otoencoder ile sıkıştırma**: Vektör boyutu 64 boyuta düşürülerek daha hızlı arama sağlanır.  
4. **Cosine similarity hesaplama**: Metinler arasındaki benzerlik ölçülür.  
5. **SQLite veritabanı kaydı**: Arama sonuçları `results.db` dosyasında saklanır.  
6. **Search fonksiyonu**: Girilen sorguya en yakın cümleleri döndürür.  

## 🚀 Özellikler
- 📥 **SBERT ile Embedding**  
- 🔄 **Otoencoder ile Sıkıştırma (Embedding boyutu → 64 boyut)**  
- 📊 **Cosine Similarity ile benzerlik hesaplama**  
- 🗄 **SQLite veritabanına kayıt**  
- 🔍 **Arama fonksiyonu ile hızlı metin bulma**

## 📦 Kullanılan Kütüphaneler
- `numpy`  
- `sqlite3`  
- `sentence_transformers`  
- `tensorflow` (Keras)  
- `scikit-learn`  

## 🛠 Kurulum
```bash
pip install numpy sqlite3 sentence-transformers tensorflow scikit-learn
````

## 🖥 Kullanım

1. `TextSimilarityAE.py` dosyasını çalıştırın.
2. Kod, metinleri işler ve modeli eğitir.
3. `search("aranan metin", 3)` ile benzer metinleri bulun.
4. Sonuçlar hem ekranda hem de `results.db` veritabanında saklanır.
