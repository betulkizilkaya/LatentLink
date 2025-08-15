import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity

# Örnek veri seti ekledim.
print("[1] Metinler yükleniyor...")
texts = [
    # Hayvanlar
    "Kedi masanın üstünde uyuyor.",
    "Masanın üzerinde bir kedi uyumakta.",
    "Köpek bahçede koşuyor.",
    "Köpek havlayarak bahçede oynuyor.",
    "Kuş gökyüzünde kanat çırpıyor.",
    "Serçe pencerenin önünde öterek bekliyor.",
    # Hava durumu
    "Bugün hava çok güzel.",
    "Dışarıda güneş parlıyor.",
    "Yağmur yavaş yavaş yağmaya başladı.",
    "Hava kapalı ve rüzgar esiyor.",
    # Yemek
    "Sabah kahvaltısında peynir ve zeytin vardı.",
    "Akşam yemeğinde makarna yaptım.",
    "Pizza fırında pişiyor.",
    "Meyve salatası çok taze görünüyor.",
    # Seyahat
    "Geçen yaz Antalya'ya tatile gittik.",
    "Uçak sabah saat yedide kalktı.",
    "Tren istasyondan ayrıldı.",
    "Otobüs yolda arıza yaptı.",
    # Eğitim
    "Matematik sınavına çalışmam gerekiyor.",
    "Bugün okulda yeni konular işledik.",
    "Kütüphanede sessizce kitap okudum.",
    "Öğretmen ödevi yarına teslim etmemizi istedi.",
    # Teknoloji
    "Telefonumun şarjı bitti.",
    "Bilgisayar çok yavaş çalışıyor.",
    "Yeni yazılım güncellemesi geldi.",
    "Tablet ekranı kırıldı.",
    # Günlük hayat
    "Marketten süt ve ekmek aldım.",
    "Parkta yürüyüş yaptık.",
    "Evde temizlik yapıyorum.",
    "Bahçeye çiçek diktim."
]

print("[2] SBERT modeli yükleniyor (ilk defa ise indirme yapılacak)...")
sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")#Embeding modeli yükle
print("[2] SBERT modeli yüklendi.")


print("[3] Embedding hesaplanıyor...")
E = sbert.encode(texts, convert_to_numpy=True).astype("float32")#Her cümleyi E matrisinde bir vektör olarak saklar.
print(f"[3] Embedding boyutu: {E.shape}")

print("[4] Otoencoder tanımlanıyor...")
#SBERT embedding’lerini sıkıştırıp (64 boyuta) daha küçük, daha anlamlı bir vektör elde etme için bir otoencoder modeli tanımlıyoruz.
in_dim     = E.shape[1]# Giriş boyutu, embedding boyutu
latent_dim = 64
h_dim      = max(128, latent_dim * 2)# Gizli katman boyutu, latent boyutunun iki katı veya 128, hangisi büyükse onu kullanır.

inp = Input(shape=(in_dim,))
h   = Dropout(0.1)(inp)#%10 düğümü rastgele kapat, overfitting’i önler.
h   = Dense(h_dim, activation="relu")(h)# Gizli katman, 128 düğüm
z   = Dense(latent_dim, activation="linear")(h)# Latent katman, 64 düğüm
h2  = Dense(h_dim, activation="relu")(z)# Latent katmandan sonra tekrar gizli katman, 128 düğüm
out = Dense(in_dim, activation="linear")(h2)# Çıkış katmanı, orijinal embedding boyutuna geri döner.

ae = Model(inp, out)# Tüm otoencoder modeli.
enc = Model(inp, z)# Encoder modeli, sadece giriş ve latent katmanları içerir.
ae.compile(optimizer=Adam(1e-3), loss="mse")# Otoencoder’i MSE kaybı ile derle.Öğrenme hızı 0.001 ile Adam optimizasyon

print("[5] Model eğitimi başlıyor...")
ae.fit(E, E, epochs=30, batch_size=16, verbose=1)# Otoencoder’i 30 epoch boyunca eğit, her batch 16 örnek içerir.
print("[5] Eğitim tamamlandı.")

print("[6] L2 normalize ve cosine similarity hesaplanıyor...")
def l2n(m):#Vektörleri normalize eder (uzunluğu 1 olur).
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)# L2 normalizasyon, her vektörü kendi normuna böler.

Z = l2n(enc.predict(E))#Encoder’dan gelen sıkıştırılmış embedding’ler.
S = cosine_similarity(Z)# Cosine benzerlik matrisini hesaplar, her vektörün diğerleriyle olan benzerliğini gösterir.

conn = sqlite3.connect("results.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS search_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT,
    result_text TEXT,
    similarity REAL
)
""")
conn.commit()

def search(query, topk=3):# Sorgu metni ile en benzer metinleri bulur.
    q = sbert.encode([query], convert_to_numpy=True).astype("float32")# Sorgu metnini embedding’e dönüştür.
    qz = l2n(enc.predict(q))# Sorgu embedding’ini encoder ile sıkıştır ve normalize et.
    sims = (Z @ qz.T).ravel()# Cosine benzerlik hesapla, Z ile sorgu embedding’i arasındaki benzerliği bulur.
    idxs = sims.argsort()[::-1][:topk]# En yüksek benzerlik değerlerine göre indeksleri alır.
    results = [(texts[i], float(sims[i])) for i in idxs]# En benzer metinleri ve benzerlik değerlerini alır.

    for res_text, sim in results:
        c.execute("INSERT INTO search_results (query_text, result_text, similarity) VALUES (?, ?, ?)",
                  (query, res_text, sim))
    conn.commit()

    return results

# Test kısmı
print(search("Masanın üstünde bir kedi var.", 3))#metinler arasında en benzer 3 cümleyi bulur.