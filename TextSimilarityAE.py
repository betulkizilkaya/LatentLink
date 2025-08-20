import numpy as np
import sqlite3
import re
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity

print("[1] tr.yazir.txt dosyası yükleniyor...")
with open("tr.yazir.txt", "r", encoding="utf-8") as f:#Dosya erişimi
    texts = [line.strip() for line in f if line.strip()]

print(f"[1] Toplam {len(texts)} metin yüklendi.")

print("[2] SBERT modeli yükleniyor (ilk defa ise indirme yapılacak)...")
sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")# SBERT modeli yükleniyor
print("[2] SBERT modeli yüklendi.")

print("[3] Embedding hesaplanıyor...")
E = sbert.encode(texts, convert_to_numpy=True).astype("float32")# Embedding hesaplanıyor
print(f"[3] Embedding boyutu: {E.shape}")

print("[4] Otoencoder tanımlanıyor...")
in_dim = E.shape[1]# Giriş boyutu
latent_dim = 64# Gizli katman boyutu yani embedding boyutu
h_dim = max(128, latent_dim * 2)# Gizli katman boyutu, latent boyutunun 2 katı veya 128, hangisi büyükse o

inp = Input(shape=(in_dim,))# Giriş katmanı
h = Dropout(0.1)(inp)# Dropout katmanı
h = Dense(h_dim, activation="relu")(h)# Gizli katman yani encoder katmanı
z = Dense(latent_dim, activation="linear")(h)  # Latent katman yani sıkıştırılmış embedding katmanı
h2 = Dense(h_dim, activation="relu")(z)
out = Dense(in_dim, activation="linear")(h2)

ae = Model(inp, out)# Otoencoder modeli
enc = Model(inp, z)# Encoder modeli
ae.compile(optimizer=Adam(1e-3), loss="mse")# Model derleme

print("[5] Model eğitimi başlıyor...")
ae.fit(E, E, epochs=30, batch_size=16, verbose=1)#Otoencoder eğitimi
print("[5] Eğitim tamamlandı.")

print("[6] L2 normalize ve cosine similarity hesaplanıyor...")
def l2n(m):
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)#Vektörlerin normunu 1 yapar

Z = l2n(enc.predict(E))# Encoder'dan latent katmanı alınıyor
S = cosine_similarity(Z)# Cosine benzerlik hesaplanıyor

conn = sqlite3.connect("results.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS search_results_yazir (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT,
    result_text TEXT,
    similarity REAL,
    UNIQUE(query_text, result_text) # Tekrar eden kayıtları engellemek için UNIQUE kısıtlaması ekleniyor
)
""")

conn.commit()

print("[7] Eski sonuçlar temizleniyor...")#Her çalıştırmada üst üste kayıt yapmasın diye önceki sonuçlar temizleniyor
c.execute("DELETE FROM search_results_yazir")
conn.commit()

def normalize_text(s: str)-> str:
    """Başındaki 'sayı|sayı|' formatını siler."""
    return re.sub(r'^\d+\|\d+\|', '', s).strip()


print("[10] Her cümle için en benzer 50 sonuç DB'ye kaydediliyor...")
for i, t in enumerate(texts):
    sims = S[i]# Benzerlik matrisinden i. satır alınır
    idxs = np.argsort(sims)[::-1][1:51]#Sıralama ters çevrilir ve en yüksek benzerlik değerlerine sahip 50 indeks alınır (Kendisini atlar).
    t_norm = normalize_text(t)# Normalizasyon işlemi yapılır

    for j in idxs:
        if j <= i:  # a-b ve b-a tekrarını engelle
            continue

        tj_norm = normalize_text(texts[j])
        sim_val = 1.0 if t_norm == tj_norm else float(sims[j])# Benzerlik değeri hesaplanır. Eğer metinler eşitse 1.0, değilse cosine benzerlik değeri kullanılır.

        # a-b ve b-a tekrarını engellemek için sıralı ekleme yaptık
        qt, rt = sorted([t, texts[j]])
        c.execute(
            "INSERT OR IGNORE INTO search_results_yazir (query_text, result_text, similarity) VALUES (?, ?, ?)",
            (qt, rt, sim_val)
        )

    if i % 100 == 0:
        print(f"{i}/{len(texts)} processed.")

conn.commit()
print("[10] Tüm sonuçlar kaydedildi.")

