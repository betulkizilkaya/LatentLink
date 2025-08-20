import numpy as np
import sqlite3
import re
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity

# 1 - Metinleri yükleme
print("[1] tr.yazir.txt dosyası yükleniyor...")
with open("tr.yazir.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

print(f"[1] Toplam {len(texts)} metin yüklendi.")

# 2 - SBERT modeli yükleme
print("[2] SBERT modeli yükleniyor (ilk defa ise indirme yapılacak)...")
sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("[2] SBERT modeli yüklendi.")

# 3 - Embedding hesaplama
print("[3] Embedding hesaplanıyor...")
E = sbert.encode(texts, convert_to_numpy=True).astype("float32")
print(f"[3] Embedding boyutu: {E.shape}")

# 4 - Otoencoder tanımlama
print("[4] Otoencoder tanımlanıyor...")
in_dim = E.shape[1]
latent_dim = 64
h_dim = max(128, latent_dim * 2)

inp = Input(shape=(in_dim,))
h = Dropout(0.1)(inp)
h = Dense(h_dim, activation="relu")(h)
z = Dense(latent_dim, activation="linear")(h)  # Latent katman
h2 = Dense(h_dim, activation="relu")(z)
out = Dense(in_dim, activation="linear")(h2)

ae = Model(inp, out)# Otoencoder modeli
enc = Model(inp, z)# Encoder modeli
ae.compile(optimizer=Adam(1e-3), loss="mse")# Model derleme

# 5 - Model eğitimi
print("[5] Model eğitimi başlıyor...")
ae.fit(E, E, epochs=30, batch_size=16, verbose=1)
print("[5] Eğitim tamamlandı.")

# 6 - L2 normalize ve cosine similarity hesaplama
print("[6] L2 normalize ve cosine similarity hesaplanıyor...")
def l2n(m):
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)

Z = l2n(enc.predict(E))
S = cosine_similarity(Z)

# 7 - SQLite veritabanı oluşturma
conn = sqlite3.connect("results.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS search_results_yazir (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT,
    result_text TEXT,
    similarity REAL,
    UNIQUE(query_text, result_text)
)
""")

conn.commit()

#Tekrar çalıştırıldığında eski veriler kalmasın.
print("[7] Eski sonuçlar temizleniyor...")
c.execute("DELETE FROM search_results_yazir")
conn.commit()

def normalize_text(s: str)-> str:
    """Başındaki 'sayı|sayı|' formatını siler."""
    return re.sub(r'^\d+\|\d+\|', '', s).strip()

# Her cümle için en benzer 50 sonuç DB'ye kaydediliyor
print("[10] Her cümle için en benzer 50 sonuç DB'ye kaydediliyor...")

for i, t in enumerate(texts):
    sims = S[i]
    idxs = np.argsort(sims)[::-1][1:51]  # en yüksek 50 benzerlik, kendisi atlandı
    t_norm = normalize_text(t)

    for j in idxs:
        if j <= i:  # a-b ve b-a tekrarını engelle
            continue

        tj_norm = normalize_text(texts[j])
        sim_val = 1.0 if t_norm == tj_norm else float(sims[j])

        # a-b ve b-a tekrarını engellemek için sıralı ekleme
        qt, rt = sorted([t, texts[j]])
        c.execute(
            "INSERT OR IGNORE INTO search_results_yazir (query_text, result_text, similarity) VALUES (?, ?, ?)",
            (qt, rt, sim_val)
        )

    if i % 100 == 0:
        print(f"{i}/{len(texts)} processed.")

conn.commit()
print("[10] Tüm sonuçlar kaydedildi.")

