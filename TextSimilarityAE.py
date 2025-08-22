import numpy as np
import sqlite3
import re
import jellyfish
from rapidfuzz.distance import Levenshtein
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

def levenshtein_similarity(s1, s2):
    max_len = max(len(s1), len(s2)) # Maksimum uzunluk hesaplanıyor
    if max_len == 0: # Eğer her iki metin de boş ise benzerlik 1.0 olarak kabul edilir
        return 1.0
    return 1 - Levenshtein.distance(s1, s2) / max_len # Levenshtein benzerliği hesaplanıyor

def jaro_winkler_similarity(a: str, b: str) -> float:
    return jellyfish.jaro_winkler_similarity(a, b)

cos_weight = 0.85# Anlam ağırlığının oranını daha fazla verdim. Kur'anda anlam benzerliği daha önemli.
lev_weight = 0.15

conn = sqlite3.connect("results.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS search_results_yazir (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT,
    result_text TEXT,
    cos_sim REAL, 
    lev_sim REAL, 
    jaro_sim REAL,
    contributions TEXT,
    final_sim REAL,
    UNIQUE(query_text, result_text) 
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

        cos_val = 1.0 if t_norm == tj_norm else float(sims[j])# Cosine benzerlik değeri alınır. Eğer metinler eşitse 1.0, değilse cosine benzerlik değeri kullanılır.
        lev_val = levenshtein_similarity(t_norm, tj_norm)# Levenshtein benzerliği hesaplanır
        jaro_val = jaro_winkler_similarity(t_norm, tj_norm)  # Jaro-Winkler benzerliği hesaplanır

        cos_contrib = 0.7 * cos_val
        lev_contrib = 0.2 * lev_val
        jaro_contrib = 0.1 * jaro_val

        final_sim = cos_contrib + lev_contrib + jaro_contrib

        contrib_str = f"cos:{cos_contrib:.3f}|lev:{lev_contrib:.3f}|jaro:{jaro_contrib:.3f}"

        # a-b ve b-a tekrarını engellemek için sıralı ekleme yaptık
        qt, rt = sorted([t, texts[j]])
        c.execute(
            "INSERT OR IGNORE INTO search_results_yazir (query_text, result_text, cos_sim, lev_sim, jaro_sim, contributions, final_sim) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (qt, rt, cos_val,lev_val,jaro_val,contrib_str,final_sim)
        )

    if i % 100 == 0:
        print(f"{i}/{len(texts)} processed.")

conn.commit()
print("[10] Tüm sonuçlar kaydedildi.")

