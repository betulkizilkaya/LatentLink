import sqlite3
import networkx as nx #Grafik kurmak ve kümeleri bulmak için.
import matplotlib.pyplot as plt #Grafikleri çizmek ve görselleştirmek için.
import math
import re
import os

output_dir = "kume_gorselleri" # Çıktı görsellerinin kaydedileceği klasör
os.makedirs(output_dir, exist_ok=True)  # klasör yoksa oluştur

conn = sqlite3.connect("results.db")
c = conn.cursor()

threshold = 0.7 # Benzerlik eşiği, 0.7 olarak ayarlandı
c.execute("SELECT query_text, result_text, final_sim FROM search_results_yazir WHERE final_sim >= ?", (threshold,))# Benzerlik eşiği 0.7 olan sonuçları alıyoruz
rows = c.fetchall()# rows değişkenine bu veriler yükleniyor.
conn.close()

def extract_ref(text):# Ayetleri yazmak yerine sure|ayet formatında yazmak için bir fonksiyon
    parts = text.split("|")
    if len(parts) >= 2:
        return f"{parts[0]}:{parts[1]}"
    return text

G = nx.Graph()# Boş bir grafik oluşturuyoruz
for a, b, sim in rows:
    if sim == 1.0:  # benzerlik 1 olanları atla. Gereksiz tekrarları engellemek için
        continue
    a_ref = extract_ref(a)# extract_ref fonksiyonu ile sure|ayet formatına çeviriyoruz
    b_ref = extract_ref(b)# extract_ref fonksiyonu ile sure|ayet formatına çeviriyoruz
    G.add_edge(a_ref, b_ref, weight=sim)# Grafiğe kenar ekliyoruz. a_ref ve b_ref düğümleri arasında sim ağırlığı ile bir kenar ekleniyor.

components = list(nx.connected_components(G))# Her kümenin düğümleri listeleniyor.
print(f"Toplam {len(components)} küme bulundu.")

two_node_components = [comp for comp in components if len(comp) == 2]# Sadece 2 düğümlü izole kümeleri filtreliyoruz.
print(f"Sadece 2 düğümlü izole kümelerin sayısı: {len(two_node_components)}")

def spiral_layout(nodes, spacing=1.5):# Spiraller şeklinde düzenleme fonksiyonu
    pos = {}# Düğüm konumlarını tutacak bir sözlük
    angle_step = 0.5# Açısal adım, her düğüm için açının ne kadar artacağını belirler
    radius_step = spacing # Her adımda artan yarıçap
    for i, node in enumerate(nodes):#i arttıkça düğümler spiral üzerinde dışa doğru açılır. Böylece düğümler üst üste binmeden daha okunabilir oluyor.
        angle = i * angle_step# Açıyı hesapla
        radius = radius_step * math.sqrt(i)# Yarıçapı hesapla, spiralin genişlemesini sağlar
        x = radius * math.cos(angle)# X koordinatını hesapla
        y = radius * math.sin(angle)# Y koordinatını hesapla
        pos[node] = (x, y)# Düğümün konumunu sözlüğe ekle
    return pos

for i, comp in enumerate(components, start=1):
    subG = G.subgraph(comp)#Her küme için ayrı bir alt grafik oluşturuluyor.
    pos = spiral_layout(list(subG.nodes()))#Spiral düzenleme ile düğümlerin konumu hesaplanıyor.

    plt.figure(figsize=(8, 8))#
    nx.draw(#Grafik çiziliyor: düğümler mavi, kenarlar gri, yazılar okunaklı.
        subG,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=2000,
        font_size=8,
        font_weight="bold",
        edge_color="gray"
    )

    plt.title(f"Küme {i} (boyut={len(comp)})", fontsize=12)# Başlığa kümenin numarası ve boyutu yazılıyor.
    plt.axis("off")
    # Görseli kaydet
    plt.savefig(os.path.join(output_dir, f"kume_{i}.png"), bbox_inches="tight")
    plt.close()

