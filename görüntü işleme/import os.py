import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Veri seti dizini ve görüntü boyutu
data_dir = r'C:\Users\sinem\Downloads\archive_extracted\train'
img_size = 128  # Görüntü boyutu

def create_data(data_dir, img_size):
    data = []
    for i in range(12):  # 0'dan 11'e kadar klasörler
        path = os.path.join(data_dir, str(i))
        class_num = i  # Etiket
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)  # Görüntüyü renkli olarak oku
                new_array = cv2.resize(img_array, (img_size, img_size))  # Görüntüyü yeniden boyutlandır
                data.append([new_array, class_num])  # Görüntü ve etiketi ekle
            except Exception as e:
                pass  # Hata varsa geç
    return data

data = create_data(data_dir, img_size)

# Verileri karıştır
random.shuffle(data)

# Özellik ve etiketleri ayır
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

# Verileri numpy array'e dönüştür ve normalize et
X = np.array(X).reshape(-1, img_size, img_size, 3)  # Renkli görüntüler olduğu için 3 kanalı kullanıyoruz
X = X / 255.0  # Normalizasyon
y = np.array(y)

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim ve test setlerinin boyutlarını yazdır
print(f'Eğitim seti boyutu: {X_train.shape}')
print(f'Test seti boyutu: {X_test.shape}')

# Kedi türlerinin isimleri
cat_breeds = [
    'American Shorthair', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Egyptian Mau', 
    'Maine Coon', 'Persian', 'Ragdoll', 'Russian Blue', 'Siamese',
    'Sphynx'
]

# İlk birkaç görüntüyü tür adlarıyla görselleştir
for i in range(5):
    plt.figure(f'Kedi Türleri {i+1}')  # İsimleri "Kedi Türleri" olarak değiştirerek numaralandırma yapar
    plt.imshow(X_train[i])
    plt.title(f'Tür: {cat_breeds[y_train[i]]}')
    plt.show()



