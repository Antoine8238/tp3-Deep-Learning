import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt



# 1. Charger le dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Nombre de classes
NUM_CLASSES = 10
INPUT_SHAPE = x_train.shape[1:]  # (32, 32, 3)

print(f"\nDonnées d'entraînement: {x_train.shape}")
print(f"Labels d'entraînement: {y_train.shape}")
print(f"Données de test: {x_test.shape}")
print(f"Labels de test: {y_test.shape}")

# 2. Normaliser les valeurs des pixels à [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 3. Convertir les labels en format One-Hot Encoding
y_train = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

print(f"\nForme des données d'entrée: {INPUT_SHAPE}")
print(f"Forme des labels après conversion One-Hot: {y_train.shape}")
print(f"Exemple de label One-Hot: {y_train[0]}")

# Noms des classes CIFAR-10
class_names = ['avion', 'automobile', 'oiseau', 'chat', 'cerf', 
               'chien', 'grenouille', 'cheval', 'bateau', 'camion']