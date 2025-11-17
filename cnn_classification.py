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

##


def build_basic_cnn(input_shape, num_classes):
    """
    Construit un CNN basique pour la classification d'images.
    
    Architecture:
    - Conv2D (32 filtres) + MaxPooling
    - Conv2D (64 filtres) + MaxPooling
    - Flatten
    - Dense (512 unités)
    - Dense (num_classes unités, sortie)
    """
    model = keras.Sequential([
        # Couche convolutive 1: 32 filtres, taille 3x3, activation ReLU
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                           input_shape=input_shape),
        # Couche de pooling 1: Max Pooling 2x2
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Couche convolutive 2: 64 filtres, taille 3x3, activation ReLU
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        # Couche de pooling 2: Max Pooling 2x2
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Couche Flatten pour passer aux couches denses
        keras.layers.Flatten(),
        
        # Couche Dense 1: 512 unités, activation ReLU
        keras.layers.Dense(512, activation='relu'),
        # Couche de sortie: num_classes unités, activation Softmax
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Construire le modèle
model = build_basic_cnn(INPUT_SHAPE, NUM_CLASSES)

# Compiler le modèle
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Afficher l'architecture du modèle
print("\nArchitecture du CNN basique:")
model.summary()

# Entraîner le modèle
print("\nEntraînement du modèle...")
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.1,  # 10% des données d'entraînement pour la validation
    verbose=1
)

# Évaluer le modèle sur les données de test
print("\n" + "="*70)
print("ÉVALUATION DU MODÈLE")
print("="*70)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nPerte sur le test: {test_loss:.4f}")
print(f"Précision sur le test: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")