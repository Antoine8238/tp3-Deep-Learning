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

## Implémentation d'un cnn basique


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

# Implémentation d'un bloc résiduel


def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    """
    Crée un bloc résiduel avec skip connection.
    
    Avantage: La skip connection permet au gradient de se propager 
    directement à travers le réseau, évitant le problème de vanishing gradient
    dans les réseaux très profonds.
    """
    # Chemin principal
    y = keras.layers.Conv2D(filters, kernel_size, strides=stride,
                           padding='same', activation='relu')(x)
    y = keras.layers.Conv2D(filters, kernel_size, padding='same')(y)
    
    # Chemin de skip connection
    if stride > 1 or x.shape[-1] != filters:
        # La skip connection doit adapter les dimensions si nécessaire
        x = keras.layers.Conv2D(filters, (1, 1), strides=stride)(x)
    
    # Addition du chemin skip avec le chemin principal
    z = keras.layers.Add()([x, y])
    z = keras.layers.Activation('relu')(z)
    
    return z

# Construire une petite architecture utilisant 3 blocs résiduels consécutifs
print("\nConstruction d'un modèle avec blocs résiduels...")

input_layer = keras.Input(shape=INPUT_SHAPE)
x = residual_block(input_layer, 32)
x = residual_block(x, 64, stride=2)
x = residual_block(x, 64)

# Ajouter les couches de classification
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(128, activation='relu')(x)
output = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

# Créer le modèle
resnet_model = keras.Model(inputs=input_layer, outputs=output)

# Compiler le modèle
resnet_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nArchitecture du ResNet simplifié:")
resnet_model.summary()

print("\n" + "="*70)
print("EXPLICATION DES BLOCS RÉSIDUELS")
print("="*70)
print("""
Avantage de la skip connection (addition de x à la sortie):
- Permet au gradient de se propager directement à travers le réseau
- Évite le problème de vanishing gradient dans les réseaux profonds
- Permet d'entraîner des réseaux beaucoup plus profonds (100+ couches)
- Le réseau peut apprendre l'identité plus facilement (si nécessaire)
- Améliore la convergence et les performances finales
""")


## Neural Style Transfer

# Charger le modèle VGG16 pré-entraîné
print("\nChargement du modèle VGG16 pré-entraîné...")
vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
vgg.trainable = False  # Important: le modèle VGG n'est pas entraîné ici

# Couches de contenu et de style pour l'extraction de features
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                'block4_conv1', 'block5_conv1']

def create_extractor(model, style_layers, content_layers):
    """
    Crée un modèle extracteur qui sort les activations des couches sélectionnées.
    """
    outputs = [model.get_layer(name).output 
               for name in style_layers + content_layers]
    return keras.Model(inputs=model.input, outputs=outputs)

extractor = create_extractor(vgg, style_layers, content_layers)

print("\nExtracteur créé avec succès!")
print(f"Couches de style: {style_layers}")
print(f"Couches de contenu: {content_layers}")

print("\n" + "="*70)
print("EXPLICATION DU TRANSFERT DE STYLE")
print("="*70)
print("""
Rôle des pertes (losses):

1. CONTENT LOSS (Perte de contenu):
   - Mesure la différence entre les features de contenu de l'image générée
     et celles de l'image de contenu originale
   - Assure que la structure/composition de l'image est préservée
   - Utilise les couches profondes du réseau (features de haut niveau)

2. STYLE LOSS (Perte de style):
   - Mesure la différence entre les statistiques de style (matrice de Gram)
     de l'image générée et celles de l'image de style
   - Capture les textures, couleurs, et motifs de l'image de style
   - Utilise plusieurs couches (features à différentes échelles)

3. OPTIMISATION:
   - On optimise les PIXELS de l'image générée (pas les poids du réseau!)
   - Objectif: minimiser content_loss + α * style_loss
   - α contrôle le compromis entre contenu et style
""")


## Visualisation de résultats 

def plot_training_history(history):
    """Affiche les courbes d'apprentissage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Précision
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Précision du modèle')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Précision')
    ax1.legend()
    ax1.grid(True)
    
    # Perte
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Perte du modèle')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perte')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\nGraphiques sauvegardés dans 'training_history.png'")
    
plot_training_history(history)

## Prédiction et visualisation


def visualize_predictions(model, x_test, y_test, class_names, num_images=10):
    
    
    # Prédictions
    predictions = model.predict(x_test[:num_images])
    
    # Créer la figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_images):
        # Afficher l'image
        axes[i].imshow(x_test[i])
        
        # Trouver les indices
        pred_idx = np.argmax(predictions[i])
        true_idx = np.argmax(y_test[i])
        
        # Noms des classes
        pred_label = class_names[pred_idx]
        true_label = class_names[true_idx]
        
        # Vérifier si correct
        if pred_idx == true_idx:
            color = 'green'
        else:
            color = 'red'
        
        # Titre
        axes[i].set_title(f'Pred: {pred_label}\nTrue: {true_label}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()
    print("Image sauvegardée!")

# Appel
visualize_predictions(model, x_test, y_test, class_names)