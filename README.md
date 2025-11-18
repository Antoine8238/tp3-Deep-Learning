# TP3 : Réseaux de Neurones Convolutifs et Vision par Ordinateur

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)


> **Département de Génie Informatique, ENSPY**  
> Module : Deep Learning - 5GI  
> Année Académique : 2024-2025

##  Table des matières

- [Description](#-description)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Structure du projet](#-structure-du-projet)
- [Utilisation](#-utilisation)
- [Résultats](#-résultats)
- [Auteurs](#-auteurs)

---

##  Description

Ce TP implémente des **Réseaux de Neurones Convolutifs (CNNs)** pour la vision par ordinateur. Il couvre :

-  **Partie 1** : Fondamentaux des CNNs (convolution, pooling, préparation CIFAR-10)
-  **Partie 2** : Implémentation d'un CNN basique et de blocs résiduels (ResNets)
-  **Partie 3** : Applications avancées (segmentation U-Net, détection d'objets, style transfer)

**Dataset** : CIFAR-10 (60,000 images 32×32, 10 classes)

**Objectifs** :
- Comprendre les opérations de convolution et pooling
- Construire et entraîner des CNNs pour la classification d'images
- Maîtriser les skip connections et ResNets
- Explorer des applications avancées en vision par ordinateur

---

##  Prérequis

- **Python** : 3.8 ou supérieur
- **Système** : Windows, macOS ou Linux
- **GPU** (optionnel) : Recommandé pour accélérer l'entraînement

---

##  Installation

### 1. Cloner le repository

```bash
git clone https://github.com/votre-username/tp3-cnn-vision.git
cd tp3-cnn-vision
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Contenu de `requirements.txt` :**
```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
```

### 4. Vérifier l'installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

---

## 📁 Structure du projet

```
tp3-cnn-vision/
│
├── cnn_classification.py    # Code principal du TP
├── requirements.txt          # Dépendances Python

│
├── results/                  # Résultats générés
│   ├── training_history.png  # Courbes d'apprentissage
│   ├── predictions.png       # Visualisations des prédictions



---

## 🚀 Utilisation



```bash
python cnn_classification.py
```

**Durée estimée** : 15-30 minutes (selon votre machine)

### Exécution par parties

Si vous souhaitez exécuter progressivement, commentez les sections non désirées dans le code.

#### Partie 1 : Préparation des données
```python
# Décommenter uniquement la Partie 1
python cnn_classification.py
```

#### Partie 2 : CNN basique
```python
# Décommenter Partie 1 + Partie 2.1
python cnn_classification.py
```

#### Partie 3 : ResNets
```python
# Décommenter Partie 1 + Partie 2.2
python cnn_classification.py
```

### Mode interactif (pour tests)

```bash
python -i cnn_classification.py
```

Les variables restent en mémoire après exécution. Vous pouvez ensuite :

```python
>>> model.summary()  # Voir l'architecture
>>> predictions = model.predict(x_test[:5])  # Tester
>>> plt.imshow(x_test[0])  # Visualiser
```

---

## 📊 Résultats

### CNN Basique

| Métrique | Valeur |
|----------|--------|
| **Précision (Train)** | ~70% |
| **Précision (Validation)** | ~65% |
| **Précision (Test)** | ~65% |
| **Paramètres** | ~1.2M |
| **Temps d'entraînement** | ~10-15 min (10 epochs) |

**Architecture** :
- Conv2D (32) + MaxPooling
- Conv2D (64) + MaxPooling
- Flatten + Dense (512) + Dense (10)

### ResNet Simplifié

| Métrique | Valeur |
|----------|--------|
| **Paramètres** | ~151k |
| **Profondeur** | 3 blocs résiduels |
| **Avantage** | Convergence plus rapide |

**Courbes d'apprentissage** : Voir `results/training_history.png`

**Exemples de prédictions** : Voir `results/predictions.png`

---

## 🎓 Concepts clés implémentés

### 1. Convolution et Pooling
- **Convolution** : Extraction de features avec filtres 3×3
- **MaxPooling** : Réduction dimensionnelle 2×2
- **Padding='same'** : Conservation des dimensions spatiales

### 2. Blocs Résiduels (ResNets)
```python
H(x) = F(x) + x  # Skip connection
```
-  Évite le vanishing gradient
-  Permet des réseaux profonds (100+ couches)
-  Meilleure convergence

### 3. Applications avancées

#### Segmentation (U-Net)
- Sortie : Carte de segmentation pixel par pixel
- Upsampling : Reconstruction de la résolution

#### Détection (Bounding Boxes)
- Format : (x, y, w, h)
- Double tête : Classification + Régression

#### Style Transfer (VGG16)
- Content Loss : Préserve la structure
- Style Loss : Capture textures (matrice de Gram)
- Optimisation : Sur les pixels, pas les poids

---

##  Améliorations possibles

- [ ] Data Augmentation (rotation, flip, zoom)
- [ ] Dropout et Batch Normalization
- [ ] ResNet plus profond (ResNet-50)
- [ ] Implémentation complète de U-Net
- [ ] YOLO pour détection temps réel
- [ ] Style Transfer avec images personnelles

---

##  Dépannage

### Erreur : "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### Erreur : Mémoire insuffisante
Réduire le batch size dans le code :
```python
batch_size = 32  # Au lieu de 64
```

### Images floues dans les visualisations
C'est normal ! CIFAR-10 contient des images 32×32 (très petites). Ajoutez :
```python
plt.imshow(image, interpolation='nearest')
```

### Entraînement trop lent
- Utiliser un GPU si disponible
- Réduire le nombre d'epochs
- Utiliser un sous-ensemble des données

---

##  Auteur

**Étudiant** : [Antoine Emmanuel ESSOMBA ESSOMBA] - Matricule [23P750]

**Encadrant** :
 Dr. Louis Fippo Fitime


**Institution** : École Nationale Supérieure Polytechnique de Yaoundé (ENSPY)  
**Département** : Génie Informatique

---

## 📚 Références

1. **ResNets** : He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
2. **U-Net** : Ronneberger, O., et al. (2015). *U-Net for Biomedical Image Segmentation*. MICCAI.
3. **Style Transfer** : Gatys, L. A., et al. (2016). *Image Style Transfer Using CNNs*. CVPR.
4. **CIFAR-10** : Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*.
5. **Documentation** : [TensorFlow/Keras](https://www.tensorflow.org/)

---



---

## 🤝 Contact

Pour toute question ou suggestion :
- **Email** : essombaantoine385@gmail.com


---

<div align="center">
  <b>⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐</b>
</div>
